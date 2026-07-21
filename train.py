"""Train and sample a line-only SVG symbol generator.

The module intentionally keeps the complete pipeline in one importable file:
raster preprocessing, a conditional beta-VAE, skeleton graph tracing, strict SVG
serialization, metric-based novelty filtering, checkpointing, and the CLI.

PyTorch is optional at import time so the geometry and validation utilities can
still be used before a device-appropriate Torch wheel has been installed.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import tempfile
import time
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:  # Preferred, but pure NumPy/Pillow fallbacks keep validation importable.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised only in minimal environments.
    cv2 = None

try:
    from skimage.morphology import skeletonize as _skimage_skeletonize  # type: ignore
except Exception:  # pragma: no cover
    _skimage_skeletonize = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - geometry-only installs are supported.
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object


SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SVG_NS = "http://www.w3.org/2000/svg"
NEIGHBORS_8 = tuple(
    (dy, dx)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dy == 0 and dx == 0)
)


class SymbolGeneratorError(RuntimeError):
    """Expected user-facing failure in the symbol pipeline."""


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 128
    margin: int = 12
    max_source_stroke_width: float = 12.0
    min_component_pixels: int = 3
    max_input_pixels: int = 40_000_000
    filled_policy: str = "outline"

    def validate(self) -> None:
        if self.image_size < 32 or self.image_size % 16:
            raise ValueError("image_size must be at least 32 and divisible by 16")
        if not 0 <= self.margin < self.image_size // 3:
            raise ValueError("margin must be non-negative and smaller than one third of image_size")
        if self.max_source_stroke_width <= 0:
            raise ValueError("max_source_stroke_width must be positive")
        if self.filled_policy not in {"outline", "reject"}:
            raise ValueError("filled_policy must be 'outline' or 'reject'")


@dataclass
class ProcessedSymbol:
    source: str
    line_mask: np.ndarray
    render_mask: np.ndarray
    normalized_gray: np.ndarray
    stroke_width: float
    conversion: dict[str, Any] = field(default_factory=dict)

    @property
    def mask(self) -> np.ndarray:
        """Compatibility alias for the canonical line mask."""

        return self.line_mask


@dataclass
class GraphStats:
    components: int
    endpoints: int
    junctions: int
    cycles: int
    path_count: int
    total_length: float
    bbox_area: float
    length_density: float

    def as_vector(self) -> np.ndarray:
        return np.asarray(
            [self.components, self.endpoints, self.junctions, self.cycles],
            dtype=np.float32,
        )


@dataclass
class TracedGraph:
    paths: list[np.ndarray]
    closed: list[bool]
    stats: GraphStats
    mask: np.ndarray


@dataclass
class QualityResult:
    valid: bool
    reasons: list[str]
    metrics: dict[str, float]


@dataclass
class NoveltyResult:
    status: str
    similarity: float
    nearest_index: int | None
    nearest_name: str | None
    components: dict[str, float]
    reason: str
    transformed_similarity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def require_torch() -> None:
    if torch is None:
        raise SymbolGeneratorError(
            "PyTorch is required for train/generate. Install the appropriate CPU or CUDA "
            "wheel from https://pytorch.org/get-started/locally/."
        )


def resolve_device(requested: str) -> "torch.device":
    require_torch()
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SymbolGeneratorError(
            "CUDA was requested, but this Torch build cannot use CUDA. Install a CUDA-enabled "
            "Torch wheel or pass --device cpu."
        )
    if requested not in {"cpu", "cuda"}:
        raise SymbolGeneratorError("--device must be auto, cpu, or cuda")
    return torch.device(requested)


def discover_images(root: str | Path, excluded: Iterable[str | Path] = ()) -> list[Path]:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise SymbolGeneratorError(f"Dataset directory does not exist: {root_path}")
    excluded_paths = [Path(item).resolve() for item in excluded]
    paths: list[Path] = []
    for path in root_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        resolved = path.resolve()
        if any(resolved == item or item in resolved.parents for item in excluded_paths):
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: p.as_posix().casefold())


def _load_grayscale(source: str | Path | Image.Image | np.ndarray, max_pixels: int) -> tuple[np.ndarray, str]:
    if isinstance(source, np.ndarray):
        array = np.asarray(source)
        source_name = "<array>"
        if array.ndim == 2:
            gray = array
        elif array.ndim == 3 and array.shape[2] in {3, 4}:
            image = Image.fromarray(array.astype(np.uint8))
            return _load_grayscale(image, max_pixels)
        else:
            raise SymbolGeneratorError("Array images must have shape HxW, HxWx3, or HxWx4")
        if np.issubdtype(gray.dtype, np.floating):
            peak = float(np.nanmax(gray)) if gray.size else 0.0
            gray = np.clip(gray * (255.0 if peak <= 1.0 else 1.0), 0, 255)
        return gray.astype(np.uint8), source_name

    close_after = False
    if isinstance(source, Image.Image):
        image = source.copy()
        source_name = getattr(source, "filename", "<PIL image>") or "<PIL image>"
    else:
        path = Path(source)
        source_name = str(path)
        try:
            image = Image.open(path)
            close_after = True
        except Exception as exc:
            raise SymbolGeneratorError(f"Could not read image {path}: {exc}") from exc

    try:
        image = ImageOps.exif_transpose(image)
        if image.width * image.height > max_pixels:
            raise SymbolGeneratorError(
                f"Image is too large ({image.width}x{image.height}); limit is {max_pixels:,} pixels"
            )
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(white, rgba).convert("L")
        else:
            image = image.convert("L")
        return np.asarray(image, dtype=np.uint8), source_name
    finally:
        if close_after:
            image.close()


def otsu_threshold(gray: np.ndarray) -> int:
    values = np.asarray(gray, dtype=np.uint8)
    if values.size == 0:
        raise SymbolGeneratorError("Image contains no pixels")
    lo, hi = int(values.min()), int(values.max())
    if lo == hi:
        raise SymbolGeneratorError("Image is blank or uniformly filled")
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probability = hist / hist.sum()
    omega = np.cumsum(probability)
    means = np.cumsum(probability * np.arange(256))
    total_mean = means[-1]
    denominator = omega * (1.0 - omega)
    denominator[denominator <= 1e-12] = np.nan
    variance = (total_mean * omega - means) ** 2 / denominator
    if np.all(np.isnan(variance)):
        return (lo + hi) // 2
    return int(np.nanargmax(variance))


def _binary_erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    result = np.ones(mask.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            result &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return result


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1)
    count = np.zeros(mask.shape, dtype=np.uint8)
    for dy, dx in NEIGHBORS_8:
        count += padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
    return count


def _zhang_suen(mask: np.ndarray) -> np.ndarray:
    """Small dependency-free Zhang-Suen thinning fallback."""

    image = mask.astype(np.uint8).copy()
    if min(image.shape) < 3:
        return image.astype(bool)
    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            p = np.pad(image, 1)
            p2 = p[:-2, 1:-1]
            p3 = p[:-2, 2:]
            p4 = p[1:-1, 2:]
            p5 = p[2:, 2:]
            p6 = p[2:, 1:-1]
            p7 = p[2:, :-2]
            p8 = p[1:-1, :-2]
            p9 = p[:-2, :-2]
            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1))
                + ((p4 == 0) & (p5 == 1))
                + ((p5 == 0) & (p6 == 1))
                + ((p6 == 0) & (p7 == 1))
                + ((p7 == 0) & (p8 == 1))
                + ((p8 == 0) & (p9 == 1))
                + ((p9 == 0) & (p2 == 1))
            )
            common = (image == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1)
            if phase == 0:
                removable = common & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                removable = common & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            if removable.any():
                image[removable] = 0
                changed = True
    return image.astype(bool)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if _skimage_skeletonize is not None:
        return np.asarray(_skimage_skeletonize(mask_bool), dtype=bool)
    return _zhang_suen(mask_bool)


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape
    for sy, sx in np.argwhere(mask):
        if visited[sy, sx]:
            continue
        queue = deque([(int(sy), int(sx))])
        visited[sy, sx] = True
        coords: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            coords.append((y, x))
            for dy, dx in NEIGHBORS_8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        components.append(np.asarray(coords, dtype=np.int32))
    return components


def _distance_inside(mask: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    # Repeated erosion is a conservative Chebyshev-distance fallback.
    current = mask.astype(bool).copy()
    distance = np.zeros(mask.shape, dtype=np.float32)
    level = 0
    while current.any():
        level += 1
        distance[current] = level
        current = _binary_erode(current)
    return distance


def _component_boundary(component: np.ndarray) -> np.ndarray:
    # Padding is important for components that originally touched a crop edge.
    padded = np.pad(component.astype(bool), 1, constant_values=False)
    boundary = padded & ~_binary_erode(padded)
    return skeletonize_mask(boundary)[1:-1, 1:-1]


def _resize_and_center(gray: np.ndarray, foreground: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    ys, xs = np.where(foreground)
    if not len(xs):
        raise SymbolGeneratorError("No dark foreground was found")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = Image.fromarray(gray[y0:y1, x0:x1], mode="L")
    available = config.image_size - 2 * config.margin
    scale = min(available / max(1, crop.width), available / max(1, crop.height))
    new_size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    crop = crop.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (config.image_size, config.image_size), 255)
    position = ((config.image_size - crop.width) // 2, (config.image_size - crop.height) // 2)
    canvas.paste(crop, position)
    return np.asarray(canvas, dtype=np.uint8)


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    result = mask.astype(bool)
    for _ in range(max(0, radius)):
        padded = np.pad(result, 1)
        expanded = np.zeros_like(result)
        for dy in range(3):
            for dx in range(3):
                expanded |= padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
        result = expanded
    return result


def preprocess_image(
    source: str | Path | Image.Image | np.ndarray,
    config: PreprocessConfig | None = None,
    *,
    size: int | None = None,
    image_size: int | None = None,
    max_source_stroke_width: float | None = None,
) -> ProcessedSymbol:
    """Convert one raster symbol into a canonical centerline representation."""

    base = config or PreprocessConfig()
    requested_size = image_size if image_size is not None else size
    if requested_size is not None or max_source_stroke_width is not None:
        values = asdict(base)
        if requested_size is not None:
            values["image_size"] = int(requested_size)
            values["margin"] = min(values["margin"], max(2, int(requested_size) // 10))
        if max_source_stroke_width is not None:
            values["max_source_stroke_width"] = float(max_source_stroke_width)
        base = PreprocessConfig(**values)
    base.validate()

    gray, source_name = _load_grayscale(source, base.max_input_pixels)
    threshold = otsu_threshold(gray)
    foreground = gray <= threshold
    coverage = float(foreground.mean())
    if coverage < 1e-5:
        raise SymbolGeneratorError("Image contains no usable foreground")
    if coverage > 0.95:
        raise SymbolGeneratorError("Image is almost completely filled; expected dark marks on white")

    normalized = _resize_and_center(gray, foreground, base)
    normalized_threshold = otsu_threshold(normalized)
    mask = normalized <= normalized_threshold
    components = _connected_components(mask)
    line_mask = np.zeros_like(mask)
    render_mask = np.zeros_like(mask)
    component_reports: list[dict[str, Any]] = []
    width_samples: list[float] = []
    solid_count = 0
    removed_count = 0

    for coords in components:
        if len(coords) < base.min_component_pixels:
            removed_count += 1
            continue
        component = np.zeros_like(mask)
        component[coords[:, 0], coords[:, 1]] = True
        ys, xs = coords[:, 0], coords[:, 1]
        bbox_area = int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        compactness = float(len(coords) / max(1, bbox_area))
        skeleton = skeletonize_mask(component)
        inside = _distance_inside(component)
        diameters = 2.0 * inside[skeleton]
        width95 = float(np.percentile(diameters, 95)) if diameters.size else 1.0
        median_width = float(np.median(diameters)) if diameters.size else 1.0
        is_solid = width95 > base.max_source_stroke_width and compactness >= 0.25
        if is_solid:
            solid_count += 1
            if base.filled_policy == "reject":
                raise SymbolGeneratorError(
                    f"Solid component detected in {source_name}; use filled_policy='outline' to convert it"
                )
            converted = _component_boundary(component)
            conversion_kind = "outline"
        else:
            converted = skeleton
            width_samples.append(median_width)
            conversion_kind = "centerline"
        line_mask |= converted
        render_mask |= component
        component_reports.append(
            {
                "pixels": int(len(coords)),
                "compactness": compactness,
                "width95": width95,
                "median_width": median_width,
                "conversion": conversion_kind,
            }
        )

    if int(line_mask.sum()) < 2:
        raise SymbolGeneratorError("No usable line geometry remained after preprocessing")
    estimated_width = float(np.median(width_samples)) if width_samples else 2.0
    estimated_width = float(np.clip(estimated_width, 1.0, 6.0))
    conversion = {
        "threshold": int(normalized_threshold),
        "source_coverage": coverage,
        "line_pixels": int(line_mask.sum()),
        "components": component_reports,
        "solid_components": solid_count,
        "removed_small_components": removed_count,
    }
    return ProcessedSymbol(
        source=source_name,
        line_mask=line_mask.astype(bool),
        render_mask=render_mask.astype(bool),
        normalized_gray=normalized,
        stroke_width=estimated_width,
        conversion=conversion,
    )


def mask_hash(mask: np.ndarray) -> str:
    canonical = np.asarray(mask, dtype=bool)
    header = np.asarray(canonical.shape, dtype=np.int32).tobytes()
    return hashlib.sha256(header + np.packbits(canonical.ravel()).tobytes()).hexdigest()


def render_line_mask(mask: np.ndarray, width: float, white_background: bool = True) -> Image.Image:
    graph = mask_to_graph(mask)
    svg = graph_to_svg(graph, stroke_width=width, size=mask.shape[0])
    rendered = svg_to_mask(svg, size=mask.shape[0])
    array = np.where(rendered, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB" if white_background else "L")


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _path_length(points: np.ndarray, closed: bool = False) -> float:
    if len(points) < 2:
        return 0.0
    work = points
    if closed and not np.allclose(points[0], points[-1]):
        work = np.vstack([points, points[0]])
    return float(np.linalg.norm(np.diff(work, axis=0), axis=1).sum())


def _dedupe_consecutive(points: Sequence[Sequence[float]]) -> np.ndarray:
    output: list[tuple[float, float]] = []
    for point in points:
        item = (float(point[0]), float(point[1]))
        if not output or item != output[-1]:
            output.append(item)
    return np.asarray(output, dtype=np.float32)


def mask_to_graph(mask: np.ndarray) -> TracedGraph:
    """Trace an 8-connected skeleton into deterministic SVG-coordinate paths."""

    skeleton = skeletonize_mask(np.asarray(mask, dtype=bool))
    pixels = {tuple(map(int, item)) for item in np.argwhere(skeleton)}
    if not pixels:
        empty_stats = GraphStats(0, 0, 0, 0, 0, 0.0, 0.0, 0.0)
        return TracedGraph([], [], empty_stats, skeleton)

    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for y, x in pixels:
        neighbors[(y, x)] = sorted(
            (y + dy, x + dx)
            for dy, dx in NEIGHBORS_8
            if (y + dy, x + dx) in pixels
        )

    special = {pixel for pixel, adjacent in neighbors.items() if len(adjacent) != 2}
    cluster_of: dict[tuple[int, int], int] = {}
    clusters: list[list[tuple[int, int]]] = []
    for start in sorted(special):
        if start in cluster_of:
            continue
        cluster_id = len(clusters)
        queue = deque([start])
        cluster_of[start] = cluster_id
        cluster: list[tuple[int, int]] = []
        while queue:
            current = queue.popleft()
            cluster.append(current)
            for nxt in neighbors[current]:
                if nxt in special and nxt not in cluster_of:
                    cluster_of[nxt] = cluster_id
                    queue.append(nxt)
        clusters.append(sorted(cluster))

    centroids: list[tuple[float, float]] = []
    for cluster in clusters:
        array = np.asarray(cluster, dtype=np.float32)
        centroids.append((float(array[:, 1].mean()), float(array[:, 0].mean())))

    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for cluster in clusters:
        for pixel in cluster:
            for nxt in neighbors[pixel]:
                if nxt in cluster_of and cluster_of[nxt] == cluster_of[pixel]:
                    visited.add(_edge_key(pixel, nxt))

    paths: list[np.ndarray] = []
    closed_flags: list[bool] = []

    def trace_from(start: tuple[int, int], nxt: tuple[int, int]) -> tuple[np.ndarray, bool]:
        start_cluster = cluster_of.get(start)
        initial = centroids[start_cluster] if start_cluster is not None else (float(start[1]), float(start[0]))
        points: list[tuple[float, float]] = [initial]
        previous = start
        current = nxt
        visited.add(_edge_key(previous, current))
        limit = len(pixels) * 4 + 4
        for _ in range(limit):
            current_cluster = cluster_of.get(current)
            if current_cluster is not None:
                points.append(centroids[current_cluster])
                return _dedupe_consecutive(points), current_cluster == start_cluster and len(points) > 2
            points.append((float(current[1]), float(current[0])))
            candidates = [
                item
                for item in neighbors[current]
                if item != previous and _edge_key(current, item) not in visited
            ]
            if not candidates:
                return _dedupe_consecutive(points), False
            following = sorted(candidates)[0]
            visited.add(_edge_key(current, following))
            previous, current = current, following
        return _dedupe_consecutive(points), False

    # Edges leaving endpoint/junction clusters are traced first.
    for cluster_id, cluster in enumerate(clusters):
        boundary: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for pixel in cluster:
            for nxt in neighbors[pixel]:
                if cluster_of.get(nxt) != cluster_id:
                    boundary.append((pixel, nxt))
        for start, nxt in sorted(boundary):
            if _edge_key(start, nxt) in visited:
                continue
            path, closed = trace_from(start, nxt)
            if len(path) >= 2 and _path_length(path, closed) > 0.25:
                paths.append(path)
                closed_flags.append(closed)

    # Every-degree-two components are cycles and have no special start node.
    all_edges = sorted(
        {
            _edge_key(pixel, nxt)
            for pixel, adjacent in neighbors.items()
            for nxt in adjacent
        }
    )
    for edge in all_edges:
        if edge in visited:
            continue
        start, current = edge
        points: list[tuple[float, float]] = [(float(start[1]), float(start[0]))]
        previous = start
        visited.add(edge)
        closed = False
        for _ in range(len(pixels) * 4 + 4):
            points.append((float(current[1]), float(current[0])))
            candidates = [
                item
                for item in neighbors[current]
                if item != previous and _edge_key(current, item) not in visited
            ]
            if not candidates:
                if start in neighbors[current]:
                    visited.add(_edge_key(current, start))
                    closed = True
                break
            following = sorted(candidates)[0]
            visited.add(_edge_key(current, following))
            previous, current = current, following
            if current == start:
                closed = True
                break
        path = _dedupe_consecutive(points)
        if len(path) >= 2 and _path_length(path, closed) > 0.25:
            paths.append(path)
            closed_flags.append(closed)

    # Stable ordering makes output bytes repeatable for a fixed mask.
    order = sorted(
        range(len(paths)),
        key=lambda index: (
            round(float(paths[index][0, 1]), 4),
            round(float(paths[index][0, 0]), 4),
            int(closed_flags[index]),
            len(paths[index]),
        ),
    )
    paths = [paths[index] for index in order]
    closed_flags = [closed_flags[index] for index in order]

    endpoint_clusters = 0
    junction_clusters = 0
    for cluster_id, cluster in enumerate(clusters):
        external = {
            _edge_key(pixel, nxt)
            for pixel in cluster
            for nxt in neighbors[pixel]
            if cluster_of.get(nxt) != cluster_id
        }
        if len(external) == 1:
            endpoint_clusters += 1
        elif len(external) >= 3:
            junction_clusters += 1
    components = len(_connected_components(skeleton))
    vertex_count = endpoint_clusters + junction_clusters
    cycles = max(0, len(paths) - max(1, vertex_count) + components)
    total_length = sum(_path_length(path, closed) for path, closed in zip(paths, closed_flags))
    ys, xs = np.where(skeleton)
    bbox_area = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) if len(xs) else 0.0
    stats = GraphStats(
        components=components,
        endpoints=endpoint_clusters,
        junctions=junction_clusters,
        cycles=cycles,
        path_count=len(paths),
        total_length=float(total_length),
        bbox_area=bbox_area,
        length_density=float(total_length / max(1.0, bbox_area)),
    )
    return TracedGraph(paths, closed_flags, stats, skeleton)


def rdp_simplify(points: np.ndarray, epsilon: float = 0.5) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= 2:
        return points.copy()
    start, end = points[0], points[-1]
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-12:
        distances = np.linalg.norm(points - start, axis=1)
    else:
        projection = np.clip(((points - start) @ segment) / denominator, 0.0, 1.0)
        closest = start + projection[:, None] * segment
        distances = np.linalg.norm(points - closest, axis=1)
    index = int(np.argmax(distances))
    if float(distances[index]) <= epsilon:
        return np.vstack([start, end])
    left = rdp_simplify(points[: index + 1], epsilon)
    right = rdp_simplify(points[index:], epsilon)
    return np.vstack([left[:-1], right])


def _fmt(value: float) -> str:
    if not math.isfinite(float(value)):
        raise ValueError("SVG coordinates must be finite")
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text if text not in {"-0", ""} else "0"


def _linear_path_d(points: np.ndarray, closed: bool) -> str:
    work = rdp_simplify(points, epsilon=0.5)
    commands = [f"M {_fmt(work[0, 0])} {_fmt(work[0, 1])}"]
    commands.extend(f"L {_fmt(point[0])} {_fmt(point[1])}" for point in work[1:])
    if closed:
        commands.append("Z")
    return " ".join(commands)


def _cubic_path_d(points: np.ndarray, closed: bool) -> str:
    work = rdp_simplify(points, epsilon=0.35)
    if len(work) < 3:
        return _linear_path_d(work, closed)
    if closed and np.allclose(work[0], work[-1]):
        work = work[:-1]
    if len(work) < 3:
        return _linear_path_d(work, closed)
    commands = [f"M {_fmt(work[0, 0])} {_fmt(work[0, 1])}"]
    count = len(work)
    segment_count = count if closed else count - 1
    for index in range(segment_count):
        p1 = work[index]
        p2 = work[(index + 1) % count]
        p0 = work[(index - 1) % count] if (closed or index > 0) else p1
        p3 = work[(index + 2) % count] if (closed or index + 2 < count) else p2
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        commands.append(
            "C "
            f"{_fmt(c1[0])} {_fmt(c1[1])} "
            f"{_fmt(c2[0])} {_fmt(c2[1])} "
            f"{_fmt(p2[0])} {_fmt(p2[1])}"
        )
    if closed:
        commands.append("Z")
    return " ".join(commands)


def _serialize_svg(path_data: Sequence[str], stroke_width: float, size: int) -> str:
    if not math.isfinite(stroke_width) or stroke_width <= 0:
        raise ValueError("stroke_width must be a positive finite number")
    lines = [
        f'<svg xmlns="{SVG_NS}" viewBox="0 0 {int(size)} {int(size)}">',
        "  <g>",
    ]
    style = (
        'fill="none" stroke="#000000" '
        f'stroke-width="{_fmt(stroke_width)}" '
        'stroke-linecap="round" stroke-linejoin="round"'
    )
    lines.extend(f'    <path d="{data}" {style}/>' for data in path_data)
    lines.extend(["  </g>", "</svg>", ""])
    return "\n".join(lines)


def _topology_signature(mask: np.ndarray) -> tuple[int, int, int, int]:
    stats = mask_to_graph(mask).stats
    return stats.components, stats.endpoints, stats.junctions, stats.cycles


def _distance_to_true(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if cv2 is not None:
        return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    height, width = mask.shape
    inf = float(height + width)
    distance = np.full(mask.shape, inf, dtype=np.float32)
    distance[mask] = 0.0
    diagonal = math.sqrt(2.0)
    for y in range(height):
        for x in range(width):
            value = distance[y, x]
            if y:
                value = min(value, distance[y - 1, x] + 1.0)
                if x:
                    value = min(value, distance[y - 1, x - 1] + diagonal)
                if x + 1 < width:
                    value = min(value, distance[y - 1, x + 1] + diagonal)
            if x:
                value = min(value, distance[y, x - 1] + 1.0)
            distance[y, x] = value
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            value = distance[y, x]
            if y + 1 < height:
                value = min(value, distance[y + 1, x] + 1.0)
                if x:
                    value = min(value, distance[y + 1, x - 1] + diagonal)
                if x + 1 < width:
                    value = min(value, distance[y + 1, x + 1] + diagonal)
            if x + 1 < width:
                value = min(value, distance[y, x + 1] + 1.0)
            distance[y, x] = value
    return distance


def _chamfer_percentile(a: np.ndarray, b: np.ndarray, percentile: float = 95.0) -> float:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if not a.any() or not b.any():
        return float("inf")
    ab = _distance_to_true(b)[a]
    ba = _distance_to_true(a)[b]
    return float(max(np.percentile(ab, percentile), np.percentile(ba, percentile)))


def graph_to_svg(
    graph: TracedGraph | Sequence[np.ndarray],
    stroke_width: float,
    size: int = 128,
    *,
    curve_error: float = 0.75,
) -> str:
    """Serialize a traced graph as path-only SVG, falling back on unsafe curves."""

    if isinstance(graph, TracedGraph):
        paths = graph.paths
        closed = graph.closed
        source_mask = graph.mask
    else:
        paths = [np.asarray(item, dtype=np.float32) for item in graph]
        closed = [bool(len(item) > 2 and np.allclose(item[0], item[-1])) for item in paths]
        source_mask = None
    valid = [(path, flag) for path, flag in zip(paths, closed) if len(path) >= 2]
    if not valid:
        raise SymbolGeneratorError("Cannot emit SVG without at least one non-empty path")

    cubic_data = [_cubic_path_d(path, flag) for path, flag in valid]
    cubic_svg = _serialize_svg(cubic_data, stroke_width, size)
    if source_mask is not None:
        try:
            rendered = skeletonize_mask(svg_to_mask(cubic_svg, size=size))
            error = _chamfer_percentile(source_mask, rendered)
            topology_matches = _topology_signature(source_mask) == _topology_signature(rendered)
            if error <= curve_error and topology_matches:
                return cubic_svg
        except Exception:
            pass
    elif all(len(path) >= 3 for path, _ in valid):
        return cubic_svg

    linear_data = [_linear_path_d(path, flag) for path, flag in valid]
    return _serialize_svg(linear_data, stroke_width, size)


_PATH_TOKEN = re.compile(r"[MLCZmlcz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def _parse_svg_path(data: str) -> tuple[list[np.ndarray], list[bool]]:
    tokens = _PATH_TOKEN.findall(data)
    index = 0
    command: str | None = None
    current = np.zeros(2, dtype=np.float64)
    start = np.zeros(2, dtype=np.float64)
    points: list[np.ndarray] = []
    paths: list[np.ndarray] = []
    closed: list[bool] = []

    def finish(is_closed: bool = False) -> None:
        nonlocal points
        if len(points) >= 2:
            paths.append(np.asarray(points, dtype=np.float32))
            closed.append(is_closed)
        points = []

    def number() -> float:
        nonlocal index
        if index >= len(tokens) or re.fullmatch(r"[A-Za-z]", tokens[index]):
            raise ValueError("Malformed SVG path data")
        value = float(tokens[index])
        index += 1
        return value

    while index < len(tokens):
        if re.fullmatch(r"[A-Za-z]", tokens[index]):
            command = tokens[index]
            index += 1
        if command is None:
            raise ValueError("SVG path must begin with a command")
        relative = command.islower()
        op = command.upper()
        if op == "M":
            if points:
                finish(False)
            target = np.asarray([number(), number()], dtype=np.float64)
            if relative:
                target += current
            current = target
            start = target.copy()
            points = [current.copy()]
            command = "l" if relative else "L"
        elif op == "L":
            target = np.asarray([number(), number()], dtype=np.float64)
            if relative:
                target += current
            current = target
            points.append(current.copy())
        elif op == "C":
            c1 = np.asarray([number(), number()], dtype=np.float64)
            c2 = np.asarray([number(), number()], dtype=np.float64)
            target = np.asarray([number(), number()], dtype=np.float64)
            if relative:
                c1 += current
                c2 += current
                target += current
            chord = float(np.linalg.norm(target - current))
            samples = max(8, min(64, int(math.ceil(chord * 2))))
            p0 = current.copy()
            for step in range(1, samples + 1):
                t = step / samples
                point = (
                    (1 - t) ** 3 * p0
                    + 3 * (1 - t) ** 2 * t * c1
                    + 3 * (1 - t) * t**2 * c2
                    + t**3 * target
                )
                points.append(point)
            current = target
        elif op == "Z":
            if points:
                points.append(start.copy())
                current = start.copy()
                finish(True)
            command = None
        else:
            raise ValueError(f"Unsupported SVG command {command!r}")
    if points:
        finish(False)
    return paths, closed


def validate_svg_schema(svg_text: str) -> None:
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        raise SymbolGeneratorError(f"Generated SVG is not valid XML: {exc}") from exc
    allowed = {"svg", "g", "path"}
    stroke_widths: set[str] = set()
    path_count = 0
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag not in allowed:
            raise SymbolGeneratorError(f"Forbidden SVG element: {tag}")
        if tag == "path":
            path_count += 1
            if element.attrib.get("fill") != "none":
                raise SymbolGeneratorError("Every SVG path must use fill='none'")
            if element.attrib.get("stroke", "").lower() not in {"#000000", "#000", "black"}:
                raise SymbolGeneratorError("Every SVG path must use a black stroke")
            if element.attrib.get("stroke-linecap") != "round" or element.attrib.get("stroke-linejoin") != "round":
                raise SymbolGeneratorError("Every SVG path must use round caps and joins")
            stroke_widths.add(element.attrib.get("stroke-width", ""))
            if not element.attrib.get("d"):
                raise SymbolGeneratorError("SVG path is missing geometry")
        for forbidden in ("href", "style", "filter", "mask", "clip-path"):
            if forbidden in element.attrib:
                raise SymbolGeneratorError(f"Forbidden SVG attribute: {forbidden}")
    if path_count == 0:
        raise SymbolGeneratorError("SVG contains no paths")
    if len(stroke_widths) != 1:
        raise SymbolGeneratorError("All paths in one symbol must have one stroke width")


def _svg_to_mask_internal(svg_text: str, size: int) -> np.ndarray:
    root = ET.fromstring(svg_text)
    viewbox = root.attrib.get("viewBox", f"0 0 {size} {size}").replace(",", " ").split()
    if len(viewbox) != 4:
        raise ValueError("SVG viewBox must contain four numbers")
    vx, vy, vw, vh = map(float, viewbox)
    if vw <= 0 or vh <= 0:
        raise ValueError("SVG viewBox dimensions must be positive")
    supersample = 4
    canvas_size = size * supersample
    image = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(image)
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "path":
            continue
        paths, closed_flags = _parse_svg_path(element.attrib.get("d", ""))
        stroke = float(element.attrib.get("stroke-width", "1"))
        width_px = max(1, round(stroke * size / max(vw, vh) * supersample))
        radius = width_px / 2.0
        for path, closed in zip(paths, closed_flags):
            transformed = np.empty_like(path)
            transformed[:, 0] = (path[:, 0] - vx) * size / vw * supersample
            transformed[:, 1] = (path[:, 1] - vy) * size / vh * supersample
            coords = [tuple(map(float, item)) for item in transformed]
            if closed and coords[0] != coords[-1]:
                coords.append(coords[0])
            draw.line(coords, fill=255, width=width_px, joint="curve")
            for point in coords if closed else (coords[0], coords[-1]):
                draw.ellipse(
                    (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                    fill=255,
                )
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8) >= 96


def svg_to_mask(svg_text: str, size: int = 128) -> np.ndarray:
    """Rerender SVG on white, preferring resvg-py and using a strict fallback."""

    validate_svg_schema(svg_text)
    try:
        import resvg_py  # type: ignore

        png = resvg_py.svg_to_bytes(
            svg_string=svg_text,
            width=int(size),
            height=int(size),
            background="white",
            shape_rendering="geometric_precision",
        )
        with Image.open(io.BytesIO(png)) as image:
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
        return gray < 224
    except ImportError:
        return _svg_to_mask_internal(svg_text, size)
    except Exception as exc:
        warnings.warn(f"resvg-py render failed; using internal path renderer: {exc}", RuntimeWarning)
        return _svg_to_mask_internal(svg_text, size)


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    segment = b - a
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, segment) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * segment)))


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(np.cross(b - a, c - a))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    return (o1 == 0 or o2 == 0 or np.sign(o1) != np.sign(o2)) and (
        o3 == 0 or o4 == 0 or np.sign(o3) != np.sign(o4)
    )


def _segment_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    )


def _angle_difference(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    first = math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))
    second = math.atan2(float(d[1] - c[1]), float(d[0] - c[0]))
    difference = abs(first - second) % math.pi
    return min(difference, math.pi - difference)


def _projected_overlap(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    direction = b - a
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return 0.0
    unit = direction / length
    first = sorted([float(np.dot(a, unit)), float(np.dot(b, unit))])
    second = sorted([float(np.dot(c, unit)), float(np.dot(d, unit))])
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _graph_segments(graph: TracedGraph) -> list[tuple[int, int, np.ndarray, np.ndarray, float]]:
    output: list[tuple[int, int, np.ndarray, np.ndarray, float]] = []
    for path_index, (path, closed) in enumerate(zip(graph.paths, graph.closed)):
        work = path
        if closed and len(path) > 2 and not np.allclose(path[0], path[-1]):
            work = np.vstack([path, path[0]])
        for segment_index, (a, b) in enumerate(zip(work[:-1], work[1:])):
            length = float(np.linalg.norm(b - a))
            if length > 1e-6:
                output.append((path_index, segment_index, a, b, length))
    return output


def compute_quality_baselines(graphs: Sequence[TracedGraph]) -> dict[str, Any]:
    if not graphs:
        return {}
    fields = {
        "components": np.asarray([item.stats.components for item in graphs], dtype=np.float32),
        "path_count": np.asarray([item.stats.path_count for item in graphs], dtype=np.float32),
        "junctions": np.asarray([item.stats.junctions for item in graphs], dtype=np.float32),
        "total_length": np.asarray([item.stats.total_length for item in graphs], dtype=np.float32),
        "length_density": np.asarray([item.stats.length_density for item in graphs], dtype=np.float32),
    }
    result: dict[str, Any] = {}
    for name, values in fields.items():
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        result[name] = {
            "median": median,
            "mad": mad,
            "p01": float(np.percentile(values, 1)),
            "p99": float(np.percentile(values, 99)),
            "upper": float(max(np.percentile(values, 99), median + 4.0 * max(mad, 1e-3))),
        }
    return result


def validate_line_quality(
    graph: TracedGraph,
    rendered_mask: np.ndarray,
    stroke_width: float,
    baselines: Mapping[str, Any] | None = None,
) -> QualityResult:
    """Reject invalid, filled-looking, or hatch-like generated line geometry."""

    reasons: list[str] = []
    rendered = np.asarray(rendered_mask, dtype=bool)
    stats = graph.stats
    metrics: dict[str, float] = {
        "components": float(stats.components),
        "path_count": float(stats.path_count),
        "junctions": float(stats.junctions),
        "total_length": float(stats.total_length),
        "length_density": float(stats.length_density),
        "foreground_fraction": float(rendered.mean()),
    }
    if not rendered.any() or stats.total_length < 4.0:
        reasons.append("empty_or_minuscule")
    if rendered.all() or rendered.mean() > 0.35:
        reasons.append("excessive_ink")
    if rendered.any() and (rendered[0].any() or rendered[-1].any() or rendered[:, 0].any() or rendered[:, -1].any()):
        reasons.append("edge_clipped")
    if stats.components > 24:
        reasons.append("excessive_fragmentation")

    if baselines:
        for field_name in ("components", "path_count", "junctions", "total_length", "length_density"):
            if field_name not in baselines:
                continue
            value = float(getattr(stats, field_name))
            upper = float(baselines[field_name].get("upper", float("inf")))
            # A little headroom prevents one outlier source from becoming a hard template.
            if value > max(upper * 1.25, upper + 1.0):
                reasons.append(f"outlier_{field_name}")

    segments = _graph_segments(graph)
    crowded_length = 0.0
    parallel_neighbors: Counter[int] = Counter()
    total_segment_length = sum(item[4] for item in segments)
    angle_limit = math.radians(15.0)
    for first_index, first in enumerate(segments):
        p1, s1, a, b, length = first
        is_crowded = False
        for second_index in range(first_index + 1, len(segments)):
            p2, s2, c, d, _ = segments[second_index]
            if p1 == p2 and abs(s1 - s2) <= 2:
                continue
            # Adjacent edges from a real shared junction are permitted.
            shared_endpoint = min(
                np.linalg.norm(a - c), np.linalg.norm(a - d), np.linalg.norm(b - c), np.linalg.norm(b - d)
            ) <= 0.75
            distance = _segment_distance(a, b, c, d)
            angle = _angle_difference(a, b, c, d)
            overlap = _projected_overlap(a, b, c, d)
            if shared_endpoint and distance <= 0.75:
                continue
            if angle <= math.radians(25.0) and distance < 1.5 * stroke_width and overlap > stroke_width:
                is_crowded = True
            if angle <= angle_limit and distance <= 2.0 * stroke_width and overlap >= 4.0 * stroke_width:
                parallel_neighbors[first_index] += 1
                parallel_neighbors[second_index] += 1
        if is_crowded:
            crowded_length += length
    crowd_fraction = crowded_length / max(1e-6, total_segment_length)
    metrics["crowded_fraction"] = float(crowd_fraction)
    metrics["max_parallel_neighbors"] = float(max(parallel_neighbors.values(), default=0))
    if crowd_fraction > 0.10:
        reasons.append("crowded_lines")
    if max(parallel_neighbors.values(), default=0) >= 2:
        reasons.append("parallel_bundle_or_hatching")

    if rendered.any():
        interior = _distance_inside(rendered)
        diameters = 2.0 * interior[rendered]
        diameter99 = float(np.percentile(diameters, 99)) if diameters.size else 0.0
        metrics["rendered_diameter_p99"] = diameter99
        if diameter99 > max(3.0, 2.2 * stroke_width):
            reasons.append("solid_looking_overlap")

    return QualityResult(valid=not reasons, reasons=sorted(set(reasons)), metrics=metrics)


def _topology_similarity(a: GraphStats, b: GraphStats) -> float:
    first = a.as_vector()
    second = b.as_vector()
    denominator = np.maximum(1.0, np.maximum(first, second))
    return float(1.0 - np.mean(np.minimum(1.0, np.abs(first - second) / denominator)))


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    denominator = int(a.sum()) + int(b.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denominator)


def _skeleton_f1(a: np.ndarray, b: np.ndarray, tolerance: float = 2.0) -> float:
    a = skeletonize_mask(a)
    b = skeletonize_mask(b)
    if not a.any() and not b.any():
        return 1.0
    if not a.any() or not b.any():
        return 0.0
    precision = float(np.mean(_distance_to_true(b)[a] <= tolerance))
    recall = float(np.mean(_distance_to_true(a)[b] <= tolerance))
    if precision + recall <= 1e-12:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def similarity_components(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a_skeleton, b_skeleton = skeletonize_mask(a), skeletonize_mask(b)
    skeleton_f1 = _skeleton_f1(a_skeleton, b_skeleton, tolerance=2.0)
    rendered_dice = _dice(_dilate_mask(a_skeleton, 1), _dilate_mask(b_skeleton, 1))
    topology = _topology_similarity(mask_to_graph(a_skeleton).stats, mask_to_graph(b_skeleton).stats)
    score = 0.60 * skeleton_f1 + 0.30 * rendered_dice + 0.10 * topology
    return {
        "skeleton_f1": float(skeleton_f1),
        "rendered_dice": float(rendered_dice),
        "topology": float(topology),
        "score": float(score),
    }


def _transform_mask(
    mask: np.ndarray,
    angle: float = 0.0,
    scale: float = 1.0,
    dx: int = 0,
    dy: int = 0,
    mirror: bool = False,
) -> np.ndarray:
    source = (np.asarray(mask, dtype=np.uint8) * 255)
    if mirror:
        source = np.fliplr(source)
    size = source.shape[0]
    image = Image.fromarray(source, mode="L")
    if scale != 1.0:
        target = max(1, round(size * scale))
        resized = image.resize((target, target), Image.Resampling.NEAREST)
        canvas = Image.new("L", (size, size), 0)
        canvas.paste(resized, ((size - target) // 2, (size - target) // 2))
        image = canvas
    if angle:
        image = image.rotate(angle, Image.Resampling.NEAREST, expand=False, fillcolor=0)
    if dx or dy:
        shifted = Image.new("L", (size, size), 0)
        shifted.paste(image, (dx, dy))
        image = shifted
    return np.asarray(image, dtype=np.uint8) >= 128


def _coarse_descriptor(mask: np.ndarray, size: int = 32) -> np.ndarray:
    image = Image.fromarray((np.asarray(mask, dtype=np.uint8) * 255), mode="L")
    small = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32).ravel()
    norm = float(np.linalg.norm(small))
    return small / norm if norm > 1e-12 else small


class NoveltyChecker:
    """Exact and geometry-aware near-duplicate checker."""

    def __init__(
        self,
        reference_masks: Sequence[np.ndarray],
        reference_names: Sequence[str] | None = None,
        *,
        duplicate_threshold: float = 0.94,
        review_threshold: float = 0.82,
        top_k: int = 64,
    ) -> None:
        if not 0.0 <= review_threshold < duplicate_threshold <= 1.0:
            raise ValueError("Expected 0 <= review_threshold < duplicate_threshold <= 1")
        self.reference_masks = [skeletonize_mask(item) for item in reference_masks]
        self.reference_names = list(reference_names or [f"reference_{i}" for i in range(len(reference_masks))])
        if len(self.reference_names) != len(self.reference_masks):
            raise ValueError("reference_names length must match reference_masks")
        self.duplicate_threshold = float(duplicate_threshold)
        self.review_threshold = float(review_threshold)
        self.top_k = max(1, int(top_k))
        self.accepted_masks: list[np.ndarray] = []
        self.accepted_names: list[str] = []
        self._rebuild()

    def _rebuild(self) -> None:
        masks = self.reference_masks + self.accepted_masks
        self._hashes: dict[str, int] = {mask_hash(mask): index for index, mask in enumerate(masks)}
        self._descriptors = (
            np.stack([_coarse_descriptor(mask) for mask in masks])
            if masks
            else np.empty((0, 32 * 32), dtype=np.float32)
        )

    def register(self, mask: np.ndarray, name: str | None = None) -> None:
        self.accepted_masks.append(skeletonize_mask(mask))
        self.accepted_names.append(name or f"generated_{len(self.accepted_masks) - 1}")
        self._rebuild()

    def _name_for_index(self, index: int) -> str:
        if index < len(self.reference_names):
            return self.reference_names[index]
        return self.accepted_names[index - len(self.reference_names)]

    @staticmethod
    def _alignment_variants(candidate: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        variants: list[tuple[np.ndarray, np.ndarray]] = []
        for angle in (-6.0, 0.0, 6.0):
            for scale in (0.96, 1.0, 1.04):
                transformed = _transform_mask(candidate, angle=angle, scale=scale)
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        variant = _transform_mask(transformed, dx=dx, dy=dy)
                        variants.append((variant, _coarse_descriptor(_dilate_mask(variant, 1), size=32)))
        return variants

    def _best_aligned(
        self,
        reference: np.ndarray,
        variants: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[dict[str, float], np.ndarray]:
        reference_descriptor = _coarse_descriptor(_dilate_mask(reference, 1), size=32)
        coarse_scores = np.asarray([float(item[1] @ reference_descriptor) for item in variants])
        # Full stroke-aware metrics are applied to several coarse finalists so resizing
        # cannot decide duplicate status by itself.
        finalists = np.argsort(-coarse_scores, kind="stable")[: min(6, len(variants))]
        best_components: dict[str, float] | None = None
        best_variant = variants[int(finalists[0])][0]
        for index in finalists:
            variant = variants[int(index)][0]
            components = similarity_components(variant, reference)
            if best_components is None or components["score"] > best_components["score"]:
                best_components = components
                best_variant = variant
        assert best_components is not None
        return best_components, best_variant

    def classify(self, mask: np.ndarray, register: bool = False, name: str | None = None) -> NoveltyResult:
        candidate = skeletonize_mask(mask)
        all_masks = self.reference_masks + self.accepted_masks
        if not candidate.any():
            return NoveltyResult("invalid", 0.0, None, None, {}, "empty")
        exact_index = self._hashes.get(mask_hash(candidate))
        if exact_index is not None:
            result = NoveltyResult(
                "duplicate",
                1.0,
                exact_index,
                self._name_for_index(exact_index),
                {"skeleton_f1": 1.0, "rendered_dice": 1.0, "topology": 1.0, "score": 1.0},
                "exact_reference" if exact_index < len(self.reference_masks) else "exact_generated",
            )
            return result
        if not all_masks:
            result = NoveltyResult("novel", 0.0, None, None, {}, "no_references")
            if register:
                self.register(candidate, name)
            return result

        descriptor = _coarse_descriptor(candidate)
        coarse = self._descriptors @ descriptor
        shortlist_size = min(len(all_masks), self.top_k)
        indices = np.argsort(-coarse, kind="stable")[:shortlist_size]
        variants = self._alignment_variants(candidate)
        best_score = -1.0
        best_components: dict[str, float] = {}
        best_index: int | None = None
        for index in indices:
            components, _ = self._best_aligned(all_masks[int(index)], variants)
            if components["score"] > best_score:
                best_score = components["score"]
                best_components = components
                best_index = int(index)

        transformed_similarity = 0.0
        if best_score < self.duplicate_threshold:
            for transformed in (
                _transform_mask(candidate, angle=90),
                _transform_mask(candidate, angle=180),
                _transform_mask(candidate, angle=270),
                _transform_mask(candidate, mirror=True),
            ):
                transformed_descriptor = _coarse_descriptor(transformed)
                transformed_coarse = self._descriptors @ transformed_descriptor
                for index in np.argsort(-transformed_coarse, kind="stable")[:shortlist_size]:
                    score = similarity_components(transformed, all_masks[int(index)])["score"]
                    transformed_similarity = max(transformed_similarity, float(score))

        if best_score >= self.duplicate_threshold:
            status = "duplicate"
            reason = "near_reference" if (best_index or 0) < len(self.reference_masks) else "near_generated"
        elif best_score >= self.review_threshold or transformed_similarity >= 0.90:
            status = "review"
            reason = "borderline_similarity" if best_score >= self.review_threshold else "rotated_or_mirrored_match"
        else:
            status = "novel"
            reason = "below_similarity_threshold"
        result = NoveltyResult(
            status=status,
            similarity=float(max(0.0, best_score)),
            nearest_index=best_index,
            nearest_name=self._name_for_index(best_index) if best_index is not None else None,
            components=best_components,
            reason=reason,
            transformed_similarity=transformed_similarity,
        )
        if register and status in {"novel", "review"}:
            self.register(candidate, name)
        return result


def changed_line_amount(base: np.ndarray, result: np.ndarray, tolerance: float = 2.0) -> dict[str, float]:
    base_skeleton = skeletonize_mask(base)
    result_skeleton = skeletonize_mask(result)
    if not base_skeleton.any():
        return {"changed_pixels": float(result_skeleton.sum()), "change_fraction": 1.0, "similarity": 0.0}
    removed = int(np.sum(_distance_to_true(result_skeleton)[base_skeleton] > tolerance))
    added = int(np.sum(_distance_to_true(base_skeleton)[result_skeleton] > tolerance))
    changed = removed + added
    fraction = float(changed / max(1, int(base_skeleton.sum())))
    return {
        "changed_pixels": float(changed),
        "added_pixels": float(added),
        "removed_pixels": float(removed),
        "change_fraction": fraction,
        "similarity": float(similarity_components(base_skeleton, result_skeleton)["score"]),
    }


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 128
    latent_dim: int = 64
    base_channels: int = 32
    min_stroke_width: float = 1.0
    max_stroke_width: float = 6.0


if nn is not None:

    class ConvDown(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            groups = min(8, out_channels)
            while out_channels % groups:
                groups -= 1
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(),
            )

        def forward(self, value: "torch.Tensor") -> "torch.Tensor":
            return self.block(value)


    class ConvUp(nn.Module):
        def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
            super().__init__()
            groups = min(8, out_channels)
            while out_channels % groups:
                groups -= 1
            self.block = nn.Sequential(
                nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(),
            )

        def forward(self, value: "torch.Tensor", skip: "torch.Tensor") -> "torch.Tensor":
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return self.block(torch.cat([value, skip], dim=1))


    class ConditionalVAE(nn.Module):
        """Conditional convolutional beta-VAE for full-symbol prediction."""

        def __init__(
            self,
            image_size: int = 128,
            latent_dim: int = 64,
            base_channels: int = 32,
            min_stroke_width: float = 1.0,
            max_stroke_width: float = 6.0,
        ) -> None:
            super().__init__()
            if image_size < 32 or image_size % 16:
                raise ValueError("image_size must be at least 32 and divisible by 16")
            self.config = ModelConfig(
                image_size=image_size,
                latent_dim=latent_dim,
                base_channels=base_channels,
                min_stroke_width=min_stroke_width,
                max_stroke_width=max_stroke_width,
            )
            channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
            self.condition_down = nn.ModuleList(
                [
                    ConvDown(2, channels[0]),
                    ConvDown(channels[0], channels[1]),
                    ConvDown(channels[1], channels[2]),
                    ConvDown(channels[2], channels[3]),
                ]
            )
            self.posterior_down = nn.ModuleList(
                [
                    ConvDown(3, channels[0]),
                    ConvDown(channels[0], channels[1]),
                    ConvDown(channels[1], channels[2]),
                    ConvDown(channels[2], channels[3]),
                ]
            )
            spatial = image_size // 16
            flat = channels[-1] * spatial * spatial
            self.to_mu = nn.Linear(flat, latent_dim)
            self.to_logvar = nn.Linear(flat, latent_dim)
            self.from_latent = nn.Linear(latent_dim, flat)
            self.bottleneck = nn.Conv2d(channels[-1] * 2, channels[-1], 1)
            self.up3 = ConvUp(channels[-1], channels[2], channels[2])
            self.up2 = ConvUp(channels[2], channels[1], channels[1])
            self.up1 = ConvUp(channels[1], channels[0], channels[0])
            final_groups = min(8, base_channels)
            while base_channels % final_groups:
                final_groups -= 1
            self.final = nn.Sequential(
                nn.Conv2d(base_channels + 2, base_channels, 3, padding=1),
                nn.GroupNorm(final_groups, base_channels),
                nn.SiLU(),
                nn.Conv2d(base_channels, 1, 1),
            )
            self.width_head = nn.Sequential(
                nn.Linear(latent_dim + channels[-1] + 1, channels[1]),
                nn.SiLU(),
                nn.Linear(channels[1], 1),
            )

        @staticmethod
        def _strength_plane(edit_strength: "torch.Tensor", image: "torch.Tensor") -> "torch.Tensor":
            if edit_strength.ndim == 1:
                edit_strength = edit_strength[:, None]
            if edit_strength.shape != (image.shape[0], 1):
                raise ValueError("edit_strength must have shape [B] or [B,1]")
            return edit_strength[:, :, None, None].expand(-1, -1, image.shape[-2], image.shape[-1])

        def encode_condition(
            self, condition: "torch.Tensor", edit_strength: "torch.Tensor"
        ) -> list["torch.Tensor"]:
            plane = self._strength_plane(edit_strength, condition)
            value = torch.cat([condition, plane], dim=1)
            features: list[torch.Tensor] = []
            for block in self.condition_down:
                value = block(value)
                features.append(value)
            return features

        def encode_posterior(
            self,
            target: "torch.Tensor",
            condition: "torch.Tensor",
            edit_strength: "torch.Tensor",
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            plane = self._strength_plane(edit_strength, target)
            value = torch.cat([target, condition, plane], dim=1)
            for block in self.posterior_down:
                value = block(value)
            flat = value.flatten(1)
            return self.to_mu(flat), self.to_logvar(flat).clamp(-12.0, 12.0)

        def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
            if self.training:
                return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            return mu

        def decode(
            self,
            z: "torch.Tensor",
            condition: "torch.Tensor",
            edit_strength: "torch.Tensor",
            condition_features: list["torch.Tensor"] | None = None,
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            features = condition_features or self.encode_condition(condition, edit_strength)
            batch = z.shape[0]
            spatial = self.config.image_size // 16
            channels = self.config.base_channels * 8
            latent = self.from_latent(z).reshape(batch, channels, spatial, spatial)
            value = self.bottleneck(torch.cat([latent, features[-1]], dim=1))
            value = self.up3(value, features[-2])
            value = self.up2(value, features[-3])
            value = self.up1(value, features[-4])
            value = F.interpolate(
                value,
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            )
            plane = self._strength_plane(edit_strength, condition)
            logits = self.final(torch.cat([value, condition, plane], dim=1))
            pooled = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
            strength = edit_strength[:, None] if edit_strength.ndim == 1 else edit_strength
            width_raw = self.width_head(torch.cat([z, pooled, strength], dim=1)).squeeze(1)
            width = self.config.min_stroke_width + (
                self.config.max_stroke_width - self.config.min_stroke_width
            ) * torch.sigmoid(width_raw)
            return logits, width

        def forward(
            self,
            target: "torch.Tensor",
            condition: "torch.Tensor",
            edit_strength: "torch.Tensor",
        ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            if target.shape != condition.shape or target.ndim != 4 or target.shape[1] != 1:
                raise ValueError("target and condition must both have shape [B,1,H,W]")
            features = self.encode_condition(condition, edit_strength)
            mu, logvar = self.encode_posterior(target, condition, edit_strength)
            z = self.reparameterize(mu, logvar)
            logits, width = self.decode(z, condition, edit_strength, features)
            return logits, mu, logvar, width

        @torch.no_grad()
        def sample(
            self,
            condition: "torch.Tensor",
            edit_strength: "torch.Tensor",
            z: "torch.Tensor | None" = None,
            temperature: float = 0.9,
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            if z is None:
                z = torch.randn(
                    condition.shape[0],
                    self.config.latent_dim,
                    device=condition.device,
                    dtype=condition.dtype,
                ) * float(temperature)
            return self.decode(z, condition, edit_strength)


else:

    class ConditionalVAE:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            require_torch()


def _draw_paths_mask(paths: Sequence[np.ndarray], size: int, width: int = 2) -> np.ndarray:
    supersample = 2
    image = Image.new("L", (size * supersample, size * supersample), 0)
    draw = ImageDraw.Draw(image)
    for path in paths:
        if len(path) < 2:
            continue
        coords = [(float(p[0] * supersample), float(p[1] * supersample)) for p in path]
        draw.line(coords, fill=255, width=max(1, width * supersample), joint="curve")
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8) >= 64


def synthesize_condition(
    line_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    edit_strength: float | None = None,
    empty_probability: float = 0.15,
) -> tuple[np.ndarray, float]:
    """Create an unlabeled partial/deformed base from a complete symbol."""

    graph = mask_to_graph(line_mask)
    if rng.random() < empty_probability:
        return np.zeros_like(line_mask, dtype=bool), 1.0
    strength = float(edit_strength if edit_strength is not None else rng.uniform(0.15, 0.85))
    strength = float(np.clip(strength, 0.0, 1.0))
    retained: list[np.ndarray] = []
    for path in graph.paths:
        if len(path) < 2 or rng.random() < strength * 0.55:
            continue
        work = path.copy()
        if len(work) > 3 and rng.random() < 0.75:
            retain_fraction = float(np.clip(1.0 - rng.uniform(0.0, strength), 0.2, 1.0))
            retain_count = max(2, round(len(work) * retain_fraction))
            if rng.random() < 0.5:
                work = work[:retain_count]
            else:
                work = work[-retain_count:]
        if strength > 0 and rng.random() < 0.65:
            jitter = rng.normal(0.0, max(0.15, strength * 1.5), size=work.shape)
            # Smooth point jitter to bend lines instead of adding pixel noise.
            if len(jitter) >= 3:
                jitter[1:-1] = (jitter[:-2] + 2 * jitter[1:-1] + jitter[2:]) / 4.0
            work = work + jitter.astype(np.float32)
        retained.append(work)
    condition = _draw_paths_mask(retained, line_mask.shape[0], width=2) if retained else np.zeros_like(line_mask)
    if condition.any() and rng.random() < 0.5:
        condition = _transform_mask(
            condition,
            angle=float(rng.uniform(-3.0, 3.0) * strength),
            dx=int(round(rng.uniform(-2.0, 2.0) * strength)),
            dy=int(round(rng.uniform(-2.0, 2.0) * strength)),
        )
    return condition.astype(bool), strength


if torch is not None:

    class SymbolDataset(Dataset):
        def __init__(self, symbols: Sequence[ProcessedSymbol], seed: int = 1337, training: bool = True) -> None:
            self.symbols = list(symbols)
            self.seed = int(seed)
            self.training = bool(training)
            self.epoch = 0

        def set_epoch(self, epoch: int) -> None:
            self.epoch = int(epoch)

        def __len__(self) -> int:
            return len(self.symbols)

        def __getitem__(self, index: int) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            symbol = self.symbols[index]
            sequence_seed = self.seed + index * 104729 + (self.epoch if self.training else 0) * 1_000_003
            rng = np.random.default_rng(sequence_seed)
            condition, strength = synthesize_condition(
                symbol.line_mask,
                rng,
                empty_probability=0.15 if self.training else 0.0,
            )
            target = _dilate_mask(symbol.line_mask, 1).astype(np.float32)
            condition_render = _dilate_mask(condition, 1).astype(np.float32)
            return (
                torch.from_numpy(target[None]),
                torch.from_numpy(condition_render[None]),
                torch.tensor(strength, dtype=torch.float32),
                torch.tensor(symbol.stroke_width, dtype=torch.float32),
            )


def _soft_erode(image: "torch.Tensor") -> "torch.Tensor":
    return -F.max_pool2d(-image, kernel_size=3, stride=1, padding=1)


def _soft_dilate(image: "torch.Tensor") -> "torch.Tensor":
    return F.max_pool2d(image, kernel_size=3, stride=1, padding=1)


def _soft_skeletonize(image: "torch.Tensor", iterations: int = 8) -> "torch.Tensor":
    opened = _soft_dilate(_soft_erode(image))
    skeleton = F.relu(image - opened)
    current = image
    for _ in range(iterations):
        current = _soft_erode(current)
        opened = _soft_dilate(_soft_erode(current))
        delta = F.relu(current - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def vae_loss(
    logits: "torch.Tensor",
    target: "torch.Tensor",
    mu: "torch.Tensor",
    logvar: "torch.Tensor",
    predicted_width: "torch.Tensor",
    target_width: "torch.Tensor",
    *,
    beta: float,
) -> tuple["torch.Tensor", dict[str, float]]:
    foreground = target.mean().clamp_min(1e-4)
    positive_weight = ((1.0 - foreground) / foreground).clamp(1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (
        probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    )).mean()
    predicted_skeleton = _soft_skeletonize(probability)
    target_skeleton = _soft_skeletonize(target)
    topology_precision = (
        (predicted_skeleton * target).sum(dim=(1, 2, 3)) + 1.0
    ) / (predicted_skeleton.sum(dim=(1, 2, 3)) + 1.0)
    topology_recall = (
        (target_skeleton * probability).sum(dim=(1, 2, 3)) + 1.0
    ) / (target_skeleton.sum(dim=(1, 2, 3)) + 1.0)
    cldice = 1.0 - (2.0 * topology_precision * topology_recall / (
        topology_precision + topology_recall + 1e-6
    )).mean()
    kl = (-0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=1)).mean()
    width = F.smooth_l1_loss(predicted_width, target_width)
    loss = bce + dice + 0.5 * cldice + float(beta) * kl + 0.1 * width
    metrics = {
        "loss": float(loss.detach()),
        "bce": float(bce.detach()),
        "dice": float(dice.detach()),
        "cldice": float(cldice.detach()),
        "kl": float(kl.detach()),
        "width": float(width.detach()),
        "beta": float(beta),
    }
    return loss, metrics
