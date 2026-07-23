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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping, Sequence

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

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    TorchTensor = torch.Tensor
    TorchDevice = torch.device
else:
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

    TorchTensor = Any
    TorchDevice = Any


SCHEMA_VERSION = 2
CHECKPOINT_KIND = "paired-family-addon-beta-vae"
LEGACY_CHECKPOINT_KIND = "line-only-symbol-beta-vae"
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
        if self.min_component_pixels < 1:
            raise ValueError("min_component_pixels must be at least 1")
        if self.max_input_pixels < 1:
            raise ValueError("max_input_pixels must be positive")
        if self.filled_policy not in {"outline", "reject"}:
            raise ValueError("filled_policy must be 'outline' or 'reject'")


@dataclass(frozen=True)
class NoveltyConfig:
    """Geometry comparison controls shared by training metadata and sampling."""

    duplicate_threshold: float = 0.94
    review_threshold: float = 0.82
    transformed_review_threshold: float = 0.90
    skeleton_weight: float = 0.60
    rendered_weight: float = 0.30
    topology_weight: float = 0.10
    skeleton_tolerance: float = 2.0
    alignment_angle: float = 6.0
    alignment_translation: int = 3
    alignment_scale: float = 0.04
    shortlist_maximum: int = 64
    precise_finalists: int = 8

    def validate(self) -> None:
        if not 0.0 <= self.review_threshold < self.duplicate_threshold <= 1.0:
            raise ValueError("Expected 0 <= review_threshold < duplicate_threshold <= 1")
        if not self.review_threshold <= self.transformed_review_threshold <= 1.0:
            raise ValueError("transformed_review_threshold must be between review_threshold and 1")
        weights = (self.skeleton_weight, self.rendered_weight, self.topology_weight)
        if any(value < 0 for value in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
            raise ValueError("novelty metric weights must be non-negative and sum to 1")
        if self.skeleton_tolerance < 0:
            raise ValueError("skeleton_tolerance must be non-negative")
        if self.alignment_angle < 0 or self.alignment_translation < 0:
            raise ValueError("alignment ranges must be non-negative")
        if not 0.0 <= self.alignment_scale < 0.5:
            raise ValueError("alignment_scale must be in [0, 0.5)")
        if self.shortlist_maximum < 1 or self.precise_finalists < 1:
            raise ValueError("shortlist_maximum and precise_finalists must be positive")


@dataclass(frozen=True)
class QualityConfig:
    """Hard line-only SVG safety limits."""

    curve_error: float = 0.75
    maximum_ink: float = 0.35
    maximum_components: int = 24
    crowded_line_limit: float = 0.10
    crowd_distance_factor: float = 1.5
    parallel_bundle_threshold: int = 3
    solid_diameter_factor: float = 2.2
    guided_noop_pixels: float = 8.0
    guided_noop_fraction: float = 0.08

    def validate(self) -> None:
        if self.curve_error <= 0:
            raise ValueError("curve_error must be positive")
        if not 0.0 < self.maximum_ink < 1.0:
            raise ValueError("maximum_ink must be between 0 and 1")
        if self.maximum_components < 1:
            raise ValueError("maximum_components must be positive")
        if not 0.0 <= self.crowded_line_limit <= 1.0:
            raise ValueError("crowded_line_limit must be between 0 and 1")
        if self.crowd_distance_factor <= 0 or self.parallel_bundle_threshold < 2:
            raise ValueError("crowding limits are invalid")
        if self.solid_diameter_factor <= 0:
            raise ValueError("solid_diameter_factor must be positive")
        if self.guided_noop_pixels < 0 or not 0.0 <= self.guided_noop_fraction <= 1.0:
            raise ValueError("guided no-op limits are invalid")


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


def resolve_device(requested: str) -> TorchDevice:
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
    config: QualityConfig | None = None,
) -> QualityResult:
    """Reject invalid, filled-looking, or hatch-like generated line geometry."""

    limits = config or QualityConfig()
    limits.validate()
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
    if rendered.all() or rendered.mean() > limits.maximum_ink:
        reasons.append("excessive_ink")
    if rendered.any() and (rendered[0].any() or rendered[-1].any() or rendered[:, 0].any() or rendered[:, -1].any()):
        reasons.append("edge_clipped")
    if stats.components > limits.maximum_components:
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
            if (
                angle <= math.radians(25.0)
                and distance < limits.crowd_distance_factor * stroke_width
                and overlap > stroke_width
            ):
                is_crowded = True
            if angle <= angle_limit and distance <= 2.0 * stroke_width and overlap >= 4.0 * stroke_width:
                parallel_neighbors[first_index] += 1
                parallel_neighbors[second_index] += 1
        if is_crowded:
            crowded_length += length
    crowd_fraction = crowded_length / max(1e-6, total_segment_length)
    metrics["crowded_fraction"] = float(crowd_fraction)
    metrics["max_parallel_neighbors"] = float(max(parallel_neighbors.values(), default=0))
    if crowd_fraction > limits.crowded_line_limit:
        reasons.append("crowded_lines")
    # Neighbor count excludes the path itself, hence threshold minus one.
    if max(parallel_neighbors.values(), default=0) >= limits.parallel_bundle_threshold - 1:
        reasons.append("parallel_bundle_or_hatching")

    if rendered.any():
        interior = _distance_inside(rendered)
        diameters = 2.0 * interior[rendered]
        diameter99 = float(np.percentile(diameters, 99)) if diameters.size else 0.0
        metrics["rendered_diameter_p99"] = diameter99
        if diameter99 > max(3.0, limits.solid_diameter_factor * stroke_width):
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


def _within_euclidean_tolerance(mask: np.ndarray, tolerance: float) -> np.ndarray:
    """Return pixels within an exact discrete Euclidean radius of ``mask``."""

    source = np.asarray(mask, dtype=bool)
    if tolerance <= 0:
        return source.copy()
    height, width = source.shape
    result = np.zeros_like(source)
    radius = int(math.floor(tolerance))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > tolerance * tolerance + 1e-9:
                continue
            source_y0, source_y1 = max(0, -dy), min(height, height - dy)
            source_x0, source_x1 = max(0, -dx), min(width, width - dx)
            target_y0, target_y1 = source_y0 + dy, source_y1 + dy
            target_x0, target_x1 = source_x0 + dx, source_x1 + dx
            result[target_y0:target_y1, target_x0:target_x1] |= source[
                source_y0:source_y1, source_x0:source_x1
            ]
    return result


def _skeleton_f1(a: np.ndarray, b: np.ndarray, tolerance: float = 2.0) -> float:
    a = skeletonize_mask(a)
    b = skeletonize_mask(b)
    if not a.any() and not b.any():
        return 1.0
    if not a.any() or not b.any():
        return 0.0
    precision = float(np.mean(_within_euclidean_tolerance(b, tolerance)[a]))
    recall = float(np.mean(_within_euclidean_tolerance(a, tolerance)[b]))
    if precision + recall <= 1e-12:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def similarity_components(
    a: np.ndarray,
    b: np.ndarray,
    *,
    skeleton_tolerance: float = 2.0,
    weights: Sequence[float] = (0.60, 0.30, 0.10),
) -> dict[str, float]:
    a_skeleton, b_skeleton = skeletonize_mask(a), skeletonize_mask(b)
    if len(weights) != 3 or any(float(value) < 0 for value in weights):
        raise ValueError("weights must contain three non-negative values")
    weight_sum = float(sum(weights))
    if weight_sum <= 0:
        raise ValueError("at least one similarity weight must be positive")
    normalized_weights = [float(value) / weight_sum for value in weights]
    skeleton_f1 = _skeleton_f1(a_skeleton, b_skeleton, tolerance=skeleton_tolerance)
    rendered_dice = _dice(_dilate_mask(a_skeleton, 1), _dilate_mask(b_skeleton, 1))
    topology = _topology_similarity(mask_to_graph(a_skeleton).stats, mask_to_graph(b_skeleton).stats)
    score = (
        normalized_weights[0] * skeleton_f1
        + normalized_weights[1] * rendered_dice
        + normalized_weights[2] * topology
    )
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
        transformed_review_threshold: float = 0.90,
        skeleton_tolerance: float = 2.0,
        metric_weights: Sequence[float] = (0.60, 0.30, 0.10),
        alignment_angle: float = 6.0,
        alignment_translation: int = 3,
        alignment_scale: float = 0.04,
        precise_finalists: int = 8,
    ) -> None:
        if not 0.0 <= review_threshold < duplicate_threshold <= 1.0:
            raise ValueError("Expected 0 <= review_threshold < duplicate_threshold <= 1")
        self.reference_masks = [skeletonize_mask(item) for item in reference_masks]
        self.reference_names = list(reference_names or [f"reference_{i}" for i in range(len(reference_masks))])
        if len(self.reference_names) != len(self.reference_masks):
            raise ValueError("reference_names length must match reference_masks")
        self.duplicate_threshold = float(duplicate_threshold)
        self.review_threshold = float(review_threshold)
        self.transformed_review_threshold = float(transformed_review_threshold)
        self.skeleton_tolerance = float(skeleton_tolerance)
        weight_sum = float(sum(metric_weights))
        if len(metric_weights) != 3 or weight_sum <= 0 or any(float(item) < 0 for item in metric_weights):
            raise ValueError("metric_weights must contain three non-negative values")
        self.metric_weights = tuple(float(item) / weight_sum for item in metric_weights)
        self.alignment_angle = max(0.0, float(alignment_angle))
        self.alignment_translation = max(0, int(alignment_translation))
        self.alignment_scale = max(0.0, float(alignment_scale))
        self.top_k = max(1, int(top_k))
        self.precise_finalists = max(1, int(precise_finalists))
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
        self._alignment_descriptors = (
            np.stack([_coarse_descriptor(_dilate_mask(mask, 1)) for mask in masks])
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

    def _alignment_variants(self, candidate: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        variants: list[tuple[np.ndarray, np.ndarray]] = []
        angles = (-self.alignment_angle, 0.0, self.alignment_angle)
        scales = (1.0 - self.alignment_scale, 1.0, 1.0 + self.alignment_scale)
        translations = range(-self.alignment_translation, self.alignment_translation + 1)
        for angle in angles:
            for scale in scales:
                transformed = _transform_mask(candidate, angle=angle, scale=scale)
                for dy in translations:
                    for dx in translations:
                        variant = _transform_mask(transformed, dx=dx, dy=dy)
                        variants.append((variant, _coarse_descriptor(_dilate_mask(variant, 1), size=32)))
        return variants

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
        variant_descriptors = np.stack([item[1] for item in variants])
        # Score all nuisance transforms against the shortlisted references in one
        # matrix multiplication, then reserve expensive graph metrics for the best
        # transform/reference pairs globally (eight by default).
        alignment_scores = variant_descriptors @ self._alignment_descriptors[indices].T
        finalist_count = min(self.precise_finalists, alignment_scores.size)
        flat_finalists = np.argsort(-alignment_scores.ravel(), kind="stable")[:finalist_count]
        best_score = -1.0
        best_components: dict[str, float] = {}
        best_index: int | None = None
        for flat_index in flat_finalists:
            variant_index, shortlist_index = np.unravel_index(int(flat_index), alignment_scores.shape)
            reference_index = int(indices[int(shortlist_index)])
            components = similarity_components(
                variants[int(variant_index)][0],
                all_masks[reference_index],
                skeleton_tolerance=self.skeleton_tolerance,
                weights=self.metric_weights,
            )
            if components["score"] > best_score:
                best_score = components["score"]
                best_components = components
                best_index = reference_index

        transformed_similarity = 0.0
        if best_score < self.duplicate_threshold:
            orientation_variants = (
                _transform_mask(candidate, angle=90),
                _transform_mask(candidate, angle=180),
                _transform_mask(candidate, angle=270),
                _transform_mask(candidate, mirror=True),
            )
            orientation_descriptors = np.stack(
                [_coarse_descriptor(item) for item in orientation_variants]
            )
            orientation_scores = orientation_descriptors @ self._descriptors.T
            orientation_finalists = np.argsort(
                -orientation_scores.ravel(), kind="stable"
            )[: min(self.precise_finalists, orientation_scores.size)]
            for flat_index in orientation_finalists:
                variant_index, reference_index = np.unravel_index(
                    int(flat_index), orientation_scores.shape
                )
                score = similarity_components(
                    orientation_variants[int(variant_index)],
                    all_masks[int(reference_index)],
                    skeleton_tolerance=self.skeleton_tolerance,
                    weights=self.metric_weights,
                )["score"]
                transformed_similarity = max(transformed_similarity, float(score))

        if best_score >= self.duplicate_threshold:
            status = "duplicate"
            reason = "near_reference" if (best_index or 0) < len(self.reference_masks) else "near_generated"
        elif best_score >= self.review_threshold or transformed_similarity >= self.transformed_review_threshold:
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
    removed = int(np.sum(base_skeleton & ~_within_euclidean_tolerance(result_skeleton, tolerance)))
    added = int(np.sum(result_skeleton & ~_within_euclidean_tolerance(base_skeleton, tolerance)))
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
    latent_dim: int = 32
    base_channels: int = 32
    min_stroke_width: float = 1.0
    max_stroke_width: float = 6.0

    def validate(self) -> None:
        if self.image_size < 32 or self.image_size % 16:
            raise ValueError("image_size must be at least 32 and divisible by 16")
        if self.latent_dim < 2 or self.base_channels < 2:
            raise ValueError("latent_dim and base_channels must be at least 2")
        if self.min_stroke_width <= 0 or self.max_stroke_width < self.min_stroke_width:
            raise ValueError("stroke width bounds are invalid")


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
        """Paired conditional beta-VAE with learned prior and edit heads.

        Family identifiers are deliberately absent from this network.  The only
        conditioning signals are the canonical base raster and requested edit
        strength, which is what makes sampling on a previously unseen base
        possible.
        """

        def __init__(
            self,
            image_size: int = 128,
            latent_dim: int = 32,
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
            self.prior_mu = nn.Linear(flat, latent_dim)
            self.prior_logvar = nn.Linear(flat, latent_dim)
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
                nn.Conv2d(base_channels, 2, 1),
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

        def encode_prior(
            self,
            condition_features: list["torch.Tensor"],
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            flat = condition_features[-1].flatten(1)
            return self.prior_mu(flat), self.prior_logvar(flat).clamp(-12.0, 12.0)

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
        ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
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
            delta_logits = self.final(torch.cat([value, condition, plane], dim=1))
            add_logits = delta_logits[:, :1]
            remove_logits = delta_logits[:, 1:2]
            add_probability = torch.sigmoid(add_logits)
            remove_probability = torch.sigmoid(remove_logits)
            composed_probability = (
                condition * (1.0 - remove_probability)
                + (1.0 - condition) * add_probability
            ).clamp(1e-5, 1.0 - 1e-5)
            composed_logits = torch.logit(composed_probability)
            pooled = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
            strength = edit_strength[:, None] if edit_strength.ndim == 1 else edit_strength
            width_raw = self.width_head(torch.cat([z, pooled, strength], dim=1)).squeeze(1)
            width = self.config.min_stroke_width + (
                self.config.max_stroke_width - self.config.min_stroke_width
            ) * torch.sigmoid(width_raw)
            return composed_logits, add_logits, remove_logits, width

        def forward(
            self,
            target: "torch.Tensor",
            condition: "torch.Tensor",
            edit_strength: "torch.Tensor",
        ) -> dict[str, "torch.Tensor"]:
            if target.shape != condition.shape or target.ndim != 4 or target.shape[1] != 1:
                raise ValueError("target and condition must both have shape [B,1,H,W]")
            features = self.encode_condition(condition, edit_strength)
            mu, logvar = self.encode_posterior(target, condition, edit_strength)
            prior_mu, prior_logvar = self.encode_prior(features)
            z = self.reparameterize(mu, logvar)
            logits, add_logits, remove_logits, width = self.decode(
                z, condition, edit_strength, features
            )
            return {
                "logits": logits,
                "add_logits": add_logits,
                "remove_logits": remove_logits,
                "mu": mu,
                "logvar": logvar,
                "prior_mu": prior_mu,
                "prior_logvar": prior_logvar,
                "width": width,
            }

        @torch.no_grad()
        def sample(
            self,
            condition: "torch.Tensor",
            edit_strength: "torch.Tensor",
            z: "torch.Tensor | None" = None,
            temperature: float = 0.9,
            return_components: bool = False,
        ) -> Any:
            features = self.encode_condition(condition, edit_strength)
            if z is None:
                prior_mu, prior_logvar = self.encode_prior(features)
                noise = torch.randn(
                    condition.shape[0],
                    self.config.latent_dim,
                    device=condition.device,
                    dtype=condition.dtype,
                )
                z = prior_mu + noise * torch.exp(0.5 * prior_logvar) * float(temperature)
            logits, add_logits, remove_logits, width = self.decode(
                z, condition, edit_strength, features
            )
            if return_components:
                return {
                    "logits": logits,
                    "add_logits": add_logits,
                    "remove_logits": remove_logits,
                    "width": width,
                }
            return logits, width


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


@dataclass
class PairedSymbol:
    """One canonical, registered base-to-deviation training example."""

    family_id: str
    base: ProcessedSymbol
    target: ProcessedSymbol
    registered_base: np.ndarray
    addition_mask: np.ndarray
    removal_mask: np.ndarray
    registration: dict[str, float]
    raw_change_ratio: float
    strength: float = 0.0
    leakage_group: str = ""
    cross_family_duplicate: bool = False
    split: str = "train"


def _delta_masks(
    condition: np.ndarray,
    target: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    condition_line = skeletonize_mask(condition)
    target_line = skeletonize_mask(target)
    addition = target_line & ~_within_euclidean_tolerance(condition_line, tolerance)
    removal = condition_line & ~_within_euclidean_tolerance(target_line, tolerance)
    return addition, removal


def _joint_pair_augmentation(
    condition: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    *,
    condition_jitter: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a small shared affine perturbation plus condition-only scan jitter."""

    angle = float(rng.uniform(-3.0, 3.0))
    scale = float(rng.uniform(0.97, 1.03))
    dx = int(rng.integers(-2, 3))
    dy = int(rng.integers(-2, 3))
    condition = _transform_mask(condition, angle=angle, scale=scale, dx=dx, dy=dy)
    target = _transform_mask(target, angle=angle, scale=scale, dx=dx, dy=dy)
    if condition_jitter and rng.random() < 0.5:
        condition = _transform_mask(
            condition,
            angle=float(rng.uniform(-0.6, 0.6)),
            dx=int(rng.integers(-1, 2)),
            dy=int(rng.integers(-1, 2)),
        )
    return skeletonize_mask(condition), skeletonize_mask(target)


def _empirical_percentile(sorted_values: np.ndarray, value: float) -> float:
    """Map a raw change score to the midpoint rank of its empirical CDF."""

    values = np.asarray(sorted_values, dtype=np.float32)
    if values.size == 0:
        return float(np.clip(value, 0.0, 1.0))
    left = int(np.searchsorted(values, value, side="left"))
    right = int(np.searchsorted(values, value, side="right"))
    return float(np.clip((left + right) / (2.0 * values.size), 0.0, 1.0))


if torch is not None:

    class PairedSymbolDataset(Dataset):
        """Deterministic family-balanced real/synthetic/identity sampler."""

        def __init__(
            self,
            pairs: Sequence[PairedSymbol],
            *,
            seed: int = 1337,
            training: bool = True,
            real_probability: float = 0.60,
            synthetic_probability: float = 0.30,
            identity_probability: float = 0.10,
            match_tolerance: float = 3.0,
        ) -> None:
            if not pairs:
                raise ValueError("PairedSymbolDataset requires at least one pair")
            self.pairs = list(pairs)
            self.seed = int(seed)
            self.training = bool(training)
            self.probabilities = (
                float(real_probability),
                float(synthetic_probability),
                float(identity_probability),
            )
            self.match_tolerance = float(match_tolerance)
            self.epoch = 0
            grouped: dict[str, list[PairedSymbol]] = defaultdict(list)
            for pair in self.pairs:
                grouped[pair.family_id].append(pair)
            self.family_ids = sorted(grouped, key=str.casefold)
            self.by_family = {
                key: sorted(grouped[key], key=lambda item: mask_hash(item.target.line_mask))
                for key in self.family_ids
            }
            self.samples_per_family = max(len(items) for items in self.by_family.values())
            self.raw_strengths = np.sort(
                np.asarray([pair.raw_change_ratio for pair in self.pairs], dtype=np.float32)
            )

        def set_epoch(self, epoch: int) -> None:
            self.epoch = int(epoch)

        def __len__(self) -> int:
            if not self.training:
                return len(self.pairs)
            return self.samples_per_family * len(self.family_ids)

        def _select_pair(self, index: int) -> PairedSymbol:
            if not self.training:
                return self.pairs[index]
            family_index = index % len(self.family_ids)
            family_id = self.family_ids[family_index]
            choices = self.by_family[family_id]
            cycle = index // len(self.family_ids)
            offset_seed = int.from_bytes(
                hashlib.sha256(
                    f"{self.seed}\0{self.epoch}\0{family_id}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            return choices[(cycle + offset_seed) % len(choices)]

        def __getitem__(self, index: int) -> dict[str, Any]:
            pair = self._select_pair(index)
            sequence_seed = self.seed + index * 104729 + self.epoch * 1_000_003
            rng = np.random.default_rng(sequence_seed)
            choice = float(rng.random()) if self.training else 0.0
            real_limit = self.probabilities[0]
            synthetic_limit = real_limit + self.probabilities[1]

            if choice < real_limit:
                example_type = "real"
                condition = pair.registered_base.copy()
                target = pair.target.line_mask.copy()
                strength = pair.strength
            elif choice < synthetic_limit:
                example_type = "synthetic"
                target = pair.target.line_mask.copy()
                condition, _ = synthesize_condition(
                    target,
                    rng,
                    edit_strength=float(rng.uniform(0.15, 0.85)),
                    empty_probability=0.0,
                )
                change = changed_line_amount(condition, target, self.match_tolerance)
                strength = _empirical_percentile(
                    self.raw_strengths, float(change["change_fraction"])
                )
            else:
                example_type = "identity"
                condition = pair.registered_base.copy()
                target = condition.copy()
                strength = 0.0

            if self.training:
                condition, target = _joint_pair_augmentation(
                    condition,
                    target,
                    rng,
                    condition_jitter=example_type != "identity",
                )
            addition, removal = _delta_masks(condition, target, self.match_tolerance)
            target_render = _dilate_mask(target, 1).astype(np.float32)
            condition_render = _dilate_mask(condition, 1).astype(np.float32)
            addition_render = _dilate_mask(addition, 1).astype(np.float32)
            removal_render = _dilate_mask(removal, 1).astype(np.float32)
            return {
                "target": torch.from_numpy(target_render[None]),
                "condition": torch.from_numpy(condition_render[None]),
                "strength": torch.tensor(strength, dtype=torch.float32),
                "target_width": torch.tensor(pair.target.stroke_width, dtype=torch.float32),
                "add_target": torch.from_numpy(addition_render[None]),
                "remove_target": torch.from_numpy(removal_render[None]),
                "family_id": pair.family_id,
                "example_type": example_type,
            }


    # Kept as a source-compatible name for callers that imported it, while the
    # accepted input is now explicitly paired rather than a flat symbol list.
    SymbolDataset = PairedSymbolDataset


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
    prior_mu: "torch.Tensor | None" = None,
    prior_logvar: "torch.Tensor | None" = None,
    add_logits: "torch.Tensor | None" = None,
    remove_logits: "torch.Tensor | None" = None,
    add_target: "torch.Tensor | None" = None,
    remove_target: "torch.Tensor | None" = None,
    condition: "torch.Tensor | None" = None,
    delta_weight: float = 0.5,
    retention_weight: float = 0.25,
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
    if prior_mu is None:
        prior_mu = torch.zeros_like(mu)
    if prior_logvar is None:
        prior_logvar = torch.zeros_like(logvar)
    # KL(q(z|base,target,strength) || p(z|base,strength)).
    variance_ratio = torch.exp(logvar - prior_logvar)
    mean_delta = (mu - prior_mu).square() * torch.exp(-prior_logvar)
    kl = (0.5 * (
        prior_logvar - logvar + variance_ratio + mean_delta - 1.0
    ).sum(dim=1)).mean()
    width = F.smooth_l1_loss(predicted_width, target_width)
    delta = torch.zeros((), device=logits.device, dtype=logits.dtype)
    retention = torch.zeros((), device=logits.device, dtype=logits.dtype)
    if (
        add_logits is not None
        and remove_logits is not None
        and add_target is not None
        and remove_target is not None
    ):
        delta = (
            F.binary_cross_entropy_with_logits(add_logits, add_target)
            + F.binary_cross_entropy_with_logits(remove_logits, remove_target)
        ) / 2.0
        if condition is not None:
            retained = condition * (1.0 - remove_target)
            retention = (torch.sigmoid(remove_logits) * retained).sum() / retained.sum().clamp_min(1.0)
    loss = (
        bce
        + dice
        + 0.5 * cldice
        + float(beta) * kl
        + 0.1 * width
        + float(delta_weight) * delta
        + float(retention_weight) * retention
    )
    metrics = {
        "loss": float(loss.detach()),
        "bce": float(bce.detach()),
        "dice": float(dice.detach()),
        "cldice": float(cldice.detach()),
        "kl": float(kl.detach()),
        "width": float(width.detach()),
        "delta": float(delta.detach()),
        "retention": float(retention.detach()),
        "beta": float(beta),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Public configuration and orchestration API
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[float, str, dict[str, Any] | None], None]


@dataclass(frozen=True)
class RegistrationConfig:
    angle_range: float = 12.0
    translation_range: int = 8
    scale_range: float = 0.12
    match_tolerance: float = 3.0
    minimum_overlap: float = 0.25

    def validate(self) -> None:
        if self.angle_range < 0 or self.angle_range > 45:
            raise ValueError("registration angle_range must be between 0 and 45 degrees")
        if self.translation_range < 0 or self.translation_range > 32:
            raise ValueError("registration translation_range must be between 0 and 32 pixels")
        if not 0.0 <= self.scale_range < 0.5:
            raise ValueError("registration scale_range must be in [0, 0.5)")
        if self.match_tolerance < 0 or self.match_tolerance > 12:
            raise ValueError("registration match_tolerance must be between 0 and 12 pixels")
        if not 0.0 <= self.minimum_overlap <= 1.0:
            raise ValueError("registration minimum_overlap must be between 0 and 1")


@dataclass(frozen=True)
class ValidationConfig:
    data: str = ""
    report: str = "validation"
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    validation_fraction: float = 0.10
    seed: int = 1337
    contact_sheet_page_size: int = 40

    def validate(self) -> None:
        self.preprocessing.validate()
        self.registration.validate()
        if not self.data:
            raise ValueError("data directory is required")
        if not self.report:
            raise ValueError("report directory is required")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if self.contact_sheet_page_size < 1:
            raise ValueError("contact_sheet_page_size must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    data: str = ""
    run: str = "runs/symbols"
    resume: str = ""
    init_checkpoint: str = ""
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    validation_fraction: float = 0.10
    device: str = "auto"
    epochs: int = 250
    batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    patience: int = 30
    seed: int = 1337
    beta_max: float = 1e-3
    beta_warmup_fraction: float = 0.25
    real_pair_probability: float = 0.60
    synthetic_pair_probability: float = 0.30
    identity_pair_probability: float = 0.10
    delta_loss_weight: float = 0.50
    retention_loss_weight: float = 0.25
    audit_sample_count: int = 32
    gradient_clip: float = 1.0
    workers: int = 0
    deterministic: bool = True
    mixed_precision: bool = True
    preview_count: int = 8
    preview_frequency: int = 10

    def validate(self) -> None:
        self.preprocessing.validate()
        self.registration.validate()
        self.model.validate()
        self.novelty.validate()
        self.quality.validate()
        if not self.data or not self.run:
            raise ValueError("data and run directories are required")
        if self.resume and self.init_checkpoint:
            raise ValueError("resume and init_checkpoint are mutually exclusive")
        if self.model.image_size != self.preprocessing.image_size:
            raise ValueError("model.image_size must match preprocessing.image_size")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size, and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.beta_max < 0 or not 0.0 < self.beta_warmup_fraction <= 1.0:
            raise ValueError("KL beta settings are invalid")
        mix = (
            self.real_pair_probability,
            self.synthetic_pair_probability,
            self.identity_pair_probability,
        )
        if any(value < 0.0 or value > 1.0 for value in mix) or not math.isclose(
            sum(mix), 1.0, abs_tol=1e-6
        ):
            raise ValueError("real, synthetic, and identity pair probabilities must sum to 1")
        if self.delta_loss_weight < 0 or self.retention_loss_weight < 0:
            raise ValueError("delta and retention loss weights must be non-negative")
        if self.audit_sample_count < 1:
            raise ValueError("audit_sample_count must be positive")
        if self.gradient_clip < 0 or self.workers < 0:
            raise ValueError("gradient_clip and workers must be non-negative")
        if self.preview_count < 1 or self.preview_frequency < 1:
            raise ValueError("preview_count and preview_frequency must be positive")


@dataclass(frozen=True)
class GenerationConfig:
    checkpoint: str = ""
    out: str = "generated"
    base: str = ""
    count: int = 50
    edit_strength: float = 0.35
    temperature: float = 0.9
    sampling_batch: int = 8
    threshold_override: float | None = None
    review_cap: int | None = None
    attempt_multiplier: int = 100
    seed: int = 1337
    device: str = "auto"
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

    def validate(self) -> None:
        self.novelty.validate()
        self.quality.validate()
        if not self.checkpoint or not self.base or not self.out:
            raise ValueError("checkpoint, base image, and output directory are required")
        if self.count < 1 or self.sampling_batch < 1 or self.attempt_multiplier < 1:
            raise ValueError("count, sampling_batch, and attempt_multiplier must be positive")
        if not 0.0 <= self.edit_strength <= 1.0:
            raise ValueError("edit_strength must be between 0 and 1")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.threshold_override is not None and not 0.05 <= self.threshold_override <= 0.95:
            raise ValueError("threshold_override must be between 0.05 and 0.95")
        if self.review_cap is not None and self.review_cap < 0:
            raise ValueError("review_cap must be non-negative or null")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")


def web_defaults() -> dict[str, Any]:
    """Return the complete JSON-safe control schema used by the local webpage."""

    return {
        "paths": {
            "data": "",
            "report": "validation",
            "run": "runs/symbols",
            "resume": "",
            "init_checkpoint": "",
            "checkpoint": "",
            "base": "",
            "out": "generated",
        },
        "preprocessing": {
            **asdict(PreprocessConfig()),
            "validation_fraction": 0.10,
        },
        "registration": asdict(RegistrationConfig()),
        "model": asdict(ModelConfig()),
        "training": {
            "device": "auto",
            "epochs": 250,
            "batch_size": 16,
            "learning_rate": 2e-4,
            "weight_decay": 1e-4,
            "patience": 30,
            "seed": 1337,
            "beta_max": 1e-3,
            "beta_warmup_fraction": 0.25,
            "real_pair_probability": 0.60,
            "synthetic_pair_probability": 0.30,
            "identity_pair_probability": 0.10,
            "delta_loss_weight": 0.50,
            "retention_loss_weight": 0.25,
            "audit_sample_count": 32,
            "gradient_clip": 1.0,
            "workers": 0,
            "deterministic": True,
            "mixed_precision": True,
            "preview_count": 8,
            "preview_frequency": 10,
        },
        "generation": {
            "count": 50,
            "edit_strength": 0.35,
            "temperature": 0.9,
            "sampling_batch": 8,
            "threshold_override": None,
            "review_cap": None,
            "attempt_multiplier": 100,
            "seed": 1337,
            "device": "auto",
        },
        "novelty": asdict(NoveltyConfig()),
        "quality": asdict(QualityConfig()),
        "safety": {
            "minimum_families": 4,
            "minimum_deviations_per_family": 20,
            "allowed_svg_elements": ["svg", "g", "path"],
            "stroke": "black",
            "fill": "none",
            "editable": False,
        },
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _section_values(config: Mapping[str, Any], sections: Sequence[str]) -> dict[str, Any]:
    """Merge flat values with selected nested web-form sections."""

    values = {key: value for key, value in config.items() if not isinstance(value, Mapping)}
    for section in sections:
        nested = config.get(section)
        if isinstance(nested, Mapping):
            values.update(nested)
    # Also accept dotted keys and HTML-friendly section_key names.
    for key, value in config.items():
        if isinstance(value, Mapping):
            continue
        for section in sections:
            dotted = f"{section}."
            prefixed = f"{section}_"
            if key.startswith(dotted):
                values[key[len(dotted) :]] = value
            elif key.startswith(prefixed):
                values[key[len(prefixed) :]] = value
    aliases = {
        "lr": "learning_rate",
        "batch": "batch_size",
        "kl_max": "beta_max",
        "warmup_fraction": "beta_warmup_fraction",
        "threshold": "threshold_override",
        "max_components": "maximum_components",
        "max_ink": "maximum_ink",
        "top_k": "shortlist_maximum",
    }
    for old, new in aliases.items():
        if old in values and new not in values:
            values[new] = values[old]
    return values


def _dataclass_kwargs(cls: type[Any], values: Mapping[str, Any]) -> dict[str, Any]:
    names = cls.__dataclass_fields__  # type: ignore[attr-defined]
    return {key: value for key, value in values.items() if key in names}


def _preprocess_config(values: Mapping[str, Any]) -> PreprocessConfig:
    typed = _dataclass_kwargs(PreprocessConfig, values)
    for key in ("image_size", "margin", "min_component_pixels", "max_input_pixels"):
        if key in typed:
            typed[key] = int(typed[key])
    if "max_source_stroke_width" in typed:
        typed["max_source_stroke_width"] = float(typed["max_source_stroke_width"])
    return PreprocessConfig(**typed)


def _model_config(values: Mapping[str, Any], image_size: int | None = None) -> ModelConfig:
    typed = _dataclass_kwargs(ModelConfig, values)
    if image_size is not None and "image_size" not in typed:
        typed["image_size"] = image_size
    for key in ("image_size", "latent_dim", "base_channels"):
        if key in typed:
            typed[key] = int(typed[key])
    for key in ("min_stroke_width", "max_stroke_width"):
        if key in typed:
            typed[key] = float(typed[key])
    return ModelConfig(**typed)


def _registration_config(values: Mapping[str, Any]) -> RegistrationConfig:
    typed = _dataclass_kwargs(RegistrationConfig, values)
    if "translation_range" in typed:
        typed["translation_range"] = int(typed["translation_range"])
    for key in set(typed) - {"translation_range"}:
        typed[key] = float(typed[key])
    return RegistrationConfig(**typed)


def _novelty_config(values: Mapping[str, Any]) -> NoveltyConfig:
    typed = _dataclass_kwargs(NoveltyConfig, values)
    for key in ("alignment_translation", "shortlist_maximum", "precise_finalists"):
        if key in typed:
            typed[key] = int(typed[key])
    for key in set(typed) - {"alignment_translation", "shortlist_maximum", "precise_finalists"}:
        typed[key] = float(typed[key])
    return NoveltyConfig(**typed)


def _quality_config(values: Mapping[str, Any]) -> QualityConfig:
    typed = _dataclass_kwargs(QualityConfig, values)
    for key in ("maximum_components", "parallel_bundle_threshold"):
        if key in typed:
            typed[key] = int(typed[key])
    for key in set(typed) - {"maximum_components", "parallel_bundle_threshold"}:
        typed[key] = float(typed[key])
    return QualityConfig(**typed)


def _coerce_validation_config(config: ValidationConfig | Mapping[str, Any]) -> ValidationConfig:
    if isinstance(config, ValidationConfig):
        result = config
    else:
        values = _section_values(config, ("paths", "preprocessing", "training", "validation"))
        result = ValidationConfig(
            data=str(values.get("data", "")),
            report=str(values.get("report", "validation")),
            preprocessing=_preprocess_config(values),
            registration=_registration_config(_section_values(config, ("registration",))),
            validation_fraction=float(values.get("validation_fraction", 0.10)),
            seed=int(values.get("seed", 1337)),
            contact_sheet_page_size=int(values.get("contact_sheet_page_size", 40)),
        )
    result.validate()
    return result


def _coerce_training_config(config: TrainingConfig | Mapping[str, Any]) -> TrainingConfig:
    if isinstance(config, TrainingConfig):
        result = config
    else:
        values = _section_values(
            config,
            ("paths", "preprocessing", "registration", "training"),
        )
        preprocessing = _preprocess_config(_section_values(config, ("preprocessing",)))
        result = TrainingConfig(
            data=str(values.get("data", "")),
            run=str(values.get("run", "runs/symbols")),
            resume=str(values.get("resume", "") or ""),
            init_checkpoint=str(values.get("init_checkpoint", "") or ""),
            preprocessing=preprocessing,
            registration=_registration_config(_section_values(config, ("registration",))),
            model=_model_config(_section_values(config, ("model",)), preprocessing.image_size),
            novelty=_novelty_config(_section_values(config, ("novelty",))),
            quality=_quality_config(_section_values(config, ("quality",))),
            validation_fraction=float(values.get("validation_fraction", 0.10)),
            device=str(values.get("device", "auto")).lower(),
            epochs=int(values.get("epochs", 250)),
            batch_size=int(values.get("batch_size", 16)),
            learning_rate=float(values.get("learning_rate", 2e-4)),
            weight_decay=float(values.get("weight_decay", 1e-4)),
            patience=int(values.get("patience", 30)),
            seed=int(values.get("seed", 1337)),
            beta_max=float(values.get("beta_max", 1e-3)),
            beta_warmup_fraction=float(values.get("beta_warmup_fraction", 0.25)),
            real_pair_probability=float(values.get("real_pair_probability", 0.60)),
            synthetic_pair_probability=float(values.get("synthetic_pair_probability", 0.30)),
            identity_pair_probability=float(values.get("identity_pair_probability", 0.10)),
            delta_loss_weight=float(values.get("delta_loss_weight", 0.50)),
            retention_loss_weight=float(values.get("retention_loss_weight", 0.25)),
            audit_sample_count=int(values.get("audit_sample_count", 32)),
            gradient_clip=float(values.get("gradient_clip", 1.0)),
            workers=int(values.get("workers", 0)),
            deterministic=_as_bool(values.get("deterministic", True)),
            mixed_precision=_as_bool(values.get("mixed_precision", True)),
            preview_count=int(values.get("preview_count", 8)),
            preview_frequency=int(values.get("preview_frequency", 10)),
        )
    result.validate()
    return result


def _coerce_generation_config(config: GenerationConfig | Mapping[str, Any]) -> GenerationConfig:
    if isinstance(config, GenerationConfig):
        result = config
    else:
        values = _section_values(config, ("paths", "generation"))
        threshold = values.get("threshold_override")
        review_cap = values.get("review_cap")
        result = GenerationConfig(
            checkpoint=str(values.get("checkpoint", "")),
            out=str(values.get("out", "generated")),
            base=str(values.get("base", "") or ""),
            count=int(values.get("count", 50)),
            edit_strength=float(values.get("edit_strength", 0.35)),
            temperature=float(values.get("temperature", 0.9)),
            sampling_batch=int(values.get("sampling_batch", values.get("batch_size", 8))),
            threshold_override=None if threshold in {None, ""} else float(threshold),
            review_cap=None if review_cap in {None, ""} else int(review_cap),
            attempt_multiplier=int(values.get("attempt_multiplier", 100)),
            seed=int(values.get("seed", 1337)),
            device=str(values.get("device", "auto")).lower(),
            novelty=_novelty_config(_section_values(config, ("novelty",))),
            quality=_quality_config(_section_values(config, ("quality",))),
        )
    result.validate()
    return result


def _emit_progress(
    callback: ProgressCallback | None,
    fraction: float,
    message: str,
    payload: dict[str, Any] | None = None,
) -> None:
    if callback is not None:
        callback(float(np.clip(fraction, 0.0, 1.0)), message, payload)


def _check_cancel(cancel_event: Any) -> None:
    if cancel_event is None:
        return
    is_set = getattr(cancel_event, "is_set", None)
    cancelled = bool(is_set()) if callable(is_set) else bool(cancel_event)
    if cancelled:
        raise SymbolGeneratorError("Operation cancelled")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for record in records for key in record})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        key: json.dumps(value, sort_keys=True, default=_json_default)
                        if isinstance(value, (dict, list, tuple))
                        else value
                        for key, value in record.items()
                    }
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass
class _PreparedDataset:
    pairs: list[PairedSymbol]
    train_pairs: list[PairedSymbol]
    validation_pairs: list[PairedSymbol]
    bases: dict[str, ProcessedSymbol]
    records: list[dict[str, Any]]
    summary: dict[str, Any]
    family_summaries: list[dict[str, Any]]
    dataset_fingerprint: str
    artifacts: dict[str, Any]


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    output = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    target_x0, target_x1 = max(0, dx), min(width, width + dx)
    target_y0, target_y1 = max(0, dy), min(height, height + dy)
    if source_x1 > source_x0 and source_y1 > source_y0:
        output[target_y0:target_y1, target_x0:target_x1] = mask[
            source_y0:source_y1, source_x0:source_x1
        ]
    return output


def register_base_to_target(
    base_mask: np.ndarray,
    target_mask: np.ndarray,
    config: RegistrationConfig | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Register a canonical base to a complete deviation with a similarity search."""

    resolved = config or RegistrationConfig()
    resolved.validate()
    base = skeletonize_mask(base_mask)
    target = skeletonize_mask(target_mask)
    target_tolerant = _within_euclidean_tolerance(target, resolved.match_tolerance)
    if not base.any() or not target.any():
        raise SymbolGeneratorError("registration requires non-empty base and target masks")
    original_pixels = int(base.sum())

    best_score = -1.0
    best = (0.0, 1.0, 0, 0)
    best_mask = base

    def consider(geometry: np.ndarray, angle: float, scale: float, dx: int, dy: int) -> None:
        nonlocal best_score, best, best_mask
        shifted = _shift_mask(geometry, dx, dy)
        if not shifted.any():
            return
        # Score against the original base length so transforms cannot improve
        # their overlap merely by clipping difficult pixels off the canvas.
        overlap = float(np.sum(shifted & target_tolerant) / original_pixels)
        # Prefer the smaller transform when scores are indistinguishable.
        complexity = abs(angle) + abs(scale - 1.0) * 20.0 + abs(dx) + abs(dy)
        old_complexity = abs(best[0]) + abs(best[1] - 1.0) * 20.0 + abs(best[2]) + abs(best[3])
        if overlap > best_score + 1e-9 or (
            math.isclose(overlap, best_score, abs_tol=1e-9) and complexity < old_complexity
        ):
            best_score = overlap
            best = (float(angle), float(scale), int(dx), int(dy))
            best_mask = shifted

    angle_values = np.linspace(
        -resolved.angle_range,
        resolved.angle_range,
        max(1, int(math.ceil(resolved.angle_range / 2.0)) + 1),
    )
    scale_values = np.linspace(
        1.0 - resolved.scale_range,
        1.0 + resolved.scale_range,
        5 if resolved.scale_range else 1,
    )
    translation_step = max(1, resolved.translation_range // 2)
    translation_values = list(
        range(-resolved.translation_range, resolved.translation_range + 1, translation_step)
    )
    if resolved.translation_range not in translation_values:
        translation_values.append(resolved.translation_range)
    for angle in angle_values:
        for scale in scale_values:
            geometry = _transform_mask(base, angle=float(angle), scale=float(scale))
            for dy in translation_values:
                for dx in translation_values:
                    consider(geometry, float(angle), float(scale), dx, dy)

    coarse_angle, coarse_scale, coarse_dx, coarse_dy = best
    fine_angles = np.unique(
        np.clip(
            np.linspace(coarse_angle - 2.0, coarse_angle + 2.0, 5),
            -resolved.angle_range,
            resolved.angle_range,
        )
    )
    fine_scales = np.unique(
        np.clip(
            np.linspace(coarse_scale - 0.03, coarse_scale + 0.03, 5),
            1.0 - resolved.scale_range,
            1.0 + resolved.scale_range,
        )
    )
    for angle in fine_angles:
        for scale in fine_scales:
            geometry = _transform_mask(base, angle=float(angle), scale=float(scale))
            for dy in range(
                max(-resolved.translation_range, coarse_dy - 2),
                min(resolved.translation_range, coarse_dy + 2) + 1,
            ):
                for dx in range(
                    max(-resolved.translation_range, coarse_dx - 2),
                    min(resolved.translation_range, coarse_dx + 2) + 1,
                ):
                    consider(geometry, float(angle), float(scale), dx, dy)

    if best_score < resolved.minimum_overlap:
        raise SymbolGeneratorError(
            f"registration overlap {best_score:.3f} is below minimum {resolved.minimum_overlap:.3f}"
        )
    return skeletonize_mask(best_mask), {
        "angle": best[0],
        "scale": best[1],
        "dx": float(best[2]),
        "dy": float(best[3]),
        "overlap": float(best_score),
    }


def _pair_contact_sheets(
    pairs: Sequence[PairedSymbol],
    directory: Path,
    page_size: int,
    max_input_pixels: int,
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    size = pairs[0].target.line_mask.shape[0] if pairs else 128

    def source_thumbnail(symbol: ProcessedSymbol) -> Image.Image:
        try:
            gray, _ = _load_grayscale(symbol.source, max_input_pixels)
            image = Image.fromarray(gray, mode="L").convert("RGB")
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            framed = Image.new("RGB", (size, size), "white")
            framed.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
            return framed
        except (OSError, SymbolGeneratorError, ValueError):
            return Image.fromarray(symbol.normalized_gray, mode="L").convert("RGB")

    for page_index, offset in enumerate(range(0, len(pairs), page_size), start=1):
        page = list(pairs[offset : offset + page_size])
        canvas = Image.new("RGB", (size * 5, len(page) * (size + 24)), "white")
        draw = ImageDraw.Draw(canvas)
        for row, pair in enumerate(page):
            y = row * (size + 24)
            base_original = source_thumbnail(pair.base)
            deviation = source_thumbnail(pair.target)
            registered = Image.fromarray(
                np.where(pair.registered_base, 0, 255).astype(np.uint8), mode="L"
            ).convert("RGB")
            processed = Image.fromarray(
                np.where(pair.target.line_mask, 0, 255).astype(np.uint8), mode="L"
            ).convert("RGB")
            overlay = np.full((size, size, 3), 255, dtype=np.uint8)
            overlay[pair.registered_base] = (80, 80, 80)
            overlay[pair.addition_mask] = (210, 36, 36)
            overlay[pair.removal_mask] = (35, 95, 205)
            for column, image in enumerate(
                (base_original, deviation, registered, processed, Image.fromarray(overlay, mode="RGB"))
            ):
                canvas.paste(image, (column * size, y))
            draw.text(
                (4, y + size + 3),
                f"{Path(pair.target.source).name[:30]}  overlap {pair.registration['overlap']:.3f}",
                fill="#333333",
            )
        path = directory / f"pairs-{page_index:03d}.png"
        canvas.save(path)
        paths.append(str(path.resolve()))
    return paths


def _assign_leakage_groups(pairs: Sequence[PairedSymbol]) -> None:
    count = len(pairs)
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    hashes: dict[str, int] = {}
    descriptors = [
        _coarse_descriptor(skeletonize_mask(pair.target.line_mask)) for pair in pairs
    ]
    tolerant = [_dilate_mask(skeletonize_mask(pair.target.line_mask), 1) for pair in pairs]
    for index, pair in enumerate(pairs):
        digest = mask_hash(pair.target.line_mask)
        if digest in hashes:
            union(index, hashes[digest])
        else:
            hashes[digest] = index
        if index:
            scores = np.asarray([float(descriptors[index] @ value) for value in descriptors[:index]])
            # Leakage safety is more important than a fixed shortlist here:
            # exhaust every descriptor-qualified prior item before splitting.
            candidates = np.flatnonzero(scores >= 0.90)
            for candidate in candidates:
                if _dice(tolerant[index], tolerant[candidate]) >= 0.94:
                    union(index, int(candidate))

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        grouped[find(index)].append(index)
    for members in grouped.values():
        member_hashes = sorted(mask_hash(pairs[index].target.line_mask) for index in members)
        group_id = hashlib.sha256("\0".join(member_hashes).encode("ascii")).hexdigest()
        families = {pairs[index].family_id.casefold() for index in members}
        for index in members:
            pairs[index].leakage_group = group_id
            pairs[index].cross_family_duplicate = len(families) > 1


def _family_split(
    pairs: Sequence[PairedSymbol],
    fraction: float,
    seed: int,
) -> tuple[list[PairedSymbol], list[PairedSymbol]]:
    by_family: dict[str, list[PairedSymbol]] = defaultdict(list)
    by_group: dict[str, list[PairedSymbol]] = defaultdict(list)
    for pair in pairs:
        by_family[pair.family_id].append(pair)
        by_group[pair.leakage_group].append(pair)
    validation_groups: set[str] = set()
    selected_by_family: Counter[str] = Counter()
    for family_id in sorted(by_family, key=str.casefold):
        family_pairs = by_family[family_id]
        needed = min(len(family_pairs) - 1, max(3, round(len(family_pairs) * fraction)))
        groups: dict[str, list[PairedSymbol]] = defaultdict(list)
        for pair in family_pairs:
            groups[pair.leakage_group].append(pair)
        ranked_groups = sorted(
            groups,
            key=lambda group: hashlib.sha256(
                f"{seed}\0{family_id.casefold()}\0{group}".encode("utf-8")
            ).digest(),
        )
        selected = selected_by_family[family_id]
        for group in ranked_groups:
            if selected >= needed:
                break
            if group in validation_groups:
                continue
            affected = Counter(pair.family_id for pair in by_group[group])
            if any(
                len(by_family[affected_family])
                - selected_by_family[affected_family]
                - affected_count
                < 1
                for affected_family, affected_count in affected.items()
            ):
                continue
            validation_groups.add(group)
            selected_by_family.update(affected)
            selected = selected_by_family[family_id]
    train, validation = [], []
    for pair in pairs:
        pair.split = "validation" if pair.leakage_group in validation_groups else "train"
        (validation if pair.split == "validation" else train).append(pair)
    return train, validation


def _dataset_fingerprint(
    pairs: Sequence[PairedSymbol],
    preprocessing: PreprocessConfig,
    registration: RegistrationConfig,
    validation_fraction: float,
    seed: int,
) -> str:
    payload = {
        "preprocessing": asdict(preprocessing),
        "registration": asdict(registration),
        "validation_fraction": float(validation_fraction),
        "split_seed": int(seed),
        "pairs": sorted(
            [
                {
                    "family_id": pair.family_id.casefold(),
                    "base_hash": mask_hash(pair.base.line_mask),
                    "target_hash": mask_hash(pair.target.line_mask),
                    "leakage_group": pair.leakage_group,
                    "split": pair.split,
                    "base_stroke_width": float(pair.base.stroke_width),
                    "target_stroke_width": float(pair.target.stroke_width),
                }
                for pair in pairs
            ],
            key=lambda item: (item["family_id"], item["target_hash"]),
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _prepare_dataset(
    config: ValidationConfig,
    *,
    progress: ProgressCallback | None,
    cancel_event: Any,
    manifest_stem: str = "manifest",
) -> _PreparedDataset:
    data_path = Path(config.data).expanduser().resolve()
    report_path = Path(config.report).expanduser().resolve()
    report_path.mkdir(parents=True, exist_ok=True)
    if not data_path.is_dir():
        raise SymbolGeneratorError(f"Dataset directory does not exist: {data_path}")
    records: list[dict[str, Any]] = []
    pairs: list[PairedSymbol] = []
    bases: dict[str, ProcessedSymbol] = {}
    structural_errors: list[str] = []
    family_directories = sorted(
        [item for item in data_path.iterdir() if item.is_dir() and item.resolve() != report_path],
        key=lambda item: item.name.casefold(),
    )
    family_casefolds: dict[str, str] = {}
    for directory in family_directories:
        folded = directory.name.casefold()
        if folded in family_casefolds:
            structural_errors.append(
                f"family IDs collide case-insensitively: {family_casefolds[folded]!r} and {directory.name!r}"
            )
        else:
            family_casefolds[folded] = directory.name

    all_images = discover_images(data_path, excluded=(report_path,))
    assigned_paths: set[Path] = set()
    total_work = max(1, len(all_images))
    processed_work = 0
    for family_directory in family_directories:
        family_id = family_directory.name
        base_candidates = sorted(
            [
                path
                for path in family_directory.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and path.stem.casefold() == "base"
            ],
            key=lambda path: path.name.casefold(),
        )
        deviation_directories = [
            path
            for path in family_directory.iterdir()
            if path.is_dir() and path.name.casefold() == "deviations"
        ]
        if len(base_candidates) != 1:
            structural_errors.append(
                f"family {family_id!r} must contain exactly one supported base.* image; found {len(base_candidates)}"
            )
        if len(deviation_directories) != 1:
            structural_errors.append(
                f"family {family_id!r} must contain exactly one deviations directory; found {len(deviation_directories)}"
            )
        if len(base_candidates) != 1 or len(deviation_directories) != 1:
            continue
        base_path = base_candidates[0]
        deviation_paths = discover_images(deviation_directories[0])
        assigned_paths.add(base_path.resolve())
        assigned_paths.update(path.resolve() for path in deviation_paths)
        base_record: dict[str, Any] = {
            "record_type": "base",
            "family_id": family_id,
            "source": str(base_path.resolve()),
            "relative_source": base_path.relative_to(data_path).as_posix(),
        }
        try:
            base = preprocess_image(base_path, config.preprocessing)
            bases[family_id] = base
            base_record.update(
                status="accepted",
                hash=mask_hash(base.line_mask),
                stroke_width=base.stroke_width,
                line_pixels=int(base.line_mask.sum()),
                conversion=base.conversion,
            )
        except Exception as exc:
            base_record.update(status="corrupt" if "read image" in str(exc).lower() else "rejected", error=str(exc))
            structural_errors.append(f"family {family_id!r} base is unusable: {exc}")
            base = None
        records.append(base_record)
        processed_work += 1
        if base is None:
            for path in deviation_paths:
                records.append(
                    {
                        "record_type": "deviation",
                        "family_id": family_id,
                        "source": str(path.resolve()),
                        "relative_source": path.relative_to(data_path).as_posix(),
                        "status": "invalid_family",
                        "error": "base image is unusable",
                    }
                )
            processed_work += len(deviation_paths)
            continue

        family_hashes: dict[str, str] = {}
        family_unique_masks: list[tuple[np.ndarray, str]] = []
        for deviation_path in deviation_paths:
            _check_cancel(cancel_event)
            relative = deviation_path.relative_to(data_path).as_posix()
            record: dict[str, Any] = {
                "record_type": "deviation",
                "family_id": family_id,
                "source": str(deviation_path.resolve()),
                "relative_source": relative,
            }
            try:
                target = preprocess_image(deviation_path, config.preprocessing)
                digest = mask_hash(target.line_mask)
                record.update(
                    hash=digest,
                    stroke_width=target.stroke_width,
                    line_pixels=int(target.line_mask.sum()),
                    conversion=target.conversion,
                )
                if digest in family_hashes:
                    record.update(status="duplicate", duplicate_of=family_hashes[digest])
                else:
                    target_tolerant = _dilate_mask(skeletonize_mask(target.line_mask), 1)
                    near_match = next(
                        (
                            source
                            for existing, source in family_unique_masks
                            if _dice(target_tolerant, existing) >= 0.995
                        ),
                        None,
                    )
                    if near_match is not None:
                        record.update(
                            status="duplicate",
                            duplicate_of=near_match,
                            duplicate_type="near",
                        )
                        records.append(record)
                        processed_work += 1
                        _emit_progress(
                            progress,
                            0.72 * processed_work / total_work,
                            f"Prepared {processed_work}/{len(all_images)} family images",
                            {"current": relative, "counts": dict(Counter(item["status"] for item in records))},
                        )
                        continue
                    registered, registration = register_base_to_target(
                        base.line_mask, target.line_mask, config.registration
                    )
                    addition, removal = _delta_masks(
                        registered, target.line_mask, config.registration.match_tolerance
                    )
                    changed = float(addition.sum() + removal.sum())
                    fraction = float(changed / max(1, int(skeletonize_mask(registered).sum())))
                    record.update(
                        registration=registration,
                        added_pixels=int(addition.sum()),
                        removed_pixels=int(removal.sum()),
                        change_fraction=fraction,
                    )
                    if (
                        changed < 8.0
                        and fraction < 0.08
                    ):
                        record.update(status="no_op", error="deviation changes fewer than 8px and 8%")
                    else:
                        family_hashes[digest] = relative
                        family_unique_masks.append((target_tolerant, relative))
                        record["status"] = "accepted"
                        pairs.append(
                            PairedSymbol(
                                family_id=family_id,
                                base=base,
                                target=target,
                                registered_base=registered,
                                addition_mask=addition,
                                removal_mask=removal,
                                registration=registration,
                                raw_change_ratio=float(
                                    (addition.sum() + 0.5 * removal.sum())
                                    / max(1, int(skeletonize_mask(registered).sum()))
                                ),
                            )
                        )
            except Exception as exc:
                message = str(exc)
                lowered = message.lower()
                if "could not read image" in lowered:
                    status = "corrupt"
                elif any(word in lowered for word in ("blank", "uniform", "no dark", "no usable foreground")):
                    status = "blank"
                elif "registration" in lowered:
                    status = "registration_failed"
                else:
                    status = "rejected"
                record.update(status=status, error=message)
            records.append(record)
            processed_work += 1
            _emit_progress(
                progress,
                0.72 * processed_work / total_work,
                f"Prepared {processed_work}/{len(all_images)} family images",
                {"current": relative, "counts": dict(Counter(item["status"] for item in records))},
            )

    for path in all_images:
        if path.resolve() not in assigned_paths:
            records.append(
                {
                    "record_type": "unassigned",
                    "source": str(path.resolve()),
                    "relative_source": path.relative_to(data_path).as_posix(),
                    "status": "unassigned",
                    "error": "supported image is outside family/base.* or family/deviations/",
                }
            )
            structural_errors.append(
                f"unassigned supported image: {path.relative_to(data_path).as_posix()}"
            )

    base_hash_families: dict[str, list[str]] = defaultdict(list)
    for family_id, base in bases.items():
        base_hash_families[mask_hash(base.line_mask)].append(family_id)
    for duplicate_families in base_hash_families.values():
        if len(duplicate_families) > 1:
            structural_errors.append(
                "duplicate canonical bases across families: " + ", ".join(sorted(duplicate_families))
            )

    _assign_leakage_groups(pairs)
    train_items, validation_items = _family_split(pairs, config.validation_fraction, config.seed)
    raw_strengths = np.asarray([pair.raw_change_ratio for pair in pairs], dtype=np.float32)
    if len(raw_strengths):
        sorted_strengths = np.sort(raw_strengths, kind="stable")
        for pair in pairs:
            pair.strength = _empirical_percentile(
                sorted_strengths, pair.raw_change_ratio
            )

    pair_by_source = {pair.target.source: pair for pair in pairs}
    for record in records:
        pair = pair_by_source.get(record.get("source", ""))
        if pair is not None and record.get("status") == "accepted":
            record.update(
                split=pair.split,
                leakage_group=pair.leakage_group,
                cross_family_duplicate=pair.cross_family_duplicate,
                strength=pair.strength,
            )

    family_summaries: list[dict[str, Any]] = []
    for family_id in sorted(bases, key=str.casefold):
        family_pairs = [pair for pair in pairs if pair.family_id == family_id]
        summary_row = {
            "family_id": family_id,
            "base": bases[family_id].source,
            "usable_deviations": len(family_pairs),
            "train": sum(pair.split == "train" for pair in family_pairs),
            "validation": sum(pair.split == "validation" for pair in family_pairs),
            "cross_family_duplicates": sum(pair.cross_family_duplicate for pair in family_pairs),
        }
        family_summaries.append(summary_row)
        if len(family_pairs) < 20:
            structural_errors.append(
                f"family {family_id!r} requires at least 20 unique usable deviations; found {len(family_pairs)}"
            )
        elif len(family_pairs) < 30:
            warnings.warn(
                f"Family {family_id!r} has only {len(family_pairs)} usable deviations; 30 or more is recommended.",
                RuntimeWarning,
            )
        if family_pairs and summary_row["train"] < 1:
            structural_errors.append(
                f"family {family_id!r} has no leakage-safe training deviation"
            )
        if len(family_pairs) >= 20 and summary_row["validation"] < 3:
            structural_errors.append(
                f"family {family_id!r} cannot supply 3 leakage-safe validation deviations"
            )
    if len([row for row in family_summaries if row["usable_deviations"] >= 20]) < 4:
        structural_errors.append("training requires at least 4 valid families")

    fingerprint = _dataset_fingerprint(
        pairs,
        config.preprocessing,
        config.registration,
        config.validation_fraction,
        config.seed,
    )

    manifest_json = report_path / f"{manifest_stem}.json"
    manifest_csv = report_path / f"{manifest_stem}.csv"
    summary: dict[str, Any] = dict(Counter(record["status"] for record in records))
    summary.update(
        {
            "discovered": len(all_images),
            "families": len(family_summaries),
            "valid_families": sum(row["usable_deviations"] >= 20 for row in family_summaries),
            "unique": len(pairs),
            "usable_deviations": len(pairs),
            "train": len(train_items),
            "validation": len(validation_items),
            "structural_errors": len(structural_errors),
        }
    )
    _write_json(
        manifest_json,
        {
            "schema_version": SCHEMA_VERSION,
            "data": str(data_path),
            "dataset_fingerprint": fingerprint,
            "summary": summary,
            "families": family_summaries,
            "errors": sorted(set(structural_errors)),
            "records": records,
        },
    )
    _write_csv(manifest_csv, records)
    contact_sheets: dict[str, list[str]] = {}
    for family_id in sorted(bases, key=str.casefold):
        safe_family = re.sub(r"[^A-Za-z0-9._-]+", "-", family_id).strip("-.") or "family"
        contact_sheets[family_id] = _pair_contact_sheets(
            [pair for pair in pairs if pair.family_id == family_id],
            report_path / "contact-sheets" / safe_family,
            config.contact_sheet_page_size,
            config.preprocessing.max_input_pixels,
        )
    config_path = report_path / "config.json"
    _write_json(config_path, asdict(config))
    artifacts = {
        "report": str(report_path),
        "config": str(config_path),
        "manifest_json": str(manifest_json),
        "manifest_csv": str(manifest_csv),
        "contact_sheets": contact_sheets,
        "dataset_fingerprint": fingerprint,
    }
    if structural_errors:
        _emit_progress(
            progress,
            1.0,
            "Dataset validation failed",
            {"summary": summary, "artifacts": artifacts, "errors": sorted(set(structural_errors))},
        )
        raise SymbolGeneratorError(
            "Dataset is not a valid paired-family dataset: " + "; ".join(sorted(set(structural_errors)))
        )
    _emit_progress(progress, 1.0, "Dataset validation complete", {"summary": summary, "families": family_summaries, "artifacts": artifacts})
    return _PreparedDataset(
        pairs,
        train_items,
        validation_items,
        bases,
        records,
        summary,
        family_summaries,
        fingerprint,
        artifacts,
    )


def validate_dataset(
    config: ValidationConfig | Mapping[str, Any],
    progress: ProgressCallback | None = None,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Validate, register, deduplicate, and split a paired-family dataset."""

    resolved = _coerce_validation_config(config)
    prepared = _prepare_dataset(resolved, progress=progress, cancel_event=cancel_event)
    return {
        "status": "complete",
        "message": (
            f"Validated {prepared.summary['unique']} deviations across "
            f"{prepared.summary['valid_families']} families"
        ),
        "summary": prepared.summary,
        "families": prepared.family_summaries,
        "dataset_fingerprint": prepared.dataset_fingerprint,
        "artifacts": prepared.artifacts,
    }


def _seed_data_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _make_loader(
    dataset: Any,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> Any:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, len(dataset))),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_data_worker if workers else None,
        generator=generator,
        # Worker-local dataset copies must be recreated after set_epoch so the
        # deterministic real/synthetic/identity mix advances each epoch.
        persistent_workers=False,
    )


def _average_metric_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(np.mean([float(row[key]) for row in rows if key in row]))
        for key in keys
    }


def _evaluate_model(
    model: Any,
    loader: Any,
    device: Any,
    beta: float,
    cancel_event: Any,
    *,
    delta_weight: float = 0.5,
    retention_weight: float = 0.25,
) -> dict[str, float]:
    model.eval()
    by_family: dict[str, list[tuple[int, dict[str, float]]]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            _check_cancel(cancel_event)
            tensors = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if hasattr(value, "to")
            }
            output = model(tensors["target"], tensors["condition"], tensors["strength"])
            family_ids = [str(value) for value in batch["family_id"]]
            for family_id in sorted(set(family_ids), key=str.casefold):
                indices = [
                    index for index, value in enumerate(family_ids) if value == family_id
                ]
                index_tensor = torch.tensor(indices, dtype=torch.long, device=device)

                def select(value: Any) -> Any:
                    return value.index_select(0, index_tensor)

                _, row = vae_loss(
                    select(output["logits"]),
                    select(tensors["target"]),
                    select(output["mu"]),
                    select(output["logvar"]),
                    select(output["width"]),
                    select(tensors["target_width"]),
                    beta=beta,
                    prior_mu=select(output["prior_mu"]),
                    prior_logvar=select(output["prior_logvar"]),
                    add_logits=select(output["add_logits"]),
                    remove_logits=select(output["remove_logits"]),
                    add_target=select(tensors["add_target"]),
                    remove_target=select(tensors["remove_target"]),
                    condition=select(tensors["condition"]),
                    delta_weight=delta_weight,
                    retention_weight=retention_weight,
                )
                by_family[family_id].append((len(indices), row))
    family_metrics: dict[str, dict[str, float]] = {}
    for family_id, entries in by_family.items():
        keys = sorted({key for _, row in entries for key in row})
        family_metrics[family_id] = {
            key: float(
                sum(count * float(row[key]) for count, row in entries if key in row)
                / max(1, sum(count for count, row in entries if key in row))
            )
            for key in keys
        }
    keys = sorted({key for row in family_metrics.values() for key in row})
    result = {
        key: float(np.mean([row[key] for row in family_metrics.values() if key in row]))
        for key in keys
    }
    result["family_count"] = float(len(family_metrics))
    return result


def _calibrate_model(
    model: Any,
    loader: Any,
    device: Any,
    cancel_event: Any,
) -> dict[str, Any]:
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            _check_cancel(cancel_event)
            target_device = batch["target"].to(device)
            condition_device = batch["condition"].to(device)
            strength_device = batch["strength"].to(device)
            output = model(target_device, condition_device, strength_device)
            probabilities.extend(torch.sigmoid(output["logits"]).cpu().numpy()[:, 0])
            targets.extend(batch["target"].numpy()[:, 0] >= 0.5)
            conditions.extend(batch["condition"].numpy()[:, 0] >= 0.5)

    threshold_scores: list[tuple[float, float]] = []
    for threshold in np.linspace(0.20, 0.80, 25):
        scores = [_dice(probability >= threshold, target) for probability, target in zip(probabilities, targets)]
        threshold_scores.append((float(np.mean(scores)), float(threshold)))
    _, best_threshold = max(threshold_scores, key=lambda item: (item[0], -abs(item[1] - 0.5)))

    changes: list[dict[str, float]] = []
    for probability, condition in zip(probabilities, conditions):
        if condition.any():
            changes.append(changed_line_amount(condition, probability >= best_threshold))
    if changes:
        similarities = np.asarray([item["similarity"] for item in changes], dtype=np.float32)
        fractions = np.asarray([item["change_fraction"] for item in changes], dtype=np.float32)
        guided_bounds = {
            "similarity_min": float(np.percentile(similarities, 5)),
            "similarity_max": float(np.percentile(similarities, 95)),
            "change_fraction_min": float(np.percentile(fractions, 5)),
            "change_fraction_max": float(np.percentile(fractions, 95)),
            "removal_fraction_max": float(
                np.percentile(
                    [item.get("removed_pixels", 0.0) / max(1.0, item.get("changed_pixels", 1.0)) for item in changes],
                    95,
                )
            ),
        }
    else:
        guided_bounds = {
            "similarity_min": 0.0,
            "similarity_max": 1.0,
            "change_fraction_min": 0.0,
            "change_fraction_max": float("inf"),
            "removal_fraction_max": 1.0,
        }
    return {
        "threshold": best_threshold,
        "threshold_scores": [
            {"threshold": threshold, "dice": score}
            for score, threshold in threshold_scores
        ],
        "guided_bounds": guided_bounds,
    }


def _save_preview(
    model: Any,
    pairs: Sequence[PairedSymbol],
    path: Path,
    *,
    count: int,
    seed: int,
    threshold: float,
    device: Any,
) -> None:
    selected = list(pairs[:count])
    if not selected:
        return
    size = model.config.image_size
    conditions = [_dilate_mask(pair.registered_base, 1).astype(np.float32) for pair in selected]
    strengths = [pair.strength for pair in selected]
    condition_tensor = torch.from_numpy(np.stack(conditions)[:, None]).to(device)
    strength_tensor = torch.tensor(strengths, dtype=torch.float32, device=device)
    generator = torch.Generator()
    generator.manual_seed(seed)
    noise = torch.randn(len(selected), model.config.latent_dim, generator=generator).to(device)
    model.eval()
    with torch.no_grad():
        condition_features = model.encode_condition(condition_tensor, strength_tensor)
        prior_mu, prior_logvar = model.encode_prior(condition_features)
        z = prior_mu + noise * torch.exp(0.5 * prior_logvar) * 0.9
        sampled = model.sample(
            condition_tensor, strength_tensor, z=z, return_components=True
        )
    output = torch.sigmoid(sampled["logits"]).cpu().numpy()[:, 0] >= threshold
    additions = torch.sigmoid(sampled["add_logits"]).cpu().numpy()[:, 0] >= threshold
    removals = torch.sigmoid(sampled["remove_logits"]).cpu().numpy()[:, 0] >= threshold

    columns = min(4, len(selected))
    rows = math.ceil(len(selected) / columns)
    canvas = Image.new("RGB", (columns * size * 5, rows * (size + 20)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (condition, addition, removal, generated, pair) in enumerate(
        zip(conditions, additions, removals, output, selected)
    ):
        x = (index % columns) * size * 5
        y = (index // columns) * (size + 20)
        masks = (condition, addition, removal, generated, pair.target.line_mask)
        for column, mask in enumerate(masks):
            image = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), mode="L").convert("RGB")
            canvas.paste(image, (x + column * size, y))
        draw.text((x + 3, y + size + 3), "base", fill="#555555")
        draw.text((x + size + 3, y + size + 3), "add", fill="#555555")
        draw.text((x + size * 2 + 3, y + size + 3), "remove", fill="#555555")
        draw.text((x + size * 3 + 3, y + size + 3), "composed", fill="#555555")
        draw.text((x + size * 4 + 3, y + size + 3), "target", fill="#555555")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _pack_reference_masks(symbols: Sequence[ProcessedSymbol]) -> Any:
    return _pack_masks([item.line_mask for item in symbols])


def _pack_masks(masks: Sequence[np.ndarray]) -> Any:
    if not masks:
        return torch.empty((0, 0), dtype=torch.uint8)
    packed = np.stack([np.packbits(np.asarray(item, dtype=bool).ravel()) for item in masks]).astype(np.uint8)
    return torch.from_numpy(np.ascontiguousarray(packed))


def _unpack_masks(packed_value: Any, shape: Sequence[int]) -> list[np.ndarray]:
    count, height, width = [int(value) for value in shape]
    packed = packed_value.cpu().numpy() if hasattr(packed_value, "cpu") else np.asarray(packed_value)
    return [
        np.unpackbits(packed[index])[: height * width].reshape(height, width).astype(bool)
        for index in range(count)
    ]


def _unpack_reference_masks(payload: Mapping[str, Any]) -> list[np.ndarray]:
    shape = [int(value) for value in payload["reference_shape"]]
    count, height, width = shape
    return _unpack_masks(payload["reference_masks_packed"], (count, height, width))


def _shortlist_recall(symbols: Sequence[ProcessedSymbol], top_k: int) -> dict[str, Any]:
    masks = [skeletonize_mask(item.line_mask) for item in symbols]
    descriptors = np.stack([_coarse_descriptor(item) for item in masks])
    hits = 0
    total = 0
    transforms = (
        {"angle": -6.0, "scale": 0.96, "dx": -3, "dy": 3},
        {"angle": 6.0, "scale": 1.04, "dx": 3, "dy": -3},
        {"angle": 0.0, "scale": 1.0, "dx": 2, "dy": 2},
    )
    for index, mask in enumerate(masks):
        for transform in transforms:
            descriptor = _coarse_descriptor(_transform_mask(mask, **transform))
            shortlist = np.argsort(-(descriptors @ descriptor), kind="stable")[: min(top_k, len(masks))]
            hits += int(index in shortlist)
            total += 1
    return {"hits": hits, "total": total, "recall": float(hits / max(1, total))}


def _generation_metadata(config: TrainingConfig, prepared: _PreparedDataset) -> dict[str, Any]:
    base_items = [prepared.bases[key] for key in sorted(prepared.bases, key=str.casefold)]
    target_items = [pair.target for pair in prepared.pairs]
    reference_items = base_items + target_items
    reference_names = (
        [f"{family_id}/base" for family_id in sorted(prepared.bases, key=str.casefold)]
        + [f"{pair.family_id}/deviations/{Path(pair.target.source).name}" for pair in prepared.pairs]
    )
    graphs = [mask_to_graph(item.line_mask) for item in reference_items]
    widths = np.asarray([item.stroke_width for item in reference_items], dtype=np.float32)
    shortlist_validation = _shortlist_recall(
        reference_items, config.novelty.shortlist_maximum
    )
    if shortlist_validation["recall"] < 1.0:
        warnings.warn(
            "Coarse novelty shortlist recall was below 100%; increase shortlist_maximum "
            "before relying on this run for duplicate filtering.",
            RuntimeWarning,
        )
    return {
        "preprocess_config": asdict(config.preprocessing),
        "registration_config": asdict(config.registration),
        "model_config": asdict(config.model),
        "novelty_config": asdict(config.novelty),
        "quality_config": asdict(config.quality),
        "reference_masks_packed": _pack_reference_masks(reference_items),
        "reference_shape": [len(reference_items), config.model.image_size, config.model.image_size],
        "reference_names": reference_names,
        "reference_hashes": [mask_hash(item.line_mask) for item in reference_items],
        "base_masks_packed": _pack_reference_masks(base_items),
        "base_shape": [len(base_items), config.model.image_size, config.model.image_size],
        "base_names": [f"{family_id}/base" for family_id in sorted(prepared.bases, key=str.casefold)],
        "base_hashes": [mask_hash(item.line_mask) for item in base_items],
        "target_masks_packed": _pack_reference_masks(target_items),
        "target_shape": [len(target_items), config.model.image_size, config.model.image_size],
        "target_names": [Path(pair.target.source).name for pair in prepared.pairs],
        "target_hashes": [mask_hash(item.line_mask) for item in target_items],
        "target_family_ids": [pair.family_id for pair in prepared.pairs],
        "family_associations": [
            {
                "family_id": pair.family_id,
                "base_hash": mask_hash(pair.base.line_mask),
                "target_hash": mask_hash(pair.target.line_mask),
                "leakage_group": pair.leakage_group,
                "cross_family_duplicate": bool(pair.cross_family_duplicate),
                "split": pair.split,
            }
            for pair in prepared.pairs
        ],
        "dataset_fingerprint": prepared.dataset_fingerprint,
        "strength_calibration": {
            "definition": "percentile of (added_length + 0.5 * removed_length) / base_length",
            "raw_values": sorted(float(pair.raw_change_ratio) for pair in prepared.pairs),
            "quantiles": {
                str(percentile): float(
                    np.percentile(
                        [pair.raw_change_ratio for pair in prepared.pairs], percentile
                    )
                )
                for percentile in (0, 5, 25, 50, 75, 95, 100)
            },
        },
        "registration_statistics": {
            "overlap_minimum": float(min(pair.registration["overlap"] for pair in prepared.pairs)),
            "overlap_median": float(np.median([pair.registration["overlap"] for pair in prepared.pairs])),
            "overlap_maximum": float(max(pair.registration["overlap"] for pair in prepared.pairs)),
        },
        "quality_baselines": compute_quality_baselines(graphs),
        "stroke_statistics": {
            "minimum": float(np.min(widths)),
            "maximum": float(np.max(widths)),
            "median": float(np.median(widths)),
            "p05": float(np.percentile(widths, 5)),
            "p95": float(np.percentile(widths, 95)),
        },
        "shortlist_validation": shortlist_validation,
    }


def _cpu_tree(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _rng_payload() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    result: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": {
            "kind": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["cuda_random_states"] = torch.cuda.get_rng_state_all()
    return result


def _restore_rng(payload: Mapping[str, Any]) -> None:
    if "python_random_state" in payload:
        random.setstate(payload["python_random_state"])
    numpy_payload = payload.get("numpy_random_state")
    if isinstance(numpy_payload, Mapping):
        np.random.set_state(
            (
                str(numpy_payload["kind"]),
                numpy_payload["state"].cpu().numpy().astype(np.uint32),
                int(numpy_payload["position"]),
                int(numpy_payload["has_gauss"]),
                float(numpy_payload["cached_gaussian"]),
            )
        )
    if "torch_random_state" in payload:
        torch.set_rng_state(payload["torch_random_state"].cpu())
    if torch.cuda.is_available() and "cuda_random_states" in payload:
        states = payload["cuda_random_states"]
        if len(states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(states)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(_cpu_tree(dict(payload)), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SymbolGeneratorError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        value = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SymbolGeneratorError(f"Could not safely load checkpoint {checkpoint_path}: {exc}") from exc
    if isinstance(value, dict) and value.get("kind") == LEGACY_CHECKPOINT_KIND:
        raise SymbolGeneratorError(
            "This checkpoint was trained by the legacy unpaired model. Retrain it with the "
            "paired-family dataset layout before resuming or generating."
        )
    if (
        not isinstance(value, dict)
        or value.get("kind") != CHECKPOINT_KIND
        or int(value.get("schema_version", -1)) != SCHEMA_VERSION
        or "model_state" not in value
    ):
        raise SymbolGeneratorError(f"Not a paired-family add-on checkpoint: {checkpoint_path}")
    return value


def _require_final_checkpoint(
    checkpoint: Mapping[str, Any], *, purpose: str
) -> None:
    stage = str(checkpoint.get("stage", ""))
    if stage != "final":
        raise SymbolGeneratorError(
            f"{purpose} requires a completed final checkpoint; found stage {stage or 'unknown'!r}. "
            "Audit-active checkpoints may only be used with --resume."
        )


def _checkpoint_payload(
    model: Any,
    config: TrainingConfig,
    metadata: Mapping[str, Any],
    *,
    epoch: int,
    best_metric: float,
    epochs_without_improvement: int,
    metrics_history: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    optimizer: Any | None = None,
    scaler: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "created_unix": float(time.time()),
        "model_state": _cpu_tree(model.state_dict()),
        "training_config": asdict(config),
        **dict(metadata),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "epochs_without_improvement": int(epochs_without_improvement),
        "metrics_history": list(metrics_history),
        "calibration": dict(calibration),
    }
    if optimizer is not None:
        payload["optimizer_state"] = _cpu_tree(optimizer.state_dict())
        payload["scaler_state"] = _cpu_tree(scaler.state_dict()) if scaler is not None else {}
        payload.update(_rng_payload())
    return payload


def _make_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older Torch compatibility.
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _training_batch_loss(
    model: Any,
    batch: Mapping[str, Any],
    device: Any,
    *,
    beta: float,
    delta_weight: float,
    retention_weight: float,
) -> tuple[Any, dict[str, float]]:
    tensors = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if hasattr(value, "to")
    }
    output = model(tensors["target"], tensors["condition"], tensors["strength"])
    return vae_loss(
        output["logits"],
        tensors["target"],
        output["mu"],
        output["logvar"],
        output["width"],
        tensors["target_width"],
        beta=beta,
        prior_mu=output["prior_mu"],
        prior_logvar=output["prior_logvar"],
        add_logits=output["add_logits"],
        remove_logits=output["remove_logits"],
        add_target=tensors["add_target"],
        remove_target=tensors["remove_target"],
        condition=tensors["condition"],
        delta_weight=delta_weight,
        retention_weight=retention_weight,
    )


def _audit_fold_assignments(pairs: Sequence[PairedSymbol], seed: int) -> list[dict[str, Any]]:
    counts = Counter(pair.family_id for pair in pairs)
    family_ids = sorted(
        counts,
        key=lambda family_id: hashlib.sha256(
            f"{seed}\0{family_id.casefold()}".encode("utf-8")
        ).digest(),
    )
    if len(family_ids) == 4:
        return [
            {"fold": index, "held_out_families": [family_id], "deviation_count": counts[family_id]}
            for index, family_id in enumerate(family_ids)
        ]
    buckets: list[list[str]] = [[] for _ in range(5)]
    totals = [0] * 5
    for family_id in sorted(family_ids, key=lambda item: (-counts[item], item.casefold())):
        bucket = min(range(5), key=lambda index: (totals[index], index))
        buckets[bucket].append(family_id)
        totals[bucket] += counts[family_id]
    return [
        {
            "fold": index,
            "held_out_families": sorted(bucket, key=str.casefold),
            "deviation_count": totals[index],
        }
        for index, bucket in enumerate(buckets)
    ]


def _score_unseen_families(
    model: Any,
    held_pairs: Sequence[PairedSymbol],
    device: Any,
    config: TrainingConfig,
    cancel_event: Any,
) -> dict[str, Any]:
    eligible = [pair for pair in held_pairs if not pair.cross_family_duplicate]
    excluded = len(held_pairs) - len(eligible)
    if not eligible:
        return {
            "eligible_deviations": 0,
            "excluded_cross_family_duplicates": excluded,
            "reconstruction_dice": 0.0,
            "prior_sample_quality": 0.0,
            "base_retention": 0.0,
            "diversity": 0.0,
            "change_distribution_agreement": 0.0,
        }
    model.eval()
    reconstruction: list[float] = []
    with torch.no_grad():
        for offset in range(0, len(eligible), config.batch_size):
            _check_cancel(cancel_event)
            page = eligible[offset : offset + config.batch_size]
            conditions = torch.from_numpy(
                np.stack([_dilate_mask(pair.registered_base, 1).astype(np.float32) for pair in page])[:, None]
            ).to(device)
            targets = torch.from_numpy(
                np.stack([_dilate_mask(pair.target.line_mask, 1).astype(np.float32) for pair in page])[:, None]
            ).to(device)
            strengths = torch.tensor([pair.strength for pair in page], dtype=torch.float32, device=device)
            output = model(targets, conditions, strengths)
            predicted = torch.sigmoid(output["logits"]).cpu().numpy()[:, 0] >= 0.5
            for mask, pair in zip(predicted, page):
                reconstruction.append(
                    _dice(skeletonize_mask(mask), skeletonize_mask(pair.target.line_mask))
                )

    family_bases: dict[str, PairedSymbol] = {}
    for pair in eligible:
        family_bases.setdefault(pair.family_id, pair)
    quality_scores: list[float] = []
    retentions: list[float] = []
    diversities: list[float] = []
    generated_changes: list[float] = []
    with torch.no_grad():
        for family_index, (family_id, representative) in enumerate(
            sorted(family_bases.items(), key=lambda item: item[0].casefold())
        ):
            _check_cancel(cancel_event)
            sample_count = config.audit_sample_count
            # The held-out target is unavailable at deployment, so prior sampling
            # must use the untouched canonical base rather than its target-derived
            # registration transform.
            audit_base = representative.base.line_mask
            condition_array = _dilate_mask(audit_base, 1).astype(np.float32)
            conditions = torch.from_numpy(
                np.repeat(condition_array[None, None], sample_count, axis=0)
            ).to(device)
            strengths = torch.linspace(
                0.5 / sample_count,
                1.0 - 0.5 / sample_count,
                sample_count,
                device=device,
            )
            torch.manual_seed(config.seed + 50_000 + family_index)
            sampled = model.sample(
                conditions,
                strengths,
                temperature=0.9,
                return_components=True,
            )
            masks = torch.sigmoid(sampled["logits"]).cpu().numpy()[:, 0] >= 0.5
            sampled_widths = sampled.get("width")
            if sampled_widths is None:
                width_values = np.full(
                    len(masks),
                    (config.model.min_stroke_width + config.model.max_stroke_width) / 2.0,
                    dtype=np.float32,
                )
            else:
                width_values = np.asarray(sampled_widths.cpu().numpy()).reshape(-1)
            for mask, stroke_width in zip(masks, width_values):
                try:
                    rendered_line = skeletonize_mask(mask)
                    quality = validate_line_quality(
                        mask_to_graph(rendered_line),
                        mask,
                        float(stroke_width),
                        {},
                        config.quality,
                    )
                    quality_scores.append(float(quality.valid))
                except (SymbolGeneratorError, ValueError):
                    quality_scores.append(0.0)
                change = changed_line_amount(audit_base, mask)
                generated_changes.append(change["change_fraction"])
                base_pixels = max(1, int(skeletonize_mask(audit_base).sum()))
                retentions.append(float(1.0 - min(1.0, change.get("removed_pixels", 0.0) / base_pixels)))
            if len(masks) > 1:
                distances = [
                    1.0 - _dice(masks[left], masks[right])
                    for left in range(len(masks))
                    for right in range(left + 1, len(masks))
                ]
                diversities.append(float(np.mean(distances)))
    empirical = np.asarray([pair.raw_change_ratio for pair in eligible], dtype=np.float32)
    generated = np.asarray(generated_changes, dtype=np.float32)
    scale = max(1e-6, float(np.percentile(empirical, 95) - np.percentile(empirical, 5)))
    distribution_agreement = float(
        np.clip(1.0 - abs(float(np.median(generated)) - float(np.median(empirical))) / scale, 0.0, 1.0)
    )
    return {
        "eligible_deviations": len(eligible),
        "excluded_cross_family_duplicates": excluded,
        "reconstruction_dice": float(np.mean(reconstruction)),
        "prior_sample_quality": float(np.mean(quality_scores)),
        "base_retention": float(np.mean(retentions)),
        "diversity": float(np.mean(diversities)) if diversities else 0.0,
        "change_distribution_agreement": distribution_agreement,
    }


def _run_unseen_family_audit(
    config: TrainingConfig,
    prepared: _PreparedDataset,
    metadata: Mapping[str, Any],
    device: Any,
    use_amp: bool,
    run_path: Path,
    progress: ProgressCallback | None,
    cancel_event: Any,
    resume_payload: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignments = _audit_fold_assignments(prepared.pairs, config.seed)
    completed: list[dict[str, Any]] = []
    resume_fold = -1
    resume_epoch = -1
    if resume_payload is not None:
        stored_assignments = list(resume_payload.get("audit_assignments", []))
        if stored_assignments != assignments:
            raise SymbolGeneratorError("Resume checkpoint audit fold assignment does not match the dataset")
        completed = [dict(item) for item in resume_payload.get("audit_metrics", [])]
        resume_fold = int(resume_payload.get("audit_fold", -1))
        resume_epoch = int(resume_payload.get("epoch", -1))

    audit_epochs = max(1, min(config.epochs, round(config.epochs * 0.20)))
    active_path = run_path / "audit-active.pt"
    for fold_index, assignment in enumerate(assignments):
        if fold_index < len(completed):
            continue
        held = set(assignment["held_out_families"])
        fold_train = [
            pair
            for pair in prepared.pairs
            if pair.family_id not in held
            and pair.split == "train"
            and not pair.cross_family_duplicate
        ]
        fold_validation = [
            pair
            for pair in prepared.pairs
            if pair.family_id not in held
            and pair.split == "validation"
            and not pair.cross_family_duplicate
        ]
        held_pairs = [pair for pair in prepared.pairs if pair.family_id in held]
        if not fold_train:
            raise SymbolGeneratorError(
                f"Audit fold {fold_index + 1} has no leakage-safe training deviations"
            )
        if not fold_validation:
            raise SymbolGeneratorError(
                f"Audit fold {fold_index + 1} has no leakage-safe internal validation deviations"
            )
        set_seed(config.seed + 10_007 * (fold_index + 1), config.deterministic)
        model = ConditionalVAE(**asdict(config.model)).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        scaler = _make_grad_scaler(use_amp)
        dataset = PairedSymbolDataset(
            fold_train,
            seed=config.seed + fold_index * 101,
            training=True,
            real_probability=config.real_pair_probability,
            synthetic_probability=config.synthetic_pair_probability,
            identity_probability=config.identity_pair_probability,
            match_tolerance=config.registration.match_tolerance,
        )
        validation_dataset = PairedSymbolDataset(
            fold_validation,
            seed=config.seed + fold_index * 101 + 17,
            training=False,
            real_probability=1.0,
            synthetic_probability=0.0,
            identity_probability=0.0,
            match_tolerance=config.registration.match_tolerance,
        )
        loader = _make_loader(
            dataset,
            batch_size=config.batch_size,
            workers=config.workers,
            shuffle=True,
            seed=config.seed + fold_index,
            pin_memory=device.type == "cuda",
        )
        validation_loader = _make_loader(
            validation_dataset,
            batch_size=config.batch_size,
            workers=config.workers,
            shuffle=False,
            seed=config.seed + fold_index + 1,
            pin_memory=device.type == "cuda",
        )
        start_epoch = 0
        if resume_payload is not None and fold_index == resume_fold:
            model.load_state_dict(resume_payload["model_state"])
            if "optimizer_state" not in resume_payload:
                raise SymbolGeneratorError("Active audit resume checkpoint lacks optimizer state")
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            scaler.load_state_dict(resume_payload.get("scaler_state", {}))
            _restore_rng(resume_payload)
            start_epoch = resume_epoch + 1
            if "data_loader_random_state" in resume_payload:
                loader.generator.set_state(resume_payload["data_loader_random_state"].cpu())

        validation_history: list[dict[str, float]] = []
        for epoch in range(start_epoch, audit_epochs):
            _check_cancel(cancel_event)
            dataset.set_epoch(epoch)
            model.train()
            rows: list[dict[str, float]] = []
            beta = config.beta_max * min(
                1.0,
                (epoch + 1) / max(1.0, audit_epochs * config.beta_warmup_fraction),
            )
            for batch_index, batch in enumerate(loader):
                _check_cancel(cancel_event)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    loss, row = _training_batch_loss(
                        model,
                        batch,
                        device,
                        beta=beta,
                        delta_weight=config.delta_loss_weight,
                        retention_weight=config.retention_loss_weight,
                    )
                if not torch.isfinite(loss):
                    raise SymbolGeneratorError(
                        f"Audit fold {fold_index + 1} produced a non-finite loss"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                rows.append(row)
                fraction = 0.15 + 0.30 * (
                    (fold_index + (epoch + (batch_index + 1) / max(1, len(loader))) / audit_epochs)
                    / len(assignments)
                )
                _emit_progress(
                    progress,
                    fraction,
                    f"Audit fold {fold_index + 1}/{len(assignments)}, epoch {epoch + 1}/{audit_epochs}",
                    {"stage": "audit", "fold": fold_index + 1, "epoch": epoch + 1, "metrics": row},
                )
            validation_metrics = _evaluate_model(
                model,
                validation_loader,
                device,
                beta,
                cancel_event,
                delta_weight=config.delta_loss_weight,
                retention_weight=config.retention_loss_weight,
            )
            validation_history.append(validation_metrics)
            active = _checkpoint_payload(
                model,
                config,
                metadata,
                epoch=epoch,
                best_metric=float(validation_metrics.get("loss", float("inf"))),
                epochs_without_improvement=0,
                metrics_history=validation_history,
                calibration={"threshold": 0.5},
                optimizer=optimizer,
                scaler=scaler,
            )
            active.update(
                {
                    "stage": "audit",
                    "audit_fold": fold_index,
                    "audit_epochs": audit_epochs,
                    "audit_assignments": assignments,
                    "audit_metrics": completed,
                    "data_loader_random_state": loader.generator.get_state(),
                }
            )
            _atomic_torch_save(active, active_path)

        scores = _score_unseen_families(
            model, held_pairs, device, config, cancel_event
        )
        completed.append(
            {
                **assignment,
                "audit_epochs": audit_epochs,
                "internal_validation": validation_history[-1] if validation_history else {},
                "scores": scores,
            }
        )
        resume_payload = None

    if active_path.exists():
        active_path.unlink()
    return assignments, completed


def train_model(
    config: TrainingConfig | Mapping[str, Any],
    progress: ProgressCallback | None = None,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Audit and train the paired-family add-on beta-VAE."""

    require_torch()
    resolved = _coerce_training_config(config)
    run_path = Path(resolved.run).expanduser().resolve()
    if not resolved.resume and any((run_path / name).exists() for name in ("best.pt", "last.pt", "audit-active.pt")):
        raise SymbolGeneratorError(
            f"Run directory already contains checkpoints: {run_path}. Choose a new run or pass resume."
        )
    run_path.mkdir(parents=True, exist_ok=True)
    _write_json(run_path / "config.json", asdict(resolved))
    validation_config = ValidationConfig(
        data=resolved.data,
        report=str(run_path),
        preprocessing=resolved.preprocessing,
        registration=resolved.registration,
        validation_fraction=resolved.validation_fraction,
        seed=resolved.seed,
    )
    prepared = _prepare_dataset(
        validation_config,
        progress=(
            (lambda fraction, message, payload: _emit_progress(progress, fraction * 0.15, message, payload))
            if progress
            else None
        ),
        cancel_event=cancel_event,
        manifest_stem="dataset-manifest",
    )
    _write_json(run_path / "config.json", asdict(resolved))
    metadata = _generation_metadata(resolved, prepared)
    device = resolve_device(resolved.device)
    use_amp = bool(resolved.mixed_precision and device.type == "cuda")

    resume_payload: dict[str, Any] | None = None
    init_payload: dict[str, Any] | None = None
    if resolved.resume:
        resume_payload = _load_checkpoint(resolved.resume)
        if resume_payload.get("dataset_fingerprint") != prepared.dataset_fingerprint:
            raise SymbolGeneratorError(
                "Exact resume requires the identical paired-family dataset fingerprint. "
                "Use init_checkpoint for an expanded dataset."
            )
        if resume_payload.get("model_config") != asdict(resolved.model):
            raise SymbolGeneratorError("Resume checkpoint model configuration does not match this run")
        if resume_payload.get("preprocess_config") != asdict(resolved.preprocessing):
            raise SymbolGeneratorError("Resume checkpoint preprocessing configuration does not match this run")
    elif resolved.init_checkpoint:
        # Validate initialization before the potentially long unseen-family
        # audit. Fold models still start from scratch; only the final model uses
        # these weights after the audit completes.
        init_payload = _load_checkpoint(resolved.init_checkpoint)
        _require_final_checkpoint(init_payload, purpose="Initialization")
        if init_payload.get("model_config") != asdict(resolved.model):
            raise SymbolGeneratorError("Initialization checkpoint model configuration does not match")
        if init_payload.get("preprocess_config") != asdict(resolved.preprocessing):
            raise SymbolGeneratorError("Initialization checkpoint preprocessing configuration does not match")

    if resume_payload is not None and resume_payload.get("stage") == "final":
        audit_assignments = [dict(item) for item in resume_payload.get("audit_assignments", [])]
        audit_metrics = [dict(item) for item in resume_payload.get("audit_metrics", [])]
    else:
        audit_assignments, audit_metrics = _run_unseen_family_audit(
            resolved,
            prepared,
            metadata,
            device,
            use_amp,
            run_path,
            progress,
            cancel_event,
            resume_payload if resume_payload is not None and resume_payload.get("stage") == "audit" else None,
        )
    audit_path = run_path / "audit.json"
    _write_json(
        audit_path,
        {
            "dataset_fingerprint": prepared.dataset_fingerprint,
            "assignments": audit_assignments,
            "folds": audit_metrics,
        },
    )

    set_seed(resolved.seed, resolved.deterministic)
    model = ConditionalVAE(**asdict(resolved.model)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=resolved.learning_rate, weight_decay=resolved.weight_decay
    )
    scaler = _make_grad_scaler(use_amp)
    train_dataset = PairedSymbolDataset(
        prepared.train_pairs,
        seed=resolved.seed,
        training=True,
        real_probability=resolved.real_pair_probability,
        synthetic_probability=resolved.synthetic_pair_probability,
        identity_probability=resolved.identity_pair_probability,
        match_tolerance=resolved.registration.match_tolerance,
    )
    validation_dataset = PairedSymbolDataset(
        prepared.validation_pairs,
        seed=resolved.seed + 17,
        training=False,
        real_probability=1.0,
        synthetic_probability=0.0,
        identity_probability=0.0,
        match_tolerance=resolved.registration.match_tolerance,
    )
    train_loader = _make_loader(
        train_dataset,
        batch_size=resolved.batch_size,
        workers=resolved.workers,
        shuffle=True,
        seed=resolved.seed,
        pin_memory=device.type == "cuda",
    )
    validation_loader = _make_loader(
        validation_dataset,
        batch_size=resolved.batch_size,
        workers=resolved.workers,
        shuffle=False,
        seed=resolved.seed + 1,
        pin_memory=device.type == "cuda",
    )
    calibration: dict[str, Any] = {
        "threshold": 0.5,
        "guided_bounds": {
            "similarity_min": 0.0,
            "similarity_max": 1.0,
            "change_fraction_min": 0.0,
            "change_fraction_max": float("inf"),
            "removal_fraction_max": 1.0,
        },
    }
    start_epoch = 0
    best_metric = float("inf")
    epochs_without_improvement = 0
    metrics_history: list[dict[str, Any]] = []
    if resume_payload is not None and resume_payload.get("stage") == "final":
        model.load_state_dict(resume_payload["model_state"])
        if "optimizer_state" not in resume_payload:
            raise SymbolGeneratorError("Resume requires a full-state last.pt checkpoint")
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        scaler.load_state_dict(resume_payload.get("scaler_state", {}))
        start_epoch = int(resume_payload.get("epoch", -1)) + 1
        best_metric = float(resume_payload.get("best_metric", float("inf")))
        epochs_without_improvement = int(resume_payload.get("epochs_without_improvement", 0))
        metrics_history = [dict(item) for item in resume_payload.get("metrics_history", [])]
        calibration = dict(resume_payload.get("calibration", calibration))
        _restore_rng(resume_payload)
        if "data_loader_random_state" in resume_payload:
            train_loader.generator.set_state(resume_payload["data_loader_random_state"].cpu())
    elif init_payload is not None:
        try:
            model.load_state_dict(init_payload["model_state"])
        except Exception as exc:
            raise SymbolGeneratorError(f"Initialization checkpoint weights are incompatible: {exc}") from exc

    best_path = run_path / "best.pt"
    last_path = run_path / "last.pt"
    metrics_json = run_path / "metrics.json"
    metrics_csv = run_path / "metrics.csv"
    stopped_early = False
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, resolved.epochs):
        _check_cancel(cancel_event)
        last_epoch = epoch
        train_dataset.set_epoch(epoch)
        model.train()
        rows: list[dict[str, float]] = []
        beta = resolved.beta_max * min(
            1.0,
            (epoch + 1) / max(1.0, resolved.epochs * resolved.beta_warmup_fraction),
        )
        for batch_index, batch in enumerate(train_loader):
            _check_cancel(cancel_event)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, row = _training_batch_loss(
                    model,
                    batch,
                    device,
                    beta=beta,
                    delta_weight=resolved.delta_loss_weight,
                    retention_weight=resolved.retention_loss_weight,
                )
            if not torch.isfinite(loss):
                raise SymbolGeneratorError(f"Training produced a non-finite loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if resolved.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), resolved.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            rows.append(row)
            fraction = 0.45 + 0.45 * (
                (epoch + (batch_index + 1) / max(1, len(train_loader))) / resolved.epochs
            )
            _emit_progress(
                progress,
                fraction,
                f"Final model epoch {epoch + 1}/{resolved.epochs}, batch {batch_index + 1}/{len(train_loader)}",
                {"stage": "final", "epoch": epoch + 1, "batch": batch_index + 1, "metrics": row},
            )
        train_metrics = _average_metric_rows(rows)
        validation_metrics = _evaluate_model(
            model,
            validation_loader,
            device,
            beta,
            cancel_event,
            delta_weight=resolved.delta_loss_weight,
            retention_weight=resolved.retention_loss_weight,
        )
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "beta": beta,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        metrics_history.append(record)
        validation_loss = float(validation_metrics["loss"])
        improved = validation_loss < best_metric - 1e-8
        if improved:
            best_metric = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint = _checkpoint_payload(
            model,
            resolved,
            metadata,
            epoch=epoch,
            best_metric=best_metric,
            epochs_without_improvement=epochs_without_improvement,
            metrics_history=metrics_history,
            calibration=calibration,
            optimizer=optimizer,
            scaler=scaler,
        )
        checkpoint.update(
            {
                "stage": "final",
                "audit_assignments": audit_assignments,
                "audit_metrics": audit_metrics,
                "data_loader_random_state": train_loader.generator.get_state(),
            }
        )
        _atomic_torch_save(checkpoint, last_path)
        if improved:
            best_checkpoint = dict(checkpoint)
            for key in (
                "optimizer_state",
                "scaler_state",
                "python_random_state",
                "numpy_random_state",
                "torch_random_state",
                "cuda_random_states",
                "data_loader_random_state",
            ):
                best_checkpoint.pop(key, None)
            _atomic_torch_save(best_checkpoint, best_path)
        _write_json(metrics_json, metrics_history)
        _write_csv(metrics_csv, metrics_history)
        if (epoch + 1) % resolved.preview_frequency == 0 or improved or epoch + 1 == resolved.epochs:
            _save_preview(
                model,
                prepared.validation_pairs,
                run_path / "previews" / f"epoch-{epoch + 1:04d}.png",
                count=resolved.preview_count,
                seed=resolved.seed + 991,
                threshold=float(calibration["threshold"]),
                device=device,
            )
        if epochs_without_improvement >= resolved.patience:
            stopped_early = True
            break

    _emit_progress(progress, 0.94, "Calibrating paired raster and edit bounds", None)
    if best_path.is_file():
        best_checkpoint = _load_checkpoint(best_path)
        model.load_state_dict(best_checkpoint["model_state"])
    else:
        best_checkpoint = _checkpoint_payload(
            model,
            resolved,
            metadata,
            epoch=last_epoch,
            best_metric=best_metric,
            epochs_without_improvement=epochs_without_improvement,
            metrics_history=metrics_history,
            calibration=calibration,
        )
        best_checkpoint.update(
            {"stage": "final", "audit_assignments": audit_assignments, "audit_metrics": audit_metrics}
        )
    calibration = _calibrate_model(model, validation_loader, device, cancel_event)
    best_checkpoint["calibration"] = calibration
    _atomic_torch_save(best_checkpoint, best_path)
    if last_path.is_file():
        last_checkpoint = _load_checkpoint(last_path)
        last_checkpoint["calibration"] = calibration
        _atomic_torch_save(last_checkpoint, last_path)

    artifacts = {
        **prepared.artifacts,
        "run": str(run_path),
        "audit": str(audit_path),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "metrics_json": str(metrics_json),
        "metrics_csv": str(metrics_csv),
        "previews": [str(item.resolve()) for item in sorted((run_path / "previews").glob("*.png"))],
    }
    result = {
        "status": "complete",
        "message": "Paired-family training complete" + (" (early stopping)" if stopped_early else ""),
        "device": str(device),
        "mixed_precision": use_amp,
        "epochs_completed": max(0, last_epoch + 1),
        "best_validation_loss": best_metric,
        "stopped_early": stopped_early,
        "calibration": calibration,
        "audit": {"assignments": audit_assignments, "folds": audit_metrics},
        "summary": prepared.summary,
        "families": prepared.family_summaries,
        "dataset_fingerprint": prepared.dataset_fingerprint,
        "artifacts": artifacts,
    }
    _emit_progress(progress, 1.0, result["message"], result)
    return result


def _checkpoint_dataclass(cls: type[Any], payload: Mapping[str, Any], key: str) -> Any:
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        raise SymbolGeneratorError(f"Checkpoint field {key!r} is invalid")
    return cls(**_dataclass_kwargs(cls, value))


def _save_generated_artifact(
    directory: Path,
    stem: str,
    svg: str,
    rendered_mask: np.ndarray,
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    svg_path = directory / f"{stem}.svg"
    png_path = directory / f"{stem}.png"
    svg_path.write_text(svg, encoding="utf-8")
    Image.fromarray(
        np.where(rendered_mask, 0, 255).astype(np.uint8), mode="L"
    ).convert("RGB").save(png_path)
    return {"svg": str(svg_path.resolve()), "png": str(png_path.resolve())}


def _manifest_snapshot(
    *,
    status: str,
    config: GenerationConfig,
    threshold: float,
    attempts: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    rejection_counts: Mapping[str, int],
    attempt_limit: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    novel_count = sum(item.get("status") == "novel" for item in outputs)
    review_count = sum(item.get("status") == "review" for item in outputs)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "requested_novel": config.count,
        "novel_count": novel_count,
        "review_count": review_count,
        "attempt_count": len(attempts),
        "attempt_limit": attempt_limit,
        "threshold": threshold,
        "input_base_hash": provenance.get("input_base_hash"),
        "requested_strength": config.edit_strength,
        "dataset_fingerprint": provenance.get("dataset_fingerprint"),
        "audit_provenance": provenance.get("audit_provenance", {}),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "outputs": list(outputs),
        "attempts": list(attempts),
    }


def generate_symbols(
    config: GenerationConfig | Mapping[str, Any],
    progress: ProgressCallback | None = None,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Sample, vectorize, quality-check, and novelty-route line-only symbols."""

    require_torch()
    resolved = _coerce_generation_config(config)
    checkpoint = _load_checkpoint(resolved.checkpoint)
    _require_final_checkpoint(checkpoint, purpose="Generation")
    preprocess_config = _checkpoint_dataclass(
        PreprocessConfig, checkpoint, "preprocess_config"
    )
    model_config = _checkpoint_dataclass(ModelConfig, checkpoint, "model_config")
    preprocess_config.validate()
    model_config.validate()
    if model_config.image_size != preprocess_config.image_size:
        raise SymbolGeneratorError("Checkpoint model and preprocessing image sizes do not match")

    # Minimal API calls inherit training-time novelty and quality controls. A full
    # web form or explicit dataclass carries its own (possibly edited) values.
    if isinstance(config, Mapping):
        novelty_keys = set(NoveltyConfig.__dataclass_fields__)
        quality_keys = set(QualityConfig.__dataclass_fields__)
        has_novelty = isinstance(config.get("novelty"), Mapping) or bool(novelty_keys & set(config))
        has_quality = isinstance(config.get("quality"), Mapping) or bool(quality_keys & set(config))
        if not has_novelty and isinstance(checkpoint.get("novelty_config"), Mapping):
            resolved = replace(
                resolved,
                novelty=_checkpoint_dataclass(NoveltyConfig, checkpoint, "novelty_config"),
            )
        if not has_quality and isinstance(checkpoint.get("quality_config"), Mapping):
            resolved = replace(
                resolved,
                quality=_checkpoint_dataclass(QualityConfig, checkpoint, "quality_config"),
            )
        resolved.validate()

    device = resolve_device(resolved.device)
    set_seed(resolved.seed, deterministic=True)
    model = ConditionalVAE(**asdict(model_config)).to(device)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except Exception as exc:
        raise SymbolGeneratorError(f"Checkpoint model weights are incompatible: {exc}") from exc
    model.eval()

    reference_masks = _unpack_reference_masks(checkpoint)
    reference_names = [str(value) for value in checkpoint.get("reference_names", [])]
    if len(reference_names) != len(reference_masks):
        reference_names = [f"reference_{index}" for index in range(len(reference_masks))]
    novelty = NoveltyChecker(
        reference_masks,
        reference_names,
        duplicate_threshold=resolved.novelty.duplicate_threshold,
        review_threshold=resolved.novelty.review_threshold,
        top_k=resolved.novelty.shortlist_maximum,
        transformed_review_threshold=resolved.novelty.transformed_review_threshold,
        skeleton_tolerance=resolved.novelty.skeleton_tolerance,
        metric_weights=(
            resolved.novelty.skeleton_weight,
            resolved.novelty.rendered_weight,
            resolved.novelty.topology_weight,
        ),
        alignment_angle=resolved.novelty.alignment_angle,
        alignment_translation=resolved.novelty.alignment_translation,
        alignment_scale=resolved.novelty.alignment_scale,
        precise_finalists=resolved.novelty.precise_finalists,
    )
    calibration = checkpoint.get("calibration", {})
    threshold = float(
        resolved.threshold_override
        if resolved.threshold_override is not None
        else calibration.get("threshold", 0.5)
    )
    if not 0.05 <= threshold <= 0.95:
        threshold = 0.5
    quality_baselines = checkpoint.get("quality_baselines", {})
    stroke_statistics = checkpoint.get("stroke_statistics", {})
    safe_width_min = max(
        model_config.min_stroke_width,
        float(stroke_statistics.get("minimum", model_config.min_stroke_width)),
    )
    safe_width_max = min(
        model_config.max_stroke_width,
        float(stroke_statistics.get("maximum", model_config.max_stroke_width)),
    )
    if safe_width_max < safe_width_min:
        safe_width_min, safe_width_max = model_config.min_stroke_width, model_config.max_stroke_width

    base_symbol = preprocess_image(resolved.base, preprocess_config)
    base_line_mask = base_symbol.line_mask
    condition_array = _dilate_mask(base_symbol.line_mask, 1).astype(np.float32)
    condition_strength = resolved.edit_strength
    provenance = {
        "input_base_hash": mask_hash(base_line_mask),
        "dataset_fingerprint": checkpoint.get("dataset_fingerprint", ""),
        "audit_provenance": {
            "assignments": checkpoint.get("audit_assignments", []),
            "folds": checkpoint.get("audit_metrics", []),
        },
    }

    out_path = Path(resolved.out).expanduser().resolve()
    novel_path = out_path / "novel"
    review_path = out_path / "review"
    if (out_path / "manifest.json").exists() or any(out_path.glob("*/*.svg")):
        raise SymbolGeneratorError(
            f"Output directory already contains a generation run: {out_path}. Choose a new output directory."
        )
    out_path.mkdir(parents=True, exist_ok=True)
    novel_path.mkdir(parents=True, exist_ok=True)
    review_path.mkdir(parents=True, exist_ok=True)
    config_path = out_path / "config.json"
    manifest_path = out_path / "manifest.json"
    effective_config = asdict(resolved)
    effective_config["preprocessing"] = asdict(preprocess_config)
    effective_config["model"] = asdict(model_config)
    effective_config["effective_threshold"] = threshold
    _write_json(config_path, effective_config)

    review_limit = resolved.count if resolved.review_cap is None else resolved.review_cap
    attempt_limit = resolved.count * resolved.attempt_multiplier
    attempts: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    novel_count = 0
    review_count = 0
    cancelled = False
    guided_bounds = calibration.get("guided_bounds", {})

    try:
        while novel_count < resolved.count and len(attempts) < attempt_limit:
            _check_cancel(cancel_event)
            batch_count = min(
                resolved.sampling_batch,
                attempt_limit - len(attempts),
                max(1, resolved.count - novel_count),
            )
            conditions = torch.from_numpy(
                np.repeat(condition_array[None, None], batch_count, axis=0)
            ).to(device)
            strengths = torch.full(
                (batch_count,), condition_strength, dtype=torch.float32, device=device
            )
            latent_rows: list[Any] = []
            with torch.no_grad():
                condition_features = model.encode_condition(conditions, strengths)
                prior_mu, prior_logvar = model.encode_prior(condition_features)
            for row_index in range(batch_count):
                row_generator = torch.Generator()
                row_generator.manual_seed(resolved.seed + len(attempts) + row_index)
                latent_rows.append(
                    torch.randn(model_config.latent_dim, generator=row_generator)
                )
            noise = torch.stack(latent_rows).to(device)
            z = prior_mu + noise * torch.exp(0.5 * prior_logvar) * resolved.temperature
            with torch.no_grad():
                sampled = model.sample(
                    conditions, strengths, z=z, return_components=True
                )
            probabilities = torch.sigmoid(sampled["logits"]).cpu().numpy()[:, 0]
            addition_probabilities = torch.sigmoid(sampled["add_logits"]).cpu().numpy()[:, 0]
            removal_probabilities = torch.sigmoid(sampled["remove_logits"]).cpu().numpy()[:, 0]
            predicted_widths = sampled["width"].cpu().numpy()

            for batch_index in range(batch_count):
                if novel_count >= resolved.count:
                    break
                _check_cancel(cancel_event)
                attempt_number = len(attempts) + 1
                sample_seed = resolved.seed + attempt_number - 1
                attempt_record: dict[str, Any] = {
                    "attempt": attempt_number,
                    "seed": sample_seed,
                    "status": "rejected",
                    "input_base_hash": provenance["input_base_hash"],
                    "requested_strength": resolved.edit_strength,
                }
                try:
                    line_mask = skeletonize_mask(probabilities[batch_index] >= threshold)
                    if int(line_mask.sum()) < 2:
                        raise SymbolGeneratorError("empty_decoded_mask")
                    graph = mask_to_graph(line_mask)
                    predicted_width = float(
                        np.clip(predicted_widths[batch_index], safe_width_min, safe_width_max)
                    )
                    stroke_width = (
                        resolved.edit_strength * predicted_width
                        + (1.0 - resolved.edit_strength) * base_symbol.stroke_width
                    )
                    stroke_width = float(np.clip(stroke_width, safe_width_min, safe_width_max))
                    svg = graph_to_svg(
                        graph,
                        stroke_width,
                        size=model_config.image_size,
                        curve_error=resolved.quality.curve_error,
                    )
                    validate_svg_schema(svg)
                    rendered_mask = svg_to_mask(svg, size=model_config.image_size)
                    rendered_line = skeletonize_mask(rendered_mask)
                    if _topology_signature(line_mask) != _topology_signature(rendered_line):
                        raise SymbolGeneratorError("topology_altered_after_rerender")
                    rendered_graph = mask_to_graph(rendered_line)
                    quality = validate_line_quality(
                        rendered_graph,
                        rendered_mask,
                        stroke_width,
                        quality_baselines,
                        resolved.quality,
                    )
                    attempt_record["quality"] = asdict(quality)
                    if not quality.valid:
                        raise SymbolGeneratorError("quality:" + ",".join(quality.reasons))

                    novelty_result = novelty.classify(rendered_line)
                    attempt_record["novelty"] = novelty_result.to_dict()
                    if novelty_result.status == "duplicate":
                        raise SymbolGeneratorError("duplicate:" + novelty_result.reason)

                    predicted_addition = skeletonize_mask(
                        addition_probabilities[batch_index] >= threshold
                    )
                    predicted_removal = skeletonize_mask(
                        removal_probabilities[batch_index] >= threshold
                    )
                    attempt_record["predicted_edit"] = {
                        "addition_pixels": int(predicted_addition.sum()),
                        "removal_pixels": int(predicted_removal.sum()),
                    }
                    change: dict[str, float] | None = None
                    forced_review_reason = ""
                    change = changed_line_amount(base_line_mask, rendered_line)
                    attempt_record["base_change"] = change
                    if (
                        change["changed_pixels"] < resolved.quality.guided_noop_pixels
                        and change["change_fraction"] < resolved.quality.guided_noop_fraction
                    ):
                        raise SymbolGeneratorError("guided_no_op")
                    if guided_bounds:
                        relation_values = (
                            ("similarity", "similarity_min", "similarity_max"),
                            ("change_fraction", "change_fraction_min", "change_fraction_max"),
                        )
                        if any(
                            float(change[value_key]) < float(guided_bounds.get(low_key, -float("inf")))
                            or float(change[value_key]) > float(guided_bounds.get(high_key, float("inf")))
                            for value_key, low_key, high_key in relation_values
                        ):
                            forced_review_reason = "outside_calibrated_base_relation"
                        removal_fraction = float(
                            change.get("removed_pixels", 0.0) / max(1.0, change["changed_pixels"])
                        )
                        if removal_fraction > float(guided_bounds.get("removal_fraction_max", 1.0)):
                            forced_review_reason = "excessive_base_removal"

                    status = (
                        "review"
                        if novelty_result.status == "review" or forced_review_reason
                        else "novel"
                    )
                    if status == "review" and review_count >= review_limit:
                        raise SymbolGeneratorError("review_cap_reached")
                    sequence = novel_count + 1 if status == "novel" else review_count + 1
                    stem = f"symbol-{sequence:04d}"
                    paths = _save_generated_artifact(
                        novel_path if status == "novel" else review_path,
                        stem,
                        svg,
                        rendered_mask,
                    )
                    attempt_record.update(
                        {
                            "status": status,
                            "reason": forced_review_reason or novelty_result.reason,
                            "stroke_width": stroke_width,
                            "artifacts": paths,
                        }
                    )
                    outputs.append(dict(attempt_record))
                    novelty.register(rendered_line, f"generated_{status}_{sequence:04d}")
                    if status == "novel":
                        novel_count += 1
                    else:
                        review_count += 1
                except SymbolGeneratorError as exc:
                    reason = str(exc)
                    if reason == "Operation cancelled":
                        raise
                    attempt_record["reason"] = reason
                    rejection_counts[reason.split(":", 1)[0]] += 1
                except Exception as exc:
                    reason = f"candidate_error:{type(exc).__name__}:{exc}"
                    attempt_record["reason"] = reason
                    rejection_counts["candidate_error"] += 1
                attempts.append(attempt_record)
                fraction = min(0.99, novel_count / resolved.count)
                _emit_progress(
                    progress,
                    fraction,
                    f"Generated {novel_count}/{resolved.count} novel symbols ({len(attempts)} attempts)",
                    {
                        "novel": novel_count,
                        "review": review_count,
                        "attempts": len(attempts),
                        "last": attempt_record,
                    },
                )
            _write_json(
                manifest_path,
                _manifest_snapshot(
                    status="running",
                    config=resolved,
                    threshold=threshold,
                    attempts=attempts,
                    outputs=outputs,
                    rejection_counts=rejection_counts,
                    attempt_limit=attempt_limit,
                    provenance=provenance,
                ),
            )
    except SymbolGeneratorError as exc:
        if str(exc) != "Operation cancelled":
            raise
        cancelled = True

    if cancelled:
        status = "cancelled"
        message = f"Generation cancelled after preserving {novel_count} novel symbols"
    elif novel_count >= resolved.count:
        status = "complete"
        message = f"Generated {novel_count} novel symbols"
    else:
        status = "shortfall"
        message = (
            f"Generated {novel_count}/{resolved.count} requested novel symbols "
            f"within {attempt_limit} attempts"
        )
    manifest = _manifest_snapshot(
        status=status,
        config=resolved,
        threshold=threshold,
        attempts=attempts,
        outputs=outputs,
        rejection_counts=rejection_counts,
        attempt_limit=attempt_limit,
        provenance=provenance,
    )
    _write_json(manifest_path, manifest)
    artifacts = {
        "out": str(out_path),
        "config": str(config_path),
        "manifest": str(manifest_path),
        "novel": [item["artifacts"] for item in outputs if item["status"] == "novel"],
        "review": [item["artifacts"] for item in outputs if item["status"] == "review"],
    }
    result = {
        "status": status,
        "message": message,
        "novel_count": novel_count,
        "review_count": review_count,
        "attempt_count": len(attempts),
        "attempt_limit": attempt_limit,
        "rejection_counts": dict(rejection_counts),
        "artifacts": artifacts,
    }
    _emit_progress(progress, 1.0, message, result)
    return result


# ---------------------------------------------------------------------------
# Command-line wrappers (the local webpage calls the same functions directly)
# ---------------------------------------------------------------------------


def _add_preprocess_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--margin", type=int, default=12)
    parser.add_argument("--max-source-stroke-width", type=float, default=12.0)
    parser.add_argument("--min-component-pixels", type=int, default=3)
    parser.add_argument("--max-input-pixels", type=int, default=40_000_000)
    parser.add_argument("--filled-policy", choices=("outline", "reject"), default="outline")


def _add_registration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--angle-range", type=float, default=12.0)
    parser.add_argument("--translation-range", type=int, default=8)
    parser.add_argument("--scale-range", type=float, default=0.12)
    parser.add_argument("--match-tolerance", type=float, default=3.0)
    parser.add_argument("--minimum-overlap", type=float, default=0.25)


def _add_novelty_arguments(
    parser: argparse.ArgumentParser, *, inherit_checkpoint: bool = False
) -> None:
    def default(value: Any) -> Any:
        return argparse.SUPPRESS if inherit_checkpoint else value

    parser.add_argument("--duplicate-threshold", type=float, default=default(0.94))
    parser.add_argument("--review-threshold", type=float, default=default(0.82))
    parser.add_argument("--transformed-review-threshold", type=float, default=default(0.90))
    parser.add_argument("--skeleton-weight", type=float, default=default(0.60))
    parser.add_argument("--rendered-weight", type=float, default=default(0.30))
    parser.add_argument("--topology-weight", type=float, default=default(0.10))
    parser.add_argument("--skeleton-tolerance", type=float, default=default(2.0))
    parser.add_argument("--alignment-angle", type=float, default=default(6.0))
    parser.add_argument("--alignment-translation", type=int, default=default(3))
    parser.add_argument("--alignment-scale", type=float, default=default(0.04))
    parser.add_argument("--shortlist-maximum", type=int, default=default(64))
    parser.add_argument("--precise-finalists", type=int, default=default(8))


def _add_quality_arguments(
    parser: argparse.ArgumentParser, *, inherit_checkpoint: bool = False
) -> None:
    def default(value: Any) -> Any:
        return argparse.SUPPRESS if inherit_checkpoint else value

    parser.add_argument("--curve-error", type=float, default=default(0.75))
    parser.add_argument("--maximum-ink", type=float, default=default(0.35))
    parser.add_argument("--maximum-components", type=int, default=default(24))
    parser.add_argument("--crowded-line-limit", type=float, default=default(0.10))
    parser.add_argument("--crowd-distance-factor", type=float, default=default(1.5))
    parser.add_argument("--parallel-bundle-threshold", type=int, default=default(3))
    parser.add_argument("--solid-diameter-factor", type=float, default=default(2.2))
    parser.add_argument("--guided-noop-pixels", type=float, default=default(8.0))
    parser.add_argument("--guided-noop-fraction", type=float, default=default(0.08))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and sample a line-only SVG symbol generator."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="preprocess a dataset and write conversion reports"
    )
    validate_parser.add_argument("--data", required=True)
    validate_parser.add_argument("--report", required=True)
    validate_parser.add_argument("--validation-fraction", type=float, default=0.10)
    validate_parser.add_argument("--seed", type=int, default=1337)
    validate_parser.add_argument("--contact-sheet-page-size", type=int, default=40)
    _add_preprocess_arguments(validate_parser)
    _add_registration_arguments(validate_parser)

    train_parser = subparsers.add_parser("train", help="train or resume the beta-VAE")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--run", required=True)
    train_parser.add_argument("--resume", default="")
    train_parser.add_argument("--init-checkpoint", default="")
    train_parser.add_argument("--validation-fraction", type=float, default=0.10)
    train_parser.add_argument("--latent-dim", type=int, default=32)
    train_parser.add_argument("--base-channels", type=int, default=32)
    train_parser.add_argument("--min-stroke-width", type=float, default=1.0)
    train_parser.add_argument("--max-stroke-width", type=float, default=6.0)
    train_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train_parser.add_argument("--epochs", type=int, default=250)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=2e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--patience", type=int, default=30)
    train_parser.add_argument("--seed", type=int, default=1337)
    train_parser.add_argument("--beta-max", "--kl-max", dest="beta_max", type=float, default=1e-3)
    train_parser.add_argument("--beta-warmup-fraction", type=float, default=0.25)
    train_parser.add_argument("--real-pair-probability", type=float, default=0.60)
    train_parser.add_argument("--synthetic-pair-probability", type=float, default=0.30)
    train_parser.add_argument("--identity-pair-probability", type=float, default=0.10)
    train_parser.add_argument("--delta-loss-weight", type=float, default=0.50)
    train_parser.add_argument("--retention-loss-weight", type=float, default=0.25)
    train_parser.add_argument("--audit-sample-count", type=int, default=32)
    train_parser.add_argument("--gradient-clip", type=float, default=1.0)
    train_parser.add_argument("--workers", type=int, default=0)
    train_parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    train_parser.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction, default=True
    )
    train_parser.add_argument("--preview-count", type=int, default=8)
    train_parser.add_argument("--preview-frequency", type=int, default=10)
    _add_preprocess_arguments(train_parser)
    _add_registration_arguments(train_parser)
    _add_novelty_arguments(train_parser)
    _add_quality_arguments(train_parser)

    generate_parser = subparsers.add_parser(
        "generate", help="sample and export novel line-only SVG symbols"
    )
    generate_parser.add_argument("--checkpoint", required=True)
    generate_parser.add_argument("--out", required=True)
    generate_parser.add_argument("--base", required=True)
    generate_parser.add_argument("--count", type=int, default=50)
    generate_parser.add_argument("--edit-strength", type=float, default=0.35)
    generate_parser.add_argument("--temperature", type=float, default=0.9)
    generate_parser.add_argument("--sampling-batch", type=int, default=8)
    generate_parser.add_argument("--threshold", dest="threshold_override", type=float, default=None)
    generate_parser.add_argument("--review-cap", type=int, default=None)
    generate_parser.add_argument("--attempt-multiplier", type=int, default=100)
    generate_parser.add_argument("--seed", type=int, default=1337)
    generate_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    _add_novelty_arguments(generate_parser, inherit_checkpoint=True)
    _add_quality_arguments(generate_parser, inherit_checkpoint=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    values = vars(args)

    def cli_progress(fraction: float, message: str, payload: dict[str, Any] | None) -> None:
        del payload
        print(f"[{fraction * 100:6.2f}%] {message}", file=sys.stderr, flush=True)

    try:
        if args.command == "validate":
            result = validate_dataset(values, progress=cli_progress)
        elif args.command == "train":
            result = train_model(values, progress=cli_progress)
        else:
            result = generate_symbols(values, progress=cli_progress)
        print(json.dumps(result, indent=2, default=_json_default))
        if result.get("status") == "shortfall":
            return 2
        if result.get("status") == "cancelled":
            return 130
        return 0
    except (SymbolGeneratorError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
