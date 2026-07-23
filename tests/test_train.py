"""Focused behavioral tests for the public API in :mod:`train`.

The tests intentionally exercise geometry and interchange contracts rather than
private implementation details.  Small compatibility helpers let preprocessing
and novelty results be represented either as dataclasses or dictionaries.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

import train


def _field(value, *names):
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    raise AssertionError(f"{type(value).__name__} has none of the fields {names!r}")


def _processed_mask(processed) -> np.ndarray:
    if isinstance(processed, np.ndarray):
        value = processed
    elif isinstance(processed, tuple):
        value = processed[0]
    else:
        value = _field(processed, "mask", "line_mask", "canonical_mask")
    value = np.asarray(value)
    assert value.shape == (128, 128)
    return value.astype(bool)


def _status(result) -> str:
    if isinstance(result, str):
        return result
    return str(_field(result, "status", "classification", "category"))


def _similarity(result) -> float:
    return float(_field(result, "similarity", "score", "max_similarity"))


def _local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _line_mask() -> np.ndarray:
    mask = np.zeros((128, 128), dtype=bool)
    mask[64, 28:100] = True
    return mask


def _asymmetric_mask() -> np.ndarray:
    mask = np.zeros((128, 128), dtype=bool)
    mask[30:96, 38] = True
    mask[95, 38:88] = True
    mask[48, 38:62] = True
    return mask


def _graph_paths(graph):
    paths = _field(graph, "paths")
    closed = _field(graph, "closed")
    assert len(paths) == len(closed)
    return paths


def _canonical_paths(graph) -> tuple[tuple[tuple[float, float], ...], ...]:
    canonical = []
    for path in _graph_paths(graph):
        points = np.asarray(path, dtype=float)
        assert points.ndim == 2 and points.shape[1] == 2
        assert len(points) >= 2
        rounded = tuple(map(tuple, np.round(points, 5)))
        reversed_points = tuple(reversed(rounded))
        canonical.append(min(rounded, reversed_points))
    return tuple(sorted(canonical))


def test_preprocess_composites_alpha_and_centers_without_stretching(tmp_path: Path):
    source = Image.new("RGBA", (80, 40), (0, 0, 0, 0))
    draw = ImageDraw.Draw(source)
    draw.line((10, 20, 70, 20), fill=(0, 0, 0, 255), width=3)
    path = tmp_path / "transparent.png"
    source.save(path)

    result = train.preprocess_image(path, size=128, max_source_stroke_width=6)
    mask = _processed_mask(result)
    ys, xs = np.nonzero(mask)

    assert np.asarray(result.render_mask).shape == (128, 128)
    assert np.isfinite(float(result.stroke_width)) and float(result.stroke_width) > 0
    assert result.conversion
    assert Path(result.source) == path
    assert mask.any()
    assert np.ptp(xs) > np.ptp(ys) * 4
    assert abs(float(xs.mean()) - 63.5) <= 2
    assert abs(float(ys.mean()) - 63.5) <= 2
    # Transparent black pixels must have been composited to white, not treated
    # as a page-sized black component.
    assert mask.mean() < 0.05


def test_preprocess_rejects_blank_input(tmp_path: Path):
    path = tmp_path / "blank.png"
    Image.new("RGB", (32, 32), "white").save(path)
    with pytest.raises((ValueError, train.SymbolGeneratorError), match="(?i)blank|foreground|empty"):
        train.preprocess_image(path)


def test_preprocess_turns_a_solid_component_into_an_unfilled_boundary(tmp_path: Path):
    source = Image.new("L", (80, 80), 255)
    ImageDraw.Draw(source).rectangle((15, 15, 64, 64), fill=0)
    path = tmp_path / "solid.png"
    source.save(path)

    mask = _processed_mask(
        train.preprocess_image(path, size=128, max_source_stroke_width=3)
    )

    assert mask.any()
    assert not mask[64, 64], "a solid source region must not remain filled"
    assert mask.mean() < 0.08, "solid regions should become thin outline loops"


def test_mask_to_graph_traces_open_disconnected_lines_deterministically():
    mask = np.zeros((128, 128), dtype=bool)
    mask[20, 10:31] = True
    mask[90, 70:111] = True

    first = train.mask_to_graph(mask)
    second = train.mask_to_graph(mask.copy())

    assert np.array_equal(np.asarray(first.mask).astype(bool), mask)
    for name in ("components", "endpoints", "junctions", "cycles"):
        assert hasattr(first.stats, name)
    assert len(_graph_paths(first)) == 2
    assert _canonical_paths(first) == _canonical_paths(second)
    endpoints = {
        tuple(np.asarray(path)[index].astype(int))
        for path in _graph_paths(first)
        for index in (0, -1)
    }
    # Traced paths use SVG coordinates (x, y), while masks index as [y, x].
    assert endpoints == {(10, 20), (30, 20), (70, 90), (110, 90)}


def test_mask_to_graph_preserves_a_closed_loop():
    mask = np.zeros((128, 128), dtype=bool)
    # A 45-degree diamond gives every loop pixel exactly two 8-neighbours;
    # unlike an axis-aligned pixel rectangle it creates no diagonal corner
    # shortcuts that could legitimately be interpreted as junction clusters.
    for offset in range(21):
        mask[30 + offset, 50 + offset] = True
        mask[50 + offset, 70 - offset] = True
        mask[70 - offset, 50 - offset] = True
        mask[50 - offset, 30 + offset] = True

    paths = train.mask_to_graph(mask)

    assert len(_graph_paths(paths)) == 1
    assert list(paths.closed) == [True]
    loop = np.asarray(paths.paths[0])
    assert len(loop) >= 5
    # Closure is represented explicitly by TracedGraph.closed; implementations
    # need not repeat the first coordinate at the end of the point array.
    assert np.linalg.norm(loop[0] - loop[-1]) <= np.sqrt(2) + 1e-6


def test_mask_to_graph_handles_a_junction_cluster_without_losing_arms():
    mask = np.zeros((128, 128), dtype=bool)
    mask[18:81, 64] = True
    mask[49, 28:101] = True

    paths = train.mask_to_graph(mask)
    endpoints = [np.asarray(path)[i] for path in paths.paths for i in (0, -1)]

    expected = np.asarray([(64, 18), (64, 80), (28, 49), (100, 49)])
    for point in expected:
        assert any(np.linalg.norm(np.asarray(endpoint) - point) <= 2 for endpoint in endpoints)
    assert _canonical_paths(paths) == _canonical_paths(train.mask_to_graph(mask))


def test_graph_to_svg_emits_only_unfilled_uniform_black_paths():
    source = np.zeros((128, 128), dtype=bool)
    source[16, 12:61] = True
    source[80, 20:101] = True
    graph = train.mask_to_graph(source)
    svg = train.graph_to_svg(graph, stroke_width=2.5, size=128)
    assert svg == train.graph_to_svg(train.mask_to_graph(source.copy()), 2.5, size=128)
    root = ET.fromstring(svg)

    assert _local_name(root) == "svg"
    assert root.attrib.get("viewBox") == "0 0 128 128"
    elements = list(root.iter())
    assert {_local_name(element) for element in elements} <= {"svg", "g", "path"}
    path_elements = [e for e in elements if _local_name(e) == "path"]
    assert len(path_elements) == 2
    for element in path_elements:
        inherited = {**root.attrib, **element.attrib}
        parent_group = next((e for e in elements if _local_name(e) == "g"), None)
        if parent_group is not None:
            inherited = {**root.attrib, **parent_group.attrib, **element.attrib}
        assert inherited.get("fill") == "none"
        assert inherited.get("stroke", "").lower() in {"black", "#000", "#000000"}
        assert float(inherited["stroke-width"]) == pytest.approx(2.5)
        assert inherited.get("stroke-linecap") == "round"
        assert inherited.get("stroke-linejoin") == "round"
        assert element.attrib.get("d", "").strip()


@pytest.mark.parametrize(
    "svg",
    [
        '<svg viewBox="0 0 128 128"><rect width="128" height="128"/></svg>',
        (
            '<svg viewBox="0 0 128 128"><path d="M 1 1 L 20 20" '
            'fill="black" stroke="black" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"/></svg>'
        ),
    ],
)
def test_svg_to_mask_rejects_fills_and_non_path_geometry(svg: str):
    with pytest.raises(train.SymbolGeneratorError):
        train.svg_to_mask(svg, size=128)


def test_svg_to_mask_rasterizes_generated_geometry_without_a_background():
    source = np.zeros((128, 128), dtype=bool)
    source[24, 20:109] = True
    svg = train.graph_to_svg(train.mask_to_graph(source), stroke_width=2.0, size=128)
    mask = np.asarray(train.svg_to_mask(svg, size=128)).astype(bool)

    assert mask.shape == (128, 128)
    assert mask.any()
    assert mask[24, 64]
    assert not mask[90, 64]
    assert mask.mean() < 0.05


def test_graph_svg_round_trip_keeps_disconnected_components():
    source = np.zeros((128, 128), dtype=bool)
    source[30, 20:55] = True
    source[88, 73:110] = True
    svg = train.graph_to_svg(train.mask_to_graph(source), stroke_width=1.0, size=128)
    rendered = np.asarray(train.svg_to_mask(svg, size=128)).astype(bool)

    # Sampling representative points is robust to the renderer's antialiasing
    # and to the deliberate one-pixel versus stroked-line representation.
    assert rendered[30, 30] and rendered[30, 45]
    assert rendered[88, 80] and rendered[88, 100]
    assert not rendered[60, 64]


def test_novelty_checker_rejects_exact_and_small_translated_duplicates():
    reference = _asymmetric_mask()
    checker = train.NoveltyChecker([reference])

    exact = checker.classify(reference.copy())
    shifted = np.zeros_like(reference)
    shifted[2:, :-2] = reference[:-2, 2:]
    aligned = checker.classify(shifted)

    assert _status(exact) == "duplicate"
    assert _similarity(exact) >= 0.94
    assert _status(aligned) == "duplicate"
    assert _similarity(aligned) >= 0.94


def test_novelty_checker_routes_rotated_copy_to_review_not_duplicate():
    reference = _asymmetric_mask()
    checker = train.NoveltyChecker([reference])

    result = checker.classify(np.rot90(reference))

    assert _status(result) == "review"
    assert _status(result) != "duplicate"


def test_novelty_checker_accepts_clearly_different_geometry():
    checker = train.NoveltyChecker([_asymmetric_mask()])
    candidate = np.zeros((128, 128), dtype=bool)
    yy, xx = np.ogrid[:128, :128]
    distance = np.sqrt((yy - 64) ** 2 + (xx - 64) ** 2)
    candidate[np.abs(distance - 25) < 0.65] = True

    result = checker.classify(candidate)

    assert _status(result) == "novel"
    assert _similarity(result) < 0.82


def test_novelty_checker_registers_accepted_outputs_for_later_comparison():
    checker = train.NoveltyChecker([_asymmetric_mask()])
    candidate = np.zeros((128, 128), dtype=bool)
    candidate[30, 25:104] = True
    candidate[88, 25:104] = True

    first = checker.classify(candidate, register=True)
    second = checker.classify(candidate.copy())

    assert _status(first) == "novel"
    assert _status(second) == "duplicate"
    assert _similarity(second) == pytest.approx(1.0)


def test_novelty_exact_hash_is_based_on_geometry_not_array_storage():
    reference = _line_mask()
    checker = train.NoveltyChecker([reference])
    for candidate in (reference.copy(order="C"), np.asfortranarray(reference)):
        result = checker.classify(candidate)
        assert _status(result) == "duplicate"
        assert _similarity(result) == pytest.approx(1.0)


def test_cli_parser_exposes_documented_commands_and_defaults(tmp_path: Path):
    parser = train.build_parser()

    validate = parser.parse_args(
        ["validate", "--data", str(tmp_path), "--report", str(tmp_path / "report")]
    )
    assert validate.command == "validate"
    assert Path(validate.data) == tmp_path
    assert validate.max_source_stroke_width > 0

    training = parser.parse_args(
        ["train", "--data", str(tmp_path), "--run", str(tmp_path / "run")]
    )
    assert training.command == "train"
    assert training.device == "auto"
    assert training.epochs == 250
    assert training.batch_size == 16
    assert training.seed == 1337
    assert training.latent_dim == 32
    assert float(_field(training, "learning_rate", "lr")) == pytest.approx(2e-4)
    assert training.patience == 30

    initialized = parser.parse_args(
        [
            "train",
            "--data",
            str(tmp_path),
            "--run",
            str(tmp_path / "expanded-run"),
            "--init-checkpoint",
            str(tmp_path / "best.pt"),
        ]
    )
    assert Path(initialized.init_checkpoint) == tmp_path / "best.pt"

    generation = parser.parse_args(
        [
            "generate",
            "--checkpoint",
            str(tmp_path / "best.pt"),
            "--base",
            str(tmp_path / "new-base.png"),
            "--out",
            str(tmp_path / "generated"),
            "--count",
            "7",
        ]
    )
    assert generation.command == "generate"
    assert generation.count == 7
    assert generation.edit_strength == pytest.approx(0.35)
    assert Path(generation.base) == tmp_path / "new-base.png"


def test_generation_cli_only_supplies_explicit_novelty_and_quality_overrides(
    tmp_path: Path,
):
    parser = train.build_parser()
    required = [
        "generate",
        "--checkpoint",
        str(tmp_path / "best.pt"),
        "--base",
        str(tmp_path / "base.png"),
        "--out",
        str(tmp_path / "generated"),
    ]

    inherited = vars(parser.parse_args(required))
    novelty_keys = set(train.NoveltyConfig.__dataclass_fields__)
    quality_keys = set(train.QualityConfig.__dataclass_fields__)
    assert novelty_keys.isdisjoint(inherited)
    assert quality_keys.isdisjoint(inherited)

    explicit = vars(
        parser.parse_args(
            required
            + [
                "--duplicate-threshold",
                "0.91",
                "--maximum-ink",
                "0.22",
            ]
        )
    )
    assert explicit["duplicate_threshold"] == pytest.approx(0.91)
    assert explicit["maximum_ink"] == pytest.approx(0.22)
    assert (novelty_keys - {"duplicate_threshold"}).isdisjoint(explicit)
    assert (quality_keys - {"maximum_ink"}).isdisjoint(explicit)


def test_conditional_vae_forward_shapes_and_finite_values():
    torch = pytest.importorskip("torch")
    model = train.ConditionalVAE(image_size=128, latent_dim=32, base_channels=8)
    model.eval()
    target = torch.zeros(1, 1, 128, 128)
    condition = torch.zeros_like(target)
    strength = torch.tensor([[0.35]])

    with torch.no_grad():
        output = model(target, condition, strength)

    assert isinstance(output, dict)
    logits = output["logits"]
    add_logits = output["add_logits"]
    remove_logits = output["remove_logits"]
    mu = output["mu"]
    logvar = output["logvar"]
    prior_mu = output["prior_mu"]
    prior_logvar = output["prior_logvar"]
    width = output["width"]
    assert logits.shape == target.shape
    assert add_logits.shape == remove_logits.shape == target.shape
    assert mu.shape == logvar.shape == prior_mu.shape == prior_logvar.shape == (1, 32)
    assert width.numel() == 1
    assert 1.0 <= float(width.item()) <= 6.0
    assert all(
        torch.isfinite(value).all()
        for value in (
            logits,
            add_logits,
            remove_logits,
            mu,
            logvar,
            prior_mu,
            prior_logvar,
            width,
        )
    )
    composed = (
        condition * (1.0 - torch.sigmoid(remove_logits))
        + (1.0 - condition) * torch.sigmoid(add_logits)
    )
    assert torch.allclose(torch.sigmoid(logits), composed, atol=1e-5)

    fixed_z = torch.zeros(1, 32)
    with torch.no_grad():
        sampled_a = model.sample(
            condition, strength, z=fixed_z, return_components=True
        )
        sampled_b = model.sample(
            condition, strength, z=fixed_z, return_components=True
        )
    assert sampled_a["logits"].shape == target.shape
    assert sampled_a["add_logits"].shape == target.shape
    assert sampled_a["remove_logits"].shape == target.shape
    assert sampled_a["width"].shape == (1,)
    for key in ("logits", "add_logits", "remove_logits", "width"):
        assert torch.equal(sampled_a[key], sampled_b[key])
