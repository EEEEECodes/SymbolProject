"""Cheap paired-family orchestration and public-contract tests.

The geometry-heavy SVG checks live in :mod:`tests.test_train`.  This module
focuses on the paired dataset shape, registration/split metadata, example
sampling, checkpoint compatibility, and the three importable workflows used by
the CLI and local webpage.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
import threading

import numpy as np
from PIL import Image, ImageDraw
import pytest

import train


def _draw_base(draw: ImageDraw.ImageDraw, family_index: int) -> None:
    """Draw four asymmetric bases with a stable outer extent."""

    draw.rectangle((7, 7, 56, 56), outline=0, width=2)
    if family_index % 4 == 0:
        draw.line((7, 7, 27, 27), fill=0, width=2)
    elif family_index % 4 == 1:
        draw.line((56, 7, 36, 27), fill=0, width=2)
    elif family_index % 4 == 2:
        draw.line((7, 34, 28, 34), fill=0, width=2)
    else:
        draw.line((35, 56, 35, 37), fill=0, width=2)


def _write_family_dataset(
    root: Path,
    *,
    family_count: int = 4,
    deviations_per_family: int = 20,
) -> Path:
    """Create complete base-plus-addition targets in the required tree."""

    for family_index in range(family_count):
        family = root / f"family-{family_index + 1:02d}"
        deviations = family / "deviations"
        deviations.mkdir(parents=True, exist_ok=True)

        base = Image.new("L", (64, 64), 255)
        _draw_base(ImageDraw.Draw(base), family_index)
        base.save(family / "base.png")

        for deviation_index in range(deviations_per_family):
            target = base.copy()
            draw = ImageDraw.Draw(target)
            x = 10 + 2 * deviation_index
            draw.line((x, 7, x, 31 + deviation_index % 5), fill=0, width=2)
            # Give each family a slightly different terminal direction without
            # changing the stable outer frame.
            branch = 5 + family_index
            draw.line((x, 31, min(54, x + branch), 31), fill=0, width=2)
            target.save(deviations / f"deviation-{deviation_index + 1:03d}.png")
    return root


def _small_config(data: Path, destination: Path) -> dict[str, object]:
    return {
        "paths": {"data": str(data), "report": str(destination)},
        "preprocessing": {
            "image_size": 32,
            "margin": 2,
            "max_source_stroke_width": 4,
            "min_component_pixels": 1,
            "max_input_pixels": 1_000_000,
            "filled_policy": "outline",
            "validation_fraction": 0.10,
        },
        # The synthetic fixture is already aligned; disabling the nuisance
        # search keeps the integration tests quick without bypassing pairing.
        "registration": {
            "angle_range": 0,
            "translation_range": 0,
            "scale_range": 0,
            "match_tolerance": 0,
            "minimum_overlap": 0.05,
        },
        "training": {"seed": 1337},
    }


def _manifest_records(result: dict[str, object]) -> list[dict[str, object]]:
    artifacts = result["artifacts"]
    assert isinstance(artifacts, dict)
    payload = json.loads(Path(artifacts["manifest_json"]).read_text(encoding="utf-8"))
    return payload["records"]


def _processed(source: str, mask: np.ndarray) -> train.ProcessedSymbol:
    gray = np.where(mask, 0, 255).astype(np.uint8)
    return train.ProcessedSymbol(
        source=source,
        line_mask=mask.copy(),
        render_mask=mask.copy(),
        normalized_gray=gray,
        stroke_width=2.0,
        conversion={},
    )


def _pair(family_id: str, index: int) -> train.PairedSymbol:
    base = np.zeros((32, 32), dtype=bool)
    base[6:27, 7] = True
    target = base.copy()
    target[10 + index % 10, 7:18 + index % 5] = True
    addition, removal = train._delta_masks(base, target, tolerance=1)
    return train.PairedSymbol(
        family_id=family_id,
        base=_processed(f"{family_id}/base.png", base),
        target=_processed(f"{family_id}/deviation-{index}.png", target),
        registered_base=base,
        addition_mask=addition,
        removal_mask=removal,
        registration={"overlap": 1.0},
        raw_change_ratio=float(addition.sum() / base.sum()),
        strength=0.4,
    )


def test_web_defaults_describe_only_the_paired_family_workflow():
    defaults = train.web_defaults()

    assert json.loads(json.dumps(defaults)) == defaults
    assert set(defaults) == {
        "paths",
        "preprocessing",
        "registration",
        "model",
        "training",
        "generation",
        "novelty",
        "quality",
        "safety",
    }
    assert defaults["paths"]["init_checkpoint"] == ""
    assert defaults["paths"]["base"] == ""
    assert defaults["registration"] == {
        "angle_range": 12.0,
        "translation_range": 8,
        "scale_range": 0.12,
        "match_tolerance": 3.0,
        "minimum_overlap": 0.25,
    }
    assert defaults["model"]["latent_dim"] == 32
    assert "empty_condition_probability" not in defaults["training"]
    assert (
        defaults["training"]["real_pair_probability"],
        defaults["training"]["synthetic_pair_probability"],
        defaults["training"]["identity_pair_probability"],
    ) == pytest.approx((0.60, 0.30, 0.10))
    assert defaults["training"]["audit_sample_count"] == 32
    assert defaults["safety"]["minimum_families"] == 4
    assert defaults["safety"]["minimum_deviations_per_family"] == 20

    defaults["model"]["latent_dim"] = 99
    assert train.web_defaults()["model"]["latent_dim"] == 32


def test_paired_config_validation_locks_cross_field_rules():
    with pytest.raises(ValueError, match="sum to 1"):
        train.TrainingConfig(
            data="dataset",
            run="run",
            real_pair_probability=0.7,
            synthetic_pair_probability=0.3,
            identity_pair_probability=0.2,
        ).validate()

    with pytest.raises(ValueError, match="mutually exclusive"):
        train.TrainingConfig(
            data="dataset",
            run="run",
            resume="last.pt",
            init_checkpoint="best.pt",
        ).validate()

    with pytest.raises(ValueError, match="base image"):
        train.GenerationConfig(checkpoint="best.pt", out="generated").validate()

    with pytest.raises(ValueError, match="minimum_overlap"):
        train.RegistrationConfig(minimum_overlap=1.1).validate()


def test_delta_masks_tolerate_redraw_jitter_but_keep_real_edits():
    base = np.zeros((32, 32), dtype=bool)
    base[6:26, 8] = True
    redrawn = np.zeros_like(base)
    redrawn[6:26, 9] = True
    redrawn[20, 9:25] = True

    addition, removal = train._delta_masks(base, redrawn, tolerance=2)

    assert addition[20, 15:25].any()
    assert int(addition.sum()) >= 8
    assert int(removal.sum()) == 0


def test_registration_recovers_scan_rotation_and_translation():
    base = np.zeros((64, 64), dtype=bool)
    base[14:51, 17] = True
    base[50, 17:48] = True
    base[27, 17:34] = True
    target = train._transform_mask(base, angle=5.0, dx=3, dy=-2)

    registered, details = train.register_base_to_target(
        base,
        target,
        train.RegistrationConfig(
            angle_range=8,
            translation_range=5,
            scale_range=0,
            match_tolerance=1,
            minimum_overlap=0.75,
        ),
    )

    assert details["overlap"] >= 0.90
    assert abs(details["angle"]) <= 8.0
    assert abs(details["dx"]) <= 5.0
    assert abs(details["dy"]) <= 5.0
    assert train.changed_line_amount(registered, target, tolerance=1)["similarity"] >= 0.9

    unrelated = np.zeros_like(base)
    unrelated[8, 8:55] = True
    with pytest.raises(train.SymbolGeneratorError, match="registration overlap"):
        train.register_base_to_target(
            base,
            unrelated,
            train.RegistrationConfig(
                angle_range=0,
                translation_range=0,
                scale_range=0,
                match_tolerance=0,
                minimum_overlap=0.9,
            ),
        )


def test_family_balanced_dataset_and_all_three_example_modes():
    torch = pytest.importorskip("torch")
    pairs = [_pair("small", 0)] + [_pair("large", index) for index in range(3)]

    balanced = train.PairedSymbolDataset(
        pairs,
        training=True,
        real_probability=1.0,
        synthetic_probability=0.0,
        identity_probability=0.0,
    )
    assert len(balanced) == 6
    counts = Counter(balanced[index]["family_id"] for index in range(len(balanced)))
    assert counts == {"large": 3, "small": 3}
    assert {balanced[index]["example_type"] for index in range(len(balanced))} == {"real"}

    synthetic = train.PairedSymbolDataset(
        pairs,
        training=True,
        real_probability=0.0,
        synthetic_probability=1.0,
        identity_probability=0.0,
    )[0]
    assert synthetic["example_type"] == "synthetic"
    assert torch.isfinite(synthetic["strength"])
    assert synthetic["add_target"].shape == synthetic["target"].shape

    identity = train.PairedSymbolDataset(
        pairs,
        training=True,
        real_probability=0.0,
        synthetic_probability=0.0,
        identity_probability=1.0,
    )[0]
    assert identity["example_type"] == "identity"
    assert float(identity["strength"]) == pytest.approx(0.0)
    assert torch.equal(identity["condition"], identity["target"])


def test_unseen_base_audit_folds_hold_out_whole_families_once():
    four_family_pairs = [
        _pair(f"family-{family_index}", deviation_index)
        for family_index in range(4)
        for deviation_index in range(2 + family_index)
    ]
    leave_one_out = train._audit_fold_assignments(four_family_pairs, seed=1337)
    assert len(leave_one_out) == 4
    assert all(len(fold["held_out_families"]) == 1 for fold in leave_one_out)
    assert sorted(
        family_id
        for fold in leave_one_out
        for family_id in fold["held_out_families"]
    ) == ["family-0", "family-1", "family-2", "family-3"]
    assert leave_one_out == train._audit_fold_assignments(four_family_pairs, seed=1337)

    counts = [9, 8, 7, 6, 5, 4, 3]
    many_family_pairs = [
        _pair(f"family-{family_index}", deviation_index)
        for family_index, count in enumerate(counts)
        for deviation_index in range(count)
    ]
    grouped = train._audit_fold_assignments(many_family_pairs, seed=7)
    assert len(grouped) == 5
    held_out = [
        family_id for fold in grouped for family_id in fold["held_out_families"]
    ]
    assert sorted(held_out) == [f"family-{index}" for index in range(7)]
    totals = [fold["deviation_count"] for fold in grouped]
    assert sum(totals) == sum(counts)
    assert max(totals) - min(totals) <= max(counts)


def test_cross_family_duplicate_groups_never_cross_final_split():
    pairs = [
        _pair(f"family-{family_index}", deviation_index)
        for family_index in range(4)
        for deviation_index in range(5)
    ]
    train._assign_leakage_groups(pairs)
    train._family_split(pairs, fraction=0.10, seed=1337)

    splits_by_group: dict[str, set[str]] = {}
    for pair in pairs:
        splits_by_group.setdefault(pair.leakage_group, set()).add(pair.split)
    assert all(len(splits) == 1 for splits in splits_by_group.values())
    assert all(pair.cross_family_duplicate for pair in pairs)


def test_oversized_leakage_group_cannot_empty_a_family_into_validation():
    family_a = [_pair("family-a", index) for index in range(3)]
    family_b = [_pair("family-b", index) for index in range(5)]
    for pair in family_a + family_b[:2]:
        pair.leakage_group = "shared-oversized-group"
    for index, pair in enumerate(family_b[2:]):
        pair.leakage_group = f"family-b-only-{index}"

    training, validation = train._family_split(
        family_a + family_b, fraction=0.40, seed=1337
    )

    assert all(pair.split == "train" for pair in family_a)
    assert {pair.family_id for pair in training} == {"family-a", "family-b"}
    assert not any(pair.family_id == "family-a" for pair in validation)


def test_dataset_fingerprint_includes_split_seed_and_fraction():
    pairs = [_pair("family-a", index) for index in range(5)]
    for index, pair in enumerate(pairs):
        pair.leakage_group = f"group-{index}"
    train._family_split(pairs, fraction=0.20, seed=7)
    preprocessing = train.PreprocessConfig(image_size=32, margin=2)
    registration = train.RegistrationConfig()

    baseline = train._dataset_fingerprint(
        pairs, preprocessing, registration, validation_fraction=0.20, seed=7
    )
    changed_seed = train._dataset_fingerprint(
        pairs, preprocessing, registration, validation_fraction=0.20, seed=8
    )
    changed_fraction = train._dataset_fingerprint(
        pairs, preprocessing, registration, validation_fraction=0.25, seed=7
    )

    assert len({baseline, changed_seed, changed_fraction}) == 3


def test_paired_loss_has_finite_prior_delta_and_retention_terms(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(train, "_soft_skeletonize", lambda value, iterations=8: value)
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:24, 12] = 1
    condition = target.clone()
    condition[:, :, 16:, 12] = 0
    add_target = (target > condition).float()
    remove_target = torch.zeros_like(target)
    logits = torch.zeros_like(target, requires_grad=True)
    add_logits = torch.zeros_like(target, requires_grad=True)
    remove_logits = torch.zeros_like(target, requires_grad=True)
    mu = torch.zeros(2, 4, requires_grad=True)
    logvar = torch.zeros_like(mu, requires_grad=True)
    prior_mu = torch.full_like(mu, 0.1)
    prior_logvar = torch.full_like(mu, -0.1)

    loss, metrics = train.vae_loss(
        logits,
        target,
        mu,
        logvar,
        torch.full((2,), 2.0, requires_grad=True),
        torch.full((2,), 2.0),
        beta=1e-3,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
        add_logits=add_logits,
        remove_logits=remove_logits,
        add_target=add_target,
        remove_target=remove_target,
        condition=condition,
        delta_weight=0.5,
        retention_weight=0.25,
    )

    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["kl"] > 0
    assert metrics["delta"] > 0
    assert metrics["retention"] > 0


def test_validate_dataset_enforces_family_structure_and_minimums(tmp_path: Path):
    too_few = _write_family_dataset(
        tmp_path / "too-few-families", family_count=3, deviations_per_family=20
    )
    with pytest.warns(RuntimeWarning, match="30 or more"):
        with pytest.raises(train.SymbolGeneratorError, match="(?i)at least 4.*famil"):
            train.validate_dataset(_small_config(too_few, tmp_path / "report-families"))

    too_small = _write_family_dataset(
        tmp_path / "too-few-deviations", family_count=4, deviations_per_family=19
    )
    with pytest.raises(train.SymbolGeneratorError, match="(?i)20.*deviation"):
        train.validate_dataset(_small_config(too_small, tmp_path / "report-deviations"))


def test_validate_dataset_reports_ambiguous_layout_before_training(tmp_path: Path):
    malformed = tmp_path / "malformed"
    for index in range(4):
        (malformed / f"family-{index}" / "deviations").mkdir(parents=True)

    with pytest.raises(train.SymbolGeneratorError, match="exactly one supported base"):
        train.validate_dataset(_small_config(malformed, tmp_path / "report"))


def test_validation_cancellation_is_cooperative(tmp_path: Path):
    dataset = _write_family_dataset(
        tmp_path / "dataset", family_count=1, deviations_per_family=1
    )
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(train.SymbolGeneratorError, match="Operation cancelled"):
        train.validate_dataset(
            _small_config(dataset, tmp_path / "report"), cancel_event=cancelled
        )


def test_validate_dataset_reports_families_registration_and_stable_splits(tmp_path: Path):
    original = _write_family_dataset(tmp_path / "original")
    first_deviation = original / "family-01" / "deviations" / "deviation-001.png"
    shutil.copyfile(first_deviation, first_deviation.with_name("duplicate.png"))
    first_deviation.with_name("corrupt.png").write_bytes(b"not an image")
    Image.new("L", (64, 64), 255).save(first_deviation.with_name("blank.png"))
    with pytest.warns(RuntimeWarning, match="30 or more"):
        first = train.validate_dataset(_small_config(original, tmp_path / "report-a"))

    assert first["status"] == "complete"
    summary = first["summary"]
    assert summary["families"] == 4
    assert summary["usable_deviations"] == 80
    assert summary["validation"] >= 12
    assert summary["duplicate"] == 1
    assert summary["corrupt"] == 1
    assert summary["blank"] == 1
    records = _manifest_records(first)
    accepted = [
        record
        for record in records
        if record["record_type"] == "deviation" and record["status"] == "accepted"
    ]
    assert len(accepted) == 80
    assert {record["family_id"] for record in accepted} == {
        "family-01",
        "family-02",
        "family-03",
        "family-04",
    }
    assert all(record.get("registration") for record in accepted)
    assert all(record.get("leakage_group") for record in accepted)
    duplicate_records = [record for record in records if record["status"] == "duplicate"]
    assert len(duplicate_records) == 1
    assert duplicate_records[0]["duplicate_of"].endswith("deviation-001.png")
    validation_by_family = Counter(
        record["family_id"] for record in accepted if record["split"] == "validation"
    )
    assert min(validation_by_family.values()) >= 3

    artifacts = first["artifacts"]
    assert all(Path(artifacts[name]).is_file() for name in ("config", "manifest_json", "manifest_csv"))
    assert artifacts["contact_sheets"]
    family_sheets = [
        path
        for paths in artifacts["contact_sheets"].values()
        for path in (paths if isinstance(paths, list) else [paths])
    ]
    assert family_sheets
    assert all(Path(path).is_file() for path in family_sheets)

    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    with pytest.warns(RuntimeWarning, match="30 or more"):
        second = train.validate_dataset(_small_config(relocated, tmp_path / "report-b"))
    split_a = {
        (record["family_id"], record["relative_source"]): record["split"]
        for record in accepted
    }
    split_b = {
        (record["family_id"], record["relative_source"]): record["split"]
        for record in _manifest_records(second)
        if record["record_type"] == "deviation" and record["status"] == "accepted"
    }
    assert split_b == split_a


def test_unassigned_image_blocks_validation(tmp_path: Path):
    dataset = _write_family_dataset(tmp_path / "dataset")
    Image.new("L", (32, 32), 255).save(dataset / "stray.png")

    with pytest.warns(RuntimeWarning, match="30 or more"):
        with pytest.raises(train.SymbolGeneratorError, match="(?i)unassigned"):
            train.validate_dataset(_small_config(dataset, tmp_path / "report"))


def test_checkpoint_loader_rejects_legacy_unpaired_payload(tmp_path: Path):
    torch = pytest.importorskip("torch")
    legacy = tmp_path / "legacy.pt"
    torch.save(
        {
            "schema_version": 1,
            "kind": train.LEGACY_CHECKPOINT_KIND,
            "model_state": {},
        },
        legacy,
    )

    with pytest.raises(train.SymbolGeneratorError, match="(?i)unpaired|retrain.*paired"):
        train._load_checkpoint(legacy)


def test_generation_rejects_an_audit_stage_checkpoint(tmp_path: Path):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "audit-active.pt"
    torch.save(
        {
            "schema_version": train.SCHEMA_VERSION,
            "kind": train.CHECKPOINT_KIND,
            "stage": "audit",
            "model_state": {},
        },
        checkpoint,
    )

    with pytest.raises(
        train.SymbolGeneratorError,
        match="Generation requires a completed final checkpoint.*Audit-active",
    ):
        train.generate_symbols(
            {
                "paths": {
                    "checkpoint": str(checkpoint),
                    "base": str(tmp_path / "base.png"),
                    "out": str(tmp_path / "generated"),
                }
            }
        )

    train._require_final_checkpoint({"stage": "final"}, purpose="Generation")


def test_tiny_cpu_train_resume_and_required_base_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    torch = pytest.importorskip("torch")
    dataset = _write_family_dataset(tmp_path / "dataset")
    run = tmp_path / "run"

    monkeypatch.setattr(train, "_soft_skeletonize", lambda value, iterations=8: value)
    monkeypatch.setattr(
        train,
        "_shortlist_recall",
        lambda symbols, top_k: {"hits": len(symbols), "total": len(symbols), "recall": 1.0},
    )

    def fake_audit(
        config,
        prepared,
        metadata,
        device,
        use_amp,
        run_path,
        progress,
        cancel_event,
        resume_payload=None,
    ):
        del metadata, device, use_amp, run_path, cancel_event, resume_payload
        assignments = train._audit_fold_assignments(prepared.pairs, config.seed)
        folds = [
            {
                **assignment,
                "audit_epochs": 0,
                "scores": {
                    "eligible_deviations": assignment["deviation_count"],
                    "excluded_cross_family_duplicates": 0,
                },
            }
            for assignment in assignments
        ]
        if progress is not None:
            progress(0.45, "Synthetic test audit complete", {"stage": "audit"})
        return assignments, folds

    # Fold training/scoring is covered by the deterministic assignment and
    # model/loss tests above. This orchestration smoke replaces it with fixed
    # metrics so CI still exercises final training and safe checkpoint state.
    monkeypatch.setattr(train, "_run_unseen_family_audit", fake_audit)

    config: dict[str, object] = {
        "paths": {"data": str(dataset), "run": str(run)},
        "preprocessing": {
            "image_size": 32,
            "margin": 2,
            "max_source_stroke_width": 4,
            "min_component_pixels": 1,
            "max_input_pixels": 1_000_000,
            "filled_policy": "outline",
            "validation_fraction": 0.10,
        },
        "registration": {
            "angle_range": 0,
            "translation_range": 0,
            "scale_range": 0,
            "match_tolerance": 0,
            "minimum_overlap": 0.05,
        },
        "model": {
            "image_size": 32,
            "latent_dim": 4,
            "base_channels": 2,
            "min_stroke_width": 1,
            "max_stroke_width": 4,
        },
        "training": {
            "device": "cpu",
            "epochs": 1,
            "batch_size": 128,
            "patience": 1,
            "seed": 7,
            "workers": 0,
            "deterministic": True,
            "mixed_precision": False,
            "preview_count": 2,
            "preview_frequency": 1,
            "audit_sample_count": 1,
        },
        "novelty": {
            "alignment_angle": 0,
            "alignment_translation": 0,
            "alignment_scale": 0,
            "shortlist_maximum": 4,
            "precise_finalists": 1,
        },
    }
    progress: list[tuple[float, str]] = []
    with pytest.warns(RuntimeWarning, match="30 or more"):
        result = train.train_model(
            config,
            progress=lambda fraction, message, _payload: progress.append(
                (fraction, message)
            ),
        )

    assert result["status"] == "complete"
    assert result["device"] == "cpu"
    assert result["mixed_precision"] is False
    assert result["epochs_completed"] == 1
    assert len(result["audit"]["assignments"]) == 4
    assert len(result["audit"]["folds"]) == 4
    assert progress[-1][0] == pytest.approx(1.0)
    assert any("audit" in message.lower() for _fraction, message in progress)

    artifacts = result["artifacts"]
    for key in (
        "config",
        "manifest_json",
        "manifest_csv",
        "audit",
        "best_checkpoint",
        "last_checkpoint",
        "metrics_json",
        "metrics_csv",
    ):
        assert Path(artifacts[key]).is_file(), key
    assert artifacts["previews"] and all(
        Path(path).is_file() for path in artifacts["previews"]
    )

    best = torch.load(
        artifacts["best_checkpoint"], map_location="cpu", weights_only=True
    )
    last = torch.load(
        artifacts["last_checkpoint"], map_location="cpu", weights_only=True
    )
    assert best["schema_version"] == 2
    assert best["kind"] == train.CHECKPOINT_KIND
    assert best["dataset_fingerprint"] == result["dataset_fingerprint"]
    assert best["base_shape"] == [4, 32, 32]
    assert best["target_shape"] == [80, 32, 32]
    assert len(best["family_associations"]) == 80
    assert best["strength_calibration"]["definition"].startswith("percentile")
    assert "optimizer_state" not in best
    assert "optimizer_state" in last
    assert "torch_random_state" in last
    assert last["stage"] == "final"

    resumed_config = json.loads(json.dumps(config))
    resumed_config["paths"]["resume"] = artifacts["last_checkpoint"]
    with pytest.warns(RuntimeWarning, match="30 or more"):
        resumed = train.train_model(resumed_config)
    assert resumed["status"] == "complete"
    assert resumed["dataset_fingerprint"] == result["dataset_fingerprint"]
    assert resumed["epochs_completed"] == 1

    new_base = Image.new("L", (64, 64), 255)
    draw = ImageDraw.Draw(new_base)
    draw.ellipse((8, 8, 56, 56), outline=0, width=2)
    draw.line((16, 40, 48, 24), fill=0, width=2)
    base_path = tmp_path / "new-base.png"
    new_base.save(base_path)
    generated_path = tmp_path / "generated"
    generated = train.generate_symbols(
        {
            "paths": {
                "checkpoint": artifacts["best_checkpoint"],
                "base": str(base_path),
                "out": str(generated_path),
            },
            "generation": {
                "count": 1,
                "sampling_batch": 1,
                "attempt_multiplier": 1,
                "threshold_override": 0.95,
                "device": "cpu",
                "seed": 9,
            },
            "novelty": {
                "alignment_angle": 0,
                "alignment_translation": 0,
                "alignment_scale": 0,
                "shortlist_maximum": 4,
                "precise_finalists": 1,
            },
        }
    )
    assert generated["status"] in {"complete", "shortfall"}
    assert generated["attempt_count"] <= 1
    manifest_path = Path(generated["artifacts"]["manifest"])
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["input_base_hash"]
    assert manifest["requested_strength"] == pytest.approx(0.35)
    assert manifest["dataset_fingerprint"] == result["dataset_fingerprint"]
    assert len(manifest["audit_provenance"]["assignments"]) == 4
    assert (generated_path / "novel").is_dir()
    assert (generated_path / "review").is_dir()


def test_generation_requires_a_real_base_before_loading_checkpoint(tmp_path: Path):
    with pytest.raises(ValueError, match="base image"):
        train.generate_symbols(
            {
                "paths": {
                    "checkpoint": str(tmp_path / "missing.pt"),
                    "base": "",
                    "out": str(tmp_path / "generated"),
                }
            }
        )
