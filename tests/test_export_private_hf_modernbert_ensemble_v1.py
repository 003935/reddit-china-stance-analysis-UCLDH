from __future__ import annotations

import json
from pathlib import Path

import huggingface_hub
import pytest
from huggingface_hub.hf_api import RepoFile

from reddit_china_stance import export_private_hf_modernbert_ensemble_v1 as exporter
from reddit_china_stance import modernbert_probability_random_candidate_v1 as candidate
from reddit_china_stance.semantic_ontology_v2 import canonical_sha256, file_sha256


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "source"
    raw_specs = (
        ("relevance", 47, 2, "a" * 64, b"relevance checkpoint"),
        ("target_stance_b4", 61, 3, "b" * 64, b"target checkpoint"),
    )
    specs: list[candidate.CheckpointSpec] = []
    receipts: list[dict[str, object]] = []
    for component, seed, epoch, trial_id, payload in raw_specs:
        checkpoint = source / "checkpoints" / component / f"seed={seed}" / "checkpoint.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(payload)
        spec = candidate.CheckpointSpec(
            component=component,
            seed=seed,
            selected_epoch=epoch,
            trial_id=trial_id,
            checkpoint_sha256=file_sha256(checkpoint),
        )
        specs.append(spec)
        receipt_body: dict[str, object] = {
            "experiment_run_id": candidate.ACQUISITION_TRAINING_RUN_ID,
            "phase_run_id": candidate.ACQUISITION_TRAINING_PHASE_ID,
            "arm": "random",
            "component": component,
            "optimiser_seed": seed,
            "training_frame_sha256": candidate.RANDOM_TRAINING_FRAME_SHA256,
            "invalid_outputs": 0,
            "locked_test_rows_accessed": 0,
            "aggregate_metrics": {"selected_epoch": epoch},
            "artifacts": {
                "checkpoint": {
                    "relative_path": "checkpoint.pt",
                    "sha256": spec.checkpoint_sha256,
                    "bytes": len(payload),
                }
            },
            "trial_id": trial_id,
        }
        receipt = {**receipt_body, "receipt_id": canonical_sha256(receipt_body)}
        receipts.append(receipt)
        _write_json(checkpoint.parent / "receipt.json", receipt)
    monkeypatch.setattr(candidate, "CHECKPOINT_SPECS", tuple(specs))
    monkeypatch.setattr(candidate, "COMPONENTS", ("relevance", "target_stance_b4"))
    monkeypatch.setattr(candidate, "SEEDS", (47, 61))
    cohort = candidate.freeze_checkpoint_cohort(receipts)
    monkeypatch.setattr(exporter, "CHECKPOINT_COHORT_ID", cohort["cohort_id"])
    _write_json(source / "provenance/checkpoint-cohort.json", cohort)

    calibration_body = {
        "bindings": {"checkpoint_cohort_id": cohort["cohort_id"]},
        "locked_test_rows_accessed": 0,
        "corpus_rows_accessed": 0,
    }
    calibration = {**calibration_body, "calibration_id": canonical_sha256(calibration_body)}
    monkeypatch.setattr(exporter, "CALIBRATION_ID", calibration["calibration_id"])
    calibration_root = source / "provenance/calibration"
    _write_json(calibration_root / "calibration.json", calibration)
    _write_json(calibration_root / "aggregate-validation.json", {"aggregate": "metadata"})
    _write_json(calibration_root / "throughput.json", {"throughput": "metadata"})
    _write_json(calibration_root / "checkpoint-provenance.json", {"members": []})
    _write_json(calibration_root / "source-bundle.json", {"source": "metadata"})
    calibration_run_id = "c" * 64
    monkeypatch.setattr(exporter, "CALIBRATION_RUN_ID", calibration_run_id)
    receipt_body = {
        "receipt_id": None,
        "calibration_run_id": calibration_run_id,
        "calibration_id": calibration["calibration_id"],
        "checkpoint_cohort_id": cohort["cohort_id"],
        "checkpoint_count": 2,
        "locked_test_rows_accessed": 0,
        "corpus_rows_accessed": 0,
        "status": "complete",
        "calibration_artifact": {
            **exporter._descriptor(calibration_root / "calibration.json"),
        },
        "aggregate_validation_artifact": {
            **exporter._descriptor(calibration_root / "aggregate-validation.json"),
        },
        "throughput_artifact": {
            **exporter._descriptor(calibration_root / "throughput.json"),
        },
    }
    del receipt_body["receipt_id"]
    receipt = {**receipt_body, "receipt_id": canonical_sha256(receipt_body)}
    monkeypatch.setattr(exporter, "CALIBRATION_RECEIPT_ID", receipt["receipt_id"])
    _write_json(calibration_root / "receipt.json", receipt)

    for name in exporter.BASE_MODEL_FILES:
        _write_json(source / "base-model" / name, {"name": name})
    config = tmp_path / "inference.toml"
    config.write_text("schema_version = '1.0.0'\n", encoding="utf-8")
    monkeypatch.setattr(exporter, "CONFIG_PATH", config)
    return source


def test_build_private_model_bundle_exactly_binds_ensemble_and_privacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"

    manifest = exporter.build_private_hf_model_bundle(source_root=source, output_dir=output)

    assert manifest["checkpoint_count"] == 2
    assert manifest["privacy"]["contains_model_weights"] is True
    assert manifest["privacy"]["contains_reddit_content"] is False
    assert sorted(output.glob("checkpoints/*/seed=*/checkpoint.pt")) == [
        output / "checkpoints/relevance/seed=47/checkpoint.pt",
        output / "checkpoints/target_stance_b4/seed=61/checkpoint.pt",
    ]
    assert "six-checkpoint ensemble" in (output / "README.md").read_text(encoding="utf-8")
    assert exporter.build_private_hf_model_bundle(source_root=source, output_dir=output) == manifest


def test_build_private_model_bundle_rejects_checkpoint_or_private_metadata_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fixture(tmp_path, monkeypatch)
    checkpoint = source / "checkpoints/relevance/seed=47/checkpoint.pt"
    checkpoint.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="checkpoint hash drifted"):
        exporter.build_private_hf_model_bundle(source_root=source, output_dir=tmp_path / "bad")

    source = _fixture(tmp_path / "second", monkeypatch)
    calibration = source / "provenance/calibration/throughput.json"
    _write_json(calibration, {"opaque_id": "private"})
    with pytest.raises(RuntimeError, match="forbidden private keys"):
        exporter.build_private_hf_model_bundle(source_root=source, output_dir=tmp_path / "private")


def test_remote_verifier_uses_lfs_oid_and_exact_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    large = tmp_path / "large.pt"
    small = tmp_path / "small.json"
    large.write_bytes(b"large")
    small.write_text("{}\n", encoding="utf-8")
    local_hashes = {"large.pt": file_sha256(large), "small.json": file_sha256(small)}
    local_sizes = {"large.pt": large.stat().st_size, "small.json": small.stat().st_size}

    class Api:
        def list_repo_tree(self, **_kwargs: object) -> list[RepoFile]:
            return [
                RepoFile(
                    path="large.pt",
                    size=large.stat().st_size,
                    oid="blob-large",
                    lfs={
                        "size": large.stat().st_size,
                        "oid": file_sha256(large),
                        "pointerSize": 128,
                    },
                ),
                RepoFile(path="small.json", size=small.stat().st_size, oid="blob-small"),
                RepoFile(path=".gitattributes", size=1, oid="server"),
            ]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **_kwargs: str(small))
    assert exporter._verify_remote_files(
        api=Api(),
        repo_id=exporter.DEFAULT_REPO_ID,
        revision="revision",
        local_hashes=local_hashes,
        local_sizes=local_sizes,
    ) == (1, 1, [".gitattributes"])
