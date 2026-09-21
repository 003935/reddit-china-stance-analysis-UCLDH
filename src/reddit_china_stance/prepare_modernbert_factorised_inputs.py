"""Build the exact private input bundle for factorised ModernBERT-v2 preparation.

The helper validates every upstream artefact locally, derives the 480-thread
bridge exposure register, and writes only metadata-only upload/spec manifests
plus the ignored private register.  It never launches Modal or a GPU.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from reddit_china_stance.modernbert_factorised_experiment import (
    build_source_bundle,
    canonical_sha256,
    file_sha256,
)
from reddit_china_stance.modernbert_factorised_training import (
    _validate_source_provenance_binding,
    _validate_teacher_receipt_binding,
    derive_bridge_exposure_register,
    validate_private_exposure_register,
)
from reddit_china_stance.modernbert_training import (
    validate_development_proxy_parquet,
)
from reddit_china_stance.privacy import assert_metadata_only
from reddit_china_stance.semantic_ontology_v2 import (
    RUBRIC_PATH,
    SCHEMA_PATH,
    validate_pilot_packet,
)
from reddit_china_stance.sol_teacher_10k_v2 import (
    validate_accepted_bridge_receipt,
    validate_teacher_packet,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = REPO_ROOT / "data/private-modernbert-factorised-v2"
TEACHER_RUN_ID = "820ed0ab1f6b65ba6f79659eb7b7e7d9172d0900d8236fc9b80822d0b726d831"
TEACHER_PACKET_ID = "94c87ac1a9e68b3977fdf7332ec96017e60e77fde3647464a0454ff144e18797"
BRIDGE_PACKET_ID = "aba1d87f00a2174160e27839c970509970863357aaf7a7438f969b2a3942452c"
BRIDGE_RUN_ID = "ff94fe4339865124cb94af573ae00b0756cf0c35ed154b257ce60b1597d2ad83"

SOURCE_PARQUET = (
    REPO_ROOT / "data/private-hf-sol-teacher-10k-v1/data/train-00000-of-00001.parquet"
)
TEACHER_PACKET_ROOT = (
    REPO_ROOT / f"data/private-sol-teacher-10k-v2/packet/packet={TEACHER_PACKET_ID}"
)
TEACHER_PRIVATE_ROOT = REPO_ROOT / "data/private-sol-teacher-10k-v2/generation"
TEACHER_PUBLIC_ROOT = REPO_ROOT / "outputs/sol-teacher-10k-v2"
TEACHER_LABELS = TEACHER_PRIVATE_ROOT / f"run={TEACHER_RUN_ID}/final/labels.parquet"
TEACHER_RECEIPT = (
    TEACHER_PUBLIC_ROOT
    / f"run={TEACHER_RUN_ID}"
    / "receipt-779adcdfe8fbe137d8fcc2296e774321b3fc435eaba145bc9afdebcd2797b4ff.json"
)
BRIDGE_PACKET_ROOT = (
    REPO_ROOT / f"data/private-semantic-ontology-v2/pilot/packet={BRIDGE_PACKET_ID}"
)
BRIDGE_AUTHORISATION = (
    REPO_ROOT
    / f"outputs/semantic-ontology-v2-bridge/run={BRIDGE_RUN_ID}"
    / "receipt-0a9f80812790a7012673abcccd69d6832a63e37f370ca3902b58d85bc6bef66d.json"
)
LEGACY_PROXY = REPO_ROOT / "data/private-modernbert-v1/inputs/development-proxy.parquet"

SOURCE_FILES = (
    "src/reddit_china_stance/modal_modernbert_factorised.py",
    "src/reddit_china_stance/modernbert_factorised_data.py",
    "src/reddit_china_stance/modernbert_factorised_experiment.py",
    "src/reddit_china_stance/modernbert_factorised_model.py",
    "src/reddit_china_stance/modernbert_factorised_splits.py",
    "src/reddit_china_stance/modernbert_factorised_training.py",
    "src/reddit_china_stance/prepare_modernbert_factorised_inputs.py",
    "src/reddit_china_stance/semantic_evaluation_v2.py",
    "src/reddit_china_stance/semantic_ontology_v2.py",
    "src/reddit_china_stance/privacy.py",
)

# Documented L4 charges since the user's ModernBERT $200 envelope began.  Failed
# paid trials remain included; provider costs without currency telemetry are not
# invented and therefore cannot appear in this measured-Modal subtotal.
MEASURED_MODAL_COSTS_USD = {
    "compatibility_preflight": Decimal("0.083"),
    "initial_asha_epoch_1": Decimal("0.734"),
    "corrected_asha": Decimal("1.714"),
    "invalid_stability": Decimal("1.868"),
    "corrective_stability": Decimal("1.810"),
    "confirmatory_curve": Decimal("4.862"),
    "locked_test_inference": Decimal("0.120"),
    "conditional_search": Decimal("7.478"),
    "two_model_cascade": Decimal("4.581824"),
}
CUMULATIVE_MEASURED_SPEND_USD = sum(MEASURED_MODAL_COSTS_USD.values(), Decimal("0"))
ACTIVE_RESERVATION_USD = Decimal("0")
PLANNED_PHASE_UPPER_USD = Decimal("25")
HARD_COST_CAP_USD = Decimal("200")
L4_RATE_USD_PER_GPU_SECOND = Decimal("0.80") / Decimal("3600")
MAX_GPU_SECONDS_BY_COMPONENT = {
    "relevance": 7_200,
    "target_stance_b4": 10_800,
    "target_stance_b2": 10_800,
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite different immutable state: {path}")
        return
    with path.open("xb") as handle:
        handle.write(encoded)


def _descriptor(
    path: Path,
    *,
    relative_path: str,
    row_count: int | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, Any] = {
        "relative_path": relative_path,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if row_count is not None:
        result["row_count"] = row_count
    return result


def _repo_descriptor(path: Path, *, row_count: int | None = None) -> dict[str, Any]:
    return _descriptor(
        path,
        relative_path=path.relative_to(REPO_ROOT).as_posix(),
        row_count=row_count,
    )


def _one_receipt(root: Path) -> Path:
    paths = sorted(root.glob("receipt-*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one content-addressed receipt under {root}")
    return paths[0]


def _normalise_output_root(output_root: Path) -> Path:
    """Resolve CLI paths into the registered ignored private namespace."""

    candidate = output_root if output_root.is_absolute() else REPO_ROOT / output_root
    resolved = candidate.resolve()
    private_root = PRIVATE_ROOT.resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as exc:
        raise ValueError(
            "factorised output_root must remain inside "
            "data/private-modernbert-factorised-v2"
        ) from exc
    return resolved


def _validate_teacher_outputs(*, packet_receipt_path: Path) -> dict[str, Any]:
    """Exact-bind final labels without replaying every already-receipted shard."""

    receipt = _read_json(TEACHER_RECEIPT)
    if TEACHER_RECEIPT.stem != f"receipt-{canonical_sha256(receipt)}":
        raise RuntimeError("teacher receipt content address drifted")
    _validate_teacher_receipt_binding(
        receipt,
        expected_teacher_run_id=TEACHER_RUN_ID,
        teacher_labels_sha256=file_sha256(TEACHER_LABELS),
        private_mapping_sha256=file_sha256(
            TEACHER_PACKET_ROOT / "private-mapping.parquet"
        ),
        source_parquet_sha256=file_sha256(SOURCE_PARQUET),
        rubric_sha256=file_sha256(RUBRIC_PATH),
        schema_sha256=file_sha256(SCHEMA_PATH),
    )
    _validate_source_provenance_binding(
        _read_json(packet_receipt_path),
        receipt_path=packet_receipt_path,
        expected_packet_id=TEACHER_PACKET_ID,
        source_parquet_sha256=file_sha256(SOURCE_PARQUET),
        private_mapping_sha256=file_sha256(
            TEACHER_PACKET_ROOT / "private-mapping.parquet"
        ),
        blinded_input_sha256=file_sha256(
            TEACHER_PACKET_ROOT / "blinded-input.json"
        ),
    )
    parquet = pq.ParquetFile(TEACHER_LABELS)
    expected_columns = (
        "opaque_id",
        "codability",
        "relevance",
        "label_json",
        "quality_tier",
        "primary_training_eligible",
    )
    expected_metadata = {
        b"kind": b"sol-teacher-10k-v2-final-labels-v1",
        b"run_id": TEACHER_RUN_ID.encode(),
        b"packet_id": TEACHER_PACKET_ID.encode(),
        b"label_schema_sha256": file_sha256(SCHEMA_PATH).encode(),
        b"source_parquet_sha256": file_sha256(SOURCE_PARQUET).encode(),
    }
    if (
        parquet.metadata.num_rows != 10_000
        or tuple(parquet.schema_arrow.names) != expected_columns
        or parquet.schema_arrow.metadata != expected_metadata
    ):
        raise RuntimeError("teacher labels Parquet schema or metadata binding drifted")
    return receipt


def build_prepare_inputs(*, output_root: Path = PRIVATE_ROOT) -> dict[str, Any]:
    """Validate upstream evidence and freeze the exact local/Volume input plan."""

    output_root = _normalise_output_root(output_root)
    teacher_packet_receipt = _one_receipt(TEACHER_PACKET_ROOT)
    teacher_packet = validate_teacher_packet(
        TEACHER_PACKET_ROOT,
        source_parquet_path=SOURCE_PARQUET,
        expected_rows=10_000,
    )
    teacher_receipt = _validate_teacher_outputs(
        packet_receipt_path=teacher_packet_receipt
    )
    if (
        teacher_packet.get("packet_id") != TEACHER_PACKET_ID
        or teacher_receipt.get("run_id") != TEACHER_RUN_ID
        or teacher_receipt.get("packet_id") != TEACHER_PACKET_ID
    ):
        raise RuntimeError("fresh-v2 teacher identity drifted")

    validate_pilot_packet(BRIDGE_PACKET_ROOT)
    validate_accepted_bridge_receipt(BRIDGE_AUTHORISATION)
    legacy = validate_development_proxy_parquet(LEGACY_PROXY)
    if legacy.get("row_count") != 452:
        raise RuntimeError("legacy proxy row count drifted")

    bridge_mapping = BRIDGE_PACKET_ROOT / "private-mapping.parquet"
    bridge_manifest = BRIDGE_PACKET_ROOT / "manifest.json"
    bridge_packet_receipt = _one_receipt(BRIDGE_PACKET_ROOT)
    bridge_register = derive_bridge_exposure_register(
        pilot_mapping_parquet_path=bridge_mapping,
        pilot_mapping_descriptor=_repo_descriptor(bridge_mapping, row_count=480),
        pilot_packet_manifest_path=bridge_manifest,
        pilot_packet_manifest_descriptor=_repo_descriptor(bridge_manifest),
        pilot_packet_receipt_path=bridge_packet_receipt,
        pilot_packet_receipt_descriptor=_repo_descriptor(bridge_packet_receipt),
        bridge_authorisation_path=BRIDGE_AUTHORISATION,
        bridge_authorisation_descriptor=_repo_descriptor(BRIDGE_AUTHORISATION),
        descriptor_root=REPO_ROOT,
    )
    validate_private_exposure_register(bridge_register, scope="bridge")

    source_bundle = build_source_bundle(REPO_ROOT, SOURCE_FILES)
    source_bundle_sha256 = canonical_sha256(source_bundle)
    input_hashes = {
        "teacher_receipt_sha256": file_sha256(TEACHER_RECEIPT),
        "teacher_labels_sha256": file_sha256(TEACHER_LABELS),
        "blinded_input_sha256": file_sha256(TEACHER_PACKET_ROOT / "blinded-input.json"),
        "private_mapping_sha256": file_sha256(
            TEACHER_PACKET_ROOT / "private-mapping.parquet"
        ),
        "source_parquet_sha256": file_sha256(SOURCE_PARQUET),
        "source_receipt_sha256": file_sha256(teacher_packet_receipt),
        "bridge_register_id": bridge_register["register_id"],
        "bridge_authorisation_sha256": file_sha256(BRIDGE_AUTHORISATION),
        "legacy_proxy_sha256": file_sha256(LEGACY_PROXY),
        "rubric_sha256": file_sha256(RUBRIC_PATH),
        "schema_sha256": file_sha256(SCHEMA_PATH),
        "source_file_hashes": {
            relative: file_sha256(REPO_ROOT / relative) for relative in SOURCE_FILES
        },
        "source_bundle_sha256": source_bundle_sha256,
        "dependency_lock_sha256": file_sha256(REPO_ROOT / "uv.lock"),
        "cost_accounting": {
            "measured_modal_costs_usd": {
                key: format(value, "f")
                for key, value in sorted(MEASURED_MODAL_COSTS_USD.items())
            },
            "cumulative_measured_spend_usd": format(
                CUMULATIVE_MEASURED_SPEND_USD, "f"
            ),
            "active_reservation_usd": format(ACTIVE_RESERVATION_USD, "f"),
            "planned_phase_upper_usd": format(PLANNED_PHASE_UPPER_USD, "f"),
            "hard_cost_cap_usd": format(HARD_COST_CAP_USD, "f"),
        },
    }
    bundle_id = canonical_sha256(input_hashes)
    local_bundle_root = output_root / f"input-bundle={bundle_id}"
    register_path = local_bundle_root / "bridge/exposure-register.json"
    _write_immutable_json(register_path, bridge_register)

    volume_root = (
        "student-modernbert-factorised-v2/input-bundles/"
        f"bundle={bundle_id}"
    )
    local_assets = {
        "teacher_receipt": TEACHER_RECEIPT,
        "teacher_ledger": TEACHER_LABELS,
        "teacher_blinded_input": TEACHER_PACKET_ROOT / "blinded-input.json",
        "teacher_private_mapping": TEACHER_PACKET_ROOT / "private-mapping.parquet",
        "source_parquet": SOURCE_PARQUET,
        "source_receipt": teacher_packet_receipt,
        "bridge_exposure_register": register_path,
        "bridge_authorisation": BRIDGE_AUTHORISATION,
        "legacy_proxy": LEGACY_PROXY,
    }
    remote_paths = {
        "teacher_receipt": f"{volume_root}/teacher/{TEACHER_RECEIPT.name}",
        "teacher_ledger": f"{volume_root}/teacher/labels.parquet",
        "teacher_blinded_input": f"{volume_root}/packet/blinded-input.json",
        "teacher_private_mapping": f"{volume_root}/packet/private-mapping.parquet",
        "source_parquet": f"{volume_root}/source/source.parquet",
        "source_receipt": f"{volume_root}/packet/{teacher_packet_receipt.name}",
        "bridge_exposure_register": f"{volume_root}/bridge/exposure-register.json",
        "bridge_authorisation": f"{volume_root}/bridge/{BRIDGE_AUTHORISATION.name}",
        "legacy_proxy": f"{volume_root}/legacy/development-proxy.parquet",
    }
    row_counts = {
        "teacher_receipt": 10_000,
        "teacher_ledger": 10_000,
        "teacher_blinded_input": 10_000,
        "teacher_private_mapping": 10_000,
        "source_parquet": 10_000,
        "bridge_exposure_register": 480,
        "bridge_authorisation": 480,
        "legacy_proxy": 452,
    }
    descriptors = {
        key: _descriptor(
            path,
            relative_path=remote_paths[key],
            row_count=row_counts.get(key),
        )
        for key, path in local_assets.items()
    }
    spec = {
        "teacher_run_id": TEACHER_RUN_ID,
        "teacher_receipt": descriptors["teacher_receipt"],
        "teacher_receipt_relative_path": remote_paths["teacher_receipt"],
        "teacher_ledger": descriptors["teacher_ledger"],
        "teacher_labels_parquet_relative_path": remote_paths["teacher_ledger"],
        "teacher_blinded_input": descriptors["teacher_blinded_input"],
        "blinded_input_json_relative_path": remote_paths["teacher_blinded_input"],
        "teacher_private_mapping": descriptors["teacher_private_mapping"],
        "private_mapping_parquet_relative_path": remote_paths[
            "teacher_private_mapping"
        ],
        "source_parquet": descriptors["source_parquet"],
        "source_parquet_relative_path": remote_paths["source_parquet"],
        "source_receipt": descriptors["source_receipt"],
        "source_receipt_relative_path": remote_paths["source_receipt"],
        "source_metadata_mapping": descriptors["teacher_private_mapping"],
        "source_metadata_mapping_relative_path": remote_paths[
            "teacher_private_mapping"
        ],
        "bridge_exposure_register": descriptors["bridge_exposure_register"],
        "bridge_exposure_register_json_relative_path": remote_paths[
            "bridge_exposure_register"
        ],
        "bridge_receipt_relative_path": remote_paths["bridge_authorisation"],
        "legacy_proxy": descriptors["legacy_proxy"],
        "legacy_proxy_parquet_relative_path": remote_paths["legacy_proxy"],
        "rubric": {
            "repo_relative_path": RUBRIC_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": file_sha256(RUBRIC_PATH),
            "bytes": RUBRIC_PATH.stat().st_size,
        },
        "schema": {
            "repo_relative_path": SCHEMA_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": file_sha256(SCHEMA_PATH),
            "bytes": SCHEMA_PATH.stat().st_size,
        },
        "bridge_authorisation": descriptors["bridge_authorisation"],
        "source_files": list(SOURCE_FILES),
        "source_bundle_sha256": source_bundle_sha256,
        "dependency_lock_sha256": input_hashes["dependency_lock_sha256"],
        "rate_card_usd_per_gpu_second": {
            "L4": format(L4_RATE_USD_PER_GPU_SECOND, "f")
        },
        "cumulative_measured_spend_usd": format(
            CUMULATIVE_MEASURED_SPEND_USD, "f"
        ),
        "active_reservation_usd": format(ACTIVE_RESERVATION_USD, "f"),
        "planned_phase_upper_usd": format(PLANNED_PHASE_UPPER_USD, "f"),
        "hard_cost_cap_usd": format(HARD_COST_CAP_USD, "f"),
        "max_gpu_seconds_by_component": dict(MAX_GPU_SECONDS_BY_COMPONENT),
    }
    upload_plan = {
        "schema_version": "1.0.0",
        "kind": "modernbert-factorised-upload-plan-v1",
        "bundle_id": bundle_id,
        "volume_name": "reddit-china-stance-data",
        "environment": "main",
        "assets": [
            {
                "name": key,
                "local_path": path.relative_to(REPO_ROOT).as_posix(),
                "remote_path": remote_paths[key],
                "sha256": descriptors[key]["sha256"],
                "bytes": descriptors[key]["bytes"],
            }
            for key, path in sorted(local_assets.items())
        ],
    }
    upload_plan = {**upload_plan, "plan_id": canonical_sha256(upload_plan)}
    assert_metadata_only(spec, where="factorised prepare input spec")
    assert_metadata_only(input_hashes, where="factorised input bundle binding")
    _write_immutable_json(local_bundle_root / "input-bindings.json", input_hashes)
    _write_immutable_json(local_bundle_root / "upload-plan.json", upload_plan)
    _write_immutable_json(output_root / "prepare-input.json", spec)
    return {
        "status": "prepared-locally",
        "bundle_id": bundle_id,
        "upload_plan_id": upload_plan["plan_id"],
        "asset_count": len(local_assets),
        "teacher_rows": 10_000,
        "bridge_threads": 480,
        "legacy_proxy_rows": 452,
        "cumulative_measured_spend_usd": format(
            CUMULATIVE_MEASURED_SPEND_USD, "f"
        ),
        "planned_phase_upper_usd": format(PLANNED_PHASE_UPPER_USD, "f"),
        "reserved_gpu_cost_usd": format(
            L4_RATE_USD_PER_GPU_SECOND
            * Decimal(
                3
                * sum(MAX_GPU_SECONDS_BY_COMPONENT.values())
            ),
            "f",
        ),
        "hard_cost_cap_usd": format(HARD_COST_CAP_USD, "f"),
        "locked_test_rows_accessed": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PRIVATE_ROOT)
    args = parser.parse_args(argv)
    result = build_prepare_inputs(output_root=_normalise_output_root(args.output_root))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ACTIVE_RESERVATION_USD",
    "CUMULATIVE_MEASURED_SPEND_USD",
    "HARD_COST_CAP_USD",
    "L4_RATE_USD_PER_GPU_SECOND",
    "MAX_GPU_SECONDS_BY_COMPONENT",
    "MEASURED_MODAL_COSTS_USD",
    "PLANNED_PHASE_UPPER_USD",
    "SOURCE_FILES",
    "build_prepare_inputs",
]
