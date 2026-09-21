from __future__ import annotations

from reddit_china_stance import modal_modernbert_probability_random_calibration_v1 as launcher


def test_source_bundle_is_exactly_scoped_and_digest_bound() -> None:
    bundle = launcher.build_source_bundle()
    assert set(bundle["files"]) == set(launcher.REQUIRED_SOURCE_FILES)
    assert launcher.validate_source_bundle(bundle) == bundle


def test_calibration_run_id_is_bound_to_current_source_bundle() -> None:
    bundle = launcher.build_source_bundle()
    run_id = launcher._calibration_run_id(bundle["source_bundle_id"])
    assert len(run_id) == 64
    assert run_id == launcher._calibration_run_id(bundle["source_bundle_id"])
    assert run_id != launcher._calibration_run_id("0" * 64)
