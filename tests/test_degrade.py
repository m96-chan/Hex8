"""Tests for hex8.degrade: Phase 3 synthetic degradation harness (Issue #13).

These tests exercise the degradation library and harness machinery itself
with a small/fast payload+radius, not the full baseline severity sweep
(that sweep lives in the generated docs/phase3-baseline.md report).

Important expected outcome: the Phase 2 decoder (hex8.decoder.decode) only
handles the ideal, zero-distortion case (see hex8.decoder.detect's module
docstring). It has no rotation/perspective correction and no color-distance
tolerance yet - that hardening is Issue #14's job. So tests below that feed
a real (non-trivial) degradation into run_harness are expected, honestly,
to observe passed=False - this is the correct baseline result today, not a
bug to be fixed here.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from hex8.degrade import (
    DEGRADATIONS,
    HarnessResult,
    apply_blur,
    apply_brightness,
    apply_combined,
    apply_jpeg_compression,
    apply_noise,
    apply_perspective_warp,
    apply_rotation,
    apply_scaling,
    format_report,
    run_harness,
)


def _sample_payload(size: int) -> bytes:
    return bytes((i * 61 + 17) % 256 for i in range(size))


def _checkerboard(size: int = 64) -> Image.Image:
    """A simple non-uniform test image so degradations have something to act on.

    Uses a block size (6px) that is deliberately *not* a multiple of JPEG's
    8x8 DCT block size, so JPEG quantization actually has high-frequency
    content to discard at low quality (an 8px-aligned block pattern would
    compress losslessly at any quality, defeating the JPEG test below).
    """
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    block = 6
    for y in range(size):
        for x in range(size):
            if ((x // block) + (y // block)) % 2 == 0:
                arr[y, x] = (255, 255, 255)
            else:
                arr[y, x] = (0, 0, 0)
    return Image.fromarray(arr, mode="RGB")


def _midtone_checkerboard(size: int = 64) -> Image.Image:
    """A checkerboard using non-extreme gray values.

    Pure black (0) and pure white (255) pixels are fixed points of a
    multiplicative brightness scale followed by clipping (0 * anything is
    still 0; 255 * anything >= 1 clips back to 255), so brightness tests
    need mid-range values to observe a visible change in both directions.
    """
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    block = 6
    for y in range(size):
        for x in range(size):
            if ((x // block) + (y // block)) % 2 == 0:
                arr[y, x] = (200, 200, 200)
            else:
                arr[y, x] = (50, 50, 50)
    return Image.fromarray(arr, mode="RGB")


def _arrays_equal(a: Image.Image, b: Image.Image) -> bool:
    if a.size != b.size:
        return False
    return np.array_equal(np.asarray(a.convert("RGB")), np.asarray(b.convert("RGB")))


# --- apply_rotation ---------------------------------------------------------


def test_apply_rotation_zero_is_near_identity():
    image = _checkerboard()
    rotated = apply_rotation(image, 0.0)

    assert rotated.size == image.size
    # Zero-degree rotation should reproduce the source exactly (no interpolation drift).
    assert _arrays_equal(image, rotated)


def test_apply_rotation_nonzero_changes_content_or_size():
    image = _checkerboard()
    rotated = apply_rotation(image, 45.0)

    assert not _arrays_equal(image, rotated)


# --- apply_scaling -----------------------------------------------------------


def test_apply_scaling_identity_factor_is_near_identity():
    image = _checkerboard()
    scaled = apply_scaling(image, 1.0)

    assert scaled.size == image.size


def test_apply_scaling_changes_output_size():
    image = _checkerboard()
    scaled_up = apply_scaling(image, 2.0)
    scaled_down = apply_scaling(image, 0.5)

    assert scaled_up.size == (image.width * 2, image.height * 2)
    assert scaled_down.size == (image.width // 2, image.height // 2)


# --- apply_blur ----------------------------------------------------------


def test_apply_blur_zero_is_near_identity():
    image = _checkerboard()
    blurred = apply_blur(image, 0.0)

    assert blurred.size == image.size
    assert _arrays_equal(image, blurred)


def test_apply_blur_nonzero_changes_content():
    image = _checkerboard()
    blurred = apply_blur(image, 3.0)

    assert blurred.size == image.size
    assert not _arrays_equal(image, blurred)


# --- apply_jpeg_compression -------------------------------------------------


def test_apply_jpeg_quality_100_vs_10_are_measurably_different():
    image = _checkerboard()
    high_quality = apply_jpeg_compression(image, 100)
    low_quality = apply_jpeg_compression(image, 10)

    high_arr = np.asarray(high_quality.convert("RGB")).astype(np.int32)
    low_arr = np.asarray(low_quality.convert("RGB")).astype(np.int32)
    orig_arr = np.asarray(image.convert("RGB")).astype(np.int32)

    high_diff = np.abs(high_arr - orig_arr).mean()
    low_diff = np.abs(low_arr - orig_arr).mean()

    assert low_diff > high_diff


# --- apply_noise ---------------------------------------------------------


def test_apply_noise_zero_sigma_is_near_identity():
    image = _checkerboard()
    noisy = apply_noise(image, 0.0)

    assert noisy.size == image.size
    assert _arrays_equal(image, noisy)


def test_apply_noise_nonzero_sigma_changes_content():
    image = _checkerboard()
    noisy = apply_noise(image, 25.0)

    assert not _arrays_equal(image, noisy)


# --- apply_brightness ------------------------------------------------------


def test_apply_brightness_factor_one_is_near_identity():
    image = _midtone_checkerboard()
    same = apply_brightness(image, 1.0)

    assert _arrays_equal(image, same)


def test_apply_brightness_nonone_changes_content():
    image = _midtone_checkerboard()
    brighter = apply_brightness(image, 1.8)
    darker = apply_brightness(image, 0.4)

    assert not _arrays_equal(image, brighter)
    assert not _arrays_equal(image, darker)


# --- apply_perspective_warp -------------------------------------------------


def test_apply_perspective_warp_zero_strength_is_near_identity():
    image = _checkerboard()
    warped = apply_perspective_warp(image, 0.0)

    assert warped.size == image.size
    assert _arrays_equal(image, warped)


def test_apply_perspective_warp_nonzero_changes_content():
    image = _checkerboard()
    warped = apply_perspective_warp(image, 0.1)

    assert not _arrays_equal(image, warped)


# --- registry ----------------------------------------------------------------


def test_degradations_registry_covers_all_seven_types():
    expected_names = {
        "rotation",
        "scaling",
        "blur",
        "jpeg",
        "noise",
        "brightness",
        "perspective",
    }
    assert set(DEGRADATIONS) == expected_names
    for name, fn in DEGRADATIONS.items():
        assert callable(fn), name


# --- apply_combined ----------------------------------------------------------


def test_apply_combined_empty_list_is_near_identity():
    image = _checkerboard()
    result = apply_combined(image, [])

    assert _arrays_equal(image, result)


def test_apply_combined_applies_steps_in_order():
    image = _checkerboard()

    rotate_then_blur = apply_combined(image, [("rotation", 30.0), ("blur", 3.0)])
    blur_then_rotate = apply_combined(image, [("blur", 3.0), ("rotation", 30.0)])

    # Order matters: rotating first then blurring is not the same as blurring
    # first then rotating (different pixels get smeared / different edges
    # from padding get introduced at each stage).
    assert not _arrays_equal(rotate_then_blur, blur_then_rotate)


def test_apply_combined_multiple_steps_matches_manual_sequential_application():
    # Deterministic steps only (no noise/randomness) so this test can assert
    # exact equality against a manually chained call.
    image = _checkerboard()

    combined = apply_combined(image, [("jpeg", 50), ("brightness", 0.7)])
    manual = apply_brightness(apply_jpeg_compression(image, 50), 0.7)

    assert _arrays_equal(combined, manual)


# --- run_harness ---------------------------------------------------------


def test_run_harness_no_degradation_passes():
    payload = _sample_payload(16)
    radius = 18
    ecc_level = 30
    cell_size = 6.0

    cases = [("identity", [])]
    results = run_harness(payload, radius, ecc_level, cell_size, cases)

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, HarnessResult)
    assert result.degradation == "identity"
    assert result.passed is True
    assert result.error is None


def test_run_harness_real_degradation_reports_failure_honestly():
    """A degradation beyond the decoder's tolerance must be reported honestly
    (passed=False, error set), not papered over. Issue #14 gave the decoder
    geometric/photometric robustness for *mild* degradation, so a case that
    still fails must now be genuinely severe: a heavy Gaussian blur (5.0 px at
    cell_size 6.0) smears whole cells together, well past the ~1.5 px mild
    blur threshold, and does not decode."""
    payload = _sample_payload(16)
    radius = 18
    ecc_level = 30
    cell_size = 6.0

    cases = [("blur", [("blur", 5.0)])]
    results = run_harness(payload, radius, ecc_level, cell_size, cases)

    assert len(results) == 1
    result = results[0]
    assert result.degradation == "blur"
    assert result.passed is False
    assert result.error is not None


def test_run_harness_records_multiple_cases():
    payload = _sample_payload(16)
    radius = 18
    ecc_level = 30
    cell_size = 6.0

    cases = [
        ("identity", []),
        # Heavy blur (well beyond the ~1.5 px mild threshold at cell_size 6.0)
        # still fails, giving a mixed pass/fail set to record.
        ("blur", [("blur", 5.0)]),
    ]
    results = run_harness(payload, radius, ecc_level, cell_size, cases)

    assert [r.degradation for r in results] == ["identity", "blur"]
    assert results[0].passed is True
    assert results[1].passed is False


# --- format_report ---------------------------------------------------------


def test_format_report_contains_each_result_row():
    results = [
        HarnessResult(degradation="identity", severity="none", passed=True, error=None),
        HarnessResult(
            degradation="rotation", severity="15.0deg", passed=False, error="boom"
        ),
    ]

    report = format_report(results)

    assert "identity" in report
    assert "none" in report
    assert "rotation" in report
    assert "15.0deg" in report
    assert "boom" in report
    # Markdown table markers.
    assert "|" in report
    assert "---" in report


def test_format_report_empty_results_still_produces_header():
    report = format_report([])

    assert "degradation" in report.lower()
    assert "|" in report


if __name__ == "__main__":
    pytest.main([__file__])
