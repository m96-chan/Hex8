"""Tests for the Issue #14 robust detection fallback in hex8.decoder.detect.

These exercise the homography fallback path: images are produced by the real
encoder (:func:`hex8.encoder.encode.encode_png`) and then degraded by the
real Phase 3 degradation library (:mod:`hex8.degrade`) at the "mild"
severities Issue #14 commits to. The fast (ideal) path cannot handle any of
these (the Issue #13 baseline recorded them all failing), so success here is
entirely due to the new fallback.

All degradations are applied at real severities - no mocking of image data.
"""

from __future__ import annotations

import numpy as np
import pytest

from hex8 import degrade
from hex8.common.layout import CellRole, build_layout
from hex8.common.symbols import PALETTE
from hex8.decoder.classify import build_observed_palette, classify_pixel
from hex8.decoder.decode import decode_image
from hex8.decoder.detect import DetectionResult, detect_marker
from hex8.encoder.encode import encode_png

RADIUS = 18
CELL_SIZE = 6.0
ECC_LEVEL = 30


def _payload(size: int = 128) -> bytes:
    return bytes((i * 37 + 11) % 256 for i in range(size))


@pytest.fixture(scope="module")
def base_image():
    return encode_png(_payload(), radius=RADIUS, ecc_level=ECC_LEVEL, cell_size=CELL_SIZE)


# The "mild degradation" severities Issue #14 commits to (see detect.py's
# module docstring). Each must both DETECT and fully DECODE.
MILD_CASES = [
    ("rotation_15deg", lambda im: degrade.apply_rotation(im, 15.0)),
    ("rotation_45deg", lambda im: degrade.apply_rotation(im, 45.0)),
    ("scaling_0.5x", lambda im: degrade.apply_scaling(im, 0.5)),
    ("scaling_2.0x", lambda im: degrade.apply_scaling(im, 2.0)),
    ("blur_1.5px", lambda im: degrade.apply_blur(im, 1.5)),
    ("jpeg_q25", lambda im: degrade.apply_jpeg_compression(im, 25)),
    ("brightness_0.5", lambda im: degrade.apply_brightness(im, 0.5)),
    ("brightness_1.5", lambda im: degrade.apply_brightness(im, 1.5)),
    ("perspective_0.05", lambda im: degrade.apply_perspective_warp(im, 0.05)),
]


@pytest.mark.parametrize(("name", "apply"), MILD_CASES, ids=[c[0] for c in MILD_CASES])
def test_detect_marker_locates_palette_cells_under_mild_degradation(base_image, name, apply):
    """The fallback recovers a transform that correctly places PALETTE cells.

    Cross-checks correctness (not merely that detection returned *something*):
    the recovered cell centers must land on the encoder's known palette-color
    assignment (symbol ``i % 8`` for the i-th palette cell). Classification
    uses the decoder's own adaptive *observed* palette, so this validates the
    recovered *geometry* independently of any colour cast (e.g. under
    brightness changes).
    """
    degraded = apply(base_image)
    result = detect_marker(degraded)

    assert isinstance(result, DetectionResult)
    assert result.radius == RADIUS
    # Note: some "mild" degradations (scaling, brightness >= 1.0) preserve the
    # exact palette colours and are still handled by the ideal fast path, so
    # we do NOT require a homography here - only that whichever path won
    # locates cells correctly. A dedicated fallback-path assertion lives in
    # test_rotation_is_handled_by_homography_fallback below.

    observed = build_observed_palette(degraded, RADIUS, result.cell_center)
    rgb = np.asarray(degraded.convert("RGB"))
    h, w = rgb.shape[:2]
    palette_cells = build_layout(RADIUS).cells_with_role(CellRole.PALETTE)

    correct = 0
    for i, (q, r) in enumerate(palette_cells):
        x, y = result.cell_center(q, r)
        px, py = int(round(x)), int(round(y))
        assert 0 <= px < w and 0 <= py < h
        color = (int(rgb[py, px, 0]), int(rgb[py, px, 1]), int(rgb[py, px, 2]))
        if classify_pixel(color, observed).symbol == i % len(PALETTE):
            correct += 1
    # Essentially all palette cells must be located correctly.
    assert correct >= len(palette_cells) - 1


@pytest.mark.parametrize(("name", "apply"), MILD_CASES, ids=[c[0] for c in MILD_CASES])
def test_decode_image_round_trips_under_mild_degradation(base_image, name, apply):
    degraded = apply(base_image)
    assert decode_image(degraded) == _payload()


def test_rotation_is_handled_by_homography_fallback(base_image):
    """A rotated image cannot use the fast path; it must resolve via homography."""
    result = detect_marker(degrade.apply_rotation(base_image, 15.0))
    assert result.homography is not None
    assert result.cell_size is None and result.canvas is None


def test_decode_image_round_trips_under_mild_noise(base_image, monkeypatch):
    """Noise is stochastic; seed the generator so the test is deterministic."""
    real_default_rng = np.random.default_rng
    monkeypatch.setattr(
        np.random, "default_rng", lambda *a, **k: real_default_rng(20240714)
    )
    degraded = degrade.apply_noise(base_image, 20.0)
    assert decode_image(degraded) == _payload()


def test_detection_result_cell_center_uses_homography_when_present():
    """cell_center must go through the homography for a fallback result."""
    from hex8.common.hexgrid import axial_to_pixel

    # With an identity homography, cell_center == the template (cell_size=1.0)
    # axial-pixel position.
    result = DetectionResult(radius=RADIUS, homography=np.eye(3, dtype=np.float64))
    for q, r in [(0, 0), (3, -2), (-5, 4)]:
        expected = axial_to_pixel(q, r, 1.0)
        got = result.cell_center(q, r)
        assert got[0] == pytest.approx(expected[0])
        assert got[1] == pytest.approx(expected[1])

    # And a non-identity (2x scale) homography actually transforms it.
    r2 = DetectionResult(radius=RADIUS, homography=np.diag([2.0, 2.0, 1.0]).astype(np.float64))
    sx, sy = r2.cell_center(3, -2)
    ex, ey = axial_to_pixel(3, -2, 1.0)
    assert sx == pytest.approx(2 * ex)
    assert sy == pytest.approx(2 * ey)


def test_detect_marker_still_raises_for_non_marker_image():
    """The fallback must not hallucinate a marker in structured non-marker data."""
    from PIL import Image

    # A checkerboard of black squares gives the distance transform plenty of
    # dark blobs, but no valid finder/palette geometry -> must still raise.
    arr = np.zeros((400, 400, 3), dtype=np.uint8)
    arr[::40] = 255
    arr[:, ::40] = 255
    image = Image.fromarray(arr, mode="RGB")
    with pytest.raises(ValueError):
        detect_marker(image)
