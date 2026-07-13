"""Tests for hex8.camera.capture: real camera capture ingestion + diagnosed
decoding (Issue #15).

Every test that can be exercised without physical camera/printer hardware
uses genuinely real images: markers produced by
:func:`hex8.encoder.encode.encode_png`, real pixel-level tampering (via
:mod:`hex8.common.layout` + the encoder's own render geometry, the same
technique used by ``tests/decoder/test_decode.py``), or the real Phase 3
degradation functions in :mod:`hex8.degrade`. No fabricated "photo-like"
pixel data is used anywhere the test claims to model a real capture -
:func:`capture_from_device`'s test below genuinely calls into `cv2` with no
camera hardware present, and honestly asserts the resulting failure, rather
than mocking a fake device.
"""

from __future__ import annotations

import pytest
from PIL import Image

from hex8 import degrade
from hex8.camera.capture import (
    CaptureResult,
    FailureCategory,
    capture_from_device,
    capture_from_file,
    decode_file_with_diagnostics,
    decode_with_diagnostics,
)
from hex8.common.layout import CellRole, build_layout
from hex8.common.symbols import color_to_symbol, symbol_to_color
from hex8.encoder.encode import encode_png
from hex8.encoder.render import cell_center_px, compute_canvas


def _sample_payload(size: int, *, seed: int = 17) -> bytes:
    return bytes((i * 61 + seed) % 256 for i in range(size))


# --- decode_with_diagnostics: success case --------------------------------


def test_decode_with_diagnostics_succeeds_on_a_valid_marker():
    payload = _sample_payload(64)
    image = encode_png(payload, radius=18, ecc_level=30, cell_size=6.0)

    result = decode_with_diagnostics(image)

    assert result == CaptureResult(
        payload=payload,
        success=True,
        failure_category=FailureCategory.NONE,
        error=None,
    )


# --- decode_with_diagnostics: no marker detected --------------------------


def test_decode_with_diagnostics_categorizes_a_non_marker_image():
    plain_image = Image.new("RGB", (400, 400), (200, 200, 200))

    result = decode_with_diagnostics(plain_image)

    assert result.success is False
    assert result.payload is None
    assert result.failure_category is FailureCategory.NO_MARKER_DETECTED
    assert result.error is not None


def test_decode_with_diagnostics_categorizes_heavy_degradation_beyond_mild_thresholds():
    """Perspective strength far beyond detect.py's documented mild ceiling
    (~0.05) is expected to defeat even the Issue #14 homography fallback -
    the finder anchors get warped past recognizability."""
    payload = _sample_payload(64)
    image = encode_png(payload, radius=18, ecc_level=30, cell_size=6.0)
    heavily_warped = degrade.apply_perspective_warp(image, 0.45)

    result = decode_with_diagnostics(heavily_warped)

    assert result.success is False
    assert result.failure_category is FailureCategory.NO_MARKER_DETECTED


# --- decode_with_diagnostics: corrupted header ----------------------------


def test_decode_with_diagnostics_categorizes_a_corrupted_header():
    """Flip every METADATA cell's real rendered pixel to a different exact
    palette color (shift its symbol by +1 mod 8), breaking the HX8M magic
    bytes the header carries, without touching FINDER/PALETTE cells (so
    marker detection itself still succeeds)."""
    payload = _sample_payload(64)
    radius = 18
    cell_size = 6.0
    image = encode_png(payload, radius=radius, ecc_level=30, cell_size=cell_size)

    layout = build_layout(radius)
    metadata_cells = layout.cells_with_role(CellRole.METADATA)
    canvas = compute_canvas(radius, cell_size)

    for q, r in metadata_cells:
        x, y = cell_center_px(q, r, cell_size, canvas)
        px, py = round(x), round(y)
        color = image.getpixel((px, py))[:3]
        symbol = color_to_symbol(color)
        image.putpixel((px, py), symbol_to_color((symbol + 1) % 8))

    result = decode_with_diagnostics(image)

    assert result.success is False
    assert result.payload is None
    assert result.failure_category is FailureCategory.HEADER_INVALID
    assert "HX8M" in result.error


# --- decode_with_diagnostics: uncorrectable Reed-Solomon corruption -------


def test_decode_with_diagnostics_categorizes_uncorrectable_rs_corruption():
    """Flip enough consecutive DATA cells to real, confidently-wrong exact
    palette colors (shift symbol by +1 mod 8 - not an ambiguous blend, so
    these classify with high confidence and are NOT treated as Reed-Solomon
    erasure hints) that the touched-byte count exceeds a single RS block's
    blind (unknown-position) correction capacity of ``nsym // 2`` bytes."""
    from hex8.common import ecc as ecc_mod

    payload = _sample_payload(32, seed=3)
    radius = 18
    cell_size = 6.0
    ecc_level = 40
    image = encode_png(payload, radius=radius, ecc_level=ecc_level, cell_size=cell_size)

    num_blocks, _block_data_len, nsym = ecc_mod._plan_blocks(len(payload), ecc_level / 100.0)
    assert num_blocks == 1, "test assumes a single RS block for a direct nsym calculation"

    layout = build_layout(radius)
    data_cells = layout.cells_with_role(CellRole.DATA)
    canvas = compute_canvas(radius, cell_size)

    # Skip the first ~35 data cells: those hold ecc.py's own internal framing
    # header (13 bytes -> ceil(13*8/3) = 35 symbols), which isn't itself
    # Reed-Solomon protected (see hex8.common.ecc's module docstring).
    skip = 40
    # 30 consecutive corrupted symbols span ~11.25 bytes, comfortably beyond
    # this block's nsym=21 -> nsym//2=10 blind-correction byte budget
    # (verified empirically against this exact payload/ecc_level/radius).
    corrupted_count = 30
    for q, r in data_cells[skip : skip + corrupted_count]:
        x, y = cell_center_px(q, r, cell_size, canvas)
        px, py = round(x), round(y)
        color = image.getpixel((px, py))[:3]
        symbol = color_to_symbol(color)
        image.putpixel((px, py), symbol_to_color((symbol + 1) % 8))

    result = decode_with_diagnostics(image)

    assert result.success is False
    assert result.payload is None
    assert result.failure_category is FailureCategory.RS_CORRECTION_FAILED
    assert "Reed-Solomon" in result.error


# --- capture_from_file -----------------------------------------------------


def test_capture_from_file_round_trips_through_decode_with_diagnostics(tmp_path):
    payload = _sample_payload(64)
    image = encode_png(payload, radius=18, ecc_level=30, cell_size=6.0)
    png_path = tmp_path / "marker.png"
    image.save(png_path, format="PNG")

    loaded = capture_from_file(png_path)
    result = decode_with_diagnostics(loaded)

    assert result.success is True
    assert result.payload == payload
    assert result.failure_category is FailureCategory.NONE


def test_capture_from_file_accepts_a_string_path(tmp_path):
    payload = _sample_payload(32)
    image = encode_png(payload, radius=18, ecc_level=30, cell_size=6.0)
    png_path = tmp_path / "marker.png"
    image.save(png_path, format="PNG")

    loaded = capture_from_file(str(png_path))

    assert loaded.mode == "RGB"
    assert decode_with_diagnostics(loaded).payload == payload


def test_decode_file_with_diagnostics_composes_capture_and_decode(tmp_path):
    payload = _sample_payload(64)
    image = encode_png(payload, radius=18, ecc_level=30, cell_size=6.0)
    png_path = tmp_path / "marker.png"
    image.save(png_path, format="PNG")

    result = decode_file_with_diagnostics(png_path)

    assert result == CaptureResult(
        payload=payload,
        success=True,
        failure_category=FailureCategory.NONE,
        error=None,
    )


# --- capture_from_device ---------------------------------------------------


def test_capture_from_device_raises_runtime_error_when_no_camera_present():
    """This is real `cv2.VideoCapture` code hitting an honest failure, not a
    mocked stand-in.

    Note: this sandbox's device index 0 is NOT a safe stand-in for "no
    camera" here - it unexpectedly resolves to a virtual v4l2loopback
    device (`/dev/video0`, reported by `v4l2-ctl --list-devices` as
    "AvataCam (platform:v4l2loopback-000)"), some sandbox-provided synthetic
    video source unrelated to real camera hardware, which `cv2.VideoCapture`
    successfully opens and reads frames from. That is not a real photograph
    of anything and must not be mistaken for one (see this module's
    docstring / the project's manual test guide for why real photos still
    require the human project owner's actual hardware). A deliberately
    out-of-range device index is used instead, which reliably has no backing
    device regardless of that sandbox artifact, to exercise the same honest
    "device cannot be opened" failure path that a genuinely camera-less
    environment (or an invalid index on real hardware) would hit."""
    with pytest.raises(RuntimeError):
        capture_from_device(device_index=999)


def test_capture_from_device_rejects_negative_warmup_frames():
    with pytest.raises(ValueError, match="warmup_frames"):
        capture_from_device(warmup_frames=-1)
