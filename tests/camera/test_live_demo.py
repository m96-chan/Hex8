"""Tests for hex8.camera.live_demo: continuous frame decode + overlay logic
(Issue #18).

Following the same convention as ``tests/camera/test_capture.py``: no
mocking of cv2/PIL - every test uses genuinely real images produced by
:func:`hex8.encoder.encode.encode_png`, and the same real pixel-tampering
technique used there to hit specific failure categories.

``run_live_demo`` itself (the ``while True: cv2.imshow(...)`` event loop) is
deliberately **not** tested here - it requires a real camera device and a
GUI-capable (non-headless) OpenCV build, neither of which this automated
test suite can rely on. It is verified manually per
``docs/phase4-manual-test-guide.md``. Everything tested here is the pure
frame-in/overlay-out decision logic and drawing calls (``cv2.polylines`` /
``cv2.putText`` are drawing primitives that work with a headless OpenCV
build too - they don't need a display), which fully determines what
``run_live_demo`` shows on screen.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from hex8.camera.live_demo import (
    FrameOverlay,
    _format_payload,
    marker_outline_points,
    process_frame,
    render_overlay,
)
from hex8.common.layout import CellRole, build_layout, corner_cells
from hex8.common.symbols import color_to_symbol, symbol_to_color
from hex8.decoder.detect import detect_marker
from hex8.encoder.encode import encode_png
from hex8.encoder.render import cell_center_px, compute_canvas


def _sample_payload(size: int, *, seed: int = 17) -> bytes:
    return bytes((i * 61 + seed) % 256 for i in range(size))


# --- process_frame: success case -------------------------------------------


def test_process_frame_succeeds_on_a_valid_marker():
    payload = b"Hello, Hex8!"
    image = encode_png(payload, radius=18, ecc_level=30, cell_size=6.0)

    overlay = process_frame(image)

    assert overlay.success is True
    assert overlay.outline_points is not None
    assert len(overlay.outline_points) == 6
    assert "12 bytes" in overlay.status_text
    assert "Hello, Hex8!" in overlay.status_text


# --- process_frame: no marker detected -------------------------------------


def test_process_frame_reports_no_marker_on_blank_image():
    plain_image = Image.new("RGB", (400, 400), (200, 200, 200))

    overlay = process_frame(plain_image)

    assert overlay.success is False
    assert overlay.outline_points is None
    assert "NO_MARKER_DETECTED" in overlay.status_text


# --- process_frame: corrupted header (marker found, decode fails) ---------


def test_process_frame_reports_header_invalid_with_outline_present():
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

    overlay = process_frame(image)

    assert overlay.success is False
    assert overlay.outline_points is not None
    assert len(overlay.outline_points) == 6
    assert "HEADER_INVALID" in overlay.status_text


# --- process_frame: uncorrectable Reed-Solomon corruption ------------------


def test_process_frame_reports_rs_correction_failed_with_outline_present():
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

    skip = 40
    corrupted_count = 30
    for q, r in data_cells[skip : skip + corrupted_count]:
        x, y = cell_center_px(q, r, cell_size, canvas)
        px, py = round(x), round(y)
        color = image.getpixel((px, py))[:3]
        symbol = color_to_symbol(color)
        image.putpixel((px, py), symbol_to_color((symbol + 1) % 8))

    overlay = process_frame(image)

    assert overlay.success is False
    assert overlay.outline_points is not None
    assert "RS_CORRECTION_FAILED" in overlay.status_text


# --- marker_outline_points --------------------------------------------------


def test_marker_outline_points_matches_the_six_grid_corners():
    payload = _sample_payload(16)
    radius = 18
    cell_size = 6.0
    image = encode_png(payload, radius=radius, ecc_level=30, cell_size=cell_size)

    detection = detect_marker(image)
    points = marker_outline_points(detection)

    expected = [
        (round(x), round(y))
        for x, y in (detection.cell_center(q, r) for q, r in corner_cells(radius))
    ]

    assert points == expected


# --- render_overlay ----------------------------------------------------------


def test_render_overlay_preserves_array_shape_and_dtype():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    overlay = FrameOverlay(outline_points=None, status_text="NO_MARKER_DETECTED", success=False)

    result = render_overlay(frame, overlay)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype


def test_render_overlay_draws_something_when_outline_and_text_present():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    points = [(10, 10), (190, 10), (190, 190), (150, 100), (10, 190), (50, 100)]
    overlay = FrameOverlay(
        outline_points=points, status_text="DECODED (4 bytes): test", success=True
    )

    result = render_overlay(frame, overlay)

    assert result.shape == frame.shape
    assert not np.array_equal(result, frame)


def test_render_overlay_status_text_is_visible_against_a_light_background():
    """A plain white status_text is invisible against a light/white frame
    (e.g. a marker's own white background, or a light-colored real-world
    scene) unless something darker is drawn behind it first - regression
    test for exactly that: on an all-white frame, some pixel in the text
    region must differ from white."""
    frame = np.full((200, 400, 3), 255, dtype=np.uint8)
    overlay = FrameOverlay(outline_points=None, status_text="DECODED (4 bytes): test", success=True)

    result = render_overlay(frame, overlay)

    text_region = result[10:40, 5:395]
    assert (text_region != 255).any()


# --- _format_payload ---------------------------------------------------------


def test_format_payload_decodes_valid_utf8():
    assert _format_payload("hello hex8".encode("utf-8")) == "hello hex8"


def test_format_payload_falls_back_to_hex_for_non_utf8_bytes():
    non_utf8 = bytes([0xFF, 0xFE, 0x00, 0x80])

    assert _format_payload(non_utf8) == non_utf8.hex()
