"""Tests for hex8.decoder.detect (Issue #9): ideal marker detection + grid
normalization.

All test images are generated with the real encoder
(``hex8.encoder.encode.encode_png``) - no mocking of the encoder or of image
data, per the ideal-case scope of Issue #9 (no rotation/perspective/blur;
that's Phase 3/4, Issues #13/#14/#15).
"""

from __future__ import annotations

import pytest
from PIL import Image

from hex8.common.canvas import CanvasInfo, cell_center_px, compute_canvas
from hex8.common.layout import CellRole, build_layout
from hex8.common import symbols
from hex8.decoder.detect import (
    DARK_THRESHOLD,
    MAX_RADIUS,
    MIN_RADIUS,
    TOLERANCE_PX,
    DetectionResult,
    detect_marker,
    normalized_cell_center,
)
from hex8.encoder.encode import encode_png

# Radius 6 is the smallest radius that build_layout() accepts at all (radii
# 1-5 all raise ValueError: not enough cells to fit the finder anchors + 16
# palette cells + 54 metadata cells) - confirmed empirically. But radii 6-8
# don't have enough DATA cell capacity to hold even a 1-byte payload at the
# default 30% ECC rate (radius 6 has only 3 data cells; radius 8 has 93,
# still short of the 96 symbols a 16-byte payload needs) - encode_png raises
# ValueError for those. Radius 9 is the smallest radius whose data-cell
# capacity actually fits a 16-byte payload at the default ECC rate; radius
# 10 (one of the issue's suggested example radii) also works and is used
# here as the "small" test radius.
SMALL_RADIUS = 10


def _sample_payload(size: int = 16) -> bytes:
    return bytes((i * 37 + 11) % 256 for i in range(size))


@pytest.mark.parametrize("radius", [SMALL_RADIUS, 18, 20])
@pytest.mark.parametrize("cell_size", [6.0, 10.0, 13.5])
def test_detect_marker_recovers_original_radius_and_cell_size(radius, cell_size):
    payload = _sample_payload()
    image = encode_png(payload, radius=radius, cell_size=cell_size)

    result = detect_marker(image)

    assert isinstance(result, DetectionResult)
    assert result.radius == radius
    assert result.cell_size == pytest.approx(cell_size, abs=1e-6)


@pytest.mark.parametrize("radius", [SMALL_RADIUS, 18, 20])
@pytest.mark.parametrize("cell_size", [6.0, 10.0, 13.5])
def test_detect_marker_canvas_matches_compute_canvas(radius, cell_size):
    payload = _sample_payload()
    image = encode_png(payload, radius=radius, cell_size=cell_size)

    result = detect_marker(image)

    expected_canvas = compute_canvas(radius, result.cell_size)
    assert isinstance(result.canvas, CanvasInfo)
    assert result.canvas == expected_canvas
    assert (result.canvas.width, result.canvas.height) == image.size


@pytest.mark.parametrize("radius", [SMALL_RADIUS, 18])
def test_detect_marker_grid_aligns_with_real_palette_cells(radius):
    # Cross-check correctness (not just width/height matching by
    # coincidence): sample palette cells at the recovered grid coordinates
    # and confirm they match the encoder's known palette color assignment
    # (symbols.PALETTE[i % 8] for the i-th palette cell).
    payload = _sample_payload()
    cell_size = 8.0
    image = encode_png(payload, radius=radius, cell_size=cell_size)

    result = detect_marker(image)

    layout = build_layout(radius)
    palette_cells = layout.cells_with_role(CellRole.PALETTE)
    pixels = image.convert("RGB").load()

    for i, (q, r) in enumerate(palette_cells):
        px, py = normalized_cell_center(result, q, r)
        expected_color = symbols.PALETTE[i % len(symbols.PALETTE)]
        assert pixels[px, py] == expected_color


def test_detect_marker_finder_cells_are_dark_at_recovered_coordinates():
    payload = _sample_payload()
    radius = 18
    cell_size = 10.0
    image = encode_png(payload, radius=radius, cell_size=cell_size)

    result = detect_marker(image)

    layout = build_layout(radius)
    finder_cells = layout.cells_with_role(CellRole.FINDER)
    pixels = image.convert("RGB").load()

    assert len(finder_cells) > 0
    for q, r in finder_cells:
        px, py = normalized_cell_center(result, q, r)
        assert sum(pixels[px, py]) <= DARK_THRESHOLD


def test_normalized_cell_center_matches_rounded_cell_center_px():
    payload = _sample_payload()
    radius = SMALL_RADIUS
    cell_size = 10.0
    image = encode_png(payload, radius=radius, cell_size=cell_size)

    result = detect_marker(image)

    for q, r in build_layout(radius).cells_with_role(CellRole.FINDER):
        expected_x, expected_y = cell_center_px(q, r, result.cell_size, result.canvas)
        assert normalized_cell_center(result, q, r) == (round(expected_x), round(expected_y))


def test_detect_marker_raises_for_solid_color_image():
    image = Image.new("RGB", (400, 300), color=(200, 200, 200))
    with pytest.raises(ValueError):
        detect_marker(image)


def test_detect_marker_raises_for_random_noise_image():
    width, height = 500, 400
    rng_bytes = bytes((i * 97 + 13) % 256 for i in range(width * height * 3))
    image = Image.frombytes("RGB", (width, height), rng_bytes)
    with pytest.raises(ValueError):
        detect_marker(image)


def test_module_constants_are_sane():
    assert MIN_RADIUS >= 1
    assert MAX_RADIUS > MIN_RADIUS
    assert TOLERANCE_PX > 0
    assert DARK_THRESHOLD >= 0
