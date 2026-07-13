"""Tests for hex8.common.canvas (moved here from hex8.encoder.render during Issue #9,
so the decoder can use the same grid-space<->image-space geometry without
depending on the encoder package). hex8.encoder.render re-exports these
names unchanged; see tests/encoder/test_render.py for the render-focused
coverage of compute_canvas/cell_center_px via that re-export.
"""

from __future__ import annotations

import pytest

from hex8.common.canvas import cell_center_px, compute_canvas
from hex8.common.hexgrid import axial_to_pixel, enumerate_cells

TEST_RADIUS = 6
CELL_SIZE = 10.0


def test_compute_canvas_returns_positive_integer_dimensions():
    canvas = compute_canvas(TEST_RADIUS, CELL_SIZE)
    assert isinstance(canvas.width, int)
    assert isinstance(canvas.height, int)
    assert canvas.width > 0
    assert canvas.height > 0


def test_compute_canvas_covers_every_cell_hexagon_with_margin():
    canvas = compute_canvas(TEST_RADIUS, CELL_SIZE)
    for q, r in enumerate_cells(TEST_RADIUS):
        cx, cy = cell_center_px(q, r, CELL_SIZE, canvas)
        assert cx - CELL_SIZE >= 0
        assert cx + CELL_SIZE <= canvas.width
        assert cy - CELL_SIZE >= 0
        assert cy + CELL_SIZE <= canvas.height


def test_cell_center_px_matches_axial_to_pixel_shifted_by_origin():
    canvas = compute_canvas(TEST_RADIUS, CELL_SIZE)
    x, y = axial_to_pixel(2, -1, CELL_SIZE)
    cx, cy = cell_center_px(2, -1, CELL_SIZE, canvas)
    assert cx == pytest.approx(x + canvas.origin_x)
    assert cy == pytest.approx(y + canvas.origin_y)


def test_compute_canvas_scales_roughly_linearly_with_cell_size():
    # Not exact: math.ceil() rounding on width/height (and on the
    # cell-size-dependent margin) can compound by a few pixels, especially
    # at small radii/cell sizes - this just checks the scaling is in the
    # right ballpark, not pixel-exact.
    canvas_1 = compute_canvas(TEST_RADIUS, 1.0)
    canvas_10 = compute_canvas(TEST_RADIUS, 10.0)
    assert canvas_10.width == pytest.approx(canvas_1.width * 10, abs=10)
    assert canvas_10.height == pytest.approx(canvas_1.height * 10, abs=10)
