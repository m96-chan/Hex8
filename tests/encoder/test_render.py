"""Tests for hex8.encoder.render: PNG/SVG marker rendering (Issue #7)."""

from __future__ import annotations

import pytest

from hex8.common.hexgrid import axial_to_pixel, enumerate_cells
from hex8.common.layout import CellRole, build_layout
from hex8.common.symbols import PALETTE, symbol_to_color
from hex8.encoder.render import (
    cell_center_px,
    compute_canvas,
    render_png,
    render_svg,
)

# build_layout(5) - the radius suggested by Issue #7 as a "small test radius"
# - raises ValueError because a radius-5 grid is too small to fit the
# metadata region (see hex8.common.layout.build_layout). Radius 6 is the
# smallest radius for which build_layout succeeds, so it is used here
# instead.
TEST_RADIUS = 6
CELL_SIZE = 10.0

_FINDER_COLOR = (10, 20, 30)
_METADATA_COLOR = symbol_to_color(1)
_DATA_COLOR = symbol_to_color(4)


def _build_layout_and_colors():
    """Build a realistic full-grid color assignment for TEST_RADIUS.

    FINDER cells get a fixed test color, PALETTE cells cycle through the
    real 8-color palette, and METADATA/DATA cells each get their own fixed
    (but distinct) palette color, so every cell in enumerate_cells(TEST_RADIUS)
    has an assigned color and different roles are easy to tell apart in tests.
    """
    layout = build_layout(TEST_RADIUS)
    palette_colors = list(PALETTE.values())
    cell_colors: dict[tuple[int, int], tuple[int, int, int]] = {}
    palette_index = 0
    for cell, role in layout.roles.items():
        if role is CellRole.FINDER:
            cell_colors[cell] = _FINDER_COLOR
        elif role is CellRole.PALETTE:
            cell_colors[cell] = palette_colors[palette_index % len(palette_colors)]
            palette_index += 1
        elif role is CellRole.METADATA:
            cell_colors[cell] = _METADATA_COLOR
        else:
            cell_colors[cell] = _DATA_COLOR
    return layout, cell_colors


@pytest.fixture(scope="module")
def layout():
    return _build_layout_and_colors()[0]


@pytest.fixture(scope="module")
def cell_colors():
    return _build_layout_and_colors()[1]


def _sample_cells_per_role(layout, count=3):
    """Yield up to `count` (q, r) cells for every CellRole, for spot-checks."""
    for role in CellRole:
        cells = layout.cells_with_role(role)
        assert cells, f"expected at least one cell with role {role}"
        yield from cells[:count]


# --- compute_canvas / cell_center_px -----------------------------------


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
        # A flat-top hexagon's vertices are at most CELL_SIZE away from its
        # center (horizontally); this is a conservative (slightly generous)
        # bound that must still land strictly inside the canvas because of
        # the quiet-zone margin.
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


# --- render_png ----------------------------------------------------------


def test_render_png_is_rgb_image_matching_compute_canvas_size(cell_colors):
    canvas = compute_canvas(TEST_RADIUS, CELL_SIZE)
    img = render_png(TEST_RADIUS, cell_colors, cell_size=CELL_SIZE)
    assert img.mode == "RGB"
    assert img.size == (canvas.width, canvas.height)


def test_render_png_pixel_colors_match_assignment_for_each_role(layout, cell_colors):
    """Acceptance criterion: sampling the pixel at a cell's center must give
    exactly the assigned color, for several cells of every role."""
    canvas = compute_canvas(TEST_RADIUS, CELL_SIZE)
    img = render_png(TEST_RADIUS, cell_colors, cell_size=CELL_SIZE)
    for q, r in _sample_cells_per_role(layout):
        x, y = cell_center_px(q, r, CELL_SIZE, canvas)
        pixel = img.getpixel((round(x), round(y)))
        assert pixel == cell_colors[(q, r)]


def test_render_png_background_pixels_match_background_color(cell_colors):
    background = (240, 240, 245)
    canvas = compute_canvas(TEST_RADIUS, CELL_SIZE)
    img = render_png(
        TEST_RADIUS, cell_colors, cell_size=CELL_SIZE, background=background
    )
    assert img.getpixel((0, 0)) == background
    assert img.getpixel((canvas.width - 1, canvas.height - 1)) == background


def test_render_png_default_background_is_white(cell_colors):
    canvas = compute_canvas(TEST_RADIUS, CELL_SIZE)
    img = render_png(TEST_RADIUS, cell_colors, cell_size=CELL_SIZE)
    assert img.getpixel((0, 0)) == (255, 255, 255)
    assert img.getpixel((canvas.width - 1, canvas.height - 1)) == (255, 255, 255)


def test_render_png_missing_cell_color_raises_clear_error(cell_colors):
    incomplete = dict(cell_colors)
    del incomplete[next(iter(incomplete))]
    with pytest.raises((KeyError, ValueError)):
        render_png(TEST_RADIUS, incomplete, cell_size=CELL_SIZE)


# --- render_svg ------------------------------------------------------------


def test_render_svg_root_element_has_expected_dimensions(cell_colors):
    canvas = compute_canvas(TEST_RADIUS, CELL_SIZE)
    svg = render_svg(TEST_RADIUS, cell_colors, cell_size=CELL_SIZE)
    assert "<svg" in svg
    assert f'width="{canvas.width}"' in svg
    assert f'height="{canvas.height}"' in svg


def test_render_svg_has_one_polygon_per_cell(cell_colors):
    svg = render_svg(TEST_RADIUS, cell_colors, cell_size=CELL_SIZE)
    assert svg.count("<polygon") == len(enumerate_cells(TEST_RADIUS))


def test_render_svg_fill_colors_match_assignment_for_sample_cells(layout, cell_colors):
    svg = render_svg(TEST_RADIUS, cell_colors, cell_size=CELL_SIZE)
    for q, r in _sample_cells_per_role(layout, count=1):
        color = cell_colors[(q, r)]
        hex_color = "#{:02X}{:02X}{:02X}".format(*color)
        assert hex_color in svg


def test_render_svg_missing_cell_color_raises_clear_error(cell_colors):
    incomplete = dict(cell_colors)
    del incomplete[next(iter(incomplete))]
    with pytest.raises((KeyError, ValueError)):
        render_svg(TEST_RADIUS, incomplete, cell_size=CELL_SIZE)
