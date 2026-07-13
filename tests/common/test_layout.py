"""Tests for hex8.common.layout (Issue #6)."""

import pytest

from hex8.common.hexgrid import enumerate_cells
from hex8.common.header import HEADER_SIZE
from hex8.common.symbols import PALETTE
from hex8.common.layout import (
    CellRole,
    METADATA_SYMBOL_COUNT,
    PALETTE_REPEATS,
    build_layout,
    corner_cells,
)


def test_metadata_symbol_count_matches_header_size():
    # ceil(HEADER_SIZE * 8 / 3), independently computed here to catch drift.
    assert METADATA_SYMBOL_COUNT == -(-(HEADER_SIZE * 8) // 3)


@pytest.mark.parametrize("radius", [5, 18, 20])
def test_corner_cells_are_six_distinct_valid_cells(radius):
    corners = corner_cells(radius)
    assert len(corners) == 6
    assert len(set(corners)) == 6
    valid = set(enumerate_cells(radius))
    for corner in corners:
        assert corner in valid


@pytest.mark.parametrize("radius", [18, 20])
def test_build_layout_covers_every_cell_exactly_once(radius):
    layout = build_layout(radius)
    all_cells = set(enumerate_cells(radius))
    assert set(layout.roles.keys()) == all_cells
    assert len(layout.roles) == len(all_cells)


@pytest.mark.parametrize("radius", [18, 20])
def test_build_layout_role_counts(radius):
    layout = build_layout(radius)
    assert len(layout.cells_with_role(CellRole.FINDER)) > 0
    assert len(layout.cells_with_role(CellRole.PALETTE)) == PALETTE_REPEATS * len(PALETTE)
    assert len(layout.cells_with_role(CellRole.METADATA)) == METADATA_SYMBOL_COUNT
    assert len(layout.cells_with_role(CellRole.DATA)) > 0


@pytest.mark.parametrize("radius", [18, 20])
def test_build_layout_is_deterministic(radius):
    first = build_layout(radius)
    second = build_layout(radius)
    assert first.roles == second.roles


def test_build_layout_too_small_radius_raises_value_error():
    with pytest.raises(ValueError):
        build_layout(1)


def test_build_layout_invalid_radius_raises_value_error():
    with pytest.raises(ValueError):
        build_layout(0)
    with pytest.raises(ValueError):
        build_layout(-1)


@pytest.mark.parametrize("radius", [18, 20])
def test_finder_cells_are_near_the_six_corners(radius):
    layout = build_layout(radius)
    finder_cells = set(layout.cells_with_role(CellRole.FINDER))
    for corner in corner_cells(radius):
        assert corner in finder_cells
