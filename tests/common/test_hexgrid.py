"""Tests for hex8.common.hexgrid: flat-top hexagonal grid geometry."""

import pytest

from hex8.common.hexgrid import (
    axial_to_pixel,
    cell_count,
    enumerate_cells,
    pixel_to_axial,
)


@pytest.mark.parametrize("radius", [1, 5, 10, 18, 20])
def test_cell_count_formula(radius):
    assert cell_count(radius) == 1 + 3 * radius * (radius + 1)


@pytest.mark.parametrize("radius", [1, 5, 10, 18])
def test_enumerate_cells_matches_cell_count(radius):
    cells = enumerate_cells(radius)
    assert len(cells) == cell_count(radius)


@pytest.mark.parametrize("radius", [1, 5, 10])
def test_enumerate_cells_within_radius(radius):
    cells = enumerate_cells(radius)
    for q, r in cells:
        s = -q - r
        assert max(abs(q), abs(r), abs(s)) <= radius


def test_enumerate_cells_no_duplicates():
    cells = enumerate_cells(5)
    assert len(cells) == len(set(cells))


def test_enumerate_cells_deterministic_order():
    """enumerate_cells must return a stable, sorted-by-(q, r) order so
    downstream modules (layout, encoder, decoder) agree on cell indexing."""
    cells = enumerate_cells(5)
    assert cells == sorted(cells)


@pytest.mark.parametrize("size", [10.0, 37.5])
def test_axial_pixel_round_trip(size):
    radius = 5
    for q, r in enumerate_cells(radius):
        x, y = axial_to_pixel(q, r, size)
        q2, r2 = pixel_to_axial(x, y, size)
        assert (q2, r2) == (q, r)


def test_axial_to_pixel_origin():
    assert axial_to_pixel(0, 0, 10.0) == (0.0, 0.0)


def test_axial_to_pixel_known_values():
    # Flat-top layout: x = size * 1.5 * q; y = size * sqrt(3) * (r + q / 2)
    import math

    size = 10.0
    x, y = axial_to_pixel(1, 0, size)
    assert x == pytest.approx(15.0)
    assert y == pytest.approx(10.0 * math.sqrt(3) * 0.5)


def test_pixel_to_axial_rounds_to_nearest_cell():
    # A point that is exactly on a cell center should round-trip exactly,
    # even with a tiny floating point perturbation.
    size = 20.0
    x, y = axial_to_pixel(3, -2, size)
    q, r = pixel_to_axial(x + 1e-9, y - 1e-9, size)
    assert (q, r) == (3, -2)
