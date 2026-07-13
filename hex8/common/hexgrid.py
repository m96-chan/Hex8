"""Flat-top hexagonal grid geometry for the Hex8 marker.

This module implements the geometry of a hexagonal grid of radius ``R``
centered at the origin, as used by the Hex8 marker (PoC v0 targets
``R`` in the 18-20 range).

Design decisions
-----------------
- **Axial coordinates** ``(q, r)`` are used to address cells, with the
  implied cube coordinate ``s = -q - r`` (so ``q + r + s == 0`` always
  holds). A cell is considered inside the grid of radius ``R`` iff
  ``max(abs(q), abs(r), abs(s)) <= R``.
- **Flat-top** hexagon orientation is used for pixel conversion. Given a
  cell "size" (the distance from the hex center to a vertex):

    x = size * 1.5 * q
    y = size * sqrt(3) * (r + q / 2)

  ``pixel_to_axial`` inverts this to fractional axial coordinates and then
  rounds to the nearest valid hex using standard cube coordinate rounding:
  each of q, r, s is rounded independently, and the component with the
  largest rounding error is recomputed from the other two so that
  ``q + r + s == 0`` is preserved exactly. This rounding step is what makes
  the axial <-> pixel round trip exact even through floating point error.
- ``enumerate_cells`` returns cells in a deterministic order (sorted by
  ``(q, r)``) so that downstream modules (layout, encoder, decoder) can
  agree on a stable cell indexing without needing to share extra state.
"""

import math

__all__ = [
    "axial_to_pixel",
    "cell_count",
    "enumerate_cells",
    "pixel_to_axial",
]


def cell_count(radius: int) -> int:
    """Return the number of cells in a hex grid of the given radius.

    Uses the closed-form formula ``1 + 3 * radius * (radius + 1)``.
    """
    return 1 + 3 * radius * (radius + 1)


def enumerate_cells(radius: int) -> list[tuple[int, int]]:
    """Return all axial ``(q, r)`` coordinates within the grid of the given radius.

    Cells are returned in a deterministic order, sorted by ``(q, r)``.
    """
    cells = []
    for q in range(-radius, radius + 1):
        r_min = max(-radius, -q - radius)
        r_max = min(radius, -q + radius)
        for r in range(r_min, r_max + 1):
            cells.append((q, r))
    cells.sort()
    return cells


def axial_to_pixel(q: int, r: int, size: float) -> tuple[float, float]:
    """Convert flat-top axial coordinates ``(q, r)`` to pixel coordinates.

    ``size`` is the distance from the hex center to a vertex.
    """
    x = size * 1.5 * q
    y = size * math.sqrt(3) * (r + q / 2)
    return (x, y)


def pixel_to_axial(x: float, y: float, size: float) -> tuple[int, int]:
    """Convert pixel coordinates back to the nearest valid axial ``(q, r)``.

    Inverts the flat-top ``axial_to_pixel`` transform to fractional axial
    coordinates, then rounds using cube coordinate rounding to obtain the
    nearest valid integer hex. Does not check whether the result lies
    within any particular grid radius; that is the caller's concern.
    """
    q_frac = (2.0 / 3.0) * x / size
    r_frac = (-1.0 / 3.0) * x / size + (math.sqrt(3) / 3.0) * y / size
    s_frac = -q_frac - r_frac

    q_round = round(q_frac)
    r_round = round(r_frac)
    s_round = round(s_frac)

    q_diff = abs(q_round - q_frac)
    r_diff = abs(r_round - r_frac)
    s_diff = abs(s_round - s_frac)

    if q_diff > r_diff and q_diff > s_diff:
        q_round = -r_round - s_round
    elif r_diff > s_diff:
        r_round = -q_round - s_round
    # else: s_round is the largest error and is simply discarded, since
    # only (q, r) are returned.

    return (int(q_round), int(r_round))
