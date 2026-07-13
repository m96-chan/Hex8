"""Finder anchor, palette calibration, and metadata cell layout for the Hex8 marker.

Given a hex grid radius ``R`` (see :mod:`hex8.common.hexgrid`), this module
partitions every cell into exactly one of four roles:

- ``FINDER``: position/scale/orientation/perspective anchors.
- ``PALETTE``: color calibration reference cells.
- ``METADATA``: cells reserved for the encoded HX8M header.
- ``DATA``: everything else - the payload/ECC symbol stream.

Design decisions
-----------------
- **Finder anchors: 6 outer vertex anchors** (chosen over the README's other
  option, 3 major anchor clusters, for better perspective/rotation
  estimation under Phase 3/4 degradation). The 6 corner cells of a
  hex-shaped grid of radius ``R`` are the axial coordinates
  ``(R, 0)``, ``(R, -R)``, ``(0, -R)``, ``(-R, 0)``, ``(-R, R)``, ``(0, R)``
  (the six permutations of the cube vector ``(R, -R, 0)``). Each corner
  anchor is a solid cluster of every cell within hex-distance
  ``ANCHOR_RADIUS`` of that corner, clipped to the grid boundary.
- **Palette cells**: ``PALETTE_REPEATS`` (2) repetitions of the 8-color
  palette (16 cells total) provide redundant calibration references. They
  are assigned to the first 16 cells *not* already claimed by a finder
  anchor, walking cells in :func:`hex8.common.hexgrid.enumerate_cells`'s
  deterministic ``(q, r)`` order.
- **Metadata cells**: enough cells to hold the HX8M header
  (:data:`hex8.common.header.HEADER_SIZE` bytes) as 3-bit symbols, i.e.
  ``ceil(HEADER_SIZE * 8 / 3)`` cells, assigned to the next unclaimed cells
  in the same deterministic order after the palette.
- **Data cells**: every remaining cell.

This "walk the deterministic cell order and claim what's needed" approach
(rather than hardcoded row/column ranges) keeps the layout well-defined for
any radius large enough to fit the reserved regions, and guarantees the
encoder and decoder agree on cell roles as long as both call
``build_layout(radius)`` with the same radius: both sides derive the
partition from the same deterministic inputs, with no extra state to share.

A human-readable diagram lives in ``docs/marker-layout.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from hex8.common.header import HEADER_SIZE
from hex8.common.hexgrid import enumerate_cells
from hex8.common.symbols import PALETTE

Cell = tuple[int, int]

#: Hex-distance (inclusive) from each corner that is claimed by that corner's
#: finder anchor cluster.
ANCHOR_RADIUS = 2

#: Number of times the full 8-color palette is repeated among the palette
#: reference cells (for redundancy).
PALETTE_REPEATS = 2

#: Number of cells reserved for the HX8M header, i.e. enough 3-bit symbols to
#: carry HEADER_SIZE bytes.
METADATA_SYMBOL_COUNT = -(-(HEADER_SIZE * 8) // 3)  # ceil(HEADER_SIZE * 8 / 3)


class CellRole(Enum):
    """The role a single hex cell plays in the marker layout."""

    FINDER = "finder"
    PALETTE = "palette"
    METADATA = "metadata"
    DATA = "data"


@dataclass(frozen=True)
class MarkerLayout:
    """A fully-resolved cell-role assignment for a given grid radius."""

    radius: int
    roles: dict[Cell, CellRole]

    def cells_with_role(self, role: CellRole) -> list[Cell]:
        """Return all cells assigned the given role, in no particular order."""
        return [cell for cell, cell_role in self.roles.items() if cell_role is role]


def _hex_distance(a: Cell, b: Cell) -> int:
    """Distance between two axial hex coordinates, in cell steps."""
    dq = a[0] - b[0]
    dr = a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def corner_cells(radius: int) -> list[Cell]:
    """Return the 6 axial corner coordinates of a hex grid of the given radius."""
    return [
        (radius, 0),
        (radius, -radius),
        (0, -radius),
        (-radius, 0),
        (-radius, radius),
        (0, radius),
    ]


def build_layout(radius: int) -> MarkerLayout:
    """Compute the deterministic finder/palette/metadata/data cell layout.

    Raises:
        ValueError: if ``radius`` is not a positive integer, or if the grid
            is too small to fit the finder anchors, palette cells, and
            metadata cells (with data cells left over).
    """
    if radius < 1:
        raise ValueError(f"radius must be a positive integer, got {radius}")

    cells = enumerate_cells(radius)
    roles: dict[Cell, CellRole] = {}

    # 1. Finder anchors: a solid cluster of cells around each of the 6 corners.
    for corner in corner_cells(radius):
        for cell in cells:
            if cell not in roles and _hex_distance(cell, corner) <= ANCHOR_RADIUS:
                roles[cell] = CellRole.FINDER

    remaining = [cell for cell in cells if cell not in roles]

    # 2. Palette cells: PALETTE_REPEATS full copies of the palette.
    palette_count = PALETTE_REPEATS * len(PALETTE)
    if len(remaining) < palette_count:
        raise ValueError(
            f"radius {radius} is too small to fit {palette_count} palette cells "
            f"({len(remaining)} cells available after finder anchors)"
        )
    for cell in remaining[:palette_count]:
        roles[cell] = CellRole.PALETTE
    remaining = remaining[palette_count:]

    # 3. Metadata cells: enough to hold the HX8M header.
    if len(remaining) < METADATA_SYMBOL_COUNT:
        raise ValueError(
            f"radius {radius} is too small to fit the {METADATA_SYMBOL_COUNT}-cell "
            f"metadata region ({len(remaining)} cells available after finder+palette)"
        )
    for cell in remaining[:METADATA_SYMBOL_COUNT]:
        roles[cell] = CellRole.METADATA
    remaining = remaining[METADATA_SYMBOL_COUNT:]

    # 4. Everything else is data.
    for cell in remaining:
        roles[cell] = CellRole.DATA

    return MarkerLayout(radius=radius, roles=roles)
