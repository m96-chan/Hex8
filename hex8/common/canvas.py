"""Grid-space <-> image-space pixel canvas geometry, shared by encoder and decoder.

:func:`hex8.common.hexgrid.axial_to_pixel` returns "grid space" pixel
coordinates centered at the origin (so most cells have negative
coordinates). :func:`compute_canvas` computes the bounding box of every cell
of a hex grid (including the full extent of each cell's hexagon, plus a
quiet-zone margin) and returns a :class:`CanvasInfo` with an
``(origin_x, origin_y)`` offset that shifts grid-space coordinates into
non-negative "image space" pixel coordinates suitable for drawing (the
encoder, :mod:`hex8.encoder.render`) or for pixel-sampling (the decoder,
:mod:`hex8.decoder.detect`). :func:`cell_center_px` applies that shift for a
single cell.

This module originally lived inside ``hex8.encoder.render`` (Issue #7), but
was moved to ``hex8.common`` while implementing Issue #9: the ideal decoder
needs this exact same geometry to map pixel coordinates back to cell
coordinates, and a decoder module importing from the encoder package would
be a backwards dependency. ``hex8.encoder.render`` re-exports these names
unchanged, so existing callers are unaffected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hex8.common.hexgrid import axial_to_pixel, enumerate_cells

Cell = tuple[int, int]

__all__ = ["CanvasInfo", "cell_center_px", "compute_canvas"]


@dataclass(frozen=True)
class CanvasInfo:
    """The pixel bounding box needed to draw or sample a full hex grid.

    Attributes:
        width: Canvas width in pixels (integer, rounded up).
        height: Canvas height in pixels (integer, rounded up).
        origin_x: Offset to add to a grid-space x coordinate (e.g. from
            :func:`hex8.common.hexgrid.axial_to_pixel`) to get a
            non-negative image-space x coordinate.
        origin_y: Offset to add to a grid-space y coordinate to get a
            non-negative image-space y coordinate.
    """

    width: int
    height: int
    origin_x: float
    origin_y: float


def compute_canvas(radius: int, cell_size: float, margin: float | None = None) -> CanvasInfo:
    """Compute the pixel bounding box needed to draw/sample a full hex grid.

    Considers every cell of ``enumerate_cells(radius)`` at the given
    ``cell_size`` (passed straight to
    :func:`hex8.common.hexgrid.axial_to_pixel`), including the full extent
    of each cell's hexagon (not just its center point), plus ``margin``
    extra pixels of quiet zone on every side.

    Args:
        radius: Hex grid radius, as accepted by ``enumerate_cells``.
        cell_size: Center-to-vertex distance of each hexagon, in pixels.
        margin: Extra quiet-zone pixels added on every side of the drawn
            grid. Defaults to ``2 * cell_size`` (documented default: two
            cell-widths of quiet zone, comfortably larger than a single
            hexagon so nothing touches the canvas edge).

    Returns:
        A :class:`CanvasInfo` describing the canvas size and the
        grid-space -> image-space pixel offset.
    """
    if margin is None:
        margin = 2.0 * cell_size

    # A flat-top hexagon with center-to-vertex distance `cell_size` extends
    # at most `cell_size` horizontally (vertices at angles 0 and 180) and at
    # most `cell_size * sqrt(3) / 2` vertically (vertices at angles 60, 120,
    # 240, 300) from its center.
    half_width = cell_size
    half_height = cell_size * math.sqrt(3) / 2

    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf
    for q, r in enumerate_cells(radius):
        x, y = axial_to_pixel(q, r, cell_size)
        min_x = min(min_x, x - half_width)
        max_x = max(max_x, x + half_width)
        min_y = min(min_y, y - half_height)
        max_y = max(max_y, y + half_height)

    origin_x = margin - min_x
    origin_y = margin - min_y

    width = math.ceil((max_x - min_x) + 2 * margin)
    height = math.ceil((max_y - min_y) + 2 * margin)

    return CanvasInfo(width=width, height=height, origin_x=origin_x, origin_y=origin_y)


def cell_center_px(
    q: int, r: int, cell_size: float, canvas: CanvasInfo
) -> tuple[float, float]:
    """Return the image-space pixel center of cell ``(q, r)``.

    Equivalent to ``axial_to_pixel(q, r, cell_size)`` shifted into image
    space via ``canvas.origin_x`` / ``canvas.origin_y``.
    """
    x, y = axial_to_pixel(q, r, cell_size)
    return (x + canvas.origin_x, y + canvas.origin_y)
