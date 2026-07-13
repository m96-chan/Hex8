"""Role-agnostic PNG/SVG rendering of a Hex8 marker cell-color assignment.

This module draws an already-resolved ``{(q, r): (r, g, b)}`` color
assignment for every cell of a hex grid (see :mod:`hex8.common.hexgrid`) as
a flat-top hexagonal marker image. It does not know anything about cell
*roles* (finder/palette/metadata/data, see :mod:`hex8.common.layout`) -
deciding which color goes where is the job of the encoder integration
(Issue #8), not this module.

Geometry
--------
Each cell is drawn as a filled flat-top hexagon: 6 vertices at angles
``60 * i`` degrees for ``i = 0..5`` (standard math convention, angle 0 along
the positive x-axis), each ``cell_size`` away from the cell's pixel center.
This matches the flat-top convention already used by
:func:`hex8.common.hexgrid.axial_to_pixel` (whose ``size`` parameter is also
the center-to-vertex distance): the two widest vertices of the hexagon (at
angles 0 and 180 degrees) point along the x-axis, and its flat sides are
horizontal (top/bottom).

Canvas coordinates
------------------
:func:`hex8.common.hexgrid.axial_to_pixel` returns "grid space" pixel
coordinates centered at the origin (so most cells have negative
coordinates). :func:`compute_canvas` computes the bounding box of every cell
of the grid (including the full extent of each cell's hexagon, plus a
quiet-zone margin) and returns a :class:`CanvasInfo` with an
``(origin_x, origin_y)`` offset that shifts grid-space coordinates into
non-negative "image space" pixel coordinates suitable for drawing.
:func:`cell_center_px` applies that shift for a single cell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image, ImageDraw

from hex8.common.hexgrid import axial_to_pixel, enumerate_cells

Cell = tuple[int, int]
RGB = tuple[int, int, int]

__all__ = [
    "CanvasInfo",
    "cell_center_px",
    "compute_canvas",
    "render_png",
    "render_svg",
]

#: Number of vertices of the flat-top hexagon used to draw each cell.
_HEX_VERTEX_COUNT = 6


@dataclass(frozen=True)
class CanvasInfo:
    """The pixel bounding box needed to draw a full hex grid.

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


def _hexagon_vertices(
    center_x: float, center_y: float, cell_size: float
) -> list[tuple[float, float]]:
    """Return the 6 vertices of a flat-top hexagon centered at (center_x, center_y).

    Vertex ``i`` (for ``i = 0..5``) is at angle ``60 * i`` degrees, at
    distance ``cell_size`` from the center.
    """
    vertices = []
    for i in range(_HEX_VERTEX_COUNT):
        angle = math.radians(60 * i)
        vertices.append(
            (
                center_x + cell_size * math.cos(angle),
                center_y + cell_size * math.sin(angle),
            )
        )
    return vertices


def compute_canvas(radius: int, cell_size: float, margin: float | None = None) -> CanvasInfo:
    """Compute the pixel bounding box needed to draw a full hex grid.

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


def _require_full_cell_colors(
    cells: list[Cell], cell_colors: dict[Cell, RGB]
) -> None:
    """Raise ValueError with a clear message if any cell lacks a color."""
    missing = [cell for cell in cells if cell not in cell_colors]
    if missing:
        raise ValueError(
            f"cell_colors is missing an entry for {len(missing)} cell(s) of "
            f"the grid, e.g. {missing[0]!r} (cell_colors must contain every "
            "cell returned by enumerate_cells(radius))."
        )


def render_png(
    radius: int,
    cell_colors: dict[Cell, RGB],
    cell_size: float = 10.0,
    background: RGB = (255, 255, 255),
) -> Image.Image:
    """Render a Hex8 marker as a Pillow RGB image.

    Draws every cell in ``enumerate_cells(radius)`` as a filled flat-top
    hexagon (see the module docstring for the exact vertex geometry),
    filled with ``cell_colors[(q, r)]``, on a ``background``-colored
    canvas sized via :func:`compute_canvas`.

    Args:
        radius: Hex grid radius.
        cell_colors: RGB color for every cell in ``enumerate_cells(radius)``.
        cell_size: Center-to-vertex distance of each hexagon, in pixels.
        background: RGB background color for the canvas.

    Returns:
        A Pillow ``Image`` in ``"RGB"`` mode.

    Raises:
        ValueError: if ``cell_colors`` is missing an entry for any cell of
            the grid.
    """
    cells = enumerate_cells(radius)
    _require_full_cell_colors(cells, cell_colors)

    canvas = compute_canvas(radius, cell_size)
    image = Image.new("RGB", (canvas.width, canvas.height), background)
    draw = ImageDraw.Draw(image)

    for q, r in cells:
        center_x, center_y = cell_center_px(q, r, cell_size, canvas)
        vertices = _hexagon_vertices(center_x, center_y, cell_size)
        draw.polygon(vertices, fill=cell_colors[(q, r)])

    return image


def _rgb_to_hex(color: RGB) -> str:
    """Format an (r, g, b) tuple as an uppercase '#RRGGBB' hex string."""
    r, g, b = color
    return f"#{r:02X}{g:02X}{b:02X}"


def render_svg(
    radius: int,
    cell_colors: dict[Cell, RGB],
    cell_size: float = 10.0,
    background: RGB = (255, 255, 255),
) -> str:
    """Render a Hex8 marker as a plain SVG XML string.

    Emits the same drawing as :func:`render_png` (no external SVG/Cairo
    library involved, by design): an ``<svg>`` root sized via
    :func:`compute_canvas`, a background ``<rect>``, and one
    ``<polygon points="x1,y1 x2,y2 ...">`` per cell with a
    ``fill="#RRGGBB"`` attribute (uppercase hex digits).

    Args:
        radius: Hex grid radius.
        cell_colors: RGB color for every cell in ``enumerate_cells(radius)``.
        cell_size: Center-to-vertex distance of each hexagon, in pixels.
        background: RGB background color for the canvas.

    Returns:
        A complete SVG document as a string.

    Raises:
        ValueError: if ``cell_colors`` is missing an entry for any cell of
            the grid.
    """
    cells = enumerate_cells(radius)
    _require_full_cell_colors(cells, cell_colors)

    canvas = compute_canvas(radius, cell_size)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{canvas.width}" height="{canvas.height}" '
        f'viewBox="0 0 {canvas.width} {canvas.height}">',
        f'<rect x="0" y="0" width="{canvas.width}" height="{canvas.height}" '
        f'fill="{_rgb_to_hex(background)}"/>',
    ]

    for q, r in cells:
        center_x, center_y = cell_center_px(q, r, cell_size, canvas)
        vertices = _hexagon_vertices(center_x, center_y, cell_size)
        points = " ".join(f"{x:.3f},{y:.3f}" for x, y in vertices)
        fill = _rgb_to_hex(cell_colors[(q, r)])
        parts.append(f'<polygon points="{points}" fill="{fill}"/>')

    parts.append("</svg>")
    return "\n".join(parts)
