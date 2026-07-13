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
Grid-space -> image-space pixel canvas geometry (``CanvasInfo``,
``compute_canvas``, ``cell_center_px``) lives in :mod:`hex8.common.canvas`
and is re-exported here unchanged, since the decoder (Issue #9) needs the
exact same geometry to map pixel coordinates back to cell coordinates and
must not depend on the encoder package.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw

from hex8.common.canvas import CanvasInfo, cell_center_px, compute_canvas
from hex8.common.hexgrid import enumerate_cells

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
