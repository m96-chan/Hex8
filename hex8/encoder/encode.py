"""Phase 1 encoder integration: payload bytes -> Hex8 marker image.

Wires together the header (:mod:`hex8.common.header`), hex grid geometry
(:mod:`hex8.common.hexgrid`), Reed-Solomon ECC (:mod:`hex8.common.ecc`),
symbol/color mapping (:mod:`hex8.common.symbols`), the cell layout
(:mod:`hex8.common.layout`), and the renderer (:mod:`hex8.encoder.render`)
into the single Phase 1 success condition from the README::

    payload bytes -> marker image

Cell color assignment (design decisions for this issue)
---------------------------------------------------------
The renderer (Issue #7) is role-agnostic; deciding which color goes in
which cell role is this module's job:

- **FINDER** cells are all rendered solid ``FINDER_COLOR`` (black), the same
  convention used by other visual fiducials (e.g. QR finder patterns).
  Since a cell's role is looked up positionally via
  ``hex8.common.layout.build_layout(radius)`` (not inferred from its color),
  reusing black - which also happens to be palette symbol 0 - creates no
  ambiguity for the encoder.
- **PALETTE** cells are assigned the 8 palette colors in symbol order,
  repeated ``PALETTE_REPEATS`` times, walking
  ``layout.cells_with_role(CellRole.PALETTE)`` in its natural (deterministic)
  order: the i-th palette cell gets ``symbols.PALETTE[i % 8]``.
- **METADATA** cells carry the packed HX8M header, converted to symbols via
  ``symbols.bits_to_symbol_stream``. The header is exactly
  ``layout.METADATA_SYMBOL_COUNT`` (54) symbols by construction (20 header
  bytes -> ceil(20*8/3) = 54), so it fills the metadata region exactly with
  no padding needed.
- **DATA** cells carry the Reed-Solomon-encoded payload, also converted to
  symbols. If there are more data cells than payload symbols (there almost
  always are, since capacity is sized for the largest supported payload),
  the leftover cells are filled with ``UNUSED_DATA_COLOR`` (white, the same
  as the renderer's background) rather than ``FINDER_COLOR``: an early test
  render with radius=20 and a small payload showed that reusing the finder
  color for unused cells produces one large contiguous black region (since
  unused cells cluster together at the tail of the deterministic cell
  order), which visually reads as a second, oversized "anchor" and would
  likely confuse position-based finder detection in Issue #9. Using the
  background color instead makes unused capacity fade into the quiet zone.
  The decoder never reads past ``header.encoded_length`` worth of symbols
  regardless, so the filler's exact value doesn't affect correctness either
  way - this choice is purely about avoiding a misleading rendered image.
"""

from __future__ import annotations

import zlib

from PIL import Image

from hex8.common import ecc as ecc_mod
from hex8.common import symbols
from hex8.common.header import Header
from hex8.common.header import pack as pack_header
from hex8.common.layout import CellRole, build_layout
from hex8.encoder.render import render_png, render_svg

Cell = tuple[int, int]
RGB = tuple[int, int, int]

__all__ = [
    "FINDER_COLOR",
    "MAX_ECC_LEVEL",
    "MIN_ECC_LEVEL",
    "UNUSED_DATA_COLOR",
    "encode_png",
    "encode_svg",
]

#: Solid fill color for FINDER anchor cells.
FINDER_COLOR: RGB = (0, 0, 0)

#: Fill color for DATA cells beyond the encoded payload's symbol count.
#: Matches the renderer's default background so unused capacity fades into
#: the quiet zone instead of forming a large, potentially-confusing block of
#: a single symbol color (see module docstring).
UNUSED_DATA_COLOR: RGB = (255, 255, 255)

#: ecc_level is stored in the HX8M header as a percentage (e.g. 30 means a
#: 30% Reed-Solomon ECC rate), matching hex8.common.ecc's supported range.
MIN_ECC_LEVEL = 25
MAX_ECC_LEVEL = 40


def _validate_ecc_level(ecc_level: int) -> None:
    if not (MIN_ECC_LEVEL <= ecc_level <= MAX_ECC_LEVEL):
        raise ValueError(
            f"ecc_level must be within [{MIN_ECC_LEVEL}, {MAX_ECC_LEVEL}], got {ecc_level!r}"
        )


def _build_cell_colors(
    payload: bytes, radius: int, ecc_level: int
) -> dict[Cell, RGB]:
    """Resolve the full per-cell color assignment for the given payload."""
    _validate_ecc_level(ecc_level)

    layout = build_layout(radius)

    encoded_payload = ecc_mod.encode(payload, ecc_rate=ecc_level / 100.0, interleave=True)
    payload_symbols = symbols.bits_to_symbol_stream(encoded_payload)

    data_cells = layout.cells_with_role(CellRole.DATA)
    if len(payload_symbols) > len(data_cells):
        raise ValueError(
            f"Encoded payload needs {len(payload_symbols)} data cells but radius "
            f"{radius} only provides {len(data_cells)}; use a larger radius, a "
            "smaller payload, or a lower ecc_level."
        )

    header = Header(
        version=1,
        flags=0,
        radius=radius,
        ecc_level=ecc_level,
        payload_length=len(payload),
        encoded_length=len(encoded_payload),
        crc32=zlib.crc32(payload),
    )
    header_symbols = symbols.bits_to_symbol_stream(pack_header(header))
    metadata_cells = layout.cells_with_role(CellRole.METADATA)
    assert len(header_symbols) == len(metadata_cells), (
        "packed header must produce exactly METADATA_SYMBOL_COUNT symbols "
        "(this is a fixed invariant of the HX8M header size, not runtime input)"
    )

    cell_colors: dict[Cell, RGB] = {}

    for cell in layout.cells_with_role(CellRole.FINDER):
        cell_colors[cell] = FINDER_COLOR

    palette_cells = layout.cells_with_role(CellRole.PALETTE)
    for i, cell in enumerate(palette_cells):
        cell_colors[cell] = symbols.PALETTE[i % len(symbols.PALETTE)]

    for cell, symbol in zip(metadata_cells, header_symbols, strict=True):
        cell_colors[cell] = symbols.symbol_to_color(symbol)

    for i, cell in enumerate(data_cells):
        if i < len(payload_symbols):
            cell_colors[cell] = symbols.symbol_to_color(payload_symbols[i])
        else:
            cell_colors[cell] = UNUSED_DATA_COLOR

    return cell_colors


def encode_png(
    payload: bytes,
    radius: int = 18,
    ecc_level: int = 30,
    cell_size: float = 10.0,
) -> Image.Image:
    """Encode `payload` bytes into a Hex8 marker PNG image.

    Args:
        payload: Arbitrary payload bytes to encode.
        radius: Hex grid radius (README target: 18-20).
        ecc_level: Reed-Solomon ECC rate as a percentage, 25-40.
        cell_size: Center-to-vertex pixel size of each hex cell.

    Returns:
        A Pillow RGB `Image` of the rendered marker.

    Raises:
        ValueError: if `ecc_level` is out of range, or the encoded payload
            does not fit in the data cells available at this `radius`.
    """
    cell_colors = _build_cell_colors(payload, radius, ecc_level)
    return render_png(radius, cell_colors, cell_size=cell_size)


def encode_svg(
    payload: bytes,
    radius: int = 18,
    ecc_level: int = 30,
    cell_size: float = 10.0,
) -> str:
    """Encode `payload` bytes into a Hex8 marker SVG document.

    Same semantics as :func:`encode_png`, emitted as an SVG string instead.
    """
    cell_colors = _build_cell_colors(payload, radius, ecc_level)
    return render_svg(radius, cell_colors, cell_size=cell_size)
