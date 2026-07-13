"""Tests for hex8.encoder.encode (Issue #8)."""

import zlib

import pytest
from PIL import Image

from hex8.common import header as header_mod
from hex8.common import symbols
from hex8.common.layout import CellRole, build_layout
from hex8.encoder.encode import (
    FINDER_COLOR,
    encode_png,
    encode_svg,
)
from hex8.encoder.render import compute_canvas


def _sample_payload(size: int) -> bytes:
    return bytes((i * 37 + 11) % 256 for i in range(size))


@pytest.mark.parametrize(
    ("payload_size", "radius"),
    [(16, 18), (128, 18), (256, 20)],
)
def test_encode_png_returns_correctly_sized_image(payload_size, radius):
    # A 256-byte payload at the default 30% ECC rate needs more data-cell
    # capacity than R=18 provides (338 raw bytes) once Reed-Solomon parity
    # and ecc.py's own small internal framing header are accounted for, so
    # it needs R=20 (426 raw bytes) instead - see docs/marker-layout.md's
    # capacity table.
    payload = _sample_payload(payload_size)
    image = encode_png(payload, radius=radius, ecc_level=30, cell_size=6.0)
    canvas = compute_canvas(radius, 6.0)
    assert isinstance(image, Image.Image)
    assert image.size == (canvas.width, canvas.height)
    assert image.mode == "RGB"


def test_finder_cells_are_rendered_as_finder_color():
    payload = _sample_payload(128)
    cell_size = 6.0
    image = encode_png(payload, radius=18, ecc_level=30, cell_size=cell_size)
    canvas = compute_canvas(18, cell_size)
    layout = build_layout(18)

    from hex8.encoder.render import cell_center_px

    for q, r in layout.cells_with_role(CellRole.FINDER)[:5]:
        x, y = cell_center_px(q, r, cell_size, canvas)
        assert image.getpixel((round(x), round(y))) == FINDER_COLOR


def test_palette_cells_cycle_through_the_8_colors_twice():
    payload = _sample_payload(128)
    cell_size = 6.0
    image = encode_png(payload, radius=18, ecc_level=30, cell_size=cell_size)
    canvas = compute_canvas(18, cell_size)
    layout = build_layout(18)

    from hex8.encoder.render import cell_center_px

    palette_cells = layout.cells_with_role(CellRole.PALETTE)
    assert len(palette_cells) == 16
    for i, (q, r) in enumerate(palette_cells):
        x, y = cell_center_px(q, r, cell_size, canvas)
        expected = symbols.PALETTE[i % 8]
        assert image.getpixel((round(x), round(y))) == expected


def test_metadata_cells_decode_back_to_the_correct_header():
    payload = _sample_payload(200)
    cell_size = 6.0
    radius = 18
    ecc_level = 25
    image = encode_png(payload, radius=radius, ecc_level=ecc_level, cell_size=cell_size)
    canvas = compute_canvas(radius, cell_size)
    layout = build_layout(radius)

    from hex8.encoder.render import cell_center_px

    metadata_cells = layout.cells_with_role(CellRole.METADATA)
    read_symbols = []
    for q, r in metadata_cells:
        x, y = cell_center_px(q, r, cell_size, canvas)
        pixel = image.getpixel((round(x), round(y)))
        read_symbols.append(symbols.color_to_symbol(pixel))

    header_bytes = symbols.symbol_stream_to_bits(read_symbols, header_mod.HEADER_SIZE)
    decoded_header = header_mod.unpack(header_bytes)

    assert decoded_header.version == header_mod.VERSION
    assert decoded_header.radius == radius
    assert decoded_header.ecc_level == ecc_level
    assert decoded_header.payload_length == len(payload)
    assert decoded_header.crc32 == zlib.crc32(payload)


def test_data_cells_carry_the_ecc_encoded_payload_symbols():
    payload = _sample_payload(64)
    cell_size = 6.0
    radius = 18
    ecc_level = 30
    image = encode_png(payload, radius=radius, ecc_level=ecc_level, cell_size=cell_size)
    canvas = compute_canvas(radius, cell_size)
    layout = build_layout(radius)

    from hex8.common import ecc as ecc_mod
    from hex8.encoder.render import cell_center_px

    expected_encoded = ecc_mod.encode(payload, ecc_rate=ecc_level / 100.0, interleave=True)
    expected_symbols = symbols.bits_to_symbol_stream(expected_encoded)

    data_cells = layout.cells_with_role(CellRole.DATA)
    assert len(expected_symbols) <= len(data_cells)

    for i, symbol in enumerate(expected_symbols):
        q, r = data_cells[i]
        x, y = cell_center_px(q, r, cell_size, canvas)
        pixel = image.getpixel((round(x), round(y)))
        assert symbols.color_to_symbol(pixel) == symbol


def test_encode_png_rejects_payload_too_large_for_radius():
    huge_payload = _sample_payload(5000)
    with pytest.raises(ValueError):
        encode_png(huge_payload, radius=18, ecc_level=30)


@pytest.mark.parametrize("bad_ecc_level", [0, 10, 24, 41, 100])
def test_encode_png_rejects_invalid_ecc_level(bad_ecc_level):
    with pytest.raises(ValueError):
        encode_png(_sample_payload(16), radius=18, ecc_level=bad_ecc_level)


def test_encode_svg_returns_valid_svg_string_matching_png_dimensions():
    payload = _sample_payload(64)
    cell_size = 6.0
    svg_text = encode_svg(payload, radius=18, ecc_level=30, cell_size=cell_size)
    canvas = compute_canvas(18, cell_size)
    assert svg_text.startswith("<svg")
    assert f'width="{canvas.width}"' in svg_text
    assert f'height="{canvas.height}"' in svg_text
    assert svg_text.count("<polygon") == len(build_layout(18).roles)
