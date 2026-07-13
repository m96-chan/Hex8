"""Tests for hex8.decoder.decode: Phase 2 decoder integration + round-trip (Issue #12)."""

from __future__ import annotations

import pytest
from PIL import Image

from hex8.decoder.decode import decode_file, decode_image


def _sample_payload(size: int) -> bytes:
    return bytes((i * 61 + 17) % 256 for i in range(size))


# --- README's Phase 2 success condition: payload bytes -> marker image -> same payload bytes ---


@pytest.mark.parametrize(
    ("payload_size", "radius"),
    [(1, 18), (128, 18), (256, 20)],
)
def test_round_trip_decode_of_encode(payload_size, radius):
    from hex8.encoder.encode import encode_png

    payload = _sample_payload(payload_size)
    image = encode_png(payload, radius=radius, ecc_level=30, cell_size=6.0)

    decoded = decode_image(image)

    assert decoded == payload


@pytest.mark.parametrize("cell_size", [4.0, 10.0, 13.5])
def test_round_trip_across_cell_sizes(cell_size):
    from hex8.encoder.encode import encode_png

    payload = _sample_payload(64)
    image = encode_png(payload, radius=18, ecc_level=30, cell_size=cell_size)

    decoded = decode_image(image)

    assert decoded == payload


def test_decode_file_reads_a_png_from_disk(tmp_path):
    from hex8.encoder.encode import encode_png

    payload = _sample_payload(64)
    image = encode_png(payload, radius=18, ecc_level=30, cell_size=6.0)
    png_path = tmp_path / "marker.png"
    image.save(png_path, format="PNG")

    decoded = decode_file(png_path)

    assert decoded == payload


def test_decode_file_rejects_svg_input(tmp_path):
    from hex8.encoder.encode import encode_svg

    payload = _sample_payload(16)
    svg_text = encode_svg(payload, radius=18, ecc_level=30, cell_size=6.0)
    svg_path = tmp_path / "marker.svg"
    svg_path.write_text(svg_text, encoding="utf-8")

    with pytest.raises(ValueError, match="SVG"):
        decode_file(svg_path)


def test_decode_image_rejects_non_marker_image():
    plain_image = Image.new("RGB", (400, 400), (255, 255, 255))

    with pytest.raises(ValueError):
        decode_image(plain_image)


def test_decode_image_recovers_from_localized_pixel_corruption():
    """A real end-to-end test (unlike Issue #11's synthetic classification-
    level test): corrupt actual DATA cell pixels in the rendered image to an
    ambiguous (roughly halfway between two palette colors) value, and
    confirm decode_image still recovers the payload via the low-confidence
    -> erasure path, as long as the corrupted count stays within Reed-
    Solomon's erasure budget for a single-block payload."""
    from hex8.common.layout import CellRole, build_layout
    from hex8.common.symbols import PALETTE
    from hex8.encoder.encode import encode_png
    from hex8.encoder.render import cell_center_px, compute_canvas

    payload = _sample_payload(16)
    radius = 18
    ecc_level = 40
    cell_size = 6.0
    image = encode_png(payload, radius=radius, ecc_level=ecc_level, cell_size=cell_size)

    layout = build_layout(radius)
    data_cells = layout.cells_with_role(CellRole.DATA)
    canvas = compute_canvas(radius, cell_size)

    # Blend two palette colors so the pixel is genuinely ambiguous, not just
    # a confident wrong exact match.
    blended = tuple((a + b) // 2 for a, b in zip(PALETTE[0], PALETTE[1], strict=True))

    # Skip the first ~35 data cells: those hold ecc.py's own internal framing
    # header (13 bytes -> ceil(13*8/3) = 35 symbols), which isn't itself
    # Reed-Solomon protected, so corrupting it would break the ECC header
    # rather than exercise the erasure-recovery path this test targets.
    corrupted_count = 5  # modest, well within a single small block's ECC budget
    for q, r in data_cells[40 : 40 + corrupted_count]:
        x, y = cell_center_px(q, r, cell_size, canvas)
        image.putpixel((round(x), round(y)), blended)

    decoded = decode_image(image)

    assert decoded == payload
