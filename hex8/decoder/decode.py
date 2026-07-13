"""Phase 2 decoder integration: marker image -> payload bytes (Issue #12).

Wires together marker detection (:mod:`hex8.decoder.detect`, Issue #9),
Lab-space color classification (:mod:`hex8.decoder.classify`, Issue #10),
and Reed-Solomon correction + CRC verification
(:mod:`hex8.decoder.correct`, Issue #11) into the single Phase 2 success
condition from the README::

    payload bytes -> marker image -> same payload bytes

Scope: PNG (raster) input only. The README's Phase 2 spec mentions
"decode(png_or_svg)", but this project's real usage never needs to decode
an SVG *as SVG*: Phase 3/4 photograph a rendered/printed image (always
raster), and this project's own SVG output
(:func:`hex8.encoder.render.render_svg`) exists to produce a vector master
for printing, not as camera-decoder input. Rasterizing an arbitrary SVG
would need a Cairo (or similar) dependency this project deliberately
avoided for the encoder (see ``hex8.encoder.render``'s module docstring);
adding one solely to decode a format nothing downstream actually produces
as decoder input isn't justified. :func:`decode_file` therefore raises a
clear error for ``.svg`` paths instead of silently failing deep inside
image loading.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from hex8.common.layout import CellRole, build_layout
from hex8.decoder.classify import build_observed_palette, classify_cells
from hex8.decoder.correct import decode_marker
from hex8.decoder.detect import detect_marker

__all__ = ["decode_file", "decode_image"]


def decode_image(image: Image.Image) -> bytes:
    """Decode a Hex8 marker from an already-loaded Pillow image.

    Args:
        image: The rendered marker image, as produced by
            :func:`hex8.encoder.encode.encode_png` (directly, or after
            being saved to and re-loaded from disk).

    Returns:
        The original payload bytes.

    Raises:
        ValueError: if no matching marker grid is found in the image (see
            :func:`hex8.decoder.detect.detect_marker`), if the recovered
            header is invalid, if Reed-Solomon correction fails, or if the
            recovered payload's CRC32 does not match the header.
    """
    detection = detect_marker(image)
    layout = build_layout(detection.radius)
    observed_palette = build_observed_palette(
        image, detection.radius, detection.cell_size, detection.canvas
    )

    metadata_cells = layout.cells_with_role(CellRole.METADATA)
    metadata_classifications_by_cell = classify_cells(
        image, metadata_cells, detection.cell_size, detection.canvas, observed_palette
    )
    metadata_classifications = [metadata_classifications_by_cell[cell] for cell in metadata_cells]

    data_cells = layout.cells_with_role(CellRole.DATA)
    data_classifications_by_cell = classify_cells(
        image, data_cells, detection.cell_size, detection.canvas, observed_palette
    )
    data_classifications = [data_classifications_by_cell[cell] for cell in data_cells]

    decoded = decode_marker(metadata_classifications, data_classifications)
    return decoded.payload


def decode_file(path: str | Path) -> bytes:
    """Decode a Hex8 marker from an image file on disk.

    Args:
        path: Path to a raster image file (e.g. PNG) produced by
            :func:`hex8.encoder.encode.encode_png`.

    Returns:
        The original payload bytes.

    Raises:
        ValueError: if `path` has a `.svg` extension (unsupported - see
            module docstring), or for the same reasons as
            :func:`decode_image`.
    """
    path = Path(path)
    if path.suffix.lower() == ".svg":
        raise ValueError(
            "Decoding SVG markers directly is not supported: rasterizing "
            "arbitrary SVG would need a Cairo-like dependency this project "
            "deliberately avoids (see hex8.encoder.render's module "
            "docstring). Render/print the SVG to a raster image first."
        )

    with Image.open(path) as image:
        return decode_image(image.convert("RGB"))
