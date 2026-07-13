"""Lab-space color classification with confidence for Hex8 marker cells.

Implements the "Color Classification" approach from the README:

1. Sample observed palette cells (:func:`build_observed_palette`).
2. Convert samples to Lab color space.
3. Convert cell samples to Lab.
4. Find nearest observed palette color.
5. Return both symbol and confidence (:func:`classify_pixel`).

Rationale: classifying against an *observed* palette (sampled from the same
image being decoded, see :func:`build_observed_palette`) rather than the
ideal :data:`hex8.common.symbols.PALETTE` RGB values lets the decoder adapt
to whatever color cast a camera/lighting/print pipeline introduces: as long
as all 8 colors are shifted in a roughly similar way, nearest-neighbor
matching against the observed palette remains robust even though exact RGB
matching (:func:`hex8.common.symbols.color_to_symbol`) would fail.

Confidence formula
-------------------
For a sample pixel, let ``d_best`` be the Lab-space Euclidean distance to
the nearest observed palette color and ``d_second`` the distance to the
second-nearest. This is a normalized margin, a standard way to turn a
"distance to best vs. distance to runner-up" comparison into a bounded
``[0, 1]`` confidence score::

    confidence = (d_second - d_best) / (d_second + d_best)

- If the best match is far closer than the runner-up (``d_best << d_second``),
  confidence approaches 1 (unambiguous match).
- If the sample is equidistant from its two nearest palette colors
  (``d_best == d_second``), confidence is exactly 0 (maximally ambiguous:
  the classifier has no basis to prefer one symbol over the other).
- The degenerate zero-sum case (``d_best == d_second == 0``, an exact tie
  at zero distance - not expected with 8 distinct palette colors, but
  handled explicitly to avoid a ``ZeroDivisionError``) is *also* treated as
  ``confidence = 0.0``: a genuine tie between two equally-good matches is
  the lowest-confidence case, not the highest, so this is consistent with
  the general formula's behavior at ``d_best == d_second``, not an
  arbitrary carve-out.

Low-confidence cells (``confidence < LOW_CONFIDENCE_THRESHOLD``) may later
be treated as erasures for Reed-Solomon correction (Issue #11) - this
module only exposes the confidence score and the ``low_confidence`` flag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

from hex8.common.canvas import CanvasInfo, cell_center_px
from hex8.common.layout import CellRole, PALETTE_REPEATS, build_layout
from hex8.common.symbols import PALETTE

Cell = tuple[int, int]
RGB = tuple[int, int, int]
Lab = tuple[float, float, float]

__all__ = [
    "LOW_CONFIDENCE_THRESHOLD",
    "Classification",
    "build_observed_palette",
    "classify_cells",
    "classify_pixel",
]

#: Documented default confidence threshold below which a classification is
#: flagged as low-confidence. May be retuned in Issue #14.
LOW_CONFIDENCE_THRESHOLD = 0.2


@dataclass(frozen=True)
class Classification:
    """The result of classifying a single sampled pixel against a palette.

    Attributes:
        symbol: The 3-bit symbol (0-7) of the nearest palette color.
        confidence: Normalized margin between the best and second-best
            Lab-distance match, in ``[0.0, 1.0]`` (see module docstring for
            the exact formula).
        low_confidence: ``True`` iff ``confidence < LOW_CONFIDENCE_THRESHOLD``.
    """

    symbol: int
    confidence: float
    low_confidence: bool


def _rgb_to_lab(pixel_rgb: RGB) -> Lab:
    """Convert a single (0-255) RGB pixel to a (L, a, b) tuple.

    ``rgb2lab`` expects float RGB in the 0-1 range and an image-shaped
    array, so the single pixel is reshaped to ``(1, 1, 3)`` and the
    ``(L, a, b)`` triple is extracted back out.
    """
    arr = np.asarray(pixel_rgb, dtype=np.float64).reshape(1, 1, 3) / 255.0
    lab = rgb2lab(arr)
    return (float(lab[0, 0, 0]), float(lab[0, 0, 1]), float(lab[0, 0, 2]))


def _sample_pixel(image: Image.Image, x: float, y: float) -> RGB:
    """Sample the image's pixel at (x, y), rounded to the nearest int pixel."""
    px = image.getpixel((round(x), round(y)))
    return (px[0], px[1], px[2])


def build_observed_palette(
    image: Image.Image, radius: int, cell_size: float, canvas: CanvasInfo
) -> dict[int, Lab]:
    """Sample the marker's palette cells and average each symbol's Lab value.

    For each symbol 0-7, averages the Lab-space values of all
    ``PALETTE_REPEATS`` palette cells assigned that symbol, sampling each
    cell's pixel at its center via :func:`hex8.common.canvas.cell_center_px`
    (rounded to the nearest int pixel).

    The i-th palette cell in
    ``build_layout(radius).cells_with_role(CellRole.PALETTE)`` corresponds
    to symbol ``i % len(PALETTE)`` (the encoder's assignment convention,
    see :mod:`hex8.encoder.encode`).

    Returns:
        A dict ``{symbol: (L, a, b)}`` covering all 8 symbols.
    """
    layout = build_layout(radius)
    palette_cells = layout.cells_with_role(CellRole.PALETTE)

    symbol_lab_sums: dict[int, list[float]] = {
        symbol: [0.0, 0.0, 0.0] for symbol in PALETTE
    }
    symbol_counts: dict[int, int] = dict.fromkeys(PALETTE, 0)

    for i, cell in enumerate(palette_cells):
        symbol = i % len(PALETTE)
        x, y = cell_center_px(cell[0], cell[1], cell_size, canvas)
        pixel_rgb = _sample_pixel(image, x, y)
        lab = _rgb_to_lab(pixel_rgb)
        sums = symbol_lab_sums[symbol]
        sums[0] += lab[0]
        sums[1] += lab[1]
        sums[2] += lab[2]
        symbol_counts[symbol] += 1

    observed_palette: dict[int, Lab] = {}
    for symbol, sums in symbol_lab_sums.items():
        count = symbol_counts[symbol]
        assert count == PALETTE_REPEATS, (
            f"expected exactly {PALETTE_REPEATS} palette cells for symbol "
            f"{symbol}, found {count} (radius {radius} layout mismatch)"
        )
        observed_palette[symbol] = (sums[0] / count, sums[1] / count, sums[2] / count)

    return observed_palette


def classify_pixel(
    pixel_rgb: RGB, observed_palette: dict[int, Lab]
) -> Classification:
    """Classify a single RGB pixel against an observed Lab palette.

    Converts ``pixel_rgb`` to Lab, computes the Euclidean distance to every
    entry of ``observed_palette``, and returns the nearest symbol along
    with a normalized-margin confidence score (see module docstring for the
    exact formula and its rationale).
    """
    pixel_lab = np.array(_rgb_to_lab(pixel_rgb))

    distances: list[tuple[float, int]] = []
    for symbol, lab in observed_palette.items():
        distance = float(np.linalg.norm(pixel_lab - np.array(lab)))
        distances.append((distance, symbol))
    distances.sort(key=lambda item: item[0])

    d_best, best_symbol = distances[0]
    d_second = distances[1][0] if len(distances) > 1 else d_best

    denominator = d_second + d_best
    if denominator > 0:
        confidence = (d_second - d_best) / denominator
    else:
        # d_best == d_second == 0: an exact tie at zero distance. A genuine
        # tie between two equally-good matches is the lowest-confidence
        # case, not the highest (see module docstring).
        confidence = 0.0

    return Classification(
        symbol=best_symbol,
        confidence=confidence,
        low_confidence=confidence < LOW_CONFIDENCE_THRESHOLD,
    )


def classify_cells(
    image: Image.Image,
    cells: list[Cell],
    cell_size: float,
    canvas: CanvasInfo,
    observed_palette: dict[int, Lab],
) -> dict[Cell, Classification]:
    """Sample and classify a batch of cells' pixel centers.

    Convenience wrapper around :func:`classify_pixel` for a batch of
    ``(q, r)`` cells (e.g. the DATA or METADATA cells from a
    :class:`hex8.common.layout.MarkerLayout`).
    """
    results: dict[Cell, Classification] = {}
    for cell in cells:
        x, y = cell_center_px(cell[0], cell[1], cell_size, canvas)
        pixel_rgb = _sample_pixel(image, x, y)
        results[cell] = classify_pixel(pixel_rgb, observed_palette)
    return results
