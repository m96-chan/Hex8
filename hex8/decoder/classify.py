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

Cell-position abstraction (Issue #14)
-------------------------------------
:func:`build_observed_palette` and :func:`classify_cells` do not compute
cell pixel positions themselves; the caller supplies a ``project`` callable
mapping an axial ``(q, r)`` cell coordinate to its ``(x, y)`` pixel center
in the image. This keeps the classifier agnostic to *how* that mapping was
recovered: the ideal fast path (Issue #9) uses a simple
``cell_size``/``canvas`` affine placement, while the Phase 3 robust fallback
(Issue #14) uses a full perspective homography. Both are exposed uniformly
via :meth:`hex8.decoder.detect.DetectionResult.cell_center`, which is the
``project`` callable normally passed here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

from hex8.common.layout import CellRole, PALETTE_REPEATS, build_layout
from hex8.common.symbols import PALETTE

Cell = tuple[int, int]
RGB = tuple[int, int, int]
Lab = tuple[float, float, float]

#: A callable mapping an axial ``(q, r)`` cell coordinate to its ``(x, y)``
#: pixel center in the image being decoded (see the module docstring's
#: "Cell-position abstraction" note). Normally
#: :meth:`hex8.decoder.detect.DetectionResult.cell_center`.
CellCenterFn = Callable[[int, int], tuple[float, float]]

__all__ = [
    "LOW_CONFIDENCE_THRESHOLD",
    "CellCenterFn",
    "Classification",
    "build_observed_palette",
    "classify_cells",
    "classify_pixel",
]

#: Confidence threshold below which a classification is flagged as
#: low-confidence (and later treated as a Reed-Solomon erasure candidate,
#: see :mod:`hex8.decoder.correct`).
#:
#: Retuned from the Issue #10 default of ``0.2`` down to ``0.1`` in Issue #14
#: (as that module's docstring anticipated). Empirically, the original ``0.2``
#: was too aggressive under mild global degradation (e.g. a small Gaussian
#: blur): it flagged so many correctly-classified-but-slightly-softened data
#: cells as erasures that the Reed-Solomon block's erasure budget was
#: exhausted and decoding failed, even though the symbols were in fact
#: recovered correctly. At ``0.1`` the correctly-classified cells under mild
#: blur/JPEG/noise sit comfortably above the threshold, while genuinely
#: ambiguous near-tie cells (e.g. a 50% black/white midpoint, whose
#: normalized-margin confidence is ~0.07) are still flagged as erasures.
LOW_CONFIDENCE_THRESHOLD = 0.1


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
    image: Image.Image, radius: int, project: CellCenterFn
) -> dict[int, Lab]:
    """Sample the marker's palette cells and average each symbol's Lab value.

    For each symbol 0-7, averages the Lab-space values of all
    ``PALETTE_REPEATS`` palette cells assigned that symbol, sampling each
    cell's pixel at the center returned by ``project`` (rounded to the
    nearest int pixel).

    The i-th palette cell in
    ``build_layout(radius).cells_with_role(CellRole.PALETTE)`` corresponds
    to symbol ``i % len(PALETTE)`` (the encoder's assignment convention,
    see :mod:`hex8.encoder.encode`).

    Args:
        image: The marker image being decoded.
        radius: The detected grid radius.
        project: Maps an axial ``(q, r)`` cell to its ``(x, y)`` pixel center
            (see :data:`CellCenterFn`).

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
        x, y = project(cell[0], cell[1])
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
    project: CellCenterFn,
    observed_palette: dict[int, Lab],
) -> dict[Cell, Classification]:
    """Sample and classify a batch of cells' pixel centers.

    Convenience wrapper around :func:`classify_pixel` for a batch of
    ``(q, r)`` cells (e.g. the DATA or METADATA cells from a
    :class:`hex8.common.layout.MarkerLayout`). Each cell's pixel center is
    located via ``project`` (see :data:`CellCenterFn`).
    """
    results: dict[Cell, Classification] = {}
    for cell in cells:
        x, y = project(cell[0], cell[1])
        pixel_rgb = _sample_pixel(image, x, y)
        results[cell] = classify_pixel(pixel_rgb, observed_palette)
    return results
