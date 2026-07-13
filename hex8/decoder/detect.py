"""Ideal-case Hex8 marker detection + grid normalization (Issue #9).

Given only a rendered marker image (its pixel width/height, no other
metadata), this module recovers enough information to reconstruct the exact
same :class:`~hex8.common.canvas.CanvasInfo` the encoder used when it drew
the marker - i.e. it figures out what ``radius`` and ``cell_size`` were
passed to :func:`hex8.encoder.encode.encode_png`, so that
:func:`hex8.common.canvas.cell_center_px` for every ``(q, r)`` in
:func:`hex8.common.hexgrid.enumerate_cells(radius)` correctly points at the
true center of that cell in the image.

Scope: this module handles the **ideal case only** - a marker image rendered
directly by our own encoder, with no rotation, perspective distortion,
blur, or noise. Robust detection under those real-world degradations is
explicitly out of scope here; that is Phase 3 (Issues #13/#14) and Phase 4
(Issue #15).

Algorithm
---------
:func:`hex8.common.canvas.compute_canvas` has width/height that scale
*linearly* in ``cell_size`` for a fixed ``radius`` (both the hex geometry and
the default quiet-zone margin scale with ``cell_size``). That means
``cell_size`` can be recovered algebraically instead of searched over:

1. For each candidate ``radius`` in ``MIN_RADIUS..MAX_RADIUS``:

   a. Compute a reference canvas at ``cell_size = 1.0``:
      ``ref = compute_canvas(radius, 1.0)``.
   b. Solve for the candidate cell size algebraically:
      ``candidate_cell_size = image.width / ref.width``.
   c. Check the height is consistent with that same cell size (within
      ``TOLERANCE_PX``, to account for ``math.ceil`` rounding in
      ``compute_canvas``); if not, this radius is wrong. This is done by
      recomputing a full canvas at ``candidate_cell_size`` (needed anyway
      for step (e)) and comparing *its* height to the image's, rather than
      naively scaling the ``cell_size=1.0`` reference canvas's height by
      ``candidate_cell_size`` - see the "Deviation from the initial design"
      note below for why.
   d. Skip this radius if :func:`hex8.common.layout.build_layout` raises
      ``ValueError`` (grid too small to fit the reserved finder/palette/
      metadata regions).
   e. Verify using the finder anchors: sample the pixel at the recovered
      center of every FINDER cell and check it is "clearly dark". A
      coincidentally-dark background pixel at one or two anchors shouldn't
      be enough to false-positive, so *every* finder cell is checked (there
      are dozens per marker; see ``hex8.common.layout.ANCHOR_RADIUS``).
   f. Also verify using the palette cells (see "Deviation from the initial
      design" note below for why this second check is necessary): sample
      every PALETTE cell and check it matches its expected exact color,
      ``hex8.common.symbols.PALETTE[i % 8]`` for the i-th palette cell.
   g. If all finder cells and all palette cells pass, this
      ``(radius, cell_size, canvas)`` is the answer.

2. If no candidate radius passes all checks, raise ``ValueError``.

Deviation from the initial design: height consistency check
--------------------------------------------------------------
The design this module was built from proposed computing
``predicted_height = candidate_cell_size * ref.height`` (i.e. scaling the
*already-rounded* ``cell_size=1.0`` reference canvas's height field) and
comparing that to the image's actual height within ``TOLERANCE_PX``.

That does not hold in general: :func:`hex8.common.canvas.compute_canvas`'s
height involves ``math.sqrt(3)`` (unlike its width, which is always an exact
integer multiple of ``cell_size`` for any radius, since it only involves
rational arithmetic), so ``ref.height`` already carries up to ~1px of
``math.ceil`` rounding error *at cell_size=1.0*. Multiplying that error by
``candidate_cell_size`` amplifies it proportionally - e.g. for radius=10,
cell_size=6.0, this produces a 3px discrepancy, already exceeding the
suggested ``TOLERANCE_PX = 1.5`` for a perfectly valid match, which would
incorrectly reject the correct radius.

Instead, this implementation recomputes a full canvas at the candidate cell
size - ``compute_canvas(radius, candidate_cell_size)`` - and compares *that*
canvas's height to the image's actual height. This is more accurate (no
amplified rounding error) and no more expensive: step (e)'s finder-anchor
verification needs this same canvas anyway. ``TOLERANCE_PX`` is kept at the
suggested ``1.5`` as a genuine floating-point/ceil-boundary safety margin.

Deviation from the initial design: added palette cross-check
--------------------------------------------------------------
Testing against real encoder output surfaced a second, more serious problem:
checking *only* that finder cells are dark is not sufficient to disambiguate
between nearby radii. A hex grid's canvas aspect ratio changes only slowly
as radius grows, so for a marker actually encoded at, say, radius=18, a
handful of *other* radii (verified empirically: 16, 17, 19, 20, ... for a
radius=18/cell_size=10.0 image) solve to a candidate cell size whose
predicted canvas height also lands within ``TOLERANCE_PX`` of the real
image's height. Worse, because each candidate's "corner" projects to
roughly the same pixel location regardless of radius (the corner is always
the extreme point of the hex shape), and the true finder anchor clusters are
physically large (``ANCHOR_RADIUS = 2`` cells wide, solid black), a wrong
radius's finder-cluster cell centers frequently land *inside* the true
image's real anchor blobs too - passing the finder-darkness check with a
false radius (concretely, radius=16 passed all 54 of its own finder-cell
samples as dark on a real radius=18/cell_size=10.0 image, purely because its
corner cluster projected inside the true radius=18 corner blobs).

Distinguishing a true match from a merely-nearby radius therefore needs a
signal that varies with the *exact* cell layout, not just canvas aspect
ratio. Palette cells provide exactly that, and - crucially - independent of
payload content: the encoder always assigns
``hex8.common.symbols.PALETTE[i % 8]`` to the i-th palette cell regardless
of what payload was encoded (see ``hex8.encoder.encode._build_cell_colors``).
For a wrong radius, the hypothesized palette-cell positions land on
essentially arbitrary real cells (true finder cells, true palette cells at a
different index, or true data cells - each of which independently matches
the *specific* expected color with only roughly 1-in-8 odds), so requiring
all 16 palette cells to match exactly drives the false-positive probability
down to about ``(1/8)**16``, while a true match always matches all 16
exactly (verified: the same radius=16-on-radius=18 false positive above
matches only 2 of 16 expected palette colors, not 16).
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from hex8.common.canvas import CanvasInfo, cell_center_px, compute_canvas
from hex8.common.layout import CellRole, build_layout
from hex8.common.symbols import PALETTE

Cell = tuple[int, int]

__all__ = [
    "DARK_THRESHOLD",
    "MAX_RADIUS",
    "MIN_RADIUS",
    "TOLERANCE_PX",
    "DetectionResult",
    "detect_marker",
    "normalized_cell_center",
]

#: Smallest grid radius searched. Radii below this cannot fit the reserved
#: finder/palette/metadata regions (see hex8.common.layout.build_layout),
#: so they are never valid Hex8 markers and are skipped without even
#: reaching build_layout.
MIN_RADIUS = 6

#: Largest grid radius searched. Generous headroom above the README's
#: largest documented example radius (60); enumerate_cells/build_layout are
#: cheap enough at this size that scanning the whole range is fast.
MAX_RADIUS = 64

#: Maximum allowed discrepancy (in pixels) between the height predicted from
#: the algebraically-solved cell size and the image's actual height, when
#: checking whether a candidate radius is consistent with the image size.
#: Accounts for math.ceil() rounding of width/height in compute_canvas().
TOLERANCE_PX = 1.5

#: Maximum sum(R, G, B) for a sampled pixel to count as "clearly dark" when
#: verifying finder anchor cells. The ideal encoder renders finder cells as
#: exact solid black (sum == 0); this threshold is generous headroom for a
#: named constant rather than a hardcoded "== (0, 0, 0)" check, decoupling
#: this decoder module from the encoder's exact FINDER_COLOR value. There is
#: no sensor noise in the ideal case, so this is not a tuned/guessed
#: tolerance - real matches will always have sum == 0.
DARK_THRESHOLD = 60


@dataclass(frozen=True)
class DetectionResult:
    """The recovered grid parameters for a detected Hex8 marker image."""

    radius: int
    cell_size: float
    canvas: CanvasInfo


def _is_clearly_dark(pixel: tuple[int, ...]) -> bool:
    return sum(pixel[:3]) <= DARK_THRESHOLD


def _try_radius(image: Image.Image, radius: int) -> DetectionResult | None:
    """Attempt to detect the marker assuming the given radius.

    Returns the DetectionResult if this radius is consistent with the
    image's size, every finder anchor cell samples as dark, and every
    palette cell samples as its expected exact color, else None.
    """
    ref = compute_canvas(radius, 1.0)
    if ref.width <= 0:
        return None

    candidate_cell_size = image.width / ref.width
    if candidate_cell_size <= 0:
        return None

    # Recompute a full canvas at the candidate cell size and check its
    # height against the image's actual height. (See the module docstring's
    # "Deviation from the initial design" note: naively scaling
    # ref.height by candidate_cell_size amplifies compute_canvas's
    # math.ceil rounding error proportionally, which is not accurate enough
    # here - recomputing directly is both more precise and no more
    # expensive, since this same canvas is needed below anyway.)
    canvas = compute_canvas(radius, candidate_cell_size)
    if abs(canvas.height - image.height) > TOLERANCE_PX:
        return None

    try:
        layout = build_layout(radius)
    except ValueError:
        return None

    finder_cells = layout.cells_with_role(CellRole.FINDER)
    if not finder_cells:
        return None

    rgb_image = image.convert("RGB")
    pixels = rgb_image.load()
    width, height = rgb_image.size

    for q, r in finder_cells:
        x, y = cell_center_px(q, r, candidate_cell_size, canvas)
        px, py = round(x), round(y)
        if not (0 <= px < width and 0 <= py < height):
            return None
        if not _is_clearly_dark(pixels[px, py]):
            return None

    # Second verification pass: palette cells (see the module docstring's
    # "Deviation from the initial design: added palette cross-check" note -
    # the finder-darkness check alone is not enough to rule out nearby
    # radii whose finder clusters happen to project inside the true image's
    # own large black anchor blobs).
    palette_cells = layout.cells_with_role(CellRole.PALETTE)
    if not palette_cells:
        return None

    for i, (q, r) in enumerate(palette_cells):
        x, y = cell_center_px(q, r, candidate_cell_size, canvas)
        px, py = round(x), round(y)
        if not (0 <= px < width and 0 <= py < height):
            return None
        expected_color = PALETTE[i % len(PALETTE)]
        if pixels[px, py][:3] != expected_color:
            return None

    return DetectionResult(radius=radius, cell_size=candidate_cell_size, canvas=canvas)


def detect_marker(
    image: Image.Image, radius_candidates: range = range(MIN_RADIUS, MAX_RADIUS + 1)
) -> DetectionResult:
    """Detect a Hex8 marker's grid radius and cell size from its rendered image.

    Args:
        image: The rendered marker image (a Pillow ``Image``), as produced
            by :func:`hex8.encoder.encode.encode_png`. Only ideal, distortion
            -free renders are supported (see module docstring).
        radius_candidates: The range of grid radii to try, in order.
            Defaults to ``MIN_RADIUS..MAX_RADIUS`` inclusive.

    Returns:
        A :class:`DetectionResult` describing the detected radius, cell
        size, and canvas geometry.

    Raises:
        ValueError: if no candidate radius in ``radius_candidates`` produces
            a canvas size and finder-anchor pattern consistent with `image`.
    """
    for radius in radius_candidates:
        result = _try_radius(image, radius)
        if result is not None:
            return result

    raise ValueError(
        "no matching Hex8 marker grid found for this image size "
        f"({image.width}x{image.height}); tried radii "
        f"{radius_candidates.start}..{radius_candidates.stop - 1}"
    )


def normalized_cell_center(result: DetectionResult, q: int, r: int) -> tuple[int, int]:
    """Return the rounded-to-nearest-int pixel coordinate of cell ``(q, r)``.

    Wraps :func:`hex8.common.canvas.cell_center_px` with the detected
    ``cell_size``/``canvas``, rounding to integer pixel coordinates ready
    for direct pixel sampling (e.g. ``image.load()[x, y]``).
    """
    x, y = cell_center_px(q, r, result.cell_size, result.canvas)
    return (round(x), round(y))
