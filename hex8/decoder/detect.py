"""Hex8 marker detection + grid normalization (Issues #9 and #14).

Ideal fast path (Issue #9) and robust fallback (Issue #14)
==========================================================
:func:`detect_marker` has two stages. It first tries the original,
zero-tolerance *ideal fast path* (unchanged from Issue #9, documented in
full below): an algebraic solve of ``cell_size`` from the image's exact
pixel width plus an exact-color verification. That path only succeeds on a
pristine, axis-aligned render straight from our own encoder, and it is kept
byte-for-byte behaviourally identical so existing callers see no change.

When the fast path finds no match (any real degradation - recompression,
rotation, perspective, blur, noise, brightness change - breaks its exact
checks), :func:`detect_marker` falls back to a general geometric solve,
:func:`_detect_via_homography`:

1. **Locate the 6 finder anchors.** Threshold the image for near-black
   pixels (``max(R,G,B) < ANCHOR_DARK_MAX``) and run a distance transform;
   the 6 finder anchor clusters are the deepest solid dark cores. Iteratively
   picking the 6 strongest, well-separated distance-transform maxima locates
   them robustly *even when a finder anchor is fused with adjacent black data
   cells* (a black data cell is only 1 cell wide, so it barely dents the
   distance transform, whereas a solid ``ANCHOR_RADIUS=2`` anchor cluster is
   several cells wide). Each anchor's sub-pixel position is refined to the
   centroid of the dark pixels in a small window around its peak.
2. **Hypothesize a radius + correspondence.** For each candidate radius, the
   6 expected anchor positions are known in a payload-independent "template"
   coordinate frame (axial-pixel coordinates at ``cell_size = 1.0``); they
   are computed by the *same* peak detector run on a synthetic finder-only
   mask, so template and image anchor points are measured consistently. Both
   the 6 template points and the 6 detected points are sorted by angle about
   their own centroid; the two hexagons are then aligned by trying all 6
   cyclic rotational offsets (and a reflection - see note below).
3. **Estimate a homography.** ``cv2.findHomography`` fits the 6 corresponded
   point pairs (over-determined vs. the 4-point minimum, so it least-squares
   over small centroid noise), producing a single transform from template
   coordinates to image pixels. Because a homography captures rotation,
   translation, scale *and* perspective together, no per-degradation special
   case (a separate "rotation correction" or "perspective correction") is
   needed - they are all subsumed.
4. **Verify the hypothesis.** Every FINDER cell is mapped through the
   homography and sampled: most must still be dark (relaxed tolerance). Then
   every PALETTE cell is mapped through, and classified with the existing
   Lab-space nearest-neighbour classifier (:mod:`hex8.decoder.classify`)
   against an observed palette sampled through the *same* homography; almost
   all must classify to their expected symbol. The palette cross-check is the
   same disambiguation principle the fast path relies on (see the
   "added palette cross-check" note below): it makes a wrong
   radius/correspondence astronomically unlikely to be accepted, since the
   expected per-cell symbols follow a payload-independent pattern that only
   the true grid reproduces.

The verified radius + homography are returned as a :class:`DetectionResult`
carrying a ``homography`` instead of a ``cell_size``/``canvas``. Downstream
callers never branch on which path succeeded: :meth:`DetectionResult.cell_center`
maps any cell to its image pixel using whichever transform is present.

Reflection: the six cyclic rotations are tried for both the detected point
order and its reverse. Our synthetic degradations (:mod:`hex8.degrade`) and a
normal photograph of a printed marker never mirror the marker, so a
reflection should not arise in practice; it is included cheaply (the correct,
non-reflected correspondence is tried first and short-circuits on success)
purely as insurance against a genuinely mirrored capture, and the palette
cross-check would reject a spurious mirrored fit anyway.

"Mild degradation" thresholds (Issue #14)
-----------------------------------------
The fallback is designed and empirically verified (see
``docs/phase3-baseline.md`` and :mod:`hex8.degrade`'s harness) to recover the
exact payload under the following per-degradation "mild" severities, at the
README's PoC target configuration (radius 18-20). Blur is inherently
relative to ``cell_size``; the figure below is for the baseline's
``cell_size = 6.0`` and scales up with larger cells.

===============  =========================================================
degradation      mild threshold (decodes correctly)
===============  =========================================================
rotation         any angle (verified 1 deg - 90 deg); README example ±15 deg
scaling          0.5x - 2.0x
blur             Gaussian radius <= 1.5 px at cell_size 6.0 (~0.25x cell_size)
jpeg             quality >= 25
noise            Gaussian sigma <= 20 (0-255 scale)
brightness       factor 0.5 - 1.5
perspective      strength <= 0.05 (fraction of width/height)
===============  =========================================================

These are the honest boundaries of the current approach: beyond them
(heavier blur that smears whole cells, stronger perspective that shrinks a
far anchor below the noise floor) recovery is not guaranteed and is left to
Phase 4. Blur tolerance in particular is *relative to the cell size* - the
1.5 px figure is for the baseline's small ``cell_size = 6.0``; at
``cell_size = 10.0`` the same decoder tolerates ~3 px (the metadata header,
which carries no error correction of its own, is the first thing a whole-cell
smear corrupts). Rotation, scaling and bright-enough brightness changes are
often still handled by the *fast* path (exact colours survive them), so those
may return a ``cell_size``/``canvas`` result rather than a homography. The
colour classifier's brightness/cast tolerance comes for free from its
Lab-space *observed*-palette design (:mod:`hex8.decoder.classify`).

Ideal fast path (Issue #9), in detail
======================================

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

import math
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from hex8.common.canvas import CanvasInfo, cell_center_px, compute_canvas
from hex8.common.hexgrid import axial_to_pixel
from hex8.common.layout import CellRole, build_layout
from hex8.common.symbols import PALETTE
from hex8.decoder.classify import build_observed_palette, classify_pixel

Cell = tuple[int, int]

__all__ = [
    "ANCHOR_DARK_MAX",
    "DARK_THRESHOLD",
    "FINDER_DARK_FRACTION",
    "MAX_RADIUS",
    "MIN_RADIUS",
    "NUM_ANCHORS",
    "PALETTE_MATCH_MIN",
    "RELAXED_FINDER_DARK_SUM",
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


# --- Robust fallback constants (Issue #14) -------------------------------

#: Number of finder anchor clusters to locate (the 6 hex-grid corners).
NUM_ANCHORS = 6

#: Maximum per-channel value (``max(R, G, B)``) for a pixel to be treated as
#: "anchor black" when building the dark mask the anchor-locating distance
#: transform runs on. Chosen to catch the solid-black finder anchors (and
#: near-black pixels after mild degradation) while excluding every saturated
#: palette colour: each of the 7 non-black palette colours has at least one
#: channel at 255, far above this threshold, so coloured cells never pollute
#: the anchor mask. This is a ``max``-channel test, not a brightness test,
#: precisely so that e.g. pure red ``(255, 0, 0)`` (low mean brightness) is
#: not mistaken for a dark anchor pixel.
ANCHOR_DARK_MAX = 110

#: Minimum distance-transform value (roughly, half the local dark-region
#: width in pixels) for a peak to be accepted as a finder-anchor candidate.
#: Filters out single stray dark specks/noise, which have a tiny inscribed
#: radius, without needing to know the (as-yet-unknown) cell size.
ANCHOR_MIN_PEAK = 1.5

#: Cell size (px) of the internal synthetic finder-only mask used to measure
#: each candidate radius's template anchor positions with the *same* peak
#: detector used on the real image (see module docstring, step 2). Any
#: comfortably-large value works; the measured points are rescaled to the
#: ``cell_size = 1.0`` template frame.
TEMPLATE_CELL_SIZE = 10.0

#: Maximum ``R + G + B`` for a homography-projected FINDER cell center to
#: count as "still dark" during fallback verification. Looser than the exact
#: fast-path ``DARK_THRESHOLD`` to tolerate mild blur/JPEG/noise softening of
#: the anchors, but still far below any saturated palette colour's channel
#: sum, so a wrong hypothesis whose finder cells land on coloured cells is
#: rejected.
RELAXED_FINDER_DARK_SUM = 250

#: Fraction of FINDER cells that must sample as dark (per
#: ``RELAXED_FINDER_DARK_SUM``) for a homography hypothesis to pass the first
#: verification gate. Below 1.0 to tolerate a few anchor-edge/perspective
#: sampling misses under mild degradation.
FINDER_DARK_FRACTION = 0.85

#: Minimum number of the 16 PALETTE cells that must classify to their
#: expected symbol (via the Lab-space observed-palette classifier) for a
#: homography hypothesis to be accepted. The 16 palette cells follow a fixed,
#: payload-independent symbol pattern, so a wrong radius/correspondence
#: matches this many only with probability on the order of ``(1/8)**14`` -
#: negligible - while a correct fit under mild degradation matches all (or
#: nearly all) 16. The small slack below 16 tolerates one or two
#: noise/JPEG-corrupted calibration cells.
PALETTE_MATCH_MIN = 14


@dataclass(frozen=True)
class DetectionResult:
    """The recovered grid parameters for a detected Hex8 marker image.

    Two shapes, one interface:

    - **Ideal fast path** (Issue #9): ``cell_size`` and ``canvas`` are set,
      ``homography`` is ``None``. Cell centers are the simple affine
      placement of the encoder's own render.
    - **Robust fallback** (Issue #14): ``homography`` is set (a 3x3
      perspective transform from ``cell_size = 1.0`` axial-pixel template
      coordinates to image pixels), while ``cell_size``/``canvas`` are
      ``None``.

    Either way, :meth:`cell_center` maps an axial ``(q, r)`` cell to its
    image pixel center, so callers need not know which path produced the
    result.
    """

    radius: int
    cell_size: float | None = None
    canvas: CanvasInfo | None = None
    homography: np.ndarray | None = field(default=None, compare=False, repr=False)

    def cell_center(self, q: int, r: int) -> tuple[float, float]:
        """Return the image-space pixel center ``(x, y)`` of cell ``(q, r)``.

        Uses the perspective ``homography`` when present (fallback path),
        otherwise the ``cell_size``/``canvas`` affine placement (fast path).
        """
        if self.homography is not None:
            tx, ty = axial_to_pixel(q, r, 1.0)
            src = np.array([[[float(tx), float(ty)]]], dtype=np.float64)
            dst = cv2.perspectiveTransform(src, self.homography)
            return (float(dst[0, 0, 0]), float(dst[0, 0, 1]))
        assert self.cell_size is not None and self.canvas is not None
        return cell_center_px(q, r, self.cell_size, self.canvas)


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
    """Detect a Hex8 marker's grid radius and geometry from its rendered image.

    Tries the ideal fast path first (exact, zero-tolerance - only matches a
    pristine render from our own encoder), then, if that finds nothing, the
    robust homography fallback (Issue #14), which tolerates mild real-world
    degradation (rotation, perspective, blur, JPEG, noise, brightness - see
    the module docstring for the specific thresholds).

    Args:
        image: The rendered marker image (a Pillow ``Image``), as produced
            by :func:`hex8.encoder.encode.encode_png`, possibly after mild
            degradation.
        radius_candidates: The range of grid radii to try, in order.
            Defaults to ``MIN_RADIUS..MAX_RADIUS`` inclusive.

    Returns:
        A :class:`DetectionResult`. On the fast path it carries
        ``cell_size``/``canvas``; on the fallback path it carries a
        ``homography`` instead. Use :meth:`DetectionResult.cell_center` to
        locate cells regardless of which path succeeded.

    Raises:
        ValueError: if neither path finds a Hex8 marker grid consistent with
            `image`.
    """
    for radius in radius_candidates:
        result = _try_radius(image, radius)
        if result is not None:
            return result

    fallback = _detect_via_homography(image, radius_candidates)
    if fallback is not None:
        return fallback

    raise ValueError(
        "no matching Hex8 marker grid found for this image size "
        f"({image.width}x{image.height}); tried radii "
        f"{radius_candidates.start}..{radius_candidates.stop - 1}"
    )


def normalized_cell_center(result: DetectionResult, q: int, r: int) -> tuple[int, int]:
    """Return the rounded-to-nearest-int pixel coordinate of cell ``(q, r)``.

    Wraps :meth:`DetectionResult.cell_center` (which transparently uses the
    fast-path affine placement or the fallback homography), rounding to
    integer pixel coordinates ready for direct pixel sampling (e.g.
    ``image.load()[x, y]``).
    """
    x, y = result.cell_center(q, r)
    return (round(x), round(y))


# --- Robust homography fallback (Issue #14) ------------------------------


def _hexagon_vertices_int(cx: float, cy: float, size: float) -> np.ndarray:
    """Integer flat-top hexagon vertices for cv2 polygon fill."""
    return np.array(
        [
            [round(cx + size * math.cos(math.radians(60 * i))),
             round(cy + size * math.sin(math.radians(60 * i)))]
            for i in range(6)
        ],
        dtype=np.int32,
    )


def _find_anchor_points(mask: np.ndarray) -> np.ndarray:
    """Locate up to ``NUM_ANCHORS`` finder-anchor centers in a dark-pixel mask.

    Runs a distance transform on the binary ``mask`` and iteratively picks
    the strongest maximum (deepest solid dark core = a finder anchor),
    refines it to the centroid of the dark pixels in a small window around
    the peak, then suppresses that neighbourhood before picking the next.
    Returns an ``(n, 2)`` float32 array of ``(x, y)`` points (``n <=
    NUM_ANCHORS``).
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    working = dist.copy()
    points: list[tuple[float, float]] = []
    for _ in range(NUM_ANCHORS):
        _, max_val, _, max_loc = cv2.minMaxLoc(working)
        if max_val <= ANCHOR_MIN_PEAK:
            break
        x0, y0 = max_loc
        window = int(round(max_val * 1.5))
        xa, xb = max(0, x0 - window), min(mask.shape[1], x0 + window + 1)
        ya, yb = max(0, y0 - window), min(mask.shape[0], y0 + window + 1)
        sub = mask[ya:yb, xa:xb]
        ys, xs = np.nonzero(sub)
        if len(xs) == 0:
            cv2.circle(working, max_loc, int(round(max_val * 2.5)), 0, -1)
            continue
        points.append((float(xs.mean() + xa), float(ys.mean() + ya)))
        cv2.circle(working, max_loc, int(round(max_val * 2.5)), 0, -1)
    return np.array(points, dtype=np.float32).reshape(-1, 2)


def _angular_sort(points: np.ndarray) -> np.ndarray:
    """Sort points counter-clockwise by angle about their own centroid."""
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


@lru_cache(maxsize=None)
def _radius_template(radius: int) -> tuple | None:
    """Precompute payload-independent template data for a candidate radius.

    Returns a tuple ``(anchor_axial_sorted, finder_template, palette_cells,
    palette_template)`` or ``None`` if the radius cannot host a marker or its
    6 anchors cannot be located on the synthetic finder mask:

    - ``anchor_axial_sorted``: ``(6, 2)`` float32 array of the 6 anchor
      centers in the ``cell_size = 1.0`` axial-pixel template frame, sorted
      angularly (to correspond against the detected anchors).
    - ``finder_template`` / ``palette_template``: ``(N, 1, 2)`` float64
      arrays of every FINDER / PALETTE cell center in the same template
      frame, ready for a single batched ``cv2.perspectiveTransform``.
    - ``palette_cells``: the PALETTE cell ``(q, r)`` list (its order defines
      each cell's expected symbol, index ``% 8``).
    """
    try:
        layout = build_layout(radius)
    except ValueError:
        return None

    finder_cells = layout.cells_with_role(CellRole.FINDER)
    palette_cells = layout.cells_with_role(CellRole.PALETTE)
    if not finder_cells or not palette_cells:
        return None

    # Measure the template anchor centers with the SAME detector used on the
    # real image, by rendering a synthetic finder-only mask.
    canvas = compute_canvas(radius, TEMPLATE_CELL_SIZE)
    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
    for q, r in finder_cells:
        cx, cy = cell_center_px(q, r, TEMPLATE_CELL_SIZE, canvas)
        cv2.fillConvexPoly(mask, _hexagon_vertices_int(cx, cy, TEMPLATE_CELL_SIZE), 1)

    anchor_px = _find_anchor_points(mask)
    if len(anchor_px) < NUM_ANCHORS:
        return None
    anchor_axial = np.array(
        [
            [(px - canvas.origin_x) / TEMPLATE_CELL_SIZE,
             (py - canvas.origin_y) / TEMPLATE_CELL_SIZE]
            for px, py in anchor_px
        ],
        dtype=np.float32,
    )
    anchor_axial_sorted = _angular_sort(anchor_axial)

    finder_template = np.array(
        [[axial_to_pixel(q, r, 1.0)] for q, r in finder_cells], dtype=np.float64
    )
    palette_template = np.array(
        [[axial_to_pixel(q, r, 1.0)] for q, r in palette_cells], dtype=np.float64
    )
    return (anchor_axial_sorted, finder_template, palette_cells, palette_template)


def _project_all(template_pts: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Map an ``(N, 1, 2)`` template point array through ``homography``."""
    return cv2.perspectiveTransform(template_pts, homography).reshape(-1, 2)


def _verify_homography(
    image: Image.Image,
    rgb: np.ndarray,
    radius: int,
    homography: np.ndarray,
    template: tuple,
) -> bool:
    """Verify a radius + homography hypothesis (finder darkness + palette match)."""
    _anchor_axial, finder_template, palette_cells, _palette_template = template
    height, width = rgb.shape[:2]

    finder_px = _project_all(finder_template, homography)
    dark = 0
    for x, y in finder_px:
        px, py = int(round(x)), int(round(y))
        if not (0 <= px < width and 0 <= py < height):
            return False
        if int(rgb[py, px, 0]) + int(rgb[py, px, 1]) + int(rgb[py, px, 2]) <= RELAXED_FINDER_DARK_SUM:
            dark += 1
    if dark < FINDER_DARK_FRACTION * len(finder_px):
        return False

    # Reuse the Lab-space observed-palette classifier for the colour cross-
    # check (the same tool the full decode uses); a wrong hypothesis will not
    # reproduce the fixed palette-symbol pattern.
    project = _homography_projector(homography)
    try:
        observed_palette = build_observed_palette(image, radius, project)
    except Exception:
        # Palette cell projected out of bounds, or layout mismatch: reject.
        return False

    matches = 0
    for i, (q, r) in enumerate(palette_cells):
        x, y = project(q, r)
        px, py = int(round(x)), int(round(y))
        if not (0 <= px < width and 0 <= py < height):
            return False
        color = (int(rgb[py, px, 0]), int(rgb[py, px, 1]), int(rgb[py, px, 2]))
        if classify_pixel(color, observed_palette).symbol == i % len(PALETTE):
            matches += 1
    return matches >= PALETTE_MATCH_MIN


def _homography_projector(homography: np.ndarray):
    """Return a ``project(q, r) -> (x, y)`` closure for the given homography."""

    def project(q: int, r: int) -> tuple[float, float]:
        tx, ty = axial_to_pixel(q, r, 1.0)
        src = np.array([[[float(tx), float(ty)]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, homography)
        return (float(dst[0, 0, 0]), float(dst[0, 0, 1]))

    return project


def _detect_via_homography(
    image: Image.Image, radius_candidates: range
) -> DetectionResult | None:
    """Robust fallback detection via finder-anchor correspondence + homography.

    See the module docstring for the full algorithm. Returns a
    homography-carrying :class:`DetectionResult` for the first radius +
    correspondence that passes verification, or ``None`` if none does.
    """
    rgb = np.asarray(image.convert("RGB"))
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None

    mask = np.all(rgb[:, :, :3] < ANCHOR_DARK_MAX, axis=2).astype(np.uint8)
    detected = _find_anchor_points(mask)
    if len(detected) < NUM_ANCHORS:
        return None
    detected_sorted = _angular_sort(detected)

    for radius in radius_candidates:
        template = _radius_template(radius)
        if template is None:
            continue
        anchor_axial_sorted = template[0]

        for reflect in (False, True):
            ordered = detected_sorted[::-1] if reflect else detected_sorted
            for rotation in range(NUM_ANCHORS):
                rolled = np.roll(ordered, rotation, axis=0)
                homography, _ = cv2.findHomography(anchor_axial_sorted, rolled, 0)
                if homography is None:
                    continue
                if _verify_homography(image, rgb, radius, homography, template):
                    return DetectionResult(radius=radius, homography=homography)

    return None
