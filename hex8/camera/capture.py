"""Real camera capture ingestion + diagnosed decoding (Issue #15).

This module is Phase 4's bridge between *real* image sources (a still image
file on disk, e.g. a phone photo, or a live camera device) and the Phase 2/3
decoder pipeline (:mod:`hex8.decoder.decode`), which already works on any
raster image and already tolerates mild rotation/perspective/blur/JPEG/noise/
brightness degradation via its homography-based fallback detector (Issue
#14). This module does not re-implement any of that: it only adds

1. real image *ingestion* (:func:`capture_from_file`, :func:`capture_from_device`)
   for camera-sourced images, as opposed to :func:`hex8.decoder.decode.decode_file`,
   whose loading logic is scoped to files already on disk, and
2. a diagnosed decode wrapper (:func:`decode_with_diagnostics`) that
   categorizes *why* a real-world capture failed to decode into a small,
   camera-specific failure taxonomy (:class:`FailureCategory`), and logs it
   via the stdlib `logging` module, so a human running a real-world capture
   session (screen-to-camera, print-to-camera - see
   ``docs/phase4-manual-test-guide.md``) gets an actionable diagnosis
   instead of a bare traceback.

Failure taxonomy: which exception maps to which category
----------------------------------------------------------
:func:`hex8.decoder.decode.decode_image` calls, in order:
``detect_marker`` -> ``build_layout`` -> ``classify_cells`` (METADATA) ->
``classify_cells`` (DATA) -> ``decode_marker`` (header + payload). Every
failure mode surfaced by that pipeline is a ``ValueError`` (verified by
reading ``hex8/decoder/detect.py``, ``hex8/decoder/correct.py``,
``hex8/common/header.py``, and ``hex8/common/ecc.py``), each with a
distinct, stable message prefix used here to categorize it:

- :func:`hex8.decoder.detect.detect_marker` raises
  ``ValueError("no matching Hex8 marker grid found for this image size ...")``
  when neither the ideal fast path nor the Issue #14 homography fallback
  locates a marker at all -> :attr:`FailureCategory.NO_MARKER_DETECTED`.
- :func:`hex8.common.header.unpack` (called from
  :func:`hex8.decoder.correct.decode_header`) raises ``ValueError`` with one
  of ``"Truncated HX8M header"``, ``"Bad HX8M magic"``, or
  ``"Unsupported HX8M version"`` when the recovered METADATA bytes do not
  form a valid HX8M header -> :attr:`FailureCategory.HEADER_INVALID`.
- :func:`hex8.common.ecc.decode` raises ``ValueError`` prefixed
  ``"Reed-Solomon decoding failed: ..."`` (wrapping the underlying
  ``reedsolo.ReedSolomonError``, e.g. ``"Too many errors to correct"``) when
  a block's corruption exceeds Reed-Solomon's correction capacity,
  and other structural ``ValueError``\\ s (``"Invalid ECC header magic"``,
  ``"Encoded data is too short to contain a valid ECC header"``, a body
  length mismatch, or an ``interleave`` mismatch) when the ECC framing
  itself is unreadable; :func:`hex8.decoder.correct.decode_payload` itself
  raises ``"Not enough data cell classifications: ..."`` when the header's
  ``encoded_length`` calls for more DATA cells than the grid has. All of
  these are pre-CRC, RS-stage failures -> :attr:`FailureCategory.RS_CORRECTION_FAILED`.
- :func:`hex8.decoder.correct.decode_payload` raises ``ValueError`` prefixed
  ``"CRC32 mismatch: ..."`` when Reed-Solomon *did* produce a payload but its
  CRC32 does not match the header's recorded checksum (RS "succeeded" on
  corrupted data that happened to still form a valid codeword, or the header
  itself was tampered with) -> :attr:`FailureCategory.CRC_MISMATCH`.
- Anything else (a non-``ValueError`` exception, or a ``ValueError`` whose
  message does not match any of the above - defensive fallback, not expected
  to be reachable given the modules read above) ->
  :attr:`FailureCategory.UNKNOWN`.

This is message-based categorization, not a new set of typed exceptions
raised deep in the decoder: the decoder modules (out of scope for Issue
#15 - see the parent task) do not expose distinct exception types per
failure mode, and inventing one now would mean duplicating
:func:`hex8.decoder.decode.decode_image`'s call sequence here just to get
per-stage `try`/`except` boundaries. The message prefixes above are stable,
descriptive literals baked into the decoder modules' own source (quoted
above), not incidental phrasing, so matching on them is robust to reads
of those modules' current implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
from PIL import Image

from hex8.decoder.decode import decode_image

__all__ = [
    "CaptureResult",
    "FailureCategory",
    "capture_from_device",
    "capture_from_file",
    "decode_file_with_diagnostics",
    "decode_with_diagnostics",
]

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    """Why a real-world capture failed to decode (or that it succeeded).

    Distinct from the *synthetic* degradation labels used by
    :mod:`hex8.degrade` (rotation, blur, jpeg, ...): those describe an
    artificial transform applied to a pristine render for testing. This
    taxonomy instead describes *where in the decode pipeline* a real capture
    failed, which is what a human debugging a bad photo (specular
    highlight, moire from a screen's pixel grid, focus blur, motion blur,
    an off-axis angle, ...) can actually act on - see the module docstring
    for exactly which exception/message maps to which category.
    """

    #: Decoding succeeded; `CaptureResult.payload` holds the recovered bytes.
    NONE = "none"
    #: `detect_marker` found no marker grid consistent with the image at all
    #: (e.g. too much perspective/blur/glare to locate the finder anchors,
    #: or the image simply does not contain a Hex8 marker).
    NO_MARKER_DETECTED = "no_marker_detected"
    #: A marker grid was located and cells were classified, but the
    #: recovered METADATA bytes do not form a valid HX8M header (bad magic
    #: or unsupported version) - typically means the METADATA region itself
    #: was misread (e.g. localized glare/moire over the header cells).
    HEADER_INVALID = "header_invalid"
    #: The header was valid, but Reed-Solomon could not correct the DATA
    #: region's errors (or the ECC framing itself was unreadable) - the
    #: photo's corruption (blur, glare, moire, low resolution, ...) exceeded
    #: the marker's configured `ecc_level` correction capacity.
    RS_CORRECTION_FAILED = "rs_correction_failed"
    #: Reed-Solomon returned a payload, but it fails the CRC32 check in the
    #: header - a rare "quietly wrong" RS correction, or a tampered/corrupted
    #: header's `crc32` field.
    CRC_MISMATCH = "crc_mismatch"
    #: Any other, unanticipated failure (defensive fallback).
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CaptureResult:
    """The outcome of attempting to decode one real-world captured image.

    Attributes:
        payload: The recovered payload bytes if `success` is True, else
            `None`.
        success: Whether decoding succeeded end-to-end.
        failure_category: :attr:`FailureCategory.NONE` on success, else the
            categorized reason for failure.
        error: `str(exception)` describing the failure, or `None` on
            success.
    """

    payload: bytes | None
    success: bool
    failure_category: FailureCategory
    error: str | None


# Stable message-prefix markers used to categorize a caught ValueError (see
# the module docstring's "Failure taxonomy" section for exactly which
# source module/line raises each one).
_NO_MARKER_MARKERS = ("no matching Hex8 marker grid found",)
_HEADER_INVALID_MARKERS = (
    "Truncated HX8M header",
    "Bad HX8M magic",
    "Unsupported HX8M version",
)
_CRC_MISMATCH_MARKERS = ("CRC32 mismatch",)
_RS_CORRECTION_FAILED_MARKERS = (
    "Reed-Solomon decoding failed",
    "Not enough data cell classifications",
    "Invalid ECC header magic",
    "Encoded data is too short to contain a valid ECC header",
    "does not match expected",
    "does not match the value used at encode time",
)


def _categorize_value_error(message: str) -> FailureCategory:
    """Map a caught ValueError's message to a :class:`FailureCategory`."""
    if any(marker in message for marker in _NO_MARKER_MARKERS):
        return FailureCategory.NO_MARKER_DETECTED
    if any(marker in message for marker in _CRC_MISMATCH_MARKERS):
        return FailureCategory.CRC_MISMATCH
    if any(marker in message for marker in _HEADER_INVALID_MARKERS):
        return FailureCategory.HEADER_INVALID
    if any(marker in message for marker in _RS_CORRECTION_FAILED_MARKERS):
        return FailureCategory.RS_CORRECTION_FAILED
    return FailureCategory.UNKNOWN


def decode_with_diagnostics(image: Image.Image) -> CaptureResult:
    """Attempt to decode a Hex8 marker from `image`, diagnosing any failure.

    Wraps :func:`hex8.decoder.decode.decode_image`: on success, returns a
    :class:`CaptureResult` carrying the recovered payload. On failure, logs
    the exception (via this module's `logging.Logger`, at `WARNING` for the
    expected `ValueError` failure modes and `ERROR` with a traceback for
    anything unexpected) and returns a `CaptureResult` categorizing why -
    see the module docstring for the full mapping from exception/message to
    :class:`FailureCategory`.

    This never raises: every failure the underlying decode pipeline can
    produce is caught and reported via the returned `CaptureResult`, which
    is the point of this wrapper for a real capture session where a human
    needs to know *why* a photo didn't decode, not just that it didn't.
    """
    try:
        payload = decode_image(image)
    except ValueError as exc:
        category = _categorize_value_error(str(exc))
        logger.warning("Hex8 decode failed (%s): %s", category.value, exc)
        return CaptureResult(
            payload=None, success=False, failure_category=category, error=str(exc)
        )
    except Exception as exc:  # noqa: BLE001 - genuinely any other failure is UNKNOWN
        logger.exception("Hex8 decode failed with an unexpected exception")
        return CaptureResult(
            payload=None,
            success=False,
            failure_category=FailureCategory.UNKNOWN,
            error=str(exc),
        )

    logger.info("Hex8 decode succeeded (%d payload byte(s))", len(payload))
    return CaptureResult(
        payload=payload, success=True, failure_category=FailureCategory.NONE, error=None
    )


def capture_from_file(path: str | Path) -> Image.Image:
    """Load an image file (e.g. a photo taken by a phone/webcam) as a Pillow Image.

    A thin, real wrapper around Pillow's own loader: unlike
    :func:`hex8.decoder.decode.decode_file`, this makes no assumption that
    `path` is one of *our own* encoder's output formats (in particular, no
    ``.svg`` special-casing - a real camera/phone photo is always a raster
    format Pillow can open directly, e.g. JPEG or PNG).

    Args:
        path: Path to a raster image file.

    Returns:
        The loaded image, converted to RGB.

    Raises:
        FileNotFoundError: if `path` does not exist.
        PIL.UnidentifiedImageError: if `path` is not a format Pillow can
            decode.
    """
    with Image.open(Path(path)) as image:
        return image.convert("RGB")


def decode_file_with_diagnostics(path: str | Path) -> CaptureResult:
    """Load an image file and attempt a diagnosed Hex8 decode of it.

    Composes :func:`capture_from_file` and :func:`decode_with_diagnostics`:
    the file-based counterpart to a live :func:`capture_from_device` capture,
    and the recommended entry point for decoding a saved camera/phone photo
    with failure diagnostics (see ``docs/phase4-manual-test-guide.md``).
    """
    image = capture_from_file(path)
    return decode_with_diagnostics(image)


def capture_from_device(device_index: int = 0, warmup_frames: int = 5) -> Image.Image:
    """Grab a single still frame from a live camera device as a Pillow Image.

    Opens camera `device_index` via `cv2.VideoCapture`, reads and discards
    `warmup_frames` initial frames (a real webcam consideration: auto-
    exposure and autofocus typically need a handful of frames to settle
    after the device is opened, so the very first frame is often
    under/over-exposed or out of focus), then reads and returns one more
    frame as the actual capture.

    Args:
        device_index: OS camera device index, as passed to
            `cv2.VideoCapture` (`0` is normally the default/first camera).
        warmup_frames: Number of initial frames to read and discard before
            capturing the frame that is returned. Must be >= 0.

    Returns:
        The captured frame as a Pillow RGB Image.

    Raises:
        ValueError: if `warmup_frames` is negative.
        RuntimeError: if the device cannot be opened, or if a frame cannot
            be read (during warmup or for the final captured frame) - e.g.
            because no camera hardware is present, the device index is
            invalid, or another process holds the device.
    """
    if warmup_frames < 0:
        raise ValueError(f"warmup_frames must be >= 0, got {warmup_frames!r}")

    capture = cv2.VideoCapture(device_index)
    try:
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open camera device {device_index}: no such device, "
                "or it is already in use by another process."
            )

        frame = None
        for frame_number in range(warmup_frames + 1):
            ok, frame = capture.read()
            if not ok:
                stage = "warmup" if frame_number < warmup_frames else "capture"
                raise RuntimeError(
                    f"Failed to read a frame from camera device {device_index} "
                    f"during {stage} (frame {frame_number + 1} of {warmup_frames + 1})."
                )

        assert frame is not None  # the loop above always runs >= 1 iteration
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    finally:
        capture.release()
