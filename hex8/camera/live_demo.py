"""Live camera preview with a real-time marker decode overlay (Issue #18).

This is Phase 4's interactive complement to :mod:`hex8.camera.capture`
(Issue #15, single still-frame capture + diagnosed decode): instead of
capturing and decoding one frame, this module continuously reads frames
from a live camera device and overlays the decode outcome - the detected
marker's outline plus the recovered payload, or a failure reason - on a GUI
preview window in real time.

Split into two layers, mirroring this project's existing testing
convention (see ``tests/camera/test_capture.py``'s module docstring, which
avoids mocking cv2/PIL entirely in favor of real image round-trips):

1. Pure, side-effect-free frame-processing/drawing logic
   (:class:`FrameOverlay`, :func:`marker_outline_points`, :func:`process_frame`,
   :func:`render_overlay`) - unit tested against real synthetic marker
   images in ``tests/camera/test_live_demo.py``, no camera or display
   needed. ``cv2.polylines``/``cv2.putText`` are drawing primitives, not GUI
   calls, so they work with a headless OpenCV build too.
2. A thin, deliberately untested GUI loop (:func:`run_live_demo`) that opens
   a real camera device and a real preview window. This needs a real camera
   and a GUI-capable (non-headless) OpenCV build - the ``demo`` extras group
   installs plain ``opencv-python`` for this, kept separate from the
   ``decoder`` extras group's ``opencv-python-headless`` (the two must never
   be installed together - they conflict at the package level). Verified
   manually per ``docs/phase4-manual-test-guide.md``, not by this test
   suite.

:func:`process_frame` calls both :func:`hex8.decoder.detect.detect_marker`
(for the marker's screen-space geometry) and
:func:`hex8.camera.capture.decode_with_diagnostics` (for the payload and
:class:`hex8.camera.capture.FailureCategory`) because neither alone exposes
both: :func:`hex8.decoder.decode.decode_image` calls `detect_marker`
internally but discards the resulting `DetectionResult` (see that module's
docstring), so there is no existing single call that returns both geometry
and payload/category together.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from hex8.camera.capture import decode_with_diagnostics
from hex8.common.layout import corner_cells
from hex8.decoder.detect import DetectionResult, detect_marker

__all__ = [
    "FrameOverlay",
    "marker_outline_points",
    "process_frame",
    "render_overlay",
    "run_live_demo",
]

#: BGR (OpenCV's channel order), not RGB.
_OUTLINE_COLOR_SUCCESS = (0, 255, 0)
_OUTLINE_COLOR_FAILURE = (0, 0, 255)
_TEXT_COLOR = (255, 255, 255)
_TEXT_ORIGIN = (10, 30)


@dataclass(frozen=True)
class FrameOverlay:
    """What to draw for one processed camera frame.

    Holds no cv2/window state, so it can be constructed and compared in
    tests without a display.

    Attributes:
        outline_points: 6 image-pixel `(x, y)` corner points of the detected
            marker's hex grid, or `None` if no marker was detected at all.
        status_text: Human-readable decode outcome, shown on screen.
        success: Whether the payload was fully recovered.
    """

    outline_points: list[tuple[int, int]] | None
    status_text: str
    success: bool


def marker_outline_points(detection: DetectionResult) -> list[tuple[int, int]]:
    """Return the 6 image-pixel corner points of the detected marker's hex grid.

    Uses the 6 grid-corner axial coordinates from
    :func:`hex8.common.layout.corner_cells`, projected through
    `detection.cell_center`. Deliberately not
    `build_layout(detection.radius).cells_with_role(CellRole.FINDER)`: that
    returns every individual cell inside each of the 6 anchor clusters
    (~54 cells at radius 18), not the 6 outline vertices those clusters
    surround.
    """
    return [
        (round(x), round(y))
        for x, y in (detection.cell_center(q, r) for q, r in corner_cells(detection.radius))
    ]


def _format_payload(payload: bytes) -> str:
    """Render recovered payload bytes for on-screen display.

    Tries UTF-8 text first; falls back to a hex dump for bytes that aren't
    valid UTF-8.
    """
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.hex()


def process_frame(image: Image.Image) -> FrameOverlay:
    """Decide what to overlay on one camera frame. Never raises."""
    try:
        detection = detect_marker(image)
        outline_points: list[tuple[int, int]] | None = marker_outline_points(detection)
    except ValueError:
        outline_points = None

    result = decode_with_diagnostics(image)

    if result.success:
        assert result.payload is not None
        status_text = f"DECODED ({len(result.payload)} bytes): {_format_payload(result.payload)}"
    else:
        status_text = result.failure_category.value.upper()

    return FrameOverlay(
        outline_points=outline_points, status_text=status_text, success=result.success
    )


def render_overlay(frame_bgr: np.ndarray, overlay: FrameOverlay) -> np.ndarray:
    """Draw `overlay` onto a copy of a BGR OpenCV frame array.

    Returns a new array of the same shape/dtype; `frame_bgr` is not
    modified in place.
    """
    frame = frame_bgr.copy()
    color = _OUTLINE_COLOR_SUCCESS if overlay.success else _OUTLINE_COLOR_FAILURE

    if overlay.outline_points is not None:
        points = np.array(overlay.outline_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        overlay.status_text, font, font_scale, thickness
    )
    text_x, text_y = _TEXT_ORIGIN
    # A solid dark background behind the text, so white text stays legible
    # regardless of what's under it - a marker's own white background, a
    # bright real-world scene, etc. (cv2.putText alone has no such backing).
    cv2.rectangle(
        frame,
        (text_x - 4, text_y - text_height - 4),
        (text_x + text_width + 4, text_y + baseline + 4),
        (0, 0, 0),
        thickness=cv2.FILLED,
    )
    cv2.putText(
        frame,
        overlay.status_text,
        _TEXT_ORIGIN,
        font,
        font_scale,
        _TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )
    return frame


def run_live_demo(device_index: int = 0, window_title: str = "Hex8 Live Demo") -> int:
    """Open a live camera preview window with a real-time decode overlay.

    Reads frames continuously from `device_index`, overlays the decode
    outcome on each one, and displays it via `cv2.imshow`. Press 'q' with
    the window focused to quit.

    Not unit tested (see this module's docstring): requires a real camera
    device and a GUI-capable OpenCV build, verified manually per
    ``docs/phase4-manual-test-guide.md``.

    Returns:
        0 on a clean exit, 1 if the camera device could not be opened or a
        frame could not be read.
    """
    capture = cv2.VideoCapture(device_index)
    if not capture.isOpened():
        # Deliberately outside the try/finally below: cv2.destroyAllWindows()
        # is a GUI call that raises on a headless OpenCV build, and no
        # window was ever opened on this path, so it must not run here.
        print(f"hex8: error: could not open camera device {device_index}", file=sys.stderr)
        capture.release()
        return 1

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print(
                    f"hex8: error: failed to read a frame from camera device {device_index}",
                    file=sys.stderr,
                )
                return 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            overlay = process_frame(Image.fromarray(rgb))
            frame = render_overlay(frame, overlay)

            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0
