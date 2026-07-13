# Phase 4 Manual Test Guide: Real Camera Capture Validation

## Status: not yet run against real photos

Everything in this guide describes validation that **this development
session could not perform**, because the sandboxed dev environment this
code was written in has no physical webcam, phone camera, or printer
attached to it. `hex8/camera/capture.py` (Issue #15) and its test suite
(`tests/camera/test_capture.py`) are genuinely complete, working code,
verified as thoroughly as possible without physical hardware: file-based
capture, diagnosed decoding, and failure categorization are all exercised
against real rendered/tampered marker images, and live device capture
(`capture_from_device`) is exercised against its honest "no such device"
failure path (see the note in that test about this sandbox's virtual
`v4l2loopback` device below).

**None of this substitutes for Issue #15's actual acceptance criterion** -
"at least one real screen-captured and one real printed-and-photographed
marker decode successfully end-to-end" - which requires a human with real
hardware. This document is that human's runbook. Issue #16 is the follow-up
that will exercise this more broadly (multiple devices/lighting conditions);
this guide covers the minimum single-photo validation for Issue #15 itself.

## Sandbox note: a virtual camera device is present, but it is not real hardware

While writing `tests/camera/test_capture.py`, `cv2.VideoCapture(0)` was
found to unexpectedly succeed in this specific sandbox: `v4l2-ctl
--list-devices` reports a device named `AvataCam
(platform:v4l2loopback-000)` at `/dev/video0`/`/dev/video1`, which OpenCV
opens and reads frames from without error. This is some sandbox-provided
synthetic/virtual video source (a `v4l2loopback` device, not physical camera
hardware), and its frames are not photographs of anything - they must not be
treated as, or reported as, a real camera capture. The test suite avoids
this device entirely (it tests the failure path via a deliberately
out-of-range device index instead) specifically so it does not
accidentally rely on this artifact. If you (the human running real
validation) are working in an environment where an index like `0` maps to
this same kind of virtual/synthetic device rather than your real webcam,
double check `v4l2-ctl --list-devices` (Linux) or your OS's camera device
list before assuming `device_index=0` is your physical camera.

## Step 1: Generate a marker

```sh
echo "hello hex8" > payload.txt
hex8 encode payload.txt marker.png --radius 18 --ecc-level 30
```

`--radius 18` and `--ecc-level 30` match the README's recommended PoC
target configuration (`R = 18-20`, `ECC = 25%-40%`). Adjust `--cell-size` if
you need a physically larger printed marker (default is `10.0` px per
cell's center-to-vertex distance; this only affects the rendered image's
pixel dimensions, not what gets captured).

## Step 2: Capture it with a real camera

Two capture scenarios, per Issue #15's scope:

1. **Screen-to-camera**: display `marker.png` at a reasonable size on a
   monitor or phone screen, then photograph the screen with a *different*
   device's camera (e.g. photograph a laptop screen with a phone). Keep the
   marker reasonably large and well-lit; avoid screen glare directly over
   the marker (specular highlights are one of the real-world failure modes
   this pipeline logs distinctly from synthetic degradation - see below).
2. **Printed-and-photographed**: print `marker.png` on paper (a normal
   office printer is fine), then photograph the printout with a phone
   camera under normal indoor lighting.

For both, aim for roughly head-on framing at first (a moderate angle is
fine and expected to work - `detect_marker`'s homography fallback tolerates
perspective warp up to roughly `strength <= 0.05`, see
`hex8/decoder/detect.py`'s module docstring - but a very steep/oblique
angle is a known, honest limit, not a bug). Save the photo (JPEG is fine;
`capture_from_file` uses Pillow, which reads any format Pillow supports) to
a file.

## Step 3: Decode the photo

Either the CLI (works as-is; `hex8 decode` already handles any raster image
Pillow can open, including a real photo, with no camera-specific code
needed) or the new diagnosed decode path:

```sh
hex8 decode photo.jpg recovered.bin
diff recovered.bin payload.txt && echo "MATCH"
```

```python
from hex8.camera.capture import decode_file_with_diagnostics

result = decode_file_with_diagnostics("photo.jpg")
print(result.success, result.failure_category, result.error)
if result.success:
    with open("recovered.bin", "wb") as f:
        f.write(result.payload)
```

`decode_file_with_diagnostics` is `capture_from_file` + `decode_with_diagnostics`
composed; use it (rather than the plain `hex8 decode` CLI) whenever you want
the failure category on a bad capture instead of just an error string. A new
`hex8` CLI subcommand was deliberately **not** added for this: `hex8 decode`
already works on any raster image including real photos (Issue #12/#14), so
the only genuinely new capability here - failure categorization - is exposed
directly from Python, where a human debugging a real capture session is
expected to be running interactively (e.g. a notebook or REPL) rather than
scripting shell invocations per photo.

For a live camera device instead of a saved file:

```python
from hex8.camera.capture import capture_from_device, decode_with_diagnostics

image = capture_from_device(device_index=0, warmup_frames=5)  # verify your
                                                                 # device index first - see the sandbox note above
result = decode_with_diagnostics(image)
print(result.success, result.failure_category, result.error)
```

## Step 4: If it fails, read `result.failure_category`

`decode_with_diagnostics` / `decode_file_with_diagnostics` log the failure
(via Python's `logging` module, logger name `hex8.camera.capture`) and
return a `CaptureResult` whose `failure_category` is one of
(`hex8.camera.capture.FailureCategory`):

| `failure_category`      | What it means                                                                 | What to check                                                                                                                                                                                  |
|--------------------------|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `NO_MARKER_DETECTED`     | No marker grid could be located at all.                                        | The most likely real-world cause given `hex8/decoder/detect.py`'s documented mild-degradation thresholds: perspective angle beyond `~0.05` strength-equivalent, blur beyond `~0.25x cell_size`, or the marker is simply out of frame/too small. Reframe more head-on, move closer, or improve focus, then retake. |
| `HEADER_INVALID`         | A marker was found and cells were classified, but the METADATA region's bytes don't form a valid header. | A localized defect (glare, moire from a screen's pixel grid interacting with the marker's own grid, a crease if printed) sitting specifically over the METADATA cells. Reposition to avoid glare directly on that region, or increase `cell_size` so METADATA cells are larger targets for the camera's sensor. |
| `RS_CORRECTION_FAILED`   | The header was fine, but Reed-Solomon couldn't correct enough of the DATA region. | The photo's overall corruption (focus blur, motion blur, moire, low resolution, JPEG compression) exceeded the marker's `--ecc-level` correction budget. Re-encode with a higher `--ecc-level` (up to 40), hold the camera steadier, or improve lighting/focus.                                              |
| `CRC_MISMATCH`           | Reed-Solomon returned a payload, but its checksum doesn't match the header.     | Rare in practice; suggests RS "corrected" onto a wrong-but-valid-looking codeword, or the header itself is tampered/corrupted in a way that still parsed as valid. Retake the photo; if persistent, treat as a bug and file a follow-up issue.                                                |
| `UNKNOWN`                | Anything else (not expected given the modules this pipeline is built on).      | File an issue with the photo and the logged exception; this is a gap in the failure taxonomy, not an expected outcome.                                                                        |

These categories describe *where in the decode pipeline* a real capture
failed - they are deliberately distinct from the *synthetic* degradation
labels used by `hex8.degrade` (`rotation`, `blur`, `jpeg`, ...), since a real
photo's specular highlights, screen moire, or focus blur don't map 1:1 to
any single synthetic degradation type; `NO_MARKER_DETECTED` /
`HEADER_INVALID` / `RS_CORRECTION_FAILED` / `CRC_MISMATCH` are what you can
actually act on when debugging a bad real-world capture.

## What happens next (Issue #16)

Once at least one screen-captured and one printed-and-photographed marker
decode successfully via the steps above, append the result (device used,
lighting, distance, angle, and whether it succeeded/what
`failure_category` it hit) as a comment on Issue #15, closing out its
acceptance criterion. Issue #16 is the broader field test across multiple
devices and lighting conditions that exercises this pipeline (and the
failure taxonomy above) much more thoroughly than this single-photo
validation does.
