"""Phase 3 synthetic degradation library + test harness (Issue #13).

This module provides:

- Seven ``apply_<name>`` functions, one per degradation type listed in the
  README's Phase 3 section (rotation, scaling, blur, JPEG compression,
  noise, brightness changes, perspective warp). Each takes a Pillow
  ``Image`` and a ``severity`` float whose meaning is documented on the
  function itself, and returns a new (degraded) Pillow ``Image``.
- A ``DEGRADATIONS`` registry mapping name -> ``apply_`` function, and an
  ``apply_combined`` helper to chain several degradations in sequence, so
  the harness can exercise degradations "independently and in combination"
  (this issue's acceptance criterion).
- A ``run_harness`` function that encodes a payload once via
  :func:`hex8.encoder.encode.encode_png`, degrades the resulting image per
  a list of named cases, and attempts :func:`hex8.decoder.decode.decode_image`
  on each degraded image, recording whether it succeeded *and* recovered
  the exact original payload.
- A ``format_report`` function rendering harness results as a markdown
  table, used to produce the committed baseline doc
  (``docs/phase3-baseline.md``).

Scope note: this module does not attempt to make the decoder more robust.
:mod:`hex8.decoder.detect` currently only handles the ideal, zero-
distortion case (see its module docstring) - no rotation/perspective
correction, no color-distance tolerance. Geometric/photometric hardening
is Issue #14's job. Most degraded cases here are therefore *expected* to
fail against today's decoder; that is the correct, honest baseline this
harness exists to measure and record, not a bug to paper over.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageFilter

from hex8.decoder.decode import decode_image
from hex8.encoder.encode import encode_png

__all__ = [
    "DEGRADATIONS",
    "HarnessResult",
    "apply_blur",
    "apply_brightness",
    "apply_combined",
    "apply_jpeg_compression",
    "apply_noise",
    "apply_perspective_warp",
    "apply_rotation",
    "apply_scaling",
    "format_report",
    "run_harness",
]


def _to_bgr_array(image: Image.Image) -> np.ndarray:
    """Convert a Pillow RGB image to an OpenCV-style BGR ``uint8`` array."""
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _from_bgr_array(array: np.ndarray) -> Image.Image:
    """Convert an OpenCV-style BGR ``uint8`` array back to a Pillow RGB image."""
    rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb, mode="RGB")


def apply_rotation(image: Image.Image, degrees: float) -> Image.Image:
    """Rotate `image` about its center by `degrees` (counter-clockwise, positive).

    `degrees` is an angle in degrees. `0.0` is a no-op (identity). The output
    canvas is expanded as needed so the full rotated content stays visible
    (no cropping); newly-exposed corners are filled white, matching the
    marker's background/quiet-zone color.

    Uses OpenCV's ``warpAffine`` for interpolation control.
    """
    if degrees % 360.0 == 0.0:
        return image.copy()

    array = _to_bgr_array(image)
    h, w = array.shape[:2]
    center = (w / 2.0, h / 2.0)

    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)

    # Expand the output canvas so the rotated content is not clipped.
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]

    rotated = cv2.warpAffine(
        array,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return _from_bgr_array(rotated)


def apply_scaling(image: Image.Image, factor: float) -> Image.Image:
    """Resize `image` by `factor` (e.g. `2.0` doubles width/height, `0.5` halves them).

    `factor` is a linear resize multiplier applied to both width and height
    (uniform scaling). `1.0` is a no-op (identity). Uses OpenCV's ``resize``
    with area interpolation for downscaling and linear interpolation for
    upscaling, matching common camera-capture resampling behavior.
    """
    if factor == 1.0:
        return image.copy()
    if factor <= 0.0:
        raise ValueError(f"scaling factor must be positive, got {factor!r}")

    array = _to_bgr_array(image)
    h, w = array.shape[:2]
    new_w = max(1, round(w * factor))
    new_h = max(1, round(h * factor))

    interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR
    scaled = cv2.resize(array, (new_w, new_h), interpolation=interpolation)
    return _from_bgr_array(scaled)


def apply_blur(image: Image.Image, radius: float) -> Image.Image:
    """Apply a Gaussian blur with the given `radius` in pixels.

    `radius` is the Gaussian blur radius in pixels, as used by Pillow's
    ``ImageFilter.GaussianBlur``. `0.0` is a no-op (identity).
    """
    if radius <= 0.0:
        return image.copy()
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Round-trip `image` through JPEG encoding at the given `quality`.

    `quality` is the JPEG quality setting, `0`-`100` (Pillow/libjpeg scale;
    higher is better quality / less compression). `100` is near-lossless;
    low values introduce visible blocking artifacts. Implemented via an
    in-memory ``io.BytesIO`` round-trip through Pillow's JPEG codec.
    """
    if not (0 <= quality <= 100):
        raise ValueError(f"JPEG quality must be within [0, 100], got {quality!r}")

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as reloaded:
        return reloaded.convert("RGB").copy()


def apply_noise(image: Image.Image, sigma: float) -> Image.Image:
    """Add zero-mean Gaussian noise with standard deviation `sigma` to each pixel.

    `sigma` is the noise standard deviation on the 0-255 pixel scale
    (applied identically to each RGB channel). `0.0` is a no-op (identity).
    The noisy result is clipped back to the valid `[0, 255]` range.
    """
    if sigma <= 0.0:
        return image.copy()

    array = np.asarray(image.convert("RGB")).astype(np.float64)
    noise = np.random.default_rng().normal(loc=0.0, scale=sigma, size=array.shape)
    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def apply_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Scale every pixel's intensity by `factor`.

    `factor` is a linear multiplier: `1.0` is a no-op (identity), values
    below `1.0` darken the image, values above `1.0` brighten it. Result is
    clipped back to the valid `[0, 255]` range.
    """
    if factor == 1.0:
        return image.copy()
    if factor < 0.0:
        raise ValueError(f"brightness factor must be non-negative, got {factor!r}")

    array = np.asarray(image.convert("RGB")).astype(np.float64)
    scaled = np.clip(array * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="RGB")


def apply_perspective_warp(image: Image.Image, strength: float) -> Image.Image:
    """Warp `image` as if viewed from an off-axis angle, by `strength`.

    `strength` is a fraction of the image's width/height (`0.0`-`~0.5`
    sensible range) used to displace three of the four corners inward
    toward the image center, simulating an off-axis camera viewpoint.
    `0.0` is a no-op (identity). The top-left corner is left fixed as an
    anchor so the warp is not just a uniform rescale. Uses OpenCV's
    ``getPerspectiveTransform`` / ``warpPerspective``; newly-exposed regions
    are filled white, matching the marker's background/quiet-zone color.
    """
    if strength == 0.0:
        return image.copy()
    if strength < 0.0:
        raise ValueError(f"perspective warp strength must be non-negative, got {strength!r}")

    array = _to_bgr_array(image)
    h, w = array.shape[:2]

    dx = strength * w
    dy = strength * h

    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [0, 0],
            [w - 1 - dx, dy],
            [w - 1 - dx, h - 1 - dy],
            [0, h - 1],
        ]
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        array,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return _from_bgr_array(warped)


#: Registry mapping degradation name -> its ``apply_`` function. Names here
#: are the canonical identifiers used in ``run_harness`` cases and
#: ``apply_combined`` steps.
DEGRADATIONS: dict[str, Callable[[Image.Image, float], Image.Image]] = {
    "rotation": apply_rotation,
    "scaling": apply_scaling,
    "blur": apply_blur,
    "jpeg": apply_jpeg_compression,
    "noise": apply_noise,
    "brightness": apply_brightness,
    "perspective": apply_perspective_warp,
}


def apply_combined(image: Image.Image, steps: list[tuple[str, float]]) -> Image.Image:
    """Apply a sequence of named degradations in order, e.g.

    ``[("rotation", 5.0), ("jpeg", 75), ("noise", 10.0)]``.

    Each step's `name` must be a key in :data:`DEGRADATIONS`. An empty
    `steps` list returns the image unchanged (a copy). Order matters:
    applying degradations in a different order can produce a different
    result (e.g. rotating before blurring smears differently than blurring
    before rotating introduces new edges from rotation padding).
    """
    result = image.copy()
    for name, severity in steps:
        if name not in DEGRADATIONS:
            raise ValueError(
                f"Unknown degradation {name!r}; known degradations: {sorted(DEGRADATIONS)}"
            )
        result = DEGRADATIONS[name](result, severity)
    return result


@dataclass(frozen=True)
class HarnessResult:
    """The outcome of running the decoder against one degraded case.

    Attributes:
        degradation: Case label, e.g. ``"rotation"`` or, for combined
            cases, a name like ``"rotation+jpeg+noise"``.
        severity: Human-readable description of the severity/combo used,
            e.g. ``"15.0deg"`` or ``"rotation=5.0,jpeg=75,noise=10.0"``.
        passed: Whether ``decode_image`` succeeded on the degraded image
            *and* recovered exactly the original payload bytes.
        error: ``str(exception)`` if `passed` is False due to an exception
            (detection/decoding failure); ``None`` if `passed` is True.
            Also set (to a descriptive message) if decoding succeeded but
            returned the wrong payload - a wrong-but-non-crashing decode
            counts as a failure, not a silent pass.
    """

    degradation: str
    severity: str
    passed: bool
    error: str | None


def _describe_severity(steps: list[tuple[str, float]]) -> str:
    if not steps:
        return "none"
    return ",".join(f"{name}={severity}" for name, severity in steps)


def run_harness(
    payload: bytes,
    radius: int,
    ecc_level: int,
    cell_size: float,
    cases: list[tuple[str, list[tuple[str, float]]]],
) -> list[HarnessResult]:
    """Run the Phase 2 decoder against a set of synthetically degraded images.

    For each ``(label, steps)`` in `cases`, encode `payload` once via
    :func:`hex8.encoder.encode.encode_png`, apply the named degradation
    step(s) via :func:`apply_combined`, attempt
    :func:`hex8.decoder.decode.decode_image` on the result, and record
    whether it succeeded and recovered the exact original payload (not just
    "didn't raise" - a wrong-but-non-crashing decode counts as a failure).

    Args:
        payload: Payload bytes to encode (same payload reused for every case).
        radius: Hex grid radius passed to ``encode_png``.
        ecc_level: Reed-Solomon ECC rate percentage passed to ``encode_png``.
        cell_size: Cell size in pixels passed to ``encode_png``.
        cases: List of ``(label, steps)`` pairs, where `steps` is a list of
            ``(degradation_name, severity)`` tuples applied in order via
            ``apply_combined`` (an empty list means no degradation at all).

    Returns:
        One :class:`HarnessResult` per case, in the same order as `cases`.
    """
    base_image = encode_png(payload, radius=radius, ecc_level=ecc_level, cell_size=cell_size)

    results: list[HarnessResult] = []
    for label, steps in cases:
        severity_desc = _describe_severity(steps)
        degraded = apply_combined(base_image, steps)

        try:
            decoded = decode_image(degraded)
        except Exception as exc:  # noqa: BLE001 - genuinely any decoder failure counts
            results.append(
                HarnessResult(
                    degradation=label,
                    severity=severity_desc,
                    passed=False,
                    error=str(exc),
                )
            )
            continue

        if decoded == payload:
            results.append(
                HarnessResult(
                    degradation=label,
                    severity=severity_desc,
                    passed=True,
                    error=None,
                )
            )
        else:
            results.append(
                HarnessResult(
                    degradation=label,
                    severity=severity_desc,
                    passed=False,
                    error="decoded payload did not match the original (silent mismatch)",
                )
            )

    return results


def format_report(results: list[HarnessResult]) -> str:
    """Render `results` as a markdown table (degradation, severity, passed, error)."""
    lines = [
        "| degradation | severity | passed | error |",
        "|---|---|---|---|",
    ]
    for result in results:
        error_cell = "" if result.error is None else result.error.replace("|", "\\|")
        lines.append(
            f"| {result.degradation} | {result.severity} | {result.passed} | {error_cell} |"
        )
    return "\n".join(lines) + "\n"
