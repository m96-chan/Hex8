# Hex8 Marker PoC

**Hex8 Marker** is a proof-of-concept for a high-density, camera-readable 2D marker based on **8-color hexagonal cells**.

The goal is to explore whether a custom marker can store more data than a traditional black-and-white QR code while keeping the design visually distinct, robust, and extensible.

This project is experimental.

---

## Concept

Hex8 Marker uses a flat-top hexagonal grid.

Each hex cell stores one of 8 colors.

```text
8 colors = 3 bits / cell
```

Compared with a binary black-and-white cell:

```text
2 colors = 1 bit / cell
8 colors = 3 bits / cell
```

This gives a theoretical 3x increase in raw symbol density before applying metadata, finder patterns, color calibration, and error correction.

---

## Design Goals

- Store binary data inside a custom visual marker
- Use 8-color hexagonal cells
- Support camera-based decoding
- Include color calibration cells
- Include error correction
- Be visually distinct from QR codes
- Leave room for future extensions such as:
  - 16-color mode
  - 120-degree split hex cells
  - animated markers
  - chunk bundles
  - neural decoding

---

## Non-Goals

This PoC is not intended to replace QR codes immediately.

It does not aim to be:

- A payment system
- A secure identity system by itself
- A standardized barcode format
- A production-ready offline file transfer format

Security, identity, and transport layers can be added later.

---

## Marker Structure

A Hex8 marker is composed of:

```text
┌────────────────────────────┐
│ finder / palette / timing  │
│                            │
│       hex data field       │
│                            │
│ finder / metadata / ecc    │
└────────────────────────────┘
```

The recommended PoC layout is:

```text
Hex8 Marker v0

- Shape: flat-top hexagon grid
- Color count: 8
- Bits per cell: 3
- Finder anchors: 6 outer anchor markers or 3 major anchors
- Palette cells: repeated 8-color reference cells
- Metadata: version, radius, payload length, ECC level
- Payload: binary data
- Error correction: Reed-Solomon
- Checksum: CRC32 or CRC32C
```

---

## Color Palette

Initial PoC palette:

| Symbol | Bits | Color |
|---|---:|---|
| 0 | `000` | Black |
| 1 | `001` | White |
| 2 | `010` | Red |
| 3 | `011` | Green |
| 4 | `100` | Blue |
| 5 | `101` | Cyan |
| 6 | `110` | Magenta |
| 7 | `111` | Yellow |

Example RGB values:

```text
Black   #000000
White   #FFFFFF
Red     #FF0000
Green   #00FF00
Blue    #0000FF
Cyan    #00FFFF
Magenta #FF00FF
Yellow  #FFFF00
```

For real-world camera decoding, fixed RGB values should not be trusted.

The decoder should read palette reference cells inside the marker and classify colors relative to the observed palette.

---

## Why Hexagons?

Hexagonal cells provide:

- A visually distinct marker shape
- Six-way neighborhood relationships
- Natural radial and spiral layouts
- Good compatibility with central-priority layouts
- Future extensibility for split-cell symbols

However, hexagons are harder to decode than square grids because they do not align perfectly with raster pixels.

For PoC v0, only the center region of each cell should be sampled.

---

## Capacity Estimate

For a hexagonal grid with radius `R`, the number of cells is:

```text
cells = 1 + 3R(R + 1)
```

Examples:

| Radius | Cells | Raw capacity at 3 bits/cell |
|---:|---:|---:|
| 10 | 331 | 993 bits / 124 bytes |
| 20 | 1261 | 3783 bits / 472 bytes |
| 40 | 4921 | 14763 bits / 1845 bytes |
| 60 | 10981 | 32943 bits / 4117 bytes |

Actual usable payload is smaller because the marker needs:

- Finder cells
- Palette cells
- Metadata
- Error correction
- Interleaving
- Checksum

A realistic first PoC target:

```text
R = 18 to 20
Payload target = 128 to 256 bytes
ECC = 25% to 40%
```

---

## Encoding Pipeline

```text
input bytes
  ↓
optional compression
  ↓
header + payload length
  ↓
CRC32 / CRC32C
  ↓
Reed-Solomon ECC
  ↓
interleaving
  ↓
3-bit symbol stream
  ↓
8-color hex cell mapping
  ↓
PNG / SVG output
```

---

## Decoding Pipeline

```text
input image
  ↓
detect marker region
  ↓
estimate orientation and perspective
  ↓
normalize grid coordinates
  ↓
read palette reference cells
  ↓
sample each hex cell center
  ↓
classify each cell into one of 8 colors
  ↓
recover 3-bit symbols
  ↓
deinterleave
  ↓
Reed-Solomon correction
  ↓
CRC verification
  ↓
output bytes
```

---

## Header Format Draft

A possible binary header:

| Field | Size | Description |
|---|---:|---|
| Magic | 4 bytes | `HX8M` |
| Version | 1 byte | Marker format version |
| Flags | 1 byte | Compression, signature, reserved |
| Radius | 1 byte | Hex grid radius |
| ECC Level | 1 byte | Error correction level |
| Payload Length | 4 bytes | Original payload size |
| Encoded Length | 4 bytes | Payload after compression/ECC |
| CRC32 | 4 bytes | Checksum of original payload |

Draft layout:

```text
HX8M
version:        u8
flags:          u8
radius:         u8
ecc_level:      u8
payload_length: u32
encoded_length: u32
crc32:          u32
payload:        bytes
ecc:            bytes
```

---

## Finder and Calibration

The marker should include:

### Finder Anchors

Used to detect position, scale, orientation, and perspective.

Possible designs:

- 3 large anchor clusters
- 6 outer vertex anchors
- outer timing ring
- center orientation core

### Palette Cells

Used for color calibration.

The marker should contain repeated examples of all 8 colors.

Example:

```text
K W R G B C M Y
K W R G B C M Y
```

The decoder should estimate observed colors from these cells and classify data cells using color distance in a perceptual color space such as Lab.

---

## Color Classification

Do not classify by fixed RGB thresholds.

Recommended approach:

1. Sample observed palette cells
2. Convert samples to Lab color space
3. Convert cell samples to Lab
4. Find nearest observed palette color
5. Return both symbol and confidence

Pseudo-code:

```python
symbol = argmin(distance_lab(sample, palette[i]))
confidence = margin_between_best_and_second_best(sample, palette)
```

Low-confidence cells may be treated as erasures for Reed-Solomon correction.

---

## Error Correction Strategy

PoC v0 should use Reed-Solomon.

Recommended:

```text
ECC rate: 25% to 40%
Interleaving: enabled
Checksum: CRC32 or CRC32C
```

Why:

- Color errors usually happen at symbol level
- Local damage should be spread across blocks
- Low-confidence cells can be marked as erasures

---

## Implementation Plan

### Phase 1: Encoder Only (done)

- Encode arbitrary bytes
- Generate ideal PNG/SVG marker
- No decoding yet

Success condition:

```text
payload bytes -> marker image
```

### Phase 2: Ideal Decoder (done)

- Decode generated PNG directly
- No camera distortion
- No blur
- No perspective correction

Success condition:

```text
payload bytes -> marker image -> same payload bytes
```

### Phase 3: Synthetic Degradation (done)

Test with:

- rotation
- scaling
- blur
- JPEG compression
- noise
- brightness changes
- perspective warp

Success condition:

```text
decoder survives mild image degradation
```

### Phase 4: Camera Test (pipeline built, pending real-hardware validation)

Test:

- screen-to-camera
- print-to-camera
- different lighting
- different phones

Success condition:

```text
real camera image can be decoded
```

---

## Recommended Tech Stack

Encoder:

- Python
- Pillow
- Plain SVG output (hand-generated XML from hex-vertex math, not CairoSVG -
  avoids a native libcairo dependency; see `hex8/encoder/render.py`)
- NumPy

Decoder:

- Python
- OpenCV
- NumPy
- scikit-image

ECC:

- Reed-Solomon library
- Later: LDPC or Fountain/Raptor-style coding

---

## Future Extensions

### 16-Color Mode

```text
16 colors = 4 bits / cell
```

Useful mainly for screen-based markers.

### 120-Degree Split Hex Cells

Each hex cell is split into 3 sectors.

With 8 colors:

```text
8 × 8 × 8 = 512 states = 9 bits / cell
```

A practical version may use only 256 safe symbols:

```text
1 split hex cell = 1 byte
```

### Animated Marker

Use multiple frames to transfer larger payloads.

```text
frame 1 -> chunk A
frame 2 -> chunk B
frame 3 -> chunk C
```

### Bundle / Chunk Mode

Split large files across many markers.

Recommended approach:

- zstd compression
- chunk splitting
- Fountain Code or Reed-Solomon parity
- hash verification
- optional signature

### Neural Decoder

Train a model using synthetic degradation:

- blur
- lighting changes
- perspective warp
- color shift
- print artifacts
- camera noise

Output:

```text
cell symbol + confidence
```

---

## Example Use Cases

- Event pass marker
- Offline data token
- VRChat world or asset tag
- Physical object metadata
- Art-like data marker
- High-density visual manifest
- Multi-marker file bundle

---

## Current Status

Phases 1-3 complete and tested; Phase 4 pipeline built, pending real-world
validation. A live camera demo (Issue #18) is also done.

```text
Hex8 Marker v0
R = 18 to 20
8 colors
single-color hex cells
Reed-Solomon ECC
PNG/SVG encoder
ideal + mild-degradation-tolerant decoder
```

- **Phase 1 (Encoder)**: done. `hex8 encode <payload> marker.png` (or
  `.svg`) via `hex8.encoder.encode`. SVG is emitted as plain XML, not via
  CairoSVG - see "Recommended Tech Stack" below for why.
- **Phase 2 (Ideal decoder)**: done. `hex8 decode marker.png out.bin`
  round-trips exactly for a pristine render.
- **Phase 3 (Synthetic degradation)**: done. `hex8.degrade` provides the 7
  degradation types plus a harness; `hex8.decoder.detect` now falls back to
  a homography-based detector (anchor correspondence + perspective
  transform) when the exact ideal path fails, recovering the payload under
  documented "mild" thresholds - see `hex8/decoder/detect.py`'s module
  docstring and `docs/phase3-baseline.md` for the full before/after
  numbers (15/40 -> 34/40 synthetic test cases).
- **Phase 4 (Camera test)**: pipeline built (`hex8.camera.capture`:
  file/live-device ingestion + failure-categorized decoding), but **not yet
  validated against a real camera or printer** by a human running the
  Issue #15/#16 field-test steps. (Earlier revisions of this README claimed
  this project's dev environment has no camera hardware at all; that turned
  out to be environment-dependent - some sandboxes for this project do have
  a real USB camera and a working display attached, see
  `docs/phase4-manual-test-guide.md`'s sandbox note - but that is not the
  same as the acceptance criterion being met.) See
  `docs/phase4-manual-test-guide.md` for the steps to run that validation
  with real hardware.
- **Live demo** (Issue #18): done. `hex8 live-demo [--device N]` opens a
  live camera preview window and overlays the decode result (marker
  outline + recovered payload or failure reason) on each frame in real
  time - green outline + decoded text (UTF-8, or a hex dump if not valid
  UTF-8) on success, red outline (or none, if no marker was found at all) +
  failure category on failure. The status text is drawn on a solid dark
  background so it stays legible regardless of what's under it (a plain
  `cv2.putText` call alone is invisible against a marker's own white
  background). Requires the separate `demo` extras group (`pip install
  hex8[demo]`), which installs plain `opencv-python` (unlike the `decoder`
  extras group's `opencv-python-headless`, this build supports GUI windows
  via `cv2.imshow`) plus `scikit-image`, so `demo` alone is enough to run
  the full decode pipeline without also installing `decoder`. The `decoder`
  and `demo` groups are mutually exclusive; do not install both in the same
  environment (their `opencv-python`/`opencv-python-headless` builds
  conflict at the package level). Manually verified end-to-end in this
  project's sandbox by feeding a rendered marker into its `v4l2loopback`
  device and confirming the preview window's overlay via screenshot - see
  `docs/phase4-manual-test-guide.md`'s "Live demo" section; this is not a
  substitute for validation against real external camera hardware (still
  Issue #16's job).
- **Cosmetic issue found while verifying Issue #18** (tracked as Issue
  #19, not fixed): a marker whose payload is much smaller than its
  radius's data-cell capacity renders with a large contiguous *blank*
  region on one side, because unused `DATA` cells are filled from a fixed
  `(q, r)`-ordered position rather than scattered evenly - see Issue #19
  for the root cause. Purely visual; `decode_image` round-trips correctly
  regardless, and it does not affect payloads near the README's own
  "realistic PoC target" (128-256 bytes at `R = 18-20`).
- One capacity note found while implementing: at the default 30% ECC rate,
  a 256-byte payload needs `R = 20` (not `R = 18`) once Reed-Solomon parity
  and framing overhead are accounted for - see `docs/marker-layout.md` for
  the exact per-radius capacity table.
- Finder anchor design decision (README originally left this open): **6
  outer vertex anchors**, not 3 major anchor clusters - see
  `docs/marker-layout.md`.

Run `pip install -e ".[decoder,dev]"` then `pytest` to verify locally.

---

## License

TBD.

---

## Name Ideas

- Hex8 Marker
- HexaCode
- SAXIA Marker
- Hex Sigil
- PrismHex
- Data Hive
