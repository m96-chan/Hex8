# Hex8 Marker Cell Layout (Issue #6)

This document describes how `hex8.common.layout.build_layout(radius)` partitions
the hex grid (from `hex8.common.hexgrid`) into finder anchors, palette
calibration cells, metadata cells, and data cells.

## Finder anchors: 6 outer vertex anchors

The README offered two options for finder anchor design: 3 major anchor
clusters, or 6 outer vertex anchors. **This project uses 6 outer vertex
anchors**, chosen for better perspective/rotation estimation, which matters
for Phase 3 (synthetic degradation) and Phase 4 (real camera) robustness.

A hex-shaped grid of radius `R` has 6 corner cells - the axial coordinates
`(R, 0)`, `(R, -R)`, `(0, -R)`, `(-R, 0)`, `(-R, R)`, `(0, R)` (the six
permutations of the cube vector `(R, -R, 0)`). Each corner anchor is a solid
cluster of every cell within hex-distance `ANCHOR_RADIUS` (2) of that corner,
clipped to the grid boundary - roughly a filled "bump" at each of the 6
points of the marker's outer hexagonal silhouette:

```text
              (0,R)
               ▓▓
        (-R,R) ▓▓        ▓▓ (R,0)
         ▓▓                ▓▓
           \    data +     /
            \  palette +  /
             \ metadata  /
           ▓▓                ▓▓
        (-R,0) ▓▓        ▓▓ (R,-R)
               ▓▓
              (0,-R)
```

## Palette cells: 2 repetitions of the 8-color palette

16 cells (`PALETTE_REPEATS=2` x 8 colors) are reserved as color calibration
references, so the decoder can build an observed-palette table (Issue #10)
independent of any single physical color patch.

## Metadata cells: the HX8M header

`METADATA_SYMBOL_COUNT = ceil(HEADER_SIZE * 8 / 3) = 54` cells, enough to
carry the 20-byte HX8M header (see `hex8.common.header`) as 3-bit symbols.

## Assignment order

Palette and metadata cells are not placed at hardcoded coordinates. Instead,
`build_layout`:

1. Claims every cell within `ANCHOR_RADIUS` of one of the 6 corners as `FINDER`.
2. Walks the remaining cells in `hex8.common.hexgrid.enumerate_cells`'s
   deterministic `(q, r)`-sorted order and claims the first 16 as `PALETTE`.
3. Continues the same walk and claims the next 54 as `METADATA`.
4. Everything left over is `DATA`.

Because both the encoder and decoder call `build_layout(radius)` with the
same radius and get the same deterministic partition back, they never need
to share anything beyond the radius itself to agree on which cell holds
what.

## Capacity at the target radii

| Radius | Total cells | Finder | Palette | Metadata | Data cells | Data bytes (raw, pre-ECC) |
|---:|---:|---:|---:|---:|---:|---:|
| 18 | 1027 | 54 | 16 | 54 | 903 | 338 |
| 20 | 1261 | 54 | 16 | 54 | 1137 | 426 |

Both comfortably exceed the 128-256 byte PoC payload target even after
Reed-Solomon ECC overhead (Issue #4) is applied to the data region.

## Small radii

`build_layout` raises `ValueError` if `radius` is too small to fit the
palette and metadata regions after finder anchors are claimed (e.g. `radius=1`).
There is no separate hardcoded minimum-radius constant - the check falls out
naturally from "are there enough unclaimed cells left".
