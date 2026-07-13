"""Command-line entry point for the Hex8 marker encoder (`hex8 encode ...`).

Decoding (`hex8 decode ...`) is added by Issue #12, once the decoder
pipeline exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hex8.encoder.encode import MAX_ECC_LEVEL, MIN_ECC_LEVEL, encode_png, encode_svg

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hex8", description="Hex8 marker encoder/decoder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser(
        "encode", help="Encode a payload file into a Hex8 marker image"
    )
    encode_parser.add_argument("input", type=Path, help="Path to the input payload file")
    encode_parser.add_argument(
        "output", type=Path, help="Output image path (.png or .svg, by extension)"
    )
    encode_parser.add_argument(
        "--radius", type=int, default=18, help="Hex grid radius (default: 18)"
    )
    encode_parser.add_argument(
        "--ecc-level",
        type=int,
        default=30,
        help=f"Reed-Solomon ECC rate as a percentage, {MIN_ECC_LEVEL}-{MAX_ECC_LEVEL} (default: 30)",
    )
    encode_parser.add_argument(
        "--cell-size",
        type=float,
        default=10.0,
        help="Center-to-vertex pixel size of each hex cell (default: 10.0)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    # `encode` is the only subcommand registered so far (add_subparsers'
    # required=True makes argparse itself reject anything else), so no
    # further command dispatch/fallback is needed until `decode` (Issue #12).
    parser = _build_parser()
    args = parser.parse_args(argv)

    payload = args.input.read_bytes()
    suffix = args.output.suffix.lower()

    if suffix == ".svg":
        svg_text = encode_svg(
            payload, radius=args.radius, ecc_level=args.ecc_level, cell_size=args.cell_size
        )
        args.output.write_text(svg_text, encoding="utf-8")
    elif suffix == ".png":
        image = encode_png(
            payload, radius=args.radius, ecc_level=args.ecc_level, cell_size=args.cell_size
        )
        image.save(args.output, format="PNG")
    else:
        print(
            f"hex8: error: unsupported output extension {suffix!r}: expected .png or .svg",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
