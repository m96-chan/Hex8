"""3-bit symbol stream <-> 8-color mapping for the Hex8 marker.

This module implements the two mappings that sit at the core of the Hex8
cell encoding:

1. Raw byte stream <-> 3-bit-per-cell symbol stream
   (``bits_to_symbol_stream`` / ``symbol_stream_to_bits``).
2. Symbol (0-7) <-> exact RGB palette color
   (``symbol_to_color`` / ``color_to_symbol``).

Only the ideal, non-camera case is handled here: colors are looked up and
matched by exact RGB value. Tolerant color-distance matching for real
camera captures is out of scope for this module and is handled by the
decoder (see Issue #10).

Padding rule (design decision, fixed by Issue #5)
--------------------------------------------------
Given ``data: bytes``, the total number of bits is ``total_bits =
len(data) * 8``. Bits are read MSB-first across the whole byte stream:
bit 7 of byte 0 first, then bit 6, ... down to bit 0 of byte 0, then bit 7
of byte 1, and so on. Bits are grouped into 3-bit symbols in that reading
order. The number of symbols produced is ``ceil(total_bits / 3)``.

If ``total_bits`` is not a multiple of 3, the final symbol is padded with
zero bits on the right (i.e. the least-significant side of that 3-bit
symbol) to fill it out to 3 bits.

Because the marker header (see Issue #2) separately carries the original
``payload_length`` in bytes, ``symbol_stream_to_bits`` does not need to
infer where the padding starts: it takes the original byte length as an
explicit parameter and reconstructs exactly that many bytes, discarding
any trailing padding bits (and any symbols beyond what is needed).
"""

import math

PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),  # Black    - 000
    1: (255, 255, 255),  # White    - 001
    2: (255, 0, 0),  # Red      - 010
    3: (0, 255, 0),  # Green    - 011
    4: (0, 0, 255),  # Blue     - 100
    5: (0, 255, 255),  # Cyan     - 101
    6: (255, 0, 255),  # Magenta  - 110
    7: (255, 255, 0),  # Yellow   - 111
}

_COLOR_TO_SYMBOL: dict[tuple[int, int, int], int] = {
    color: symbol for symbol, color in PALETTE.items()
}

__all__ = [
    "PALETTE",
    "bits_to_symbol_stream",
    "color_to_symbol",
    "symbol_stream_to_bits",
    "symbol_to_color",
]


def bits_to_symbol_stream(data: bytes) -> list[int]:
    """Convert a raw byte stream into a stream of 3-bit symbols (0-7).

    Bits are read MSB-first across the whole byte stream and grouped into
    3-bit symbols in that order. If the total bit count is not a multiple
    of 3, the final symbol is padded with zero bits on its least-significant
    side. See the module docstring for the full padding rule.
    """
    total_bits = len(data) * 8
    num_symbols = math.ceil(total_bits / 3) if total_bits else 0

    symbols: list[int] = []
    for symbol_index in range(num_symbols):
        symbol = 0
        for bit_offset in range(3):
            bit_index = symbol_index * 3 + bit_offset
            if bit_index < total_bits:
                byte_index = bit_index // 8
                bit_in_byte = 7 - (bit_index % 8)
                bit_value = (data[byte_index] >> bit_in_byte) & 1
            else:
                bit_value = 0  # right-padding for the final partial symbol
            symbol = (symbol << 1) | bit_value
        symbols.append(symbol)
    return symbols


def symbol_stream_to_bits(symbols: list[int], original_byte_length: int) -> bytes:
    """Reconstruct the original byte stream from a 3-bit symbol stream.

    Takes ``original_byte_length`` (in bytes) explicitly, since the header
    carries this value separately and no padding-detection heuristic is
    needed. Discards any trailing padding bits (and any excess symbols
    beyond what is required to reconstruct that many bytes).

    Raises:
        ValueError: if ``symbols`` does not contain enough bits to
            reconstruct ``original_byte_length`` bytes.
    """
    required_bits = original_byte_length * 8
    available_bits = len(symbols) * 3
    if available_bits < required_bits:
        raise ValueError(
            f"Not enough symbols to reconstruct {original_byte_length} byte(s): "
            f"need {required_bits} bits but only {available_bits} are available."
        )

    result = bytearray(original_byte_length)
    for bit_index in range(required_bits):
        symbol_index = bit_index // 3
        bit_in_symbol = 2 - (bit_index % 3)
        bit_value = (symbols[symbol_index] >> bit_in_symbol) & 1

        byte_index = bit_index // 8
        bit_in_byte = 7 - (bit_index % 8)
        result[byte_index] |= bit_value << bit_in_byte

    return bytes(result)


def symbol_to_color(symbol: int) -> tuple[int, int, int]:
    """Look up the RGB color for a 3-bit symbol (0-7).

    Raises:
        ValueError: if ``symbol`` is not in range 0-7.
    """
    if symbol not in PALETTE:
        raise ValueError(f"Invalid symbol: {symbol!r} (must be an integer 0-7).")
    return PALETTE[symbol]


def color_to_symbol(color: tuple[int, int, int]) -> int:
    """Look up the 3-bit symbol for an exact RGB palette color.

    This is the ideal (non-camera) reverse lookup: only an exact match
    against one of the 8 palette colors is accepted.

    Raises:
        ValueError: if ``color`` is not one of the 8 exact palette values.
    """
    if color not in _COLOR_TO_SYMBOL:
        raise ValueError(f"Invalid color: {color!r} (not one of the 8 palette colors).")
    return _COLOR_TO_SYMBOL[color]
