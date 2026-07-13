"""Deinterleave + Reed-Solomon correction + CRC verification (Issue #11).

Completes the Phase 2 decode pipeline started by Issues #9 (marker
detection) and #10 (color classification): given the per-cell
:class:`~hex8.decoder.classify.Classification` results for the METADATA and
DATA regions, recover the packed HX8M header and the original payload
bytes, verified against the header's CRC32.

Deinterleaving and the actual Reed-Solomon correction happen inside
:func:`hex8.common.ecc.decode` (Issue #4) - this module's job is the
integration layer around it: converting cell classifications into byte
streams, translating low-confidence cells into Reed-Solomon erasure hints,
and verifying the result's integrity.

Low-confidence cells as erasures
---------------------------------
A single Hex8 "symbol" is 3 bits, but a Reed-Solomon "symbol" (in GF(256))
is a full byte, so a low-confidence 3-bit cell doesn't map 1:1 to an
erasable RS byte: its 3 bits can fall within one or two bytes of the
encoded payload's bit stream (since 3 doesn't divide 8). This module
conservatively marks *every* byte touched by a low-confidence symbol's bit
range as an erasure candidate - over-marking is safe (Reed-Solomon can
correct up to `nsym` erasures per block; a few extra candidate positions
just use up more of that budget) as long as the total stays within the
configured ECC rate's correction capacity.

Bytes within the encoded payload's first ``ecc.HEADER_SIZE`` bytes (the
Reed-Solomon module's own small internal framing header - number of
blocks, block size, etc.) are never marked as erasures: that header isn't
itself Reed-Solomon protected (see :mod:`hex8.common.ecc`'s docstring), so
there's nothing `ecc.decode` could do with an erasure hint there anyway.

Known limitation (candidate for Issue #14): this conservative "mark every
touched byte" approach works cleanly for spatially *contiguous* low-
confidence regions (a scratch, glare, a localized print defect) - the
number of touched bytes stays close to the number of genuinely-uncertain
symbols. A pathologically *sparse*, evenly-spaced pattern of low-confidence
cells can cascade into flagging far more bytes than are actually uncertain
(each flagged byte pulls in its neighboring symbols, which in turn touch
further bytes), potentially exhausting the Reed-Solomon block's erasure
budget (`nsym`) even though the real amount of damage was within its
capacity. Real-world degradation (Phase 3/4) is expected to be spatially
localized rather than adversarially sparse, so this is not addressed here.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

from hex8.common import ecc as ecc_mod
from hex8.common import symbols
from hex8.common.header import Header
from hex8.common.header import HEADER_SIZE as HX8M_HEADER_SIZE
from hex8.common.header import unpack as unpack_header
from hex8.decoder.classify import Classification

__all__ = ["DecodedMarker", "decode_header", "decode_marker", "decode_payload"]


@dataclass(frozen=True)
class DecodedMarker:
    """The fully-decoded result of a Hex8 marker: its header and payload."""

    header: Header
    payload: bytes


def decode_header(metadata_classifications: list[Classification]) -> Header:
    """Recover the HX8M header from the METADATA region's classifications.

    Args:
        metadata_classifications: Classifications for the METADATA cells,
            in the same order as
            ``hex8.common.layout.build_layout(radius).cells_with_role(CellRole.METADATA)``.

    Returns:
        The unpacked :class:`~hex8.common.header.Header`.

    Raises:
        ValueError: if the recovered bytes don't form a valid HX8M header
            (bad magic/version, see :func:`hex8.common.header.unpack`).
    """
    metadata_symbols = [c.symbol for c in metadata_classifications]
    header_bytes = symbols.symbol_stream_to_bits(metadata_symbols, HX8M_HEADER_SIZE)
    return unpack_header(header_bytes)


def _symbol_touched_byte_indices(symbol_index: int) -> tuple[int, ...]:
    """Return the 1 or 2 byte indices that a 3-bit symbol's bits fall into.

    Symbol `i` occupies bit range [3*i, 3*i + 2] (MSB-first across the byte
    stream, per hex8.common.symbols). A byte index is `bit_index // 8`.
    """
    start_bit = symbol_index * 3
    end_bit = start_bit + 2
    start_byte = start_bit // 8
    end_byte = end_bit // 8
    if start_byte == end_byte:
        return (start_byte,)
    return (start_byte, end_byte)


def decode_payload(header: Header, data_classifications: list[Classification]) -> bytes:
    """Recover and verify the original payload from the DATA region's classifications.

    Args:
        header: The already-decoded HX8M header (see :func:`decode_header`),
            which supplies ``encoded_length``, ``ecc_level``, and ``crc32``.
        data_classifications: Classifications for the DATA cells, in the
            same order as
            ``hex8.common.layout.build_layout(radius).cells_with_role(CellRole.DATA)``.
            Only the first ``ceil(header.encoded_length * 8 / 3)`` entries
            are used; any cells beyond that (unused capacity, see
            :mod:`hex8.encoder.encode`) are ignored.

    Returns:
        The original payload bytes.

    Raises:
        ValueError: if there aren't enough data classifications for
            ``header.encoded_length``, if Reed-Solomon correction fails
            (see :func:`hex8.common.ecc.decode`), or if the recovered
            payload's CRC32 does not match ``header.crc32``.
    """
    needed_symbols = math.ceil(header.encoded_length * 8 / 3)
    if len(data_classifications) < needed_symbols:
        raise ValueError(
            f"Not enough data cell classifications: need {needed_symbols} for "
            f"encoded_length={header.encoded_length}, got {len(data_classifications)}"
        )

    used = data_classifications[:needed_symbols]
    data_symbols = [c.symbol for c in used]
    encoded_payload = symbols.symbol_stream_to_bits(data_symbols, header.encoded_length)

    erasure_body_positions: set[int] = set()
    for i, classification in enumerate(used):
        if not classification.low_confidence:
            continue
        for byte_index in _symbol_touched_byte_indices(i):
            if byte_index < ecc_mod.HEADER_SIZE:
                # Within ecc's own internal (unprotected) framing header -
                # no erasure hint can help there; see module docstring.
                continue
            erasure_body_positions.add(byte_index - ecc_mod.HEADER_SIZE)

    payload = ecc_mod.decode(
        encoded_payload,
        ecc_rate=header.ecc_level / 100.0,
        interleave=True,
        erasure_body_positions=erasure_body_positions or None,
    )

    if zlib.crc32(payload) != header.crc32:
        raise ValueError(
            "CRC32 mismatch: decoded payload failed integrity check "
            "(marker header may be corrupted or tampered with)"
        )

    return payload


def decode_marker(
    metadata_classifications: list[Classification],
    data_classifications: list[Classification],
) -> DecodedMarker:
    """Decode both the header and payload, returning both as one result."""
    header = decode_header(metadata_classifications)
    payload = decode_payload(header, data_classifications)
    return DecodedMarker(header=header, payload=payload)
