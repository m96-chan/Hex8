"""Reed-Solomon ECC encoding/decoding with interleaving.

This module implements the "Reed-Solomon ECC" + "interleaving" steps of the
Hex8 encoding pipeline described in the README, built on top of the
``reedsolo`` library's ``RSCodec``.

GF(256) block size constraint
------------------------------
``reedsolo.RSCodec`` operates over GF(256): a single RS codeword (message
bytes + parity/ECC bytes) cannot exceed 255 bytes. The target Hex8 payloads
are 128-256 bytes with 25%-40% ECC overhead, which can require *more* than
255 bytes once ECC symbols are added. To support this, arbitrary-length
input data is split into one or more equally-sized RS blocks, each block is
RS-encoded independently, and (optionally) the resulting blocks are
interleaved symbol-by-symbol so that a contiguous burst of corruption in the
interleaved stream is spread across many blocks (hitting at most one symbol
per block per burst byte) instead of wiping out a single block entirely.

Block splitting scheme
-----------------------
Given ``ecc_rate`` (the fraction of a block's *total* size -- data + ECC --
that is spent on ECC symbols), we choose the smallest number of blocks
``num_blocks`` such that the data can be split into ``num_blocks`` equal
chunks (the last one zero-padded if needed) of length ``block_data_len``,
with a per-block ECC symbol count::

    nsym = round((block_data_len + nsym) * ecc_rate)

(a small fixed-point loop is used to solve this, since ``nsym`` appears on
both sides -- the formula is defined in terms of the resulting *total*
block size, per the ECC rate definition), such that::

    block_data_len + nsym <= 255

``num_blocks`` starts at 1 and increases until this constraint is satisfied
(fewer, larger blocks are always preferred, since larger RS blocks are more
efficient and correct proportionally larger bursts).

All blocks in a given ``encode()`` call share the same ``block_data_len``
and ``nsym`` (the last block is zero-padded up to ``block_data_len`` before
RS encoding if the data doesn't divide evenly). Padding is stripped again
on decode using the original data length recorded in the header below.

Internal framing (header)
--------------------------
``decode()`` only receives the bytes produced by ``encode()`` -- it has no
other side channel for the original data length or block layout. So
``encode()`` prepends a small, fixed-size, *not* RS-protected metadata
header to the returned bytes:

======================= ======= ===========================================
Field                   Size    Description
======================= ======= ===========================================
Magic                   4 bytes ``b"H8EC"``, sanity check for decode()
data_length             4 bytes Original ``data`` length (big-endian u32)
num_blocks              2 bytes Number of RS blocks (big-endian u16)
block_data_len          1 byte  Data bytes per block, pre-padding (u8)
nsym                    1 byte  ECC symbols per block (u8)
interleaved             1 byte  1 if blocks were interleaved, else 0
======================= ======= ===========================================

Total header size: 13 bytes. The header is not itself Reed-Solomon
protected: it is small, and protecting it would require its own separate
ECC scheme out of scope for this module. This is a deliberate PoC-level
design tradeoff -- if the header is corrupted, decoding fails outright
(this mirrors, at smaller scale, the existing HX8M marker header design).

After the header, the RS-encoded block bytes follow: either interleaved
(symbol ``i`` of block 0, symbol ``i`` of block 1, ..., symbol ``i`` of
block N-1, then symbol ``i+1`` of block 0, ...) or, if
``interleave=False``, simply concatenated block-by-block.
"""

from __future__ import annotations

import struct

from reedsolo import RSCodec, ReedSolomonError

MAGIC = b"H8EC"

# GF(256) hard limit: a single RS codeword (data + ECC symbols) cannot
# exceed this many bytes.
MAX_BLOCK_TOTAL_SIZE = 255

# Valid ECC rate range, per the README's "Error Correction Strategy" section.
MIN_ECC_RATE = 0.25
MAX_ECC_RATE = 0.40

_HEADER_STRUCT = struct.Struct(">4sIHBBB")
HEADER_SIZE = _HEADER_STRUCT.size


def _validate_ecc_rate(ecc_rate: float) -> None:
    if not (MIN_ECC_RATE <= ecc_rate <= MAX_ECC_RATE):
        raise ValueError(
            f"ecc_rate must be within [{MIN_ECC_RATE}, {MAX_ECC_RATE}], got {ecc_rate!r}"
        )


def _nsym_for_block_data_len(block_data_len: int, ecc_rate: float) -> int:
    """Solve nsym = round((block_data_len + nsym) * ecc_rate) via fixed-point
    iteration. This converges because ecc_rate < 1 makes the update a
    contraction mapping.

    The result is always at least 1: `reedsolo`'s own `RSCodec.decode()`
    slices the corrected message out of the codeword as `data[:-nsym]`, and
    `data[:-0]` is `data[:0]` in Python (negative-zero slicing does not mean
    "no bytes removed from the end" - it means "keep zero bytes"), which
    silently returns an empty message instead of the real data whenever
    `nsym == 0`. A tiny payload's `round(block_data_len * ecc_rate)` can
    legitimately compute to 0 (e.g. a 1-byte block at any rate in
    [0.25, 0.40] rounds to 0), so this floor is required for correctness,
    not just for "some" error correction capability.
    """
    nsym = round(block_data_len * ecc_rate)
    for _ in range(32):
        new_nsym = round((block_data_len + nsym) * ecc_rate)
        if new_nsym == nsym:
            break
        nsym = new_nsym
    return max(1, nsym)


def _plan_blocks(data_len: int, ecc_rate: float) -> tuple[int, int, int]:
    """Return (num_blocks, block_data_len, nsym) for splitting `data_len`
    bytes of data into equal-size RS blocks (the last zero-padded to match)
    such that each block's total size (block_data_len + nsym) fits within
    the GF(256) 255-byte limit.
    """
    if data_len == 0:
        return 0, 0, 0

    num_blocks = 1
    while True:
        block_data_len = -(-data_len // num_blocks)  # ceil division
        nsym = _nsym_for_block_data_len(block_data_len, ecc_rate)
        if block_data_len + nsym <= MAX_BLOCK_TOTAL_SIZE:
            return num_blocks, block_data_len, nsym
        num_blocks += 1


def _interleave(blocks: list[bytes]) -> bytes:
    """Interleave equal-length blocks symbol-by-symbol (column-major):
    output byte at position i * num_blocks + j is blocks[j][i]."""
    num_blocks = len(blocks)
    block_len = len(blocks[0])
    out = bytearray(num_blocks * block_len)
    for j, block in enumerate(blocks):
        out[j::num_blocks] = block
    return bytes(out)


def _deinterleave(data: bytes, num_blocks: int, block_len: int) -> list[bytes]:
    """Inverse of _interleave: recover the num_blocks original blocks, each
    of length block_len, from the interleaved byte stream."""
    return [bytes(data[j::num_blocks]) for j in range(num_blocks)]


def encode(data: bytes, ecc_rate: float = 0.3, interleave: bool = True) -> bytes:
    """Encode `data` with Reed-Solomon ECC, splitting it across one or more
    RS blocks as needed to respect the GF(256) 255-byte block limit, and
    optionally interleaving the resulting blocks symbol-by-symbol.

    Args:
        data: Arbitrary input bytes to protect.
        ecc_rate: Fraction of each block's total size (data + ECC) spent on
            ECC symbols. Must be within [0.25, 0.40].
        interleave: If True (default), interleave RS-encoded blocks
            symbol-by-symbol so contiguous corruption spreads across
            blocks. If False, blocks are simply concatenated.

    Returns:
        A self-describing byte string: a small internal header (see module
        docstring) followed by the (optionally interleaved) RS-encoded
        block data. Pass the full return value to `decode()`.
    """
    _validate_ecc_rate(ecc_rate)

    num_blocks, block_data_len, nsym = _plan_blocks(len(data), ecc_rate)

    header = _HEADER_STRUCT.pack(
        MAGIC, len(data), num_blocks, block_data_len, nsym, 1 if interleave else 0
    )

    if num_blocks == 0:
        return header

    padded_len = num_blocks * block_data_len
    padded_data = data + b"\x00" * (padded_len - len(data))

    rsc = RSCodec(nsym)
    encoded_blocks = [
        rsc.encode(padded_data[i * block_data_len : (i + 1) * block_data_len])
        for i in range(num_blocks)
    ]

    body = _interleave(encoded_blocks) if interleave else b"".join(encoded_blocks)

    return header + body


def _body_position_to_block_position(
    position: int, num_blocks: int, block_total_len: int, interleave: bool
) -> tuple[int, int]:
    """Map a byte offset within the (possibly interleaved) body to
    (block_index, within_block_index).

    Inverse of `_interleave`'s `out[j::num_blocks] = block` assignment: body
    position `p` in the interleaved stream belongs to block `p % num_blocks`
    at within-block index `p // num_blocks`. When not interleaved, blocks are
    simply concatenated, so block index is `p // block_total_len`.
    """
    if interleave:
        return position % num_blocks, position // num_blocks
    return position // block_total_len, position % block_total_len


def decode(
    data: bytes,
    ecc_rate: float = 0.3,
    interleave: bool = True,
    erasure_body_positions: set[int] | None = None,
) -> bytes:
    """Decode bytes produced by `encode()`, correcting errors introduced by
    RS ECC and reversing interleaving, returning the original data.

    Args:
        data: The full byte string returned by `encode()` (header + body).
        ecc_rate: Must be within [0.25, 0.40], as with `encode()`. The
            actual block layout used for decoding is recovered from the
            internal header embedded by `encode()` -- this parameter is
            validated for API symmetry with `encode()`, but is not itself
            required to reproduce the block layout.
        interleave: Must match the value passed to the corresponding
            `encode()` call. Cross-checked against the internal header;
            a mismatch raises `ValueError`.
        erasure_body_positions: Optional 0-indexed byte offsets within the
            body (`data[HEADER_SIZE:]`, i.e. the same coordinate space as
            the encoded bytes themselves - *not* per-block indices) that
            are already known to be unreliable (e.g. from low-confidence
            color classification upstream), and should be treated as
            Reed-Solomon erasures rather than unknown errors. Reed-Solomon
            can correct up to `nsym` erasures per block, twice as many as
            the `nsym // 2` unknown-position errors it can correct without
            this hint. Positions outside the valid body range are silently
            ignored (callers may pass slightly-approximate hints). Ignored
            entirely if `None` (the default).

    Returns:
        The original data bytes.

    Raises:
        ValueError: If `ecc_rate` is out of range, the header is malformed
            or inconsistent with `interleave`, or a block's errors exceed
            Reed-Solomon's correction capability.
    """
    _validate_ecc_rate(ecc_rate)

    if len(data) < HEADER_SIZE:
        raise ValueError("Encoded data is too short to contain a valid ECC header")

    magic, data_len, num_blocks, block_data_len, nsym, interleaved_flag = (
        _HEADER_STRUCT.unpack(data[:HEADER_SIZE])
    )

    if magic != MAGIC:
        raise ValueError(f"Invalid ECC header magic: {magic!r}")

    header_interleaved = bool(interleaved_flag)
    if header_interleaved != interleave:
        raise ValueError(
            f"interleave={interleave} does not match the value used at encode time "
            f"({header_interleaved}); pass the matching value to decode()"
        )

    if num_blocks == 0:
        return b""

    block_total_len = block_data_len + nsym
    body = data[HEADER_SIZE:]
    expected_body_len = num_blocks * block_total_len
    if len(body) != expected_body_len:
        raise ValueError(
            f"Encoded body length {len(body)} does not match expected "
            f"{expected_body_len} for {num_blocks} block(s) of size {block_total_len}"
        )

    if interleave:
        blocks = _deinterleave(body, num_blocks, block_total_len)
    else:
        blocks = [
            body[i * block_total_len : (i + 1) * block_total_len] for i in range(num_blocks)
        ]

    erase_pos_by_block: dict[int, list[int]] = {i: [] for i in range(num_blocks)}
    if erasure_body_positions:
        for position in erasure_body_positions:
            if not (0 <= position < expected_body_len):
                continue  # out-of-range hint, silently ignored
            block_index, within_block_index = _body_position_to_block_position(
                position, num_blocks, block_total_len, interleave
            )
            erase_pos_by_block[block_index].append(within_block_index)

    rsc = RSCodec(nsym)
    decoded_chunks = []
    for i, block in enumerate(blocks):
        erase_pos = erase_pos_by_block[i] or None
        try:
            decoded_msg, _decoded_msgecc, _errata_pos = rsc.decode(block, erase_pos=erase_pos)
        except ReedSolomonError as exc:
            raise ValueError(f"Reed-Solomon decoding failed: {exc}") from exc
        decoded_chunks.append(bytes(decoded_msg))

    padded_data = b"".join(decoded_chunks)
    return padded_data[:data_len]
