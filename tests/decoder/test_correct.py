"""Tests for hex8.decoder.correct (Issue #11)."""

from __future__ import annotations

import zlib

import pytest

from hex8.common import ecc as ecc_mod
from hex8.common import header as header_mod
from hex8.common import symbols
from hex8.decoder.classify import Classification
from hex8.decoder.correct import decode_header, decode_payload


def _sample_payload(size: int) -> bytes:
    return bytes((i * 29 + 3) % 256 for i in range(size))


def _make_header(payload: bytes, radius: int, ecc_level: int) -> header_mod.Header:
    encoded = ecc_mod.encode(payload, ecc_rate=ecc_level / 100.0, interleave=True)
    return header_mod.Header(
        version=header_mod.VERSION,
        flags=0,
        radius=radius,
        ecc_level=ecc_level,
        payload_length=len(payload),
        encoded_length=len(encoded),
        crc32=zlib.crc32(payload),
    )


def _classifications_for(symbols_list: list[int]) -> list[Classification]:
    return [Classification(symbol=s, confidence=1.0, low_confidence=False) for s in symbols_list]


# --- decode_header -----------------------------------------------------


def test_decode_header_round_trips_a_packed_header():
    header = header_mod.Header(
        version=header_mod.VERSION,
        flags=0,
        radius=18,
        ecc_level=30,
        payload_length=200,
        encoded_length=286,
        crc32=0xDEADBEEF,
    )
    header_bytes = header_mod.pack(header)
    metadata_symbols = symbols.bits_to_symbol_stream(header_bytes)
    classifications = _classifications_for(metadata_symbols)

    decoded = decode_header(classifications)

    assert decoded == header


# --- decode_payload: normal round trip ----------------------------------


@pytest.mark.parametrize("payload_size", [1, 16, 128])
@pytest.mark.parametrize("ecc_level", [25, 30, 40])
def test_decode_payload_round_trip_no_corruption(payload_size, ecc_level):
    payload = _sample_payload(payload_size)
    header = _make_header(payload, radius=18, ecc_level=ecc_level)
    encoded = ecc_mod.encode(payload, ecc_rate=ecc_level / 100.0, interleave=True)
    data_symbols = symbols.bits_to_symbol_stream(encoded)
    classifications = _classifications_for(data_symbols)

    decoded = decode_payload(header, classifications)

    assert decoded == payload


# --- decode_payload: CRC mismatch ----------------------------------------


def test_decode_payload_crc_mismatch_raises_value_error():
    payload = _sample_payload(32)
    header = _make_header(payload, radius=18, ecc_level=30)
    # Tamper the header's CRC directly (simulating a corrupted/forged
    # header field whose own metadata symbols happened to survive RS/CRC
    # checks at the header level - Issue #2's header has no ECC of its
    # own, so this models the payload-level integrity check catching a
    # bad CRC field regardless of why it's wrong).
    tampered_header = header_mod.Header(**{**header.__dict__, "crc32": header.crc32 ^ 0xFFFFFFFF})

    encoded = ecc_mod.encode(payload, ecc_rate=0.3, interleave=True)
    data_symbols = symbols.bits_to_symbol_stream(encoded)
    classifications = _classifications_for(data_symbols)

    with pytest.raises(ValueError, match="CRC"):
        decode_payload(tampered_header, classifications)


# --- decode_payload: low-confidence cells become RS erasures -------------


def test_decode_payload_uses_low_confidence_flags_as_erasures():
    """Corrupt enough bytes to exceed RS's blind (unknown-position) error
    correction budget, but flag exactly those bytes' corresponding data-cell
    classifications as low_confidence. decode_payload must still recover
    the original payload by treating them as erasures; without the
    low_confidence flags, the same corruption must fail outright."""
    payload = _sample_payload(32)
    ecc_level = 40
    header = _make_header(payload, radius=18, ecc_level=ecc_level)

    encoded = bytearray(ecc_mod.encode(payload, ecc_rate=ecc_level / 100.0, interleave=True))
    num_blocks, block_data_len, nsym = ecc_mod._plan_blocks(len(payload), ecc_level / 100.0)
    assert num_blocks == 1, "test assumes a single RS block for a direct nsym calculation"

    body_len = len(encoded) - ecc_mod.HEADER_SIZE
    num_corrupted = max(1, nsym - 1)  # beyond nsym // 2, within nsym erasures
    # A contiguous burst (as opposed to e.g. a sparse, strided pattern) keeps
    # the number of *symbols* touching a corrupted byte close to the number
    # of corrupted bytes themselves - realistic for real-world localized
    # damage (a scratch, glare, a misprint), and avoids the conservative
    # byte-marking in decode_payload ballooning far past `nsym`'s erasure
    # budget, which a maximally-adversarial sparse pattern can trigger.
    assert num_corrupted <= body_len
    corrupt_positions = list(range(num_corrupted))

    corrupted = bytearray(encoded)
    for pos in corrupt_positions:
        corrupted[ecc_mod.HEADER_SIZE + pos] ^= 0xFF

    true_symbols = symbols.bits_to_symbol_stream(bytes(encoded))
    corrupted_symbols = symbols.bits_to_symbol_stream(bytes(corrupted))

    # Symbols whose 3 bits overlap a corrupted byte are exactly the ones a
    # real color classifier would plausibly flag as unreliable.
    corrupted_byte_indices = {ecc_mod.HEADER_SIZE + p for p in corrupt_positions}

    def _symbol_touches_corrupted_byte(symbol_index: int) -> bool:
        start_bit = symbol_index * 3
        for bit_offset in range(3):
            if (start_bit + bit_offset) // 8 in corrupted_byte_indices:
                return True
        return False

    confident_wrong_classifications = []
    flagged_low_confidence_classifications = []
    for i, symbol in enumerate(corrupted_symbols):
        touched = _symbol_touches_corrupted_byte(i)
        confident_wrong_classifications.append(
            Classification(symbol=symbol, confidence=1.0, low_confidence=False)
        )
        flagged_low_confidence_classifications.append(
            Classification(symbol=symbol, confidence=0.05 if touched else 1.0, low_confidence=touched)
        )

    # Sanity check: the corruption actually changed some symbols, so this
    # test would be vacuous otherwise.
    assert corrupted_symbols != true_symbols

    # Without low-confidence hints, this much corruption should fail outright.
    with pytest.raises(ValueError):
        decode_payload(header, confident_wrong_classifications)

    # With low-confidence hints treated as erasures, it must recover.
    decoded = decode_payload(header, flagged_low_confidence_classifications)
    assert decoded == payload
