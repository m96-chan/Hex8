"""Tests for Reed-Solomon ECC encoding/decoding with interleaving.

See hex8/common/ecc.py for the block-splitting and interleaving scheme.
"""

import pytest

from hex8.common.ecc import decode, encode

PAYLOAD_SIZES = (16, 128, 256)
ECC_RATES = (0.25, 0.3, 0.4)


@pytest.mark.parametrize("size", PAYLOAD_SIZES)
@pytest.mark.parametrize("ecc_rate", ECC_RATES)
@pytest.mark.parametrize("interleave", (True, False))
def test_round_trip(size, ecc_rate, interleave):
    data = bytes((i * 37 + 11) % 256 for i in range(size))

    encoded = encode(data, ecc_rate=ecc_rate, interleave=interleave)
    decoded = decode(encoded, ecc_rate=ecc_rate, interleave=interleave)

    assert decoded == data


@pytest.mark.parametrize("ecc_rate", ECC_RATES)
def test_round_trip_tiny_payload(ecc_rate):
    """A 1-byte payload: `round(1 * ecc_rate)` is 0 for every rate in
    ECC_RATES, which - discovered while implementing Issue #11 - used to
    produce nsym=0. reedsolo's own decode() slices the corrected message as
    `data[:-nsym]`, and `data[:-0]` is `data[:0]` (Python's negative-zero
    slicing gotcha), which silently returns an empty message instead of the
    real data. nsym must never be allowed to reach 0."""
    data = bytes([42])

    encoded = encode(data, ecc_rate=ecc_rate, interleave=True)
    decoded = decode(encoded, ecc_rate=ecc_rate, interleave=True)

    assert decoded == data


def test_round_trip_empty_data():
    data = b""

    encoded = encode(data, ecc_rate=0.3, interleave=True)
    decoded = decode(encoded, ecc_rate=0.3, interleave=True)

    assert decoded == data


@pytest.mark.parametrize("interleave", (True, False))
@pytest.mark.parametrize("ecc_rate", ECC_RATES)
def test_corruption_recovery_single_block(ecc_rate, interleave):
    """A 16-byte payload fits in a single RS block; corrupt roughly nsym // 2
    bytes (the maximum correctable symbol-error count) and confirm recovery."""
    data = bytes((i * 53 + 7) % 256 for i in range(16))

    encoded = bytearray(encode(data, ecc_rate=ecc_rate, interleave=interleave))

    # Corrupt bytes starting after the internal header, spread across the body,
    # in a count that should remain within RS's correction capability.
    header_len = len(encode(b"", ecc_rate=ecc_rate, interleave=interleave))
    body = encoded[header_len:]
    nsym = max(1, round(len(body) * ecc_rate))
    correctable_errors = max(1, nsym // 2)

    corrupted_body = bytearray(body)
    for i in range(correctable_errors):
        pos = (i * 7) % len(corrupted_body)
        corrupted_body[pos] ^= 0xFF

    corrupted = bytes(encoded[:header_len]) + bytes(corrupted_body)
    decoded = decode(corrupted, ecc_rate=ecc_rate, interleave=interleave)

    assert decoded == data


@pytest.mark.parametrize("interleave", (True, False))
def test_corruption_recovery_multi_block(interleave):
    """A 256-byte payload requires multiple RS blocks. Corrupt a contiguous
    burst spanning the interleaved stream and confirm recovery."""
    ecc_rate = 0.4
    data = bytes((i * 97 + 3) % 256 for i in range(256))

    encoded = bytearray(encode(data, ecc_rate=ecc_rate, interleave=interleave))

    # Flip a modest, contiguous run of bytes near the start of the encoded body.
    header_len = len(encode(b"", ecc_rate=ecc_rate, interleave=interleave))
    burst_len = 6
    for offset in range(burst_len):
        encoded[header_len + offset] ^= 0xFF

    decoded = decode(bytes(encoded), ecc_rate=ecc_rate, interleave=interleave)

    assert decoded == data


@pytest.mark.parametrize("ecc_rate", (0.0, 0.1, 0.24, 0.41, 0.5, 1.0))
def test_encode_invalid_ecc_rate_raises_value_error(ecc_rate):
    with pytest.raises(ValueError):
        encode(b"hello world", ecc_rate=ecc_rate)


@pytest.mark.parametrize("ecc_rate", (0.0, 0.1, 0.24, 0.41, 0.5, 1.0))
def test_decode_invalid_ecc_rate_raises_value_error(ecc_rate):
    encoded = encode(b"hello world", ecc_rate=0.3)
    with pytest.raises(ValueError):
        decode(encoded, ecc_rate=ecc_rate)


def test_decode_too_short_raises_value_error():
    with pytest.raises(ValueError, match="too short"):
        decode(b"\x00\x01\x02", ecc_rate=0.3)


def test_decode_bad_magic_raises_value_error():
    encoded = bytearray(encode(b"hello world", ecc_rate=0.3))
    encoded[0:4] = b"XXXX"
    with pytest.raises(ValueError, match="magic"):
        decode(bytes(encoded), ecc_rate=0.3)


def test_decode_interleave_mismatch_raises_value_error():
    encoded = encode(b"hello world", ecc_rate=0.3, interleave=True)
    with pytest.raises(ValueError, match="interleave"):
        decode(encoded, ecc_rate=0.3, interleave=False)


def test_decode_truncated_body_raises_value_error():
    encoded = encode(b"hello world", ecc_rate=0.3)
    with pytest.raises(ValueError, match="does not match expected"):
        decode(encoded[:-1], ecc_rate=0.3)


def test_decode_uncorrectable_errors_raises_value_error():
    """Corrupting far more bytes than a block's ECC symbols can correct
    should surface as a clear ValueError, not silently return wrong data."""
    ecc_rate = 0.4
    data = bytes(range(64))
    encoded = bytearray(encode(data, ecc_rate=ecc_rate, interleave=False))

    header_len = len(encode(b"", ecc_rate=ecc_rate, interleave=False))
    # Corrupt almost the entire body -- far beyond nsym // 2 correctable errors.
    for i in range(header_len, len(encoded)):
        encoded[i] ^= 0xFF

    with pytest.raises(ValueError, match="Reed-Solomon decoding failed"):
        decode(bytes(encoded), ecc_rate=ecc_rate, interleave=False)


# --- erasure support (Issue #11: known-position corruption, e.g. flagged by
# low-confidence color classification, is far cheaper for Reed-Solomon to
# correct than unknown-position errors: up to `nsym` erasures vs only
# `nsym // 2` unknown errors per block) ------------------------------------


@pytest.mark.parametrize("interleave", (True, False))
def test_erasures_correct_more_errors_than_unknown_positions_allow(interleave):
    """Corrupt close to `nsym` bytes (well beyond the nsym // 2 unknown-error
    limit) but tell decode() exactly where they are; it must still recover,
    whereas the same corruption without erasure info would fail."""
    ecc_rate = 0.4
    data = bytes((i * 41 + 5) % 256 for i in range(32))

    encoded = bytearray(encode(data, ecc_rate=ecc_rate, interleave=interleave))
    header_len = len(encode(b"", ecc_rate=ecc_rate, interleave=interleave))
    body_len = len(encoded) - header_len

    nsym = max(1, round(body_len * ecc_rate))
    # Just under nsym corrupted bytes: recoverable with erasures, but well
    # beyond the nsym // 2 unknown-error correction limit.
    num_corrupted = max(1, nsym - 1)

    positions = [(i * 3) % body_len for i in range(num_corrupted)]
    positions = sorted(set(positions))

    corrupted = bytearray(encoded)
    for pos in positions:
        corrupted[header_len + pos] ^= 0xFF

    # Without erasure hints, this much corruption should fail outright.
    with pytest.raises(ValueError, match="Reed-Solomon decoding failed"):
        decode(bytes(corrupted), ecc_rate=ecc_rate, interleave=interleave)

    # With erasure hints at the exact corrupted positions, it must recover.
    decoded = decode(
        bytes(corrupted),
        ecc_rate=ecc_rate,
        interleave=interleave,
        erasure_body_positions=set(positions),
    )
    assert decoded == data


def test_erasures_out_of_range_positions_are_ignored():
    """Erasure positions beyond the body's actual length are just ignored,
    not an error -- callers may pass slightly-approximate hints."""
    data = b"hello world"
    encoded = encode(data, ecc_rate=0.3, interleave=True)

    decoded = decode(
        encoded,
        ecc_rate=0.3,
        interleave=True,
        erasure_body_positions={10_000, 20_000},
    )
    assert decoded == data


def test_erasures_default_to_none_and_do_not_change_normal_round_trip():
    data = bytes((i * 13 + 1) % 256 for i in range(64))
    encoded = encode(data, ecc_rate=0.3, interleave=True)
    decoded = decode(encoded, ecc_rate=0.3, interleave=True, erasure_body_positions=None)
    assert decoded == data
