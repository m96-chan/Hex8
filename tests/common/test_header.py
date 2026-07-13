"""Tests for the HX8M binary header pack/unpack functions.

See hex8/common/header.py for the header layout and design rationale.
"""

import struct

import pytest

from hex8.common.header import HEADER_SIZE, MAGIC, VERSION, Header, pack, unpack


def make_header(**overrides):
    """Build a representative Header, allowing individual fields to be overridden."""
    fields = {
        "version": VERSION,
        "flags": 0,
        "radius": 5,
        "ecc_level": 2,
        "payload_length": 1024,
        "encoded_length": 2048,
        "crc32": 0xDEADBEEF,
    }
    fields.update(overrides)
    return Header(**fields)


def test_round_trip_pack_unpack():
    header = make_header()
    data = pack(header)

    assert len(data) == HEADER_SIZE == 20
    assert data[:4] == MAGIC

    result = unpack(data)

    assert result == header


def test_unpack_bad_magic_raises_value_error():
    header = make_header()
    data = pack(header)
    corrupted = b"XXXX" + data[4:]

    with pytest.raises(ValueError):
        unpack(corrupted)


def test_unpack_bad_version_raises_value_error():
    header = make_header()
    data = bytearray(pack(header))
    # Version is the 5th byte, right after the 4-byte magic.
    data[4] = VERSION + 1

    with pytest.raises(ValueError):
        unpack(bytes(data))


def test_unpack_truncated_input_raises_value_error():
    header = make_header()
    data = pack(header)
    truncated = data[:-1]

    with pytest.raises(ValueError):
        unpack(truncated)


def test_unpack_empty_input_raises_value_error():
    with pytest.raises(ValueError):
        unpack(b"")


def test_pack_rejects_reserved_flags_bits():
    # Bit 0 (compression) and bit 1 (signature) are reserved for future use;
    # pack() must refuse a header that sets either of them.
    header_compression = make_header(flags=0b0000_0001)
    header_signature = make_header(flags=0b0000_0010)
    header_reserved = make_header(flags=0b0000_0100)

    with pytest.raises(ValueError):
        pack(header_compression)

    with pytest.raises(ValueError):
        pack(header_signature)

    with pytest.raises(ValueError):
        pack(header_reserved)


def test_unpack_does_not_reject_reserved_flags_bits():
    # unpack() must tolerate the reserved bits being set (e.g. produced by a
    # future encoder version) and simply expose the raw flags byte as-is.
    header = make_header()
    data = bytearray(pack(header))
    flags_offset = 4 + 1  # after magic (4) and version (1)
    data[flags_offset] = 0b0000_0011

    result = unpack(bytes(data))

    assert result.flags == 0b0000_0011


def test_unpack_extra_trailing_bytes_are_ignored():
    header = make_header()
    data = pack(header) + b"trailing-payload-bytes"

    result = unpack(data)

    assert result == header


def test_header_field_values_are_encoded_big_endian():
    header = make_header(
        version=VERSION,
        flags=0,
        radius=7,
        ecc_level=1,
        payload_length=0x01020304,
        encoded_length=0x05060708,
        crc32=0x090A0B0C,
    )
    data = pack(header)

    magic, version, flags, radius, ecc_level, payload_length, encoded_length, crc32 = (
        struct.unpack(">4sBBBBIII", data)
    )

    assert magic == MAGIC
    assert version == VERSION
    assert flags == 0
    assert radius == 7
    assert ecc_level == 1
    assert payload_length == 0x01020304
    assert encoded_length == 0x05060708
    assert crc32 == 0x090A0B0C
