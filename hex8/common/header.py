"""HX8M binary header - pack/unpack for the Hex8 marker format.

The HX8M header is a fixed-size, 20-byte binary structure prepended to every
Hex8 marker payload. It is intentionally simple (a flat C-style struct) so
that both the Python reference implementation and future non-Python
decoders (e.g. embedded/camera pipelines) can parse it without a general
purpose serialization library.

Layout (20 bytes total, big-endian / network byte order)::

    Offset  Size  Field            Type
    ------  ----  ---------------  --------
    0       4     Magic            bytes    (must be b"HX8M")
    4       1     Version          uint8
    5       1     Flags            uint8    (bit 0: compression, bit 1: signature)
    6       1     Radius           uint8
    7       1     ECC Level        uint8
    8       4     Payload Length   uint32   (original payload size)
    12      4     Encoded Length   uint32   (payload after compression/ECC)
    16      4     CRC32            uint32   (checksum of the original payload)

Design decisions
-----------------
- **Byte order**: big-endian (network byte order) is used for all
  multi-byte fields, matching the ``>`` prefix convention of the ``struct``
  module. This keeps the on-wire format architecture-independent.
- **Magic / version constants**: ``MAGIC = b"HX8M"`` identifies the format,
  and ``VERSION = 1`` is the current format version. ``Header`` does not
  carry the magic bytes as a field - they are a fixed constant that
  ``pack()`` writes and ``unpack()`` validates.
- **Flags byte**: bit 0 (compression enabled) and bit 1 (signature present)
  are reserved for future use. No compression or signature logic exists
  anywhere in this codebase yet, so:

  - ``pack()`` only accepts a flags value with bits 0 and 1 (and all other
    reserved bits 2-7) unset, and raises ``ValueError`` otherwise. This
    prevents producing a header that claims a feature this codebase does
    not implement.
  - ``unpack()`` does not error if those bits are set (e.g. a header
    produced by a future encoder version); it simply exposes the raw
    flags byte via ``Header.flags`` unchanged, with no interpretation.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

#: Fixed 4-byte magic value identifying an HX8M header.
MAGIC: bytes = b"HX8M"

#: Current HX8M header format version.
VERSION: int = 1

#: Total size, in bytes, of a packed HX8M header.
HEADER_SIZE: int = 20

# Struct format: 4s (magic) + B (version) + B (flags) + B (radius)
# + B (ecc_level) + I (payload_length) + I (encoded_length) + I (crc32),
# all big-endian / network byte order, no implicit padding.
_STRUCT_FORMAT = ">4sBBBBIII"

# Bits 0 and 1 of the flags byte are reserved for compression/signature
# support that is not yet implemented anywhere in this codebase. Bits 2-7
# are reserved for future use as well. pack() must reject any of them.
_FLAGS_RESERVED_MASK = 0b1111_1111


@dataclass
class Header:
    """In-memory representation of an HX8M header.

    Note that the 4-byte magic is intentionally not a field here: it is a
    fixed constant (``MAGIC``) that ``pack()`` writes and ``unpack()``
    validates, rather than a value callers choose.
    """

    version: int
    flags: int
    radius: int
    ecc_level: int
    payload_length: int
    encoded_length: int
    crc32: int


def pack(header: Header) -> bytes:
    """Serialize a `Header` into the 20-byte HX8M binary header.

    Raises:
        ValueError: if any reserved flags bit (0-7, since no bits are
            currently allocated to real features) is set.
    """
    if header.flags & _FLAGS_RESERVED_MASK:
        raise ValueError(
            "Header.flags has reserved bit(s) set: "
            f"{header.flags:#010b}. Compression and signature support are "
            "not implemented yet, so all flags bits must currently be 0."
        )

    return struct.pack(
        _STRUCT_FORMAT,
        MAGIC,
        header.version,
        header.flags,
        header.radius,
        header.ecc_level,
        header.payload_length,
        header.encoded_length,
        header.crc32,
    )


def unpack(data: bytes) -> Header:
    """Parse the first `HEADER_SIZE` bytes of `data` into a `Header`.

    Any bytes beyond `HEADER_SIZE` (i.e. the marker payload that follows
    the header) are ignored.

    Raises:
        ValueError: if `data` is shorter than `HEADER_SIZE`, if the magic
            bytes do not match `MAGIC`, or if the version does not match
            the currently supported `VERSION`.
    """
    if len(data) < HEADER_SIZE:
        raise ValueError(
            f"Truncated HX8M header: expected at least {HEADER_SIZE} bytes, "
            f"got {len(data)}."
        )

    (
        magic,
        version,
        flags,
        radius,
        ecc_level,
        payload_length,
        encoded_length,
        crc32,
    ) = struct.unpack(_STRUCT_FORMAT, data[:HEADER_SIZE])

    if magic != MAGIC:
        raise ValueError(f"Bad HX8M magic: expected {MAGIC!r}, got {magic!r}.")

    if version != VERSION:
        raise ValueError(
            f"Unsupported HX8M version: expected {VERSION}, got {version}."
        )

    return Header(
        version=version,
        flags=flags,
        radius=radius,
        ecc_level=ecc_level,
        payload_length=payload_length,
        encoded_length=encoded_length,
        crc32=crc32,
    )
