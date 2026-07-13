"""Tests for hex8.common.symbols: 3-bit symbol stream <-> 8-color mapping."""

import math

import pytest

from hex8.common.symbols import (
    PALETTE,
    bits_to_symbol_stream,
    color_to_symbol,
    symbol_stream_to_bits,
    symbol_to_color,
)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00",
        b"\xff",
        b"\x01\x02",
        b"\xab\xcd\xef",
        b"hello world",
        bytes(range(10)),
        bytes([0xFF] * 10),
    ],
)
def test_round_trip_various_lengths(data):
    symbols = bits_to_symbol_stream(data)
    assert symbol_stream_to_bits(symbols, len(data)) == data


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_symbol_stream_length_matches_ceil_formula(length):
    data = bytes(range(length))
    symbols = bits_to_symbol_stream(data)
    expected_length = math.ceil(length * 8 / 3)
    assert len(symbols) == expected_length


def test_padding_edge_case_one_byte():
    # 1 byte = 8 bits. 8 / 3 = 2.67 -> ceil = 3 symbols (9 bits total,
    # last symbol padded with 1 zero bit on the least-significant side).
    data = b"\xff"
    symbols = bits_to_symbol_stream(data)
    assert len(symbols) == 3
    # bits: 11111111 -> grouped as 111 111 11(0 padding)
    assert symbols == [0b111, 0b111, 0b110]
    assert symbol_stream_to_bits(symbols, len(data)) == data


def test_padding_edge_case_two_bytes():
    # 2 bytes = 16 bits. 16 / 3 = 5.33 -> ceil = 6 symbols (18 bits total,
    # last symbol padded with 2 zero bits).
    data = b"\x01\x02"
    symbols = bits_to_symbol_stream(data)
    expected_length = math.ceil(16 / 3)
    assert len(symbols) == expected_length
    assert symbol_stream_to_bits(symbols, len(data)) == data


def test_bits_to_symbol_stream_empty():
    assert bits_to_symbol_stream(b"") == []


def test_bits_to_symbol_stream_exact_multiple_of_three_bytes():
    # 3 bytes = 24 bits, evenly divisible by 3 -> no padding needed.
    data = b"\x01\x02\x03"
    symbols = bits_to_symbol_stream(data)
    assert len(symbols) == 8
    assert symbol_stream_to_bits(symbols, len(data)) == data


def test_symbol_stream_to_bits_known_values():
    # 0b101_010_011 -> byte 0b10101001 = 0xA9, with 1 leftover padding bit
    # (the trailing 1 bit) discarded.
    symbols = [0b101, 0b010, 0b011]
    result = symbol_stream_to_bits(symbols, 1)
    assert result == bytes([0b10101001])


def test_symbol_stream_to_bits_insufficient_symbols_raises():
    with pytest.raises(ValueError):
        symbol_stream_to_bits([0, 1], 2)


def test_symbol_stream_to_bits_zero_length():
    assert symbol_stream_to_bits([], 0) == b""


@pytest.mark.parametrize("symbol", range(8))
def test_symbol_to_color_round_trip(symbol):
    color = symbol_to_color(symbol)
    assert color_to_symbol(color) == symbol


def test_palette_matches_spec():
    expected = {
        0: (0, 0, 0),
        1: (255, 255, 255),
        2: (255, 0, 0),
        3: (0, 255, 0),
        4: (0, 0, 255),
        5: (0, 255, 255),
        6: (255, 0, 255),
        7: (255, 255, 0),
    }
    assert PALETTE == expected


@pytest.mark.parametrize("symbol", [-1, 8, 100])
def test_symbol_to_color_invalid_raises(symbol):
    with pytest.raises(ValueError):
        symbol_to_color(symbol)


@pytest.mark.parametrize(
    "color",
    [
        (1, 0, 0),
        (255, 255, 255, 0),
        (0, 0, 0, 0),
        (128, 128, 128),
    ],
)
def test_color_to_symbol_invalid_raises(color):
    with pytest.raises(ValueError):
        color_to_symbol(color)
