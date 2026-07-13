"""Tests for hex8.cli (Issue #8's `hex8 encode` and Issue #12's `hex8 decode` CLI)."""

from PIL import Image

from hex8.cli import main


def _write_payload(tmp_path, size=64):
    payload = bytes((i * 13 + 7) % 256 for i in range(size))
    input_path = tmp_path / "payload.bin"
    input_path.write_bytes(payload)
    return input_path, payload


def test_cli_encode_writes_a_valid_png(tmp_path):
    input_path, _payload = _write_payload(tmp_path)
    output_path = tmp_path / "marker.png"

    exit_code = main(["encode", str(input_path), str(output_path)])

    assert exit_code == 0
    assert output_path.exists()
    with Image.open(output_path) as image:
        assert image.format == "PNG"
        assert image.size[0] > 0 and image.size[1] > 0


def test_cli_encode_writes_a_valid_svg(tmp_path):
    input_path, _payload = _write_payload(tmp_path)
    output_path = tmp_path / "marker.svg"

    exit_code = main(["encode", str(input_path), str(output_path)])

    assert exit_code == 0
    text = output_path.read_text(encoding="utf-8")
    assert text.startswith("<svg")


def test_cli_encode_respects_radius_and_ecc_level_options(tmp_path):
    input_path, _payload = _write_payload(tmp_path, size=16)
    output_path = tmp_path / "marker.png"

    exit_code = main(
        [
            "encode",
            str(input_path),
            str(output_path),
            "--radius",
            "20",
            "--ecc-level",
            "25",
            "--cell-size",
            "4",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()


def test_cli_encode_rejects_unsupported_output_extension(tmp_path):
    input_path, _payload = _write_payload(tmp_path)
    output_path = tmp_path / "marker.bmp"

    exit_code = main(["encode", str(input_path), str(output_path)])

    assert exit_code != 0
    assert not output_path.exists()


def test_cli_decode_round_trips_a_cli_encoded_marker(tmp_path):
    input_path, payload = _write_payload(tmp_path)
    marker_path = tmp_path / "marker.png"
    output_path = tmp_path / "recovered.bin"

    assert main(["encode", str(input_path), str(marker_path)]) == 0
    exit_code = main(["decode", str(marker_path), str(output_path)])

    assert exit_code == 0
    assert output_path.read_bytes() == payload


def test_cli_decode_rejects_svg_input(tmp_path):
    input_path, _payload = _write_payload(tmp_path, size=16)
    marker_path = tmp_path / "marker.svg"
    output_path = tmp_path / "recovered.bin"

    assert main(["encode", str(input_path), str(marker_path)]) == 0
    exit_code = main(["decode", str(marker_path), str(output_path)])

    assert exit_code != 0
    assert not output_path.exists()
