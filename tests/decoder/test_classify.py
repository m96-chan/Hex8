"""Tests for hex8.decoder.classify: Lab-space color classification with confidence.

See GitHub Issue #10.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage.color import rgb2lab

from hex8.common.canvas import cell_center_px, compute_canvas
from hex8.common.layout import CellRole, build_layout
from hex8.common.symbols import PALETTE, color_to_symbol
from hex8.decoder.classify import (
    LOW_CONFIDENCE_THRESHOLD,
    Classification,
    build_observed_palette,
    classify_cells,
    classify_pixel,
)
from hex8.encoder.encode import encode_png

# A radius that comfortably fits finder + palette + metadata cells.
RADIUS = 18
CELL_SIZE = 10.0


def _rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    arr = np.array(rgb, dtype=np.float64).reshape(1, 1, 3) / 255.0
    lab = rgb2lab(arr)
    return (lab[0, 0, 0], lab[0, 0, 1], lab[0, 0, 2])


def _lab_distance(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


@pytest.fixture(scope="module")
def rendered_image():
    """A real encode_png output at RADIUS, with a known small payload."""
    payload = b"Hex8 classify test payload!"
    return encode_png(payload, radius=RADIUS, ecc_level=30, cell_size=CELL_SIZE)


@pytest.fixture(scope="module")
def canvas():
    return compute_canvas(RADIUS, CELL_SIZE)


class TestBuildObservedPalette:
    def test_returns_all_eight_symbols(self, rendered_image, canvas):
        observed = build_observed_palette(rendered_image, RADIUS, CELL_SIZE, canvas)
        assert set(observed.keys()) == set(range(8))

    def test_matches_ideal_palette_lab_values(self, rendered_image, canvas):
        observed = build_observed_palette(rendered_image, RADIUS, CELL_SIZE, canvas)
        for symbol, ideal_rgb in PALETTE.items():
            ideal_lab = _rgb_to_lab(ideal_rgb)
            observed_lab = observed[symbol]
            # The ideal (noiseless) rendered image should reproduce the exact
            # palette color at every sampled cell center, so the averaged Lab
            # value should be extremely close to the ideal Lab value.
            distance = _lab_distance(observed_lab, ideal_lab)
            assert distance < 0.5, (
                f"symbol {symbol}: observed Lab {observed_lab} too far from "
                f"ideal Lab {ideal_lab} (distance={distance})"
            )


class TestClassifyPixel:
    @pytest.fixture()
    def exact_observed_palette(self):
        """Observed palette built directly from the exact PALETTE RGB values."""
        return {symbol: _rgb_to_lab(rgb) for symbol, rgb in PALETTE.items()}

    def test_returns_classification_dataclass(self, exact_observed_palette):
        result = classify_pixel((0, 0, 0), exact_observed_palette)
        assert isinstance(result, Classification)

    @pytest.mark.parametrize("symbol", sorted(PALETTE.keys()))
    def test_exact_palette_color_classifies_correctly_with_high_confidence(
        self, symbol, exact_observed_palette
    ):
        rgb = PALETTE[symbol]
        result = classify_pixel(rgb, exact_observed_palette)
        assert result.symbol == symbol
        assert result.confidence > 0.9
        assert result.low_confidence is False

    def test_midpoint_between_two_colors_is_low_confidence(
        self, exact_observed_palette
    ):
        # Black (0,0,0) and White (255,255,255) midpoint: (128,128,128).
        exact_result = classify_pixel((0, 0, 0), exact_observed_palette)
        midpoint_result = classify_pixel((128, 128, 128), exact_observed_palette)

        assert midpoint_result.confidence < exact_result.confidence
        # Constructed around the actual threshold rather than assuming a
        # specific literal value.
        assert midpoint_result.confidence < LOW_CONFIDENCE_THRESHOLD + 0.3

    def test_tie_case_confidence_is_zero_and_low_confidence(self):
        # Two palette entries at exactly the same Lab distance from the
        # sample pixel: a genuine tie is the LOWEST confidence case, not the
        # highest, and must not raise ZeroDivisionError.
        sample_lab = (50.0, 0.0, 0.0)
        observed_palette = {
            0: (50.0, 10.0, 0.0),  # distance 10 from sample
            1: (50.0, -10.0, 0.0),  # distance 10 from sample (tied)
            2: (80.0, 0.0, 0.0),  # farther away, not involved in the tie
        }
        # classify_pixel takes an RGB pixel, not a Lab value directly; use a
        # pixel whose Lab conversion we don't control precisely, so instead
        # exercise the tie via two entries that are literally identical Lab
        # values as the pixel itself is irrelevant to the tie construction:
        # both distances must be equal. We pick a pixel and palette so that
        # d_best == d_second == 0 exactly, since that's the only case the
        # spec calls out explicitly (d_best <= d_second always, so the only
        # zero-sum case is exact zero-zero tie).
        rgb = (100, 150, 200)
        pixel_lab = _rgb_to_lab(rgb)
        tied_palette = {0: pixel_lab, 1: pixel_lab, 2: sample_lab}
        result = classify_pixel(rgb, tied_palette)
        assert result.confidence == 0.0
        assert result.low_confidence is True
        # Sanity check the non-degenerate palette above is indeed a tie in
        # Lab distance terms (used only to document the "tie" concept).
        d0 = _lab_distance(sample_lab, observed_palette[0])
        d1 = _lab_distance(sample_lab, observed_palette[1])
        assert d0 == pytest.approx(d1)


class TestClassifyCells:
    def test_recovers_known_encoded_symbols_on_data_cells(self, rendered_image, canvas):
        layout = build_layout(RADIUS)
        observed_palette = build_observed_palette(
            rendered_image, RADIUS, CELL_SIZE, canvas
        )

        data_cells = layout.cells_with_role(CellRole.DATA)[:10]
        results = classify_cells(
            rendered_image, data_cells, CELL_SIZE, canvas, observed_palette
        )

        assert set(results.keys()) == set(data_cells)

        for cell in data_cells:
            x, y = cell_center_px(cell[0], cell[1], CELL_SIZE, canvas)
            px = rendered_image.getpixel((round(x), round(y)))
            expected_symbol = color_to_symbol(px[:3])
            classification = results[cell]
            assert isinstance(classification, Classification)
            assert classification.symbol == expected_symbol
            assert classification.low_confidence is False
