"""Gridding RACS-all images, which span three ASKAP bands with gaps between them.

The bands, each of 1 MHz channels:

- RACS-low3: 799.5-1087.5 MHz
- RACS-mid: 1295.5-1439.5 MHz (the RFI-free upper half of its 288 MHz band)
- RACS-high: 1511.5-1799.5 MHz

wsclean labels an image of channels ``a..b`` with ``(f[a] + f[b]) / 2``, so
images straddling a gap, or of uneven width, land on a grid much finer than
their spacing. fitscube 2.10 refused most of these.
"""

from __future__ import annotations

import logging
import math
from functools import reduce

import astropy.units as u
import numpy as np
import pytest
from fitscube.combine_fits import even_spacing


def racs_all_channels(width: float) -> tuple[np.ndarray, np.ndarray]:
    """Channel index on the native grid, and centre frequency, of every channel."""
    per_mhz = 1e6 / width
    starts = np.array([0, 496, 712]) * per_mhz
    counts = np.array([288, 144, 288]) * per_mhz
    index = np.concatenate(
        [start + np.arange(count) for start, count in zip(starts, counts, strict=False)]
    ).astype(np.int64)
    return index, 799.5e6 + width / 2 + index * width


def image_centres(
    index: np.ndarray, frequencies: np.ndarray, bins: list[tuple[int, int]]
) -> tuple[np.ndarray, int]:
    """wsclean's frequency for each image, and the planes of the exact grid."""
    centres = np.array([(frequencies[a] + frequencies[b]) / 2 for a, b in bins])
    # Centres sit on a half-channel grid, so integers give the exact answer
    halves = sorted(int(index[a] + index[b]) for a, b in bins)
    step = reduce(math.gcd, np.diff(halves).tolist())
    return centres, (halves[-1] - halves[0]) // step + 1


def random_bins(n: int, size: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Contiguous bins of ``size`` channels, give or take a quarter."""
    bins = []
    start = 0
    while start < n:
        end = min(n, start + max(1, round(size * rng.uniform(0.75, 1.25))))
        bins.append((start, end - 1))
        start = end
    return bins


def check_grid(centres: np.ndarray, planes: int) -> None:
    new_specs, missing_chan_idx = even_spacing(centres * u.Hz)
    assert len(new_specs) == planes
    assert (~missing_chan_idx).sum() == len(centres)
    present = new_specs.to(u.Hz).value[~missing_chan_idx]
    assert np.allclose(present, np.sort(centres), rtol=0, atol=1)


@pytest.mark.parametrize("width", [0.5e6, 1e6, 2e6])
def test_uneven_bins(width: float) -> None:
    rng = np.random.default_rng(0)
    index, frequencies = racs_all_channels(width)
    for size in range(2, 41):
        bins = random_bins(len(index), size, rng)
        check_grid(*image_centres(index, frequencies, bins))


def test_wsclean_default_splits() -> None:
    """wsclean's own split into ``channels_out`` images of near-equal width."""
    index, frequencies = racs_all_channels(1e6)
    n = len(index)
    for channels_out in range(2, 85):
        bins = [
            (i * n // channels_out, (i + 1) * n // channels_out - 1)
            for i in range(channels_out)
        ]
        check_grid(*image_centres(index, frequencies, bins))


def test_jittered_centres_grid_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rng = np.random.default_rng(1)
    index, frequencies = racs_all_channels(1e6)
    for size in (5, 8, 16, 20):
        centres, planes = image_centres(
            index, frequencies, random_bins(len(index), size, rng)
        )
        centres += rng.uniform(-0.25, 0.25, len(centres))
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="fitscube"):
            check_grid(centres, planes)
        assert "not exactly on a regular grid" in caplog.text


def test_fine_channels() -> None:
    """ASKAP's native 18.5 kHz channels, imaged 54 at a time."""
    rng = np.random.default_rng(2)
    index, frequencies = racs_all_channels(1e6 / 54)
    centres, planes = image_centres(
        index, frequencies, random_bins(len(index), 54, rng)
    )
    assert planes > 100_000
    check_grid(centres, planes)
