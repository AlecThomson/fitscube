"""Tests for optional cube compression"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from fitscube.combine_fits import combine_fits, compress_cube


@pytest.mark.parametrize("method", ["gzip", "pgzip"])
def test_compress_cube_roundtrip(
    file_list_onebeam: list[Path], tmp_path: Path, method: str
) -> None:
    """Compressing a finished cube should be lossless and remove the original"""
    output_file = tmp_path / "test.fits"
    combine_fits(
        file_list=file_list_onebeam,
        out_cube=output_file,
        time_domain_mode=True,
        overwrite=True,
    )

    with fits.open(output_file) as hdul:
        original_data = hdul[0].data.copy()
        original_beams = hdul[1].data.copy()

    compressed_path = compress_cube(output_file, method=method)

    assert compressed_path.name == output_file.name + ".gz"
    assert not output_file.exists()
    assert compressed_path.exists()

    with fits.open(compressed_path) as hdul:
        np.testing.assert_array_equal(hdul[0].data, original_data)
        assert hdul[1].name == "BEAMS"
        np.testing.assert_array_equal(hdul[1].data["BMAJ"], original_beams["BMAJ"])
