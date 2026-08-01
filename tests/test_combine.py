"""Tests related to specific combine functionality"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from fitscube.combine_fits import check_for_any_beam, combine_fits


def test_check_for_any_beams_no_beams(file_list) -> None:
    """See if we can confirm is all beams are in fits files"""
    # file_list returns fits files without beam information
    assert not check_for_any_beam(file_list=file_list)


def test_check_for_any_beam_real_images(time_image_paths) -> None:
    """See if beam is in any of these images"""
    assert check_for_any_beam(file_list=time_image_paths)


def test_check_for_any_beam_one_beam(file_list_onebeam) -> None:
    """See if beam is in any of these images. Only one of the files should have the beamn properties"""
    assert check_for_any_beam(file_list=file_list_onebeam)


def test_combine_beam_not_in_first_file(
    file_list_onebeam: list[Path], output_file: Path
) -> None:
    """A beam only on the fourth input should still reach the output cube"""
    combine_fits(
        file_list=file_list_onebeam,
        out_cube=output_file,
        time_domain_mode=True,
        overwrite=True,
    )

    with fits.open(output_file) as hdul:
        assert hdul[1].name == "BEAMS"
        bmaj = hdul[1].data["BMAJ"]
        # Only plane 3 carries a beam; the rest are the NaN sentinel
        assert np.isclose(bmaj[3], 3600.0)
        assert np.all(bmaj[np.arange(len(bmaj)) != 3] < 1e-30)
