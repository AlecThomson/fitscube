"""Tests related to specific combine functionality"""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time
from fitscube.combine_fits import check_for_any_beam, combine_fits, get_polarisation


@pytest.mark.filterwarnings("ignore:'datfix' made the change")
def test_get_polarisation_uses_crval(headers: dict[str, str]) -> None:
    """POL must track CRVAL4 (the actual Stokes code)"""
    header = fits.Header.fromstring(headers["beams"])
    assert get_polarisation(header) == 0  # CRVAL4 == 1.0 -> Stokes I

    header["CRVAL4"] = 2.0  # Stokes Q
    assert get_polarisation(header) == 1


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


@pytest.mark.parametrize(
    ("stokes_code", "expected_pol"),
    [(1, 0), (2, 1), (3, 2), (4, 3)],  # I, Q, U, V
)
def test_combine_beam_polarisation_matches_stokes(
    tmp_path: Path,
    even_specs: u.Quantity,
    stokes_code: int,
    expected_pol: int,
) -> None:
    """Each Stokes plane's own beam table must carry its own POL, not always 0."""
    image = np.ones((1, 1, 10, 10))
    file_list = []
    for i, spec in enumerate(even_specs):
        header = fits.Header()
        header["CRVAL3"] = spec.to(u.s).value
        header["CDELT3"] = 9.98
        header["CRPIX3"] = 1
        header["CTYPE3"] = "TIME"
        header["CUNIT3"] = "s"
        header["CTYPE4"] = "STOKES"
        header["CRPIX4"] = 1.0
        header["CRVAL4"] = float(stokes_code)
        header["CDELT4"] = 1.0
        header["DATE-OBS"] = Time(spec.to(u.d).value, format="mjd").isot
        header["MJD-OBS"] = Time(spec.to(u.d).value, format="mjd").mjd
        # Vary the beam per plane so it isn't collapsed to a single constant beam.
        header["BMAJ"] = 1.0 + i * 0.1
        header["BMIN"] = 1.0
        header["BPA"] = 1.0

        path = tmp_path / f"plane_{i}.fits"
        fits.PrimaryHDU(image * i, header=header).writeto(path, overwrite=True)
        file_list.append(path)

    out_cube = tmp_path / "out.fits"
    combine_fits(
        file_list=file_list,
        out_cube=out_cube,
        time_domain_mode=True,
        overwrite=True,
    )

    with fits.open(out_cube) as hdul:
        assert hdul[0].header["CRVAL4"] == stokes_code
        assert hdul[1].name == "BEAMS"
        assert set(hdul[1].data["POL"].tolist()) == {expected_pol}
