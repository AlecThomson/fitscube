"""Tests related to specific combine functionality"""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time
from fitscube.combine_fits import (
    check_for_any_beam,
    combine_fits,
    get_polarisation,
    make_beam_table,
)
from radio_beam import Beams


@pytest.mark.filterwarnings("ignore:'datfix' made the change")
def test_get_polarisation_is_axis_index(headers: dict[str, str]) -> None:
    """POL values are 0-based indices along the Stokes axis, not Stokes codes"""
    header = fits.Header.fromstring(headers["beams"])
    header["NAXIS4"] = 1
    header["CRVAL4"] = 3.0  # Stokes U, but a single plane
    assert get_polarisation(header).tolist() == [0]

    header["NAXIS4"] = 3
    assert get_polarisation(header).tolist() == [0, 1, 2]


@pytest.mark.filterwarnings("ignore:'datfix' made the change")
def test_make_beam_table_rejects_multi_stokes(headers: dict[str, str]) -> None:
    """Multi-Stokes beam tables are not supported yet"""
    header = fits.Header.fromstring(headers["beams"])
    header["NAXIS4"] = 3
    beams = Beams(
        major=np.ones(4) * u.arcsec,
        minor=np.ones(4) * u.arcsec,
        pa=np.zeros(4) * u.deg,
    )

    with pytest.raises(NotImplementedError):
        make_beam_table(beams, header)


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
    "stokes_code",
    [1, 2, 3, 4],  # I, Q, U, V
)
def test_combine_beam_pol_is_zero_index_for_single_stokes(
    tmp_path: Path,
    even_specs: u.Quantity,
    stokes_code: int,
) -> None:
    """POL/CHAN are 0-based axis indices, not FITS Stokes codes."""
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
        assert hdul[0].header["NAXIS3"] == len(even_specs)
        assert hdul[0].header["NAXIS4"] == 1
        assert np.all(hdul[1].data["POL"] == 0)
        assert np.array_equal(hdul[1].data["CHAN"], np.arange(len(even_specs)))
