"""Tests related to specific combine functionality"""

from __future__ import annotations

from itertools import permutations
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


N_CHAN = 3
MJD_START = 60000.0
STOKES_CODES = (1, 2, 3, 4, -5)  # I, Q, U, V, XX
CELESTIAL_ORDERS = (("RA", "DEC"), ("DEC", "RA"))


def _layouts() -> list[tuple[bool, tuple[str, ...], tuple[str, ...]]]:
    """Every supported (time_domain_mode, celestial, extra axes) input layout.

    RA/DEC stay on axes 1 and 2 in either order. Any ordered subset of FREQ,
    TIME and STOKES follows. Frequency mode needs a FREQ axis unless the image
    is 2D (REFFREQ); time mode reads DATE-OBS, so any subset works there and a
    missing TIME axis is appended to the cube.
    """
    extra_layouts = [
        order
        for size in range(4)
        for order in permutations(("FREQ", "TIME", "STOKES"), size)
    ]
    return [
        (time_domain_mode, celestial, extra)
        for time_domain_mode in (False, True)
        for celestial in CELESTIAL_ORDERS
        for extra in extra_layouts
        if time_domain_mode or not extra or "FREQ" in extra
    ]


def _layout_id(layout: tuple[bool, tuple[str, ...], tuple[str, ...]]) -> str:
    time_domain_mode, celestial, extra = layout
    mode = "time" if time_domain_mode else "freq"
    return f"{mode}-{','.join((*celestial, *extra))}"


LAYOUTS = _layouts()


def _write_planes(
    tmp_path: Path,
    celestial: tuple[str, ...],
    extra_axes: tuple[str, ...],
    stokes_code: int,
    n_chan: int = N_CHAN,
) -> list[Path]:
    """Write single-channel images in the given axis layout.

    Plane ``i`` holds the value ``i`` and its own beam, so the cube's plane
    order can be checked and a BEAMS table is always written.
    """
    file_list = []
    for i in range(n_chan):
        time = Time(MJD_START + i * 10 / 86400, format="mjd")
        axes = {
            "RA": {"CTYPE": "RA---SIN", "CRVAL": 0.0, "CDELT": -1e-3, "CRPIX": 5.0},
            "DEC": {"CTYPE": "DEC--SIN", "CRVAL": 0.0, "CDELT": 1e-3, "CRPIX": 5.0},
            "FREQ": {
                "CTYPE": "FREQ",
                "CRVAL": 1e9 + i * 1e6,
                "CDELT": 1e6,
                "CRPIX": 1.0,
                "CUNIT": "Hz",
            },
            "TIME": {
                "CTYPE": "TIME",
                "CRVAL": time.mjd * 86400,
                "CDELT": 10.0,
                "CRPIX": 1.0,
                "CUNIT": "s",
            },
            "STOKES": {
                "CTYPE": "STOKES",
                "CRVAL": float(stokes_code),
                "CDELT": 1.0,
                "CRPIX": 1.0,
            },
        }
        header = fits.Header()
        for fits_axis, name in enumerate((*celestial, *extra_axes), start=1):
            for key, value in axes[name].items():
                header[f"{key}{fits_axis}"] = value
        header["DATE-OBS"] = time.isot
        header["MJD-OBS"] = time.mjd
        if not extra_axes:
            header["REFFREQ"] = 1e9 + i * 1e6
        # Vary the beam per plane so a beam table is written
        header["BMAJ"] = 1e-3 * (1 + i)
        header["BMIN"] = 1e-3
        header["BPA"] = 0.0

        shape = (1,) * len(extra_axes) + (10, 10)
        path = tmp_path / f"plane_{i}.fits"
        fits.PrimaryHDU(np.full(shape, float(i)), header=header).writeto(path)
        file_list.append(path)
    return file_list


@pytest.mark.parametrize(
    ("layout", "stokes_code"),
    [
        (layout, STOKES_CODES[idx % len(STOKES_CODES)])
        for idx, layout in enumerate(LAYOUTS)
    ],
    ids=[_layout_id(layout) for layout in LAYOUTS],
)
def test_combine_axis_layouts(
    tmp_path: Path,
    layout: tuple[bool, tuple[str, ...], tuple[str, ...]],
    stokes_code: int,
) -> None:
    """Every input axis layout gives the right cube axes, plane order and BEAMS"""
    time_domain_mode, celestial, extra_axes = layout
    out_cube = tmp_path / "out.fits"
    combine_fits(
        file_list=_write_planes(tmp_path, celestial, extra_axes, stokes_code),
        out_cube=out_cube,
        time_domain_mode=time_domain_mode,
        overwrite=True,
    )

    with fits.open(out_cube) as hdul:
        header = hdul[0].header
        n_axes = header["NAXIS"]
        ctypes = [header[f"CTYPE{axis}"] for axis in range(1, n_axes + 1)]
        combine_ctype = "TIME" if time_domain_mode else "FREQ"
        assert ctypes.count(combine_ctype) == 1
        combine_axis = ctypes.index(combine_ctype) + 1

        # Axes: the combine axis holds every plane, the rest are degenerate
        for axis in range(3, n_axes + 1):
            expected = N_CHAN if axis == combine_axis else 1
            assert header[f"NAXIS{axis}"] == expected, ctypes
        # No Stokes axis is invented for inputs that lack one
        assert ("STOKES" in ctypes) == ("STOKES" in extra_axes)
        if "STOKES" in ctypes:
            assert header[f"CRVAL{ctypes.index('STOKES') + 1}"] == stokes_code

        # Plane i of the combine axis holds input i
        planes = np.moveaxis(hdul[0].data, n_axes - combine_axis, 0)
        for i, plane in enumerate(planes.reshape(N_CHAN, -1)):
            assert np.all(plane == i)

        beam_table = hdul["BEAMS"]
        assert beam_table.header["NCHAN"] == N_CHAN
        assert beam_table.header["NPOL"] == 1
        assert np.array_equal(beam_table.data["CHAN"], np.arange(N_CHAN))
        assert np.all(beam_table.data["POL"] == 0)


def test_combine_rejects_multi_stokes_beams(tmp_path: Path) -> None:
    """A multi-Stokes cube with varying beams fails before any plane is written"""
    file_list = []
    for i in range(N_CHAN):
        header = fits.Header()
        header["CTYPE3"], header["CRVAL3"] = "STOKES", 1.0
        header["CDELT3"], header["CRPIX3"] = 1.0, 1.0
        header["CTYPE4"], header["CRVAL4"] = "FREQ", 1e9 + i * 1e6
        header["CDELT4"], header["CRPIX4"], header["CUNIT4"] = 1e6, 1.0, "Hz"
        header["BMAJ"], header["BMIN"], header["BPA"] = 1e-3 * (1 + i), 1e-3, 0.0
        path = tmp_path / f"plane_{i}.fits"
        fits.PrimaryHDU(np.full((1, 2, 10, 10), 1.0 + i), header=header).writeto(path)
        file_list.append(path)

    out_cube = tmp_path / "out.fits"
    with pytest.raises(NotImplementedError, match="single-Stokes"):
        combine_fits(file_list=file_list, out_cube=out_cube, overwrite=True)

    # The data were never written: every plane is still the blank fill
    assert np.all(fits.getdata(out_cube) == 0)
