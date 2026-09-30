"""End-to-end combines of image pairs across input axis layouts

Each case combines two single-plane images and states the cube it expects in
the notation ``(RA, DEC, FREQ) + (RA, DEC, FREQ) -> (RA, DEC, FREQ)``, or the
error it expects. Axes are named by CTYPE. A Stokes symbol (e.g. ``I`` or
``XX``) is a STOKES axis holding that parameter, and ``FOO`` is an axis type
fitscube knows nothing about.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import StokesCoord
from astropy.coordinates.polarization import FITS_STOKES_VALUE_SYMBOL_MAP
from astropy.io import fits
from astropy.time import Time
from fitscube.combine_fits import combine_fits
from fitscube.exceptions import AxisMismatchException, StokesMismatchException

Axes = tuple[str, ...]
WritePlane = Callable[[Axes, int], Path]


def _is_stokes(axis: str) -> bool:
    return axis in {stokes.symbol for stokes in FITS_STOKES_VALUE_SYMBOL_MAP.values()}


def _axes_id(value: object) -> str:
    if isinstance(value, bool):
        return "time" if value else "freq"
    if isinstance(value, tuple):
        return f"({','.join(value)})"
    if isinstance(value, type):
        return value.__name__
    return str(value)


@pytest.fixture
def write_plane(tmp_path: Path) -> WritePlane:
    """Write the ``index``-th single-plane image with the given axes.

    Plane ``index`` holds the value ``index`` and its own beam, is one step
    later in frequency and time than the plane before, and each non-celestial
    axis has length one.
    """

    def _write_plane(axes: Axes, index: int) -> Path:
        time = Time(60000.0 + index * 10 / 86400, format="mjd")
        cards = {
            "RA": {"CTYPE": "RA---SIN", "CRVAL": 0.0, "CDELT": -1e-3, "CRPIX": 5.0},
            "DEC": {"CTYPE": "DEC--SIN", "CRVAL": 0.0, "CDELT": 1e-3, "CRPIX": 5.0},
            "FREQ": {
                "CTYPE": "FREQ",
                "CRVAL": 1e9 + index * 1e6,
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
            "FOO": {"CTYPE": "FOO", "CRVAL": 1.0, "CDELT": 1.0, "CRPIX": 1.0},
        }
        header = fits.Header()
        for fits_axis, axis in enumerate(axes, start=1):
            axis_cards = (
                {
                    "CTYPE": "STOKES",
                    "CRVAL": float(StokesCoord(axis).value),
                    "CDELT": 1.0,
                    "CRPIX": 1.0,
                }
                if _is_stokes(axis)
                else cards[axis]
            )
            for key, value in axis_cards.items():
                header[f"{key}{fits_axis}"] = value
        header["DATE-OBS"] = time.isot
        header["MJD-OBS"] = time.mjd
        if len(axes) == 2:
            header["REFFREQ"] = 1e9 + index * 1e6
        # Vary the beam per plane so a beam table is written
        header["BMAJ"] = 1e-3 * (1 + index)
        header["BMIN"] = 1e-3
        header["BPA"] = 0.0

        shape = (1,) * (len(axes) - 2) + (10, 10)
        path = tmp_path / f"plane_{index}.fits"
        fits.PrimaryHDU(np.full(shape, float(index)), header=header).writeto(path)
        return path

    return _write_plane


@pytest.fixture
def out_cube(tmp_path: Path) -> Path:
    return tmp_path / "cube.fits"


@pytest.mark.parametrize(
    ("time_domain_mode", "first", "second", "expected"),
    [
        # Frequency cubes
        (False, ("RA", "DEC"), ("RA", "DEC"), ("RA", "DEC", "FREQ")),
        (False, ("RA", "DEC", "FREQ"), ("RA", "DEC", "FREQ"), ("RA", "DEC", "FREQ")),
        (False, ("DEC", "RA", "FREQ"), ("DEC", "RA", "FREQ"), ("DEC", "RA", "FREQ")),
        (
            False,
            ("RA", "DEC", "FOO", "FREQ"),
            ("RA", "DEC", "FOO", "FREQ"),
            ("RA", "DEC", "FOO", "FREQ"),
        ),
        (
            False,
            ("RA", "DEC", "FREQ", "FOO"),
            ("RA", "DEC", "FREQ", "FOO"),
            ("RA", "DEC", "FREQ", "FOO"),
        ),
        # wsclean axis order
        (
            False,
            ("RA", "DEC", "FREQ", "I"),
            ("RA", "DEC", "FREQ", "I"),
            ("RA", "DEC", "FREQ", "STOKES"),
        ),
        # CASA axis order
        (
            False,
            ("RA", "DEC", "U", "FREQ"),
            ("RA", "DEC", "U", "FREQ"),
            ("RA", "DEC", "STOKES", "FREQ"),
        ),
        (
            False,
            ("RA", "DEC", "FREQ", "XX"),
            ("RA", "DEC", "FREQ", "XX"),
            ("RA", "DEC", "FREQ", "STOKES"),
        ),
        (
            False,
            ("RA", "DEC", "TIME", "FREQ"),
            ("RA", "DEC", "TIME", "FREQ"),
            ("RA", "DEC", "TIME", "FREQ"),
        ),
        (
            False,
            ("RA", "DEC", "FREQ", "TIME", "V"),
            ("RA", "DEC", "FREQ", "TIME", "V"),
            ("RA", "DEC", "FREQ", "TIME", "STOKES"),
        ),
        # Time cubes. A missing TIME axis is appended.
        (True, ("RA", "DEC"), ("RA", "DEC"), ("RA", "DEC", "TIME")),
        (True, ("RA", "DEC", "TIME"), ("RA", "DEC", "TIME"), ("RA", "DEC", "TIME")),
        (
            True,
            ("RA", "DEC", "FREQ"),
            ("RA", "DEC", "FREQ"),
            ("RA", "DEC", "FREQ", "TIME"),
        ),
        (
            True,
            ("RA", "DEC", "FREQ", "I"),
            ("RA", "DEC", "FREQ", "I"),
            ("RA", "DEC", "FREQ", "STOKES", "TIME"),
        ),
        (
            True,
            ("RA", "DEC", "Q", "TIME"),
            ("RA", "DEC", "Q", "TIME"),
            ("RA", "DEC", "STOKES", "TIME"),
        ),
        (
            True,
            ("RA", "DEC", "TIME", "FOO"),
            ("RA", "DEC", "TIME", "FOO"),
            ("RA", "DEC", "TIME", "FOO"),
        ),
    ],
    ids=_axes_id,
)
def test_combine_pair(
    write_plane: WritePlane,
    out_cube: Path,
    time_domain_mode: bool,
    first: Axes,
    second: Axes,
    expected: Axes,
) -> None:
    """first + second -> expected, with the planes in order and a valid BEAMS table"""
    combine_fits(
        file_list=[write_plane(first, 0), write_plane(second, 1)],
        out_cube=out_cube,
        time_domain_mode=time_domain_mode,
        overwrite=True,
    )

    with fits.open(out_cube) as hdul:
        header = hdul[0].header
        n_axes = header["NAXIS"]
        ctypes = tuple(
            header[f"CTYPE{axis}"].split("-")[0] for axis in range(1, n_axes + 1)
        )
        assert ctypes == expected

        combine_axis = ctypes.index("TIME" if time_domain_mode else "FREQ") + 1
        naxes = tuple(header[f"NAXIS{axis}"] for axis in range(1, n_axes + 1))
        assert naxes == tuple(
            10 if ctype in ("RA", "DEC") else 2 if axis == combine_axis else 1
            for axis, ctype in enumerate(ctypes, start=1)
        )
        stokes = [axis for axis in first if _is_stokes(axis)]
        if stokes:
            stokes_axis = ctypes.index("STOKES") + 1
            assert header[f"CRVAL{stokes_axis}"] == StokesCoord(stokes[0]).value

        # Plane i along the combine axis holds input i
        planes = np.moveaxis(hdul[0].data, n_axes - combine_axis, 0)
        assert np.all(planes[0] == 0)
        assert np.all(planes[1] == 1)

        beam_table = hdul["BEAMS"]
        assert beam_table.header["NCHAN"] == 2
        assert beam_table.header["NPOL"] == 1
        assert beam_table.data["CHAN"].tolist() == [0, 1]
        assert beam_table.data["POL"].tolist() == [0, 0]


@pytest.mark.parametrize(
    ("time_domain_mode", "first", "second", "error"),
    [
        (
            False,
            ("RA", "DEC", "FREQ"),
            ("DEC", "RA", "FREQ"),
            AxisMismatchException,
        ),
        (False, ("RA", "DEC"), ("RA", "DEC", "FREQ"), AxisMismatchException),
        (
            False,
            ("RA", "DEC", "FREQ"),
            ("RA", "DEC", "FOO", "FREQ"),
            AxisMismatchException,
        ),
        (
            False,
            ("RA", "DEC", "FREQ", "I"),
            ("RA", "DEC", "I", "FREQ"),
            AxisMismatchException,
        ),
        (
            False,
            ("RA", "DEC", "FREQ", "I"),
            ("RA", "DEC", "FREQ"),
            AxisMismatchException,
        ),
        (
            False,
            ("RA", "DEC", "FREQ", "I"),
            ("RA", "DEC", "FREQ", "Q"),
            StokesMismatchException,
        ),
        (
            False,
            ("RA", "DEC", "FREQ", "XX"),
            ("RA", "DEC", "FREQ", "YY"),
            StokesMismatchException,
        ),
        # Frequency mode needs a FREQ axis, or REFFREQ on a 2D image
        (False, ("RA", "DEC", "FOO"), ("RA", "DEC", "FOO"), ValueError),
        (True, ("RA", "DEC", "TIME"), ("RA", "DEC", "FREQ"), AxisMismatchException),
        (
            True,
            ("RA", "DEC", "TIME", "I"),
            ("RA", "DEC", "TIME", "Q"),
            StokesMismatchException,
        ),
    ],
    ids=_axes_id,
)
def test_combine_pair_raises(
    write_plane: WritePlane,
    out_cube: Path,
    time_domain_mode: bool,
    first: Axes,
    second: Axes,
    error: type[Exception],
) -> None:
    """first + second -> error, before any output is written"""
    with pytest.raises(error):
        combine_fits(
            file_list=[write_plane(first, 0), write_plane(second, 1)],
            out_cube=out_cube,
            time_domain_mode=time_domain_mode,
            overwrite=True,
        )
    assert not out_cube.exists()


def test_combine_rejects_multi_stokes_beams(tmp_path: Path, out_cube: Path) -> None:
    """A multi-Stokes cube with varying beams fails before any plane is written"""
    file_list = []
    for index in range(2):
        header = fits.Header()
        header["CTYPE3"], header["CRVAL3"] = "STOKES", 1.0
        header["CDELT3"], header["CRPIX3"] = 1.0, 1.0
        header["CTYPE4"], header["CRVAL4"] = "FREQ", 1e9 + index * 1e6
        header["CDELT4"], header["CRPIX4"], header["CUNIT4"] = 1e6, 1.0, "Hz"
        header["BMAJ"] = 1e-3 * (1 + index)
        header["BMIN"], header["BPA"] = 1e-3, 0.0
        path = tmp_path / f"plane_{index}.fits"
        fits.PrimaryHDU(np.full((1, 2, 10, 10), 1.0 + index), header=header).writeto(
            path
        )
        file_list.append(path)

    with pytest.raises(NotImplementedError, match="single-Stokes"):
        combine_fits(file_list=file_list, out_cube=out_cube, overwrite=True)

    # The data were never written: every plane is still the blank fill
    assert np.all(fits.getdata(out_cube) == 0)
