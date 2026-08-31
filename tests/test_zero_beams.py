"""Tests for blanking planes whose restoring beam is exactly zero.

wsclean writes BMAJ = BMIN = 0 when a plane holds no fitted PSF. With
`-fit-spectral-pol` that plane is the model image, which looks like real
data but is not comparable to the rest of the cube.
"""

from __future__ import annotations

import logging
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from fitscube.combine_fits import (
    cli,
    combine_fits,
    find_zero_beams,
    get_parser,
    nan_zero_beams,
)
from radio_beam import Beams

from tests.test_combine_guards import make_plane


@pytest.fixture
def specs() -> u.Quantity:
    return np.arange(4) * 1e6 * u.Hz + 1e9 * u.Hz


@pytest.fixture
def zero_beam_file_list(tmp_path: Path, specs: u.Quantity) -> list[Path]:
    """Four planes, where the second carries a zero beam."""
    return [
        make_plane(
            tmp_path / f"plane_{i}.fits",
            spec,
            value=float(i + 1),
            beam=0.0 if i == 1 else 1e-3 * (i + 1),
        )
        for i, spec in enumerate(specs)
    ]


def test_find_zero_beams() -> None:
    beams = Beams(
        major=[1.0, 0.0, 2.0, np.nan] * u.arcsec,
        minor=[1.0, 0.0, 0.0, np.nan] * u.arcsec,
        pa=[0.0, 0.0, 0.0, np.nan] * u.deg,
    )
    # A zero major or a zero minor is degenerate; a NaN beam is not a zero beam,
    # and a zero position angle is perfectly legitimate.
    assert np.array_equal(find_zero_beams(beams), [False, True, True, False])


def test_nan_zero_beams() -> None:
    beams = Beams(
        major=[1.0, 0.0] * u.arcsec,
        minor=[1.0, 0.0] * u.arcsec,
        pa=[10.0, 0.0] * u.deg,
    )
    blanked = nan_zero_beams(beams, np.array([False, True]))

    assert np.isclose(blanked.major[0].to(u.arcsec).value, 1.0)
    assert np.isnan(blanked.major[1].value)
    assert np.isnan(blanked.minor[1].value)
    assert np.isnan(blanked.pa[1].value)
    # The input must not be mutated
    assert np.isclose(beams.major[1].to(u.arcsec).value, 0.0)


def test_zero_beam_plane_is_blanked(
    tmp_path: Path, zero_beam_file_list: list[Path]
) -> None:
    """The zero-beam plane is NaN-ed out; every other plane is untouched."""
    out_cube = tmp_path / "cube.fits"
    combine_fits(file_list=zero_beam_file_list, out_cube=out_cube, overwrite=True)

    with fits.open(out_cube) as hdu_list:
        cube = np.squeeze(hdu_list[0].data)
        beam_table = hdu_list["BEAMS"]
        majors = beam_table.data["BMAJ"]
        minors = beam_table.data["BMIN"]

    assert np.isnan(cube[1]).all()
    assert np.allclose(cube[0], 1.0)
    assert np.allclose(cube[2], 3.0)
    assert np.allclose(cube[3], 4.0)

    # The blanked beam is stored with the same NaN sentinel as any other NaN PSF.
    # Compare exactly: np.isclose(0.0, tiny) is True, so it cannot tell the
    # sentinel apart from the zero it is meant to replace.
    tiny = np.finfo(np.float32).tiny
    assert majors[1] == tiny
    assert minors[1] == tiny
    assert np.isclose(majors[0], (1e-3 * u.deg).to(u.arcsec).value)


def test_zero_beam_warns(
    tmp_path: Path,
    zero_beam_file_list: list[Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="fitscube"):
        combine_fits(
            file_list=zero_beam_file_list,
            out_cube=tmp_path / "cube.fits",
            overwrite=True,
        )

    assert "restoring beam of exactly zero" in caplog.text
    assert "[1]" in caplog.text


def test_zero_beam_opt_out(tmp_path: Path, zero_beam_file_list: list[Path]) -> None:
    """With blank_zero_beams=False the image data is kept."""
    out_cube = tmp_path / "cube.fits"
    combine_fits(
        file_list=zero_beam_file_list,
        out_cube=out_cube,
        overwrite=True,
        blank_zero_beams=False,
    )

    with fits.open(out_cube) as hdu_list:
        cube = np.squeeze(hdu_list[0].data)
        majors = hdu_list["BEAMS"].data["BMAJ"]

    assert np.allclose(cube[1], 2.0)
    # The data survives, but a literal zero must never reach the beam table
    assert majors[1] == np.finfo(np.float32).tiny


def test_beam_table_never_holds_a_zero_beam(
    tmp_path: Path, zero_beam_file_list: list[Path]
) -> None:
    """A zero BMAJ/BMIN always becomes the sentinel, whichever way we are called."""
    tiny = np.finfo(np.float32).tiny

    for blank in (True, False):
        out_cube = tmp_path / f"cube_{blank}.fits"
        combine_fits(
            file_list=zero_beam_file_list,
            out_cube=out_cube,
            overwrite=True,
            blank_zero_beams=blank,
        )

        with fits.open(out_cube) as hdu_list:
            table = hdu_list["BEAMS"].data

        assert not (table["BMAJ"] == 0.0).any()
        assert not (table["BMIN"] == 0.0).any()
        assert table["BMAJ"][1] == tiny
        assert table["BMIN"][1] == tiny
        assert table["BPA"][1] == tiny
        # A zero position angle on a real beam is legitimate and must survive.
        # make_plane writes BPA = 0 for every plane, so the untouched channels
        # prove the sentinel is applied per-beam, not per-column.
        assert (table["BPA"][[0, 2, 3]] == 0.0).all()


def test_zero_beam_opt_out_still_warns(
    tmp_path: Path,
    zero_beam_file_list: list[Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="fitscube"):
        combine_fits(
            file_list=zero_beam_file_list,
            out_cube=tmp_path / "cube.fits",
            overwrite=True,
            blank_zero_beams=False,
        )

    assert "restoring beam of exactly zero" in caplog.text
    assert "kept as-is" in caplog.text


def test_no_zero_beams_are_untouched(tmp_path: Path, specs: u.Quantity) -> None:
    """Cubes without a zero beam behave exactly as before."""
    file_list = [
        make_plane(
            tmp_path / f"plane_{i}.fits", spec, value=float(i + 1), beam=1e-3 * (i + 1)
        )
        for i, spec in enumerate(specs)
    ]
    out_cube = tmp_path / "cube.fits"
    combine_fits(file_list=file_list, out_cube=out_cube, overwrite=True)

    with fits.open(out_cube) as hdu_list:
        cube = np.squeeze(hdu_list[0].data)

    assert not np.isnan(cube).any()


def test_zero_beam_and_blank_channels(tmp_path: Path, specs: u.Quantity) -> None:
    """A zero beam and a --create-blanks gap must not shift each other."""
    gapped = [specs[0], specs[2], specs[3]]
    beams = [1e-3, 0.0, 3e-3]
    file_list = [
        make_plane(tmp_path / f"gap_{i}.fits", spec, value=float(i + 1), beam=beam)
        for i, (spec, beam) in enumerate(zip(gapped, beams, strict=True))
    ]

    out_cube = tmp_path / "cube.fits"
    combine_fits(
        file_list=file_list, out_cube=out_cube, create_blanks=True, overwrite=True
    )

    with fits.open(out_cube) as hdu_list:
        cube = np.squeeze(hdu_list[0].data)
        majors = hdu_list["BEAMS"].data["BMAJ"]

    tiny = np.finfo(np.float32).tiny
    # Channel 1 is the missing channel, channel 2 is the zero-beam plane
    assert np.allclose(cube[0], 1.0)
    assert np.isnan(cube[1]).all()
    assert np.isnan(cube[2]).all()
    assert np.allclose(cube[3], 3.0)
    assert majors[1] == tiny
    assert majors[2] == tiny
    assert np.isclose(majors[3], (3e-3 * u.deg).to(u.arcsec).value)


def test_cli_no_blank_zero_beams(
    tmp_path: Path, zero_beam_file_list: list[Path]
) -> None:
    out_cube = tmp_path / "cube.fits"
    parser = get_parser()
    args = parser.parse_args(
        [*[str(f) for f in zero_beam_file_list], str(out_cube), "--no-blank-zero-beams"]
    )
    cli(args)

    with fits.open(out_cube) as hdu_list:
        cube = np.squeeze(hdu_list[0].data)

    assert np.allclose(cube[1], 2.0)


def test_cli_blanks_zero_beams_by_default(
    tmp_path: Path, zero_beam_file_list: list[Path]
) -> None:
    out_cube = tmp_path / "cube.fits"
    parser = get_parser()
    args = parser.parse_args([*[str(f) for f in zero_beam_file_list], str(out_cube)])
    cli(args)

    with fits.open(out_cube) as hdu_list:
        cube = np.squeeze(hdu_list[0].data)

    assert np.isnan(cube[1]).all()
