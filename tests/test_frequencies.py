from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from fitscube.combine_fits import combine_fits, even_spacing, parse_specs


@pytest.fixture
def even_specs() -> u.Quantity:
    rng = np.random.default_rng()
    start = rng.integers(1, 3)
    end = rng.integers(4, 6)
    num = rng.integers(6, 10)
    return np.linspace(start, end, num) * u.GHz


@pytest.fixture
def output_file():
    yield Path("test.fits")
    Path("test.fits").unlink()


@pytest.fixture
def file_list(even_specs: u.Quantity):
    image = np.ones((1, 10, 10))
    for i, spec in enumerate(even_specs):
        header = fits.Header()
        header["CRVAL3"] = spec.to(u.Hz).value
        header["CDELT3"] = 1e8
        header["CRPIX3"] = 1
        header["CTYPE3"] = "FREQ"
        header["CUNIT3"] = "Hz"
        hdu = fits.PrimaryHDU(image * i, header=header)
        hdu.writeto(f"plane_{i}.fits", overwrite=True)

    yield [Path(f"plane_{i}.fits") for i in range(len(even_specs))]

    for i in range(len(even_specs)):
        Path(f"plane_{i}.fits").unlink()


def test_parse_specs(file_list: list[Path], even_specs: u.Quantity):
    file_specs, specs, missing_chan_idx = parse_specs(file_list)
    assert np.array_equal(file_specs, even_specs)


def test_uneven(file_list: list[Path], even_specs: u.Quantity):
    unven_specs = np.concatenate([even_specs[0:1], even_specs[3:]])
    file_array = np.array(file_list)
    uneven_files = np.concatenate([file_array[0:1], file_array[3:]]).tolist()
    file_specs, specs, missing_chan_idx = parse_specs(uneven_files, create_blanks=True)
    assert np.array_equal(file_specs, unven_specs)
    assert missing_chan_idx[1]
    assert np.allclose(specs.to(u.Hz).value, even_specs.to(u.Hz).value)


def test_even_combine(file_list: list[Path], even_specs: u.Quantity, output_file: Path):
    specs = combine_fits(
        file_list=file_list,
        out_cube=output_file,
        create_blanks=False,
        overwrite=True,
    )

    assert np.array_equal(specs, even_specs)

    cube = fits.getdata(output_file, verify="exception")
    for chan in range(len(specs)):
        image = fits.getdata(file_list[chan])
        plane = cube[chan]
        assert np.allclose(plane, image)


def test_uneven_combine(
    file_list: list[Path], even_specs: u.Quantity, output_file: Path
):
    # unven_specs = np.concatenate([even_specs[0:1], even_specs[3:]])
    file_array = np.array(file_list)
    uneven_files = np.concatenate([file_array[0:1], file_array[3:]]).tolist()
    specs = combine_fits(
        file_list=uneven_files,
        out_cube=output_file,
        create_blanks=True,
        overwrite=True,
    )

    assert np.allclose(specs.to(u.Hz).value, even_specs.to(u.Hz).value)
    expected_spectrum = np.arange(len(even_specs)).astype(float)
    expected_spectrum[1:3] = np.nan

    cube = fits.getdata(output_file)
    cube_spectrum = cube[:, 0, 0]
    assert cube.shape[0] == len(even_specs)
    assert cube.shape[0] == len(specs)
    for i in range(len(even_specs)):
        if np.isnan(expected_spectrum[i]):
            assert np.isnan(cube_spectrum[i])
        else:
            assert np.isclose(cube_spectrum[i], expected_spectrum[i])
    for chan in range(len(specs)):
        image = fits.getdata(file_list[chan], verify="exception")
        plane = cube[chan]
        if np.isnan(plane).all():
            assert chan in (1, 2)
            continue
        assert np.allclose(plane, image)


def test_even_spacing_non_adjacent_gaps():
    # Present channels 0, 3, 5 GHz on a 1 GHz grid (1, 2, 4 missing). No two
    # surviving channels are adjacent, so min(diffs) == 2 GHz would mis-grid the
    # axis to [0, 2, 4] and drop the real channels. The gcd of the diffs
    # [3, 2] GHz recovers the true 1 GHz step.
    specs = np.array([0.0, 3.0, 5.0]) * u.GHz
    new_specs, missing_chan_idx = even_spacing(specs)
    assert len(new_specs) == 6, "should rebuild the full 0..5 GHz grid"
    assert missing_chan_idx.sum() == 3
    expected_missing = np.array([False, True, True, False, True, False])
    assert np.array_equal(missing_chan_idx, expected_missing)
