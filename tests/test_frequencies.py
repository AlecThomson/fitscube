from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from fitscube.combine_fits import combine_fits, parse_freqs


@pytest.fixture()
def even_freqs() -> u.Quantity:
    rng = np.random.default_rng()
    start = rng.integers(1, 3)
    end = rng.integers(4, 6)
    num = rng.integers(6, 10)
    return np.linspace(start, end, num) * u.GHz


@pytest.fixture()
def output_file():
    yield Path("test.fits")
    Path("test.fits").unlink()


@pytest.fixture()
def file_list(even_freqs: u.Quantity):
    image = np.ones((1, 10, 10))
    for i, freq in enumerate(even_freqs):
        header = fits.Header()
        header["CRVAL3"] = freq.to(u.Hz).value
        header["CDELT3"] = 1e8
        header["CRPIX3"] = 1
        header["CTYPE3"] = "FREQ"
        header["CUNIT3"] = "Hz"
        hdu = fits.PrimaryHDU(image * i, header=header)
        hdu.writeto(f"plane_{i}.fits", overwrite=True)

    yield [Path(f"plane_{i}.fits") for i in range(len(even_freqs))]

    for i in range(len(even_freqs)):
        Path(f"plane_{i}.fits").unlink()


def test_parse_freqs(file_list: list[Path], even_freqs: u.Quantity):
    file_freqs, freqs, missing_chan_idx = parse_freqs(file_list)
    assert np.array_equal(file_freqs, even_freqs)


def test_uneven(file_list: list[Path], even_freqs: u.Quantity):
    unven_freqs = np.concatenate([even_freqs[0:1], even_freqs[3:]])
    file_array = np.array(file_list)
    uneven_files = np.concatenate([file_array[0:1], file_array[3:]]).tolist()
    file_freqs, freqs, missing_chan_idx = parse_freqs(uneven_files, create_blanks=True)
    assert np.array_equal(file_freqs, unven_freqs)
    assert missing_chan_idx[1]
    assert np.allclose(freqs.to(u.Hz).value, even_freqs.to(u.Hz).value)


def test_even_combine(file_list: list[Path], even_freqs: u.Quantity, output_file: Path):
    freqs = combine_fits(
        file_list=file_list,
        out_cube=output_file,
        create_blanks=False,
        overwrite=True,
    )

    assert np.array_equal(freqs, even_freqs)

    cube = fits.getdata(output_file)
    for chan in range(len(freqs)):
        image = fits.getdata(file_list[chan])
        plane = cube[chan]
        assert np.allclose(plane, image)


def test_uneven_combine(
    file_list: list[Path], even_freqs: u.Quantity, output_file: Path
):
    # unven_freqs = np.concatenate([even_freqs[0:1], even_freqs[3:]])
    file_array = np.array(file_list)
    uneven_files = np.concatenate([file_array[0:1], file_array[3:]]).tolist()
    freqs = combine_fits(
        file_list=uneven_files,
        out_cube=output_file,
        create_blanks=True,
        overwrite=True,
    )

    assert np.allclose(freqs.to(u.Hz).value, even_freqs.to(u.Hz).value)
    expected_spectrum = np.arange(len(even_freqs)).astype(float)
    expected_spectrum[1:3] = np.nan

    cube = fits.getdata(output_file)
    cube_spectrum = cube[:, 0, 0]
    assert cube.shape[0] == len(even_freqs)
    assert cube.shape[0] == len(freqs)
    for i in range(len(even_freqs)):
        if np.isnan(expected_spectrum[i]):
            assert np.isnan(cube_spectrum[i])
        else:
            assert np.isclose(cube_spectrum[i], expected_spectrum[i])
    for chan in range(len(freqs)):
        image = fits.getdata(file_list[chan])
        plane = cube[chan]
        if np.isnan(plane).all():
            assert chan in (1, 2)
            continue
        assert np.allclose(plane, image)
