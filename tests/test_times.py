from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from fitscube.combine_fits import combine_fits, even_spacing, parse_specs
from fitscube.exceptions import IrregularSpacingException


def test_parse_specs(file_list: list[Path], even_specs: u.Quantity):
    file_specs, _specs, _missing_chan_idx = parse_specs(
        file_list, time_domain_mode=True
    )
    # assert np.array_equal(file_specs, even_specs)
    assert np.allclose(file_specs, even_specs, rtol=1e-10, atol=1e-9)


def test_uneven(file_list: list[Path], even_specs: u.Quantity):
    uneven_specs = np.concatenate([even_specs[0:1], even_specs[3:]])
    file_array = np.array(file_list)
    uneven_files = np.concatenate([file_array[0:1], file_array[3:]]).tolist()
    file_specs, specs, missing_chan_idx = parse_specs(
        uneven_files, create_blanks=True, time_domain_mode=True
    )
    assert np.allclose(file_specs, uneven_specs, rtol=1e-10, atol=1e-9)
    # assert np.array_equal(file_specs, uneven_specs)
    assert missing_chan_idx[1]
    assert np.allclose(
        specs.to(u.s).value, even_specs.to(u.s).value, rtol=1e-10, atol=1e-9
    )


def test_even_combine(file_list: list[Path], even_specs: u.Quantity, output_file: Path):
    specs = combine_fits(
        file_list=file_list,
        out_cube=output_file,
        create_blanks=False,
        overwrite=True,
        time_domain_mode=True,
    )

    assert np.allclose(specs, even_specs, rtol=1e-12, atol=1e-13)

    cube = fits.getdata(output_file, verify="exception")
    for chan in range(len(specs)):
        image = fits.getdata(file_list[chan])
        plane = cube[chan]
        assert np.allclose(plane, image)


def test_uneven_combine(
    file_list: list[Path], even_specs: u.Quantity, output_file: Path
):
    """!!!! NOTE: For some unknown reason thios test *SOMETIMES* raises the
    following error:

    > FAILED tests/test_times.py::test_uneven - ValueError: operands could not be broadcast together with shapes (5,) (6,)

    Completely unclear to me why
    """

    # uneven_specs = np.concatenate([even_specs[0:1], even_specs[3:]])
    file_array = np.array(file_list)
    uneven_files = np.concatenate([file_array[0:1], file_array[3:]]).tolist()
    specs = combine_fits(
        file_list=uneven_files,
        out_cube=output_file,
        create_blanks=True,
        overwrite=True,
        time_domain_mode=True,
    )
    print(specs)
    print(even_specs)
    assert np.allclose(specs.to(u.s).value, even_specs.to(u.s).value)
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
            assert np.isclose(
                cube_spectrum[i], expected_spectrum[i]
            )  # , atol=1e-11, rtol=1e-10)
    for chan in range(len(specs)):
        image = fits.getdata(file_list[chan], verify="exception")
        plane = cube[chan]
        if np.isnan(plane).all():
            assert chan in (1, 2)
            continue
        assert np.allclose(plane, image)


@pytest.mark.filterwarnings("ignore:'datfix' made the change")
def test_wsclean_images_create_axis(time_image_paths, tmpdir) -> None:
    """Ensure that the combined cube conforms to the input data"""

    tmpdir = Path(tmpdir) / "time_cube_combine"
    tmpdir.mkdir(parents=True, exist_ok=True)
    out_cube = tmpdir / "time_cube_mate.fits"

    combine_fits(
        file_list=time_image_paths,
        out_cube=out_cube,
        overwrite=True,
        time_domain_mode=True,
    )

    cube_data = fits.getdata(out_cube)
    for i, time_image_path in enumerate(time_image_paths):
        image_data = fits.getdata(time_image_path)
        # The TIME axis will be appended as a new dimension
        cube_image_data = cube_data[i]
        assert np.allclose(image_data.squeeze(), cube_image_data.squeeze())


@pytest.mark.filterwarnings("ignore:'datfix' made the change")
def test_wsclean_images_create_axis_floating(time_image_paths, tmpdir) -> None:
    """Ensure that the combined cube conforms to the input data"""

    tmpdir = Path(tmpdir) / "time_cube_combine"
    tmpdir.mkdir(parents=True, exist_ok=True)
    out_cube = tmpdir / "time_cube_mate.fits"

    for float_length, float_rep in ((32, ">f4"), (64, ">f8")):
        combine_fits(
            file_list=time_image_paths,
            out_cube=out_cube,
            overwrite=True,
            time_domain_mode=True,
            float_length=float_length,  # type: ignore [arg-type]
        )

        cube_data = fits.getdata(out_cube)
        assert cube_data.dtype == float_rep
        for i, time_image_path in enumerate(time_image_paths):
            image_data = fits.getdata(time_image_path)
            # The TIME axis will be appended as a new dimension
            cube_image_data = cube_data[i]
            assert np.allclose(image_data.squeeze(), cube_image_data.squeeze())


@pytest.mark.filterwarnings("ignore:'datfix' made the change")
def test_wsclean_images_create_axis_withbb(time_image_paths, tmpdir) -> None:
    """Ensure that the combined cube conforms to the input data. This will use the
    bounding box option. This should led to no change."""

    tmpdir = Path(tmpdir) / "time_cube_combine"
    tmpdir.mkdir(parents=True, exist_ok=True)
    out_cube = tmpdir / "time_cube_mate.fits"

    combine_fits(
        file_list=time_image_paths,
        out_cube=out_cube,
        overwrite=True,
        time_domain_mode=True,
        bounding_box=True,
    )

    cube_data = fits.getdata(out_cube)
    for i, time_image_path in enumerate(time_image_paths):
        image_data = fits.getdata(time_image_path)
        # The TIME axis will be appended as a new dimension
        cube_image_data = cube_data[i]
        assert np.allclose(image_data.squeeze(), cube_image_data.squeeze())


@pytest.mark.filterwarnings("ignore:'datfix' made the change")
def test_wsclean_images_create_axis_withbb_noinvalidate0(
    time_image_paths_withzeroborder, tmpdir
) -> None:
    """Ensure that the combined cube conforms to the input data. This will use the
    bounding box option. This should led to no change."""

    tmpdir = Path(tmpdir) / "time_cube_combine_zeros"
    tmpdir.mkdir(parents=True, exist_ok=True)
    out_cube = tmpdir / "time_cube_mate.fits"

    combine_fits(
        file_list=time_image_paths_withzeroborder,
        out_cube=out_cube,
        overwrite=True,
        time_domain_mode=True,
        bounding_box=True,
        invalidate_zeros=False,
    )

    cube_data = fits.getdata(out_cube)
    assert cube_data.squeeze().shape[-2:] == (100, 100)
    for i, time_image_path in enumerate(time_image_paths_withzeroborder):
        image_data = fits.getdata(time_image_path)
        # The TIME axis will be appended as a new dimension
        cube_image_data = cube_data[i]
        assert np.allclose(image_data.squeeze(), cube_image_data.squeeze())


@pytest.mark.filterwarnings("ignore:'datfix' made the change")
def test_wsclean_images_create_axis_withbb_invalidatezeros(
    time_image_paths_withzeroborder, tmpdir
) -> None:
    """Ensure that the combined cube conforms to the input data. This will use the
    bounding box option. This should led to no change. Here we also get fitscube to
    mark pixels that are exactly 0 to be nans"""

    tmpdir = Path(tmpdir) / "time_cube_combine_zeros"
    tmpdir.mkdir(parents=True, exist_ok=True)
    out_cube = tmpdir / "time_cube_mate.fits"

    combine_fits(
        file_list=time_image_paths_withzeroborder,
        out_cube=out_cube,
        overwrite=True,
        time_domain_mode=True,
        bounding_box=True,
        invalidate_zeros=True,
    )

    cube_data = fits.getdata(out_cube)
    assert cube_data.squeeze().shape[-2:] == (90, 90)
    for i, time_image_path in enumerate(time_image_paths_withzeroborder):
        image_data = fits.getdata(time_image_path)
        # The TIME axis will be appended as a new dimension
        cube_image_data = cube_data[i]
        assert np.allclose(image_data.squeeze()[5:-5, 5:-5], cube_image_data.squeeze())


@pytest.mark.filterwarnings("ignore:'datfix' made the change")
@pytest.mark.filterwarnings("ignore:ERFA function")
def test_time_axis_reference_is_set(time_image_paths, tmpdir) -> None:
    """MJDREF and TIMESYS were copied from the first input, shifting the axis"""
    for path in time_image_paths:
        with fits.open(path, mode="update") as hdu_list:
            hdu_list[0].header["MJDREF"] = 51544.0
            hdu_list[0].header["TIMESYS"] = "TT"

    out_cube = Path(tmpdir) / "time_cube_ref.fits"
    combine_fits(
        file_list=time_image_paths,
        out_cube=out_cube,
        overwrite=True,
        time_domain_mode=True,
    )

    header = fits.getheader(out_cube)
    assert header["MJDREF"] == 0.0
    assert header["TIMESYS"] == "UTC"
    wcs = WCS(header)
    first_time = wcs.pixel_to_world(*([0] * wcs.pixel_n_dim))[-1]
    date_obs = Time(fits.getheader(time_image_paths[0])["DATE-OBS"], scale="utc")
    # UTC MJD seconds skip leap seconds, which a FITS reader counts (37 s since 1972)
    assert abs(first_time - date_obs) < 60 * u.s


def test_even_spacing_one_second_steps():
    """rtol and atol were passed to np.isclose swapped, giving a ~5 s tolerance
    that matched every grid point of a 1 s grid"""
    times = (5.18e9 + np.array([0.0, 1.0, 3.0, 4.0])) * u.s
    new_times, missing_chan_idx = even_spacing(times, time_domain_mode=True)
    assert len(new_times) == 5
    assert np.array_equal(missing_chan_idx, [False, False, True, False, False])


ASKAP_CADENCE = 9.95328  # s


def askap_times(n: int = 1564) -> np.ndarray:
    """Start times of a run of ASKAP integrations, in MJD seconds."""
    return 5.2597e9 + ASKAP_CADENCE * np.arange(n)


@pytest.mark.parametrize("perturb", ["rounded", "jittered"])
def test_imprecise_times_keep_their_cadence(perturb: str) -> None:
    """Times a little off a long cadence were put on a ~1 s grid (issue #77).

    wsclean writes DATE-OBS to 0.1 s, so the smallest gap was 9.9 s rather
    than 9.953 s, and numbering every time against it drifted by a whole step
    along the run. The ten missing in a row check a long gap is numbered right.
    """
    missing = [100, 500, 501, *range(800, 810)]
    times = np.delete(askap_times(), missing)
    if perturb == "rounded":
        times = np.round(times, 1)
    else:
        times += np.random.default_rng(0).uniform(-0.3, 0.3, len(times))

    new_times, missing_chan_idx = even_spacing(times * u.s, time_domain_mode=True)

    assert len(new_times) == 1564
    assert np.flatnonzero(missing_chan_idx).tolist() == missing
    assert np.allclose(np.diff(new_times.to(u.s).value), ASKAP_CADENCE, atol=1e-3)


def test_long_hole_between_short_runs() -> None:
    """Numbering from the 9.9 s smallest gap settled on one step too many."""
    present = np.r_[0:50, 150:200]
    times = np.round(5.2597e9 + ASKAP_CADENCE * present, 1)

    new_times, missing_chan_idx = even_spacing(times * u.s, time_domain_mode=True)

    assert len(new_times) == 200
    assert np.flatnonzero(~missing_chan_idx).tolist() == present.tolist()


def test_time_off_the_cadence_is_refused() -> None:
    """An image 1 s off the cadence put the run on a ~1 s grid, 90% blank."""
    times = np.round(askap_times(), 1)
    times[700] += 1.0
    with pytest.raises(IrregularSpacingException, match="smallest gap"):
        even_spacing(times * u.s, time_domain_mode=True)


def test_times_are_not_split_below_their_smallest_gap() -> None:
    """Frequencies 2 and 3 steps apart are gridded, but times must share a cadence."""
    offsets = np.array([0.0, 20.0, 50.0])
    new_specs, _ = even_spacing((1e9 + offsets * 1e6) * u.Hz)
    assert len(new_specs) == 6
    with pytest.raises(IrregularSpacingException, match="smallest gap"):
        even_spacing((5.18e9 + offsets) * u.s, time_domain_mode=True)
