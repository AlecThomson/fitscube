"""Tests for extracting planes"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from fitscube.exceptions import (
    ChannelMissingException,
    ShapeMismatchException,
    TargetAxisMissingException,
)
from fitscube.extract import (
    ExtractOptions,
    ExtractPlanesOptions,
    TargetIndex,
    _check_extract_mode,
    count_cube_planes,
    create_plane_target_wcs,
    create_target_index,
    extract_plane_from_cube,
    extract_planes_from_cube,
    find_target_axis,
    fits_file_contains_beam_table,
    get_output_path,
    read_beam_table,
    update_header_for_target_axis,
)


def test_target_index() -> None:
    """Simple test of the target container"""
    target_index = create_target_index(channel_index=2)
    assert isinstance(target_index, TargetIndex)
    assert target_index.axis_name == "FREQ"
    assert target_index.axis_index == 2

    target_index = create_target_index(time_index=20)
    assert isinstance(target_index, TargetIndex)
    assert target_index.axis_name == "TIME"
    assert target_index.axis_index == 20

    # int of 0 is False
    target_index = create_target_index(time_index=0)
    assert isinstance(target_index, TargetIndex)
    assert target_index.axis_name == "TIME"
    assert target_index.axis_index == 0

    with pytest.raises(ValueError, match="index"):
        create_target_index()

    with pytest.raises(ValueError, match="index"):
        create_target_index(channel_index=2, time_index=4)


def test_check_mode_for_consistency() -> None:
    """See if the basic consistency checks for the extraction mode makes sense"""
    extract_options = ExtractOptions()
    with pytest.raises(ValueError, match="index"):
        _check_extract_mode(extract_options=extract_options)

    extract_options = ExtractOptions(channel_index=3, time_index=4)
    with pytest.raises(ValueError, match="index"):
        _check_extract_mode(extract_options=extract_options)


def test_get_output_path() -> None:
    """Make sure the output path generated is correct"""

    target_index = create_target_index(channel_index=10)
    in_fits = Path("some.example.cube.fits")
    expected_fits = Path("some.example.cube.channel-10.fits")

    assert expected_fits == get_output_path(
        input_path=in_fits, target_index=target_index
    )

    target_index = create_target_index(time_index=10)
    in_fits = Path("some.example.cube.fits")
    expected_fits = Path("some.example.cube.timestep-10.fits")

    assert expected_fits == get_output_path(
        input_path=in_fits, target_index=target_index
    )

    target_index = create_target_index(channel_index=10)
    for compressed in ("some.example.cube.fits.gz", "some.example.cube.fits.fz"):
        assert Path("some.example.cube.channel-10.fits") == get_output_path(
            input_path=Path(compressed), target_index=target_index
        )


def test_header(example_header) -> None:
    """Puliing together the header"""

    header = fits.header.Header.fromstring(example_header)

    assert header["NAXIS"] == 4


def test_find_freq_axis(example_header) -> None:
    """Find the components associated with frequency from the header"""
    header = fits.header.Header.fromstring(example_header)

    freq_wcs = find_target_axis(header=header)
    assert freq_wcs.axis == 4
    assert freq_wcs.crpix == 1
    assert freq_wcs.crval == 801490740.740741
    assert freq_wcs.cdelt == 4000000.0


def test_create_plane_freq_wcs(example_header) -> None:
    """Update the freq wcs to indicate a plane"""
    header = fits.header.Header.fromstring(example_header)

    freq_wcs = find_target_axis(header=header)
    plane_wcs = create_plane_target_wcs(original_freq_wcs=freq_wcs, target_index=1)

    assert plane_wcs.axis == freq_wcs.axis
    assert plane_wcs.crpix == 1
    assert plane_wcs.crval == 805490740.740741
    assert plane_wcs.cdelt == freq_wcs.cdelt

    plane_wcs = create_plane_target_wcs(original_freq_wcs=freq_wcs, target_index=0)

    assert plane_wcs.axis == freq_wcs.axis
    assert plane_wcs.crpix == 1
    assert plane_wcs.crval == 801490740.740741
    assert plane_wcs.cdelt == freq_wcs.cdelt


def test_update_header_for_frequency(example_header) -> None:
    """Update the fits header to denote the change for a
    extract channel"""

    header = fits.header.Header.fromstring(example_header)

    freq_wcs = find_target_axis(header=header)
    target_index = create_target_index(channel_index=1)
    new_header = update_header_for_target_axis(
        header=header, target_wcs=freq_wcs, target_index=target_index
    )
    assert new_header["CRPIX4"] == 1
    assert new_header["CRVAL4"] == 805490740.740741
    assert new_header["CDELT4"] == 4000000

    keys = header.keys()
    for key in keys:
        if key in ("CPIX4", "CRVAL4", "CDELT4"):
            continue
        assert header[key] == new_header[key]


def test_fits_file_contains_beam_table(example_header) -> None:
    """See if the header / fits file contains a beam table"""

    header = fits.header.Header.fromstring(example_header)

    assert not fits_file_contains_beam_table(header=header)


def test_fits_file_contains_beam_table_2(headers) -> None:
    """More tests for beam"""
    header = fits.header.Header.fromstring(headers["beams"])

    assert fits_file_contains_beam_table(header=header)


def test_cube_file(cube_path) -> None:
    """Just a check to see if the fits cube packaged can be pulled out"""
    assert cube_path.exists()


def test_image_files(image_paths) -> None:
    """Just a check to see if the fits cube packaged can be pulled out"""

    assert all(f.exists() for f in image_paths)


def test_fits_file_contains_beam_table_from_file(cube_path, image_paths) -> None:
    """Make sure that the fits cube can be examined from a path and determine
    whether a beam table exists"""
    assert fits_file_contains_beam_table(cube_path)
    assert not fits_file_contains_beam_table(image_paths[0])


def test_compare_extracted_to_image(cube_path, image_paths, tmpdir) -> None:
    """Perform a single plane extraction and compare it to the base
    data it was formed from"""

    output_file = Path(tmpdir) / "extract" / "test.fits"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    channel = 0

    extract_options = ExtractOptions(
        hdu_index=0, channel_index=channel, output_path=output_file
    )

    sub_path = extract_plane_from_cube(
        fits_cube=cube_path, extract_options=extract_options
    )

    assert sub_path == output_file
    sub_data = fits.getdata(sub_path)
    image_data = fits.getdata(image_paths[channel])

    assert np.allclose(sub_data, image_data)

    sub_header = fits.getheader(sub_path)
    image_header = fits.getheader(image_paths[channel])

    assert np.isclose(sub_header["BMAJ"], image_header["BMAJ"])
    assert np.isclose(sub_header["BMAJ"], image_header["BMAJ"])
    assert np.isclose(sub_header["BPA"], image_header["BPA"])


def test_compare_extracted_to_image_bad_channel(cube_path, tmpdir) -> None:
    """Capture error if a channel error is raised"""

    output_file = Path(tmpdir) / "extract" / "test.fits"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    channel = 9999

    extract_options = ExtractOptions(
        hdu_index=0, channel_index=channel, output_path=output_file
    )
    with pytest.raises(ChannelMissingException):
        extract_plane_from_cube(fits_cube=cube_path, extract_options=extract_options)


def test_extract_time_from_freq(cube_path, tmpdir) -> None:
    """Attempt to extract a timestep from a cube without TIME. Should
    raise an error
    """
    output_file = Path(tmpdir) / "extract" / "test.fits"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    timestep = 0

    extract_options = ExtractOptions(
        hdu_index=0, time_index=timestep, output_path=output_file
    )

    with pytest.raises(TargetAxisMissingException):
        extract_plane_from_cube(fits_cube=cube_path, extract_options=extract_options)


def test_extract_time_cube(timecube_path, tmpdir) -> None:
    """Attempt to extract a timestep from a cube without TIME. Should
    raise an error
    """
    output_file = Path(tmpdir) / "extract" / "test.fits"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    timestep = 200

    extract_options = ExtractOptions(
        hdu_index=0, time_index=timestep, output_path=output_file
    )

    timecube = extract_plane_from_cube(
        fits_cube=timecube_path, extract_options=extract_options
    )
    assert timecube.exists()


def test_count_cube_planes(cube_path) -> None:
    """The channel count comes off the frequency axis"""
    header = fits.getheader(cube_path)
    assert count_cube_planes(header=header) == header["NAXIS3"]


def test_read_beam_table(cube_path, tmpdir) -> None:
    """A cube with a beam table hands back one beam per channel"""
    with fits.open(cube_path) as open_fits:
        beams = read_beam_table(open_fits=open_fits)
        assert beams is not None
        assert len(beams.major) == open_fits[0].header["NAXIS3"]

    # A cube that does not claim a beam table has none to read
    no_table = Path(tmpdir) / "no_table.fits"
    with fits.open(cube_path) as open_fits:
        header = open_fits[0].header.copy()
        del header["CASAMBM"]
        fits.writeto(no_table, data=open_fits[0].data, header=header)

    with fits.open(no_table) as open_fits:
        assert read_beam_table(open_fits=open_fits) is None


def test_extract_planes_matches_single_plane(cube_path, tmpdir) -> None:
    """Extracting every plane at once must give exactly what extracting them
    one at a time does, beam table and all"""
    tmp_path = Path(tmpdir)
    channels = count_cube_planes(header=fits.getheader(cube_path))

    one_at_a_time = [
        extract_plane_from_cube(
            fits_cube=cube_path,
            extract_options=ExtractOptions(
                channel_index=channel,
                output_path=tmp_path / f"single.{channel}.fits",
                overwrite=True,
            ),
        )
        for channel in range(channels)
    ]

    all_at_once = extract_planes_from_cube(
        cube_path,
        ExtractPlanesOptions(
            overwrite=True,
            output_paths=[
                tmp_path / f"batch.{channel}.fits" for channel in range(channels)
            ],
        ),
    )

    assert len(all_at_once) == channels
    for single, batch in zip(one_at_a_time, all_at_once, strict=True):
        assert np.array_equal(fits.getdata(single), fits.getdata(batch))
        single_header, batch_header = fits.getheader(single), fits.getheader(batch)
        assert set(single_header) == set(batch_header)
        for key in single_header:
            assert single_header[key] == batch_header[key], f"{key} differs"


def test_extract_planes_default_paths(cube_path) -> None:
    """Without output paths the planes are named after the cube"""
    channels = count_cube_planes(header=fits.getheader(cube_path))

    planes = extract_planes_from_cube(cube_path, ExtractPlanesOptions(overwrite=True))

    assert len(planes) == channels
    assert all(plane.exists() for plane in planes)
    assert planes[0] == get_output_path(
        input_path=cube_path, target_index=create_target_index(channel_index=0)
    )


def test_extract_planes_output_path_count(cube_path, tmpdir) -> None:
    """One output path per channel, or none at all"""
    with pytest.raises(ShapeMismatchException):
        extract_planes_from_cube(
            cube_path,
            ExtractPlanesOptions(output_paths=[Path(tmpdir) / "only_one.fits"]),
        )


def _write_crpix_cube(path: Path, crpix: float | None) -> Path:
    """A 4 channel (FREQ, DEC, RA) cube whose reference pixel is not 1"""
    header = fits.Header()
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CTYPE3"] = "FREQ"
    header["CRVAL3"] = 1e9
    header["CDELT3"] = 1e6
    header["CUNIT3"] = "Hz"
    if crpix is not None:
        header["CRPIX3"] = crpix
    data = np.arange(4, dtype=np.float32)[:, None, None] * np.ones((4, 2, 2))
    fits.writeto(path, data=data, header=header)
    return path


@pytest.mark.parametrize(("crpix", "expected"), [(3.0, 0.998e9), (None, 1.001e9)])
def test_extract_respects_crpix(tmpdir, crpix, expected) -> None:
    """The plane value used to ignore CRPIX, i.e. assume CRPIX = 1"""
    cube = _write_crpix_cube(Path(tmpdir) / "cube.fits", crpix)

    output_file = Path(tmpdir) / "plane.fits"
    extract_plane_from_cube(
        fits_cube=cube,
        extract_options=ExtractOptions(channel_index=0, output_path=output_file),
    )
    header = fits.getheader(output_file)
    assert header["CRPIX3"] == 1
    assert np.isclose(header["CRVAL3"], expected)

    plane_paths = extract_planes_from_cube(
        fits_cube=cube,
        extract_planes_options=ExtractPlanesOptions(
            output_paths=[Path(tmpdir) / f"p{i}.fits" for i in range(4)]
        ),
    )
    for channel, plane_path in enumerate(plane_paths):
        assert np.isclose(
            fits.getheader(plane_path)["CRVAL3"], expected + channel * 1e6
        )


@pytest.mark.parametrize("channel", [4, -1])
def test_extract_index_outside_axis(tmpdir, channel) -> None:
    """One past the end hit an IndexError, and -1 got the wrong CRVAL"""
    cube = _write_crpix_cube(Path(tmpdir) / "cube.fits", 1.0)
    with pytest.raises(ChannelMissingException):
        extract_plane_from_cube(
            fits_cube=cube,
            extract_options=ExtractOptions(
                channel_index=channel, output_path=Path(tmpdir) / "plane.fits"
            ),
        )


def test_extract_refuses_to_overwrite_cube(tmpdir) -> None:
    """With overwrite set, an output path naming the cube replaced it with one plane"""
    cube = _write_crpix_cube(Path(tmpdir) / "cube.fits", 1.0)
    # Same file under another spelling
    same_cube = Path(tmpdir) / "." / "cube.fits"

    with pytest.raises(ValueError, match="is the input cube"):
        extract_plane_from_cube(
            fits_cube=cube,
            extract_options=ExtractOptions(
                channel_index=2, output_path=same_cube, overwrite=True
            ),
        )
    with pytest.raises(ValueError, match="is the input cube"):
        extract_planes_from_cube(
            fits_cube=cube,
            extract_planes_options=ExtractPlanesOptions(
                output_paths=[Path(tmpdir) / f"p{i}.fits" for i in range(3)]
                + [same_cube],
                overwrite=True,
            ),
        )
    assert fits.getdata(cube).shape == (4, 2, 2)


@pytest.mark.filterwarnings("ignore:'cdfix' made the change")
@pytest.mark.parametrize(
    "keywords",
    [
        {"CDELT3": None, "CUNIT3": None, "CD3_3": 1e6},
        {"CDELT3": 5e5, "PC3_3": 2.0},
        {"CDELT3": None, "CRVAL3": 1e6, "CRPIX3": 1000.0},
    ],
)
def test_extract_follows_fits_wcs(tmpdir, keywords) -> None:
    """Missing CDELT or CUNIT raised a KeyError, and CDi_i and PCi_i were ignored"""
    cube = _write_crpix_cube(Path(tmpdir) / "cube.fits", 1.0)
    with fits.open(cube, mode="update") as hdu_list:
        for key, value in keywords.items():
            if value is None:
                del hdu_list[0].header[key]
            else:
                hdu_list[0].header[key] = value
    cube_wcs = WCS(fits.getheader(cube))

    for channel in (0, 3):
        output_file = Path(tmpdir) / f"plane_{channel}.fits"
        extract_plane_from_cube(
            fits_cube=cube,
            extract_options=ExtractOptions(
                channel_index=channel, output_path=output_file
            ),
        )
        plane_wcs = WCS(fits.getheader(output_file))
        assert np.isclose(
            plane_wcs.spectral.pixel_to_world_values(0),
            cube_wcs.spectral.pixel_to_world_values(channel),
        )


def test_extract_planes_time_cube(timecube_path, tmpdir) -> None:
    """Only a FREQ axis was looked for, so time cubes could not be split"""
    output_paths = [Path(tmpdir) / f"t{i}.fits" for i in range(724)]
    extract_planes_from_cube(
        fits_cube=timecube_path,
        extract_planes_options=ExtractPlanesOptions(output_paths=output_paths),
    )

    single = Path(tmpdir) / "single.fits"
    extract_plane_from_cube(
        fits_cube=timecube_path,
        extract_options=ExtractOptions(time_index=200, output_path=single),
    )
    assert np.array_equal(
        fits.getdata(output_paths[200]), fits.getdata(single), equal_nan=True
    )
    assert fits.getheader(output_paths[200]) == fits.getheader(single)
    assert get_output_path(
        timecube_path, create_target_index(time_index=3)
    ).name.endswith("timestep-3.fits")


def test_extract_planes_skips_degenerate_freq(tmpdir) -> None:
    """A time cube built from images keeps their one-plane FREQ axis"""
    cube = Path(tmpdir) / "cube.fits"
    header = fits.Header()
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CTYPE3"] = "FREQ"
    header["CTYPE4"] = "TIME"
    header["CDELT4"] = 10.0
    fits.writeto(
        cube,
        np.arange(3, dtype=np.float32)[:, None, None, None] * np.ones((3, 1, 2, 2)),
        header,
    )

    output_paths = extract_planes_from_cube(
        fits_cube=cube,
        extract_planes_options=ExtractPlanesOptions(
            output_paths=[Path(tmpdir) / f"t{i}.fits" for i in range(3)]
        ),
    )
    for timestep, path in enumerate(output_paths):
        assert fits.getdata(path).shape == (1, 1, 2, 2)
        assert np.all(fits.getdata(path) == timestep)
