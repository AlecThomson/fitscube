"""Tests for extracting planes"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from fitscube.exceptions import ChannelMissingException
from fitscube.extract import (
    ExtractOptions,
    TargetIndex,
    _check_extract_mode,
    create_plane_target_wcs,
    create_target_index,
    extract_plane_from_cube,
    find_target_axis,
    fits_file_contains_beam_table,
    get_output_path,
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
