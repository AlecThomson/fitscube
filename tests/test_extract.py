"""Tests for extracting planes"""

from __future__ import annotations

from pathlib import Path

from astropy.io import fits
from fitscube.extract import (
    create_plane_freq_wcs,
    find_freq_axis,
    fits_file_contains_beam_table,
    get_output_path,
    update_header_for_frequency,
)


def test_get_output_path() -> None:
    """Make sure the output path generated is correct"""

    in_fits = Path("some.example.cube.fits")
    channel_index = 10
    expected_fits = Path("some.example.cube.channel-10.fits")

    assert expected_fits == get_output_path(
        input_path=in_fits, channel_index=channel_index
    )


def test_header(example_header) -> None:
    """Puliing together the header"""

    header = fits.header.Header.fromstring(example_header)

    assert header["NAXIS"] == 4


def test_find_freq_axis(example_header) -> None:
    """Find the components associated with frequency from the header"""
    header = fits.header.Header.fromstring(example_header)

    freq_wcs = find_freq_axis(header=header)
    assert freq_wcs.axis == 4
    assert freq_wcs.crpix == 1
    assert freq_wcs.crval == 801490740.740741
    assert freq_wcs.cdelt == 4000000.0


def test_create_plane_freq_wcs(example_header) -> None:
    """Update the freq wcs to indicate a plane"""
    header = fits.header.Header.fromstring(example_header)

    freq_wcs = find_freq_axis(header=header)
    plane_wcs = create_plane_freq_wcs(original_freq_wcs=freq_wcs, channel_index=1)

    assert plane_wcs.axis == freq_wcs.axis
    assert plane_wcs.crpix == 1
    assert plane_wcs.crval == 805490740.740741
    assert plane_wcs.cdelt == freq_wcs.cdelt

    plane_wcs = create_plane_freq_wcs(original_freq_wcs=freq_wcs, channel_index=0)

    assert plane_wcs.axis == freq_wcs.axis
    assert plane_wcs.crpix == 1
    assert plane_wcs.crval == 801490740.740741
    assert plane_wcs.cdelt == freq_wcs.cdelt


def test_update_header_for_frequency(example_header) -> None:
    """Update the fits header to denote the change for a
    extract channel"""

    header = fits.header.Header.fromstring(example_header)

    freq_wcs = find_freq_axis(header=header)

    new_header = update_header_for_frequency(
        header=header, freq_wcs=freq_wcs, channel_index=1
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
