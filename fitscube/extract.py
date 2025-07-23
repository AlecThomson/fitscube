"""Extract a plane out of a larger fits cube"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from fitscube.exceptions import FREQMissingException
from fitscube.logging import logger


@dataclass
class ExtractOptions:
    """Basic options to extract a plane of data"""

    hdu_index: int = 0
    """The HDU in the fits cube to access (e.g. for header and data)"""
    channel_index: int = 0
    """The channel of the cube to extract"""


def get_output_path(input_path: Path, channel_index: int) -> Path:
    # The input_path suffix returns a string with a period
    channel_suffix = f".channel-{channel_index}{input_path.suffix}"
    output_path = input_path.with_suffix(channel_suffix)

    logger.debug(f"The formed {output_path=}")

    return output_path


@dataclass
class FreqWCS:
    """Exxtract of frequency information in the WCS taken straight from
    the fits header."""

    axis: int
    """The axis count that the frequency corresponds to"""
    ctype: str
    """The FITS WCS key name"""
    crpix: int
    """The reference index position"""
    crval: float
    """The reference value stored in the header"""
    cdelt: float
    """The step between planes"""
    cunit: str
    """the unit of the frequency information"""


def fits_file_contains_beam_table(header: fits.header.Header | Path) -> bool:
    loaded_header: fits.header.Header = (
        fits.getheader(header) if isinstance(header, Path) else header
    )

    if "CASAMBM" not in loaded_header:
        return False

    return bool(loaded_header["CASAMBM"])


def find_freq_axis(header: fits.header.Header) -> FreqWCS:
    """Attempt to find the axies of the channel in the data
    cube that corresponds to frequency/channels.

    Args:
        header (fits.header.Header): The header from the fits cube

    Returns:
        FreqWCS: The information in the FITS header describing the frequency of the cube
    """

    naxis = header["NAXIS"]
    # Remember that range upper limit is exclusive
    for axis in range(1, naxis + 1):
        if "FREQ" in header[f"CTYPE{axis}"]:
            logger.info(f"Found FREQ at {axis=}")
            return FreqWCS(
                axis=axis,
                ctype=header[f"CTYPE{axis}"],
                crpix=header[f"CRPIX{axis}"],
                crval=header[f"CRVAL{axis}"],
                cdelt=header[f"CDELT{axis}"],
                cunit=header[f"CUNIT{axis}"],
            )

    msg = "Did not find the frequency axis"
    raise FREQMissingException(msg)


def create_plane_freq_wcs(original_freq_wcs: FreqWCS, channel_index: int) -> FreqWCS:
    channel_freq = original_freq_wcs.crval + (channel_index * original_freq_wcs.cdelt)
    return FreqWCS(
        axis=original_freq_wcs.axis,
        ctype=original_freq_wcs.ctype,
        crpix=1,
        crval=channel_freq,
        cdelt=original_freq_wcs.cdelt,
        cunit=original_freq_wcs.cunit,
    )


def update_header_for_frequency(
    header: fits.header.Header, freq_wcs: FreqWCS, channel_index: int
) -> fits.header.Header:
    # Get the new wcs items for the channels
    plane_freq_wcs = create_plane_freq_wcs(
        original_freq_wcs=freq_wcs, channel_index=channel_index
    )
    _idx = freq_wcs.axis
    out_header = header.copy()
    out_header[f"CTYPE{_idx}"] = plane_freq_wcs.ctype
    out_header[f"CRPIX{_idx}"] = plane_freq_wcs.crpix
    out_header[f"CRVAL{_idx}"] = plane_freq_wcs.crval
    out_header[f"CDELT{_idx}"] = plane_freq_wcs.cdelt
    out_header[f"CUNIT{_idx}"] = plane_freq_wcs.cunit

    return out_header


def extract_plane_from_cube(fits_cube: Path, extract_options: ExtractOptions) -> Path:
    output_path: Path = get_output_path(
        input_path=fits_cube, channel_index=extract_options.channel_index
    )

    logger.info(f"Opening {fits_cube=}")
    with fits.open(
        name=fits_cube, mode="readonly", memmap=True, lazy_load_hdus=True
    ) as open_fits:
        logger.info(
            f"Extracting header and data for hdu_index={extract_options.hdu_index}"
        )
        header = open_fits[extract_options.hdu_index].header
        data = open_fits[extract_options.hdu_index].data[
            ..., extract_options.channel_index, :, :
        ]

    logger.info("Extracted header")
    logger.info(header)

    logger.info(f"\nData shape: {data.shape}")
    freq_axis_wcs = find_freq_axis(header=header)
    freq_cube_index = len(data.shape) - freq_axis_wcs.axis

    # Get the channel index requested
    freq_plane_data = np.take(data, extract_options.channel_index, axis=freq_cube_index)
    # and pad it back so dimensions match
    freq_plane_data = np.expand_dims(freq_plane_data, axis=freq_cube_index)
    freq_plane_header = update_header_for_frequency(
        header=header,
        freq_wcs=freq_axis_wcs,
        channel_index=extract_options.channel_index,
    )
    logger.info(f"Formed new header: {freq_plane_header}")

    return output_path


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Extract a plane from a fits cube")

    parser.add_argument("fitscube", type=Path, help="The cube to extract a plane from")
    parser.add_argument("--channel", type=int, default=0, help="The channel to extract")

    return parser


def cli() -> None:
    parser = get_parser()

    _ = parser.parse_args()


if __name__ == "__main__":
    cli()
