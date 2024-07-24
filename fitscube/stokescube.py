#!/usr/bin/env python3
"""Fitscube: Combine single-Stokes FITS files into a Stokes cube.

Assumes:
- All files have the same WCS
- All files have the same shape / pixel grid
- All the relevant information is in the first header of the first image

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from fitscube.fitscube import logger
from fitscube.logging import set_verbosity


def combine_stokes(
    stokes_i_file: Path,
    stokes_q_file: Path,
    stokes_u_file: Path,
    stokes_v_file: Path | None = None,
) -> fits.HDUList:
    """Combine single-Stokes FITS files into a Stokes cube.

    Args:
        stokes_i_file (Path): Path to Stokes I file
        stokes_q_file (Path): Path to Stokes Q file
        stokes_u_file (Path): Path to Stokes U file
        stokes_v_file (Path | None, optional): Path to the Stokes V file. Defaults to None.

    Raises:
        ValueError: If the headers are not the same for Stokes I and Q
        ValueError: If the data are not the same shape for Stokes I and Q
        ValueError: If the headers are not the same for Stokes I and U
        ValueError: If the data are not the same shape for Stokes I and U
        ValueError: If the headers are not the same for Stokes I and V
        ValueError: If the data are not the same shape for Stokes I and V

    Returns:
        fits.HDUList: The combined Stokes cube
    """
    # Read in the data
    stokes_i = fits.getdata(stokes_i_file)
    stokes_q = fits.getdata(stokes_q_file)
    stokes_u = fits.getdata(stokes_u_file)
    stokes_v = fits.getdata(stokes_v_file) if stokes_v_file is not None else None

    # Get the header
    stokes_i_header = fits.getheader(stokes_i_file)
    stokes_q_header = fits.getheader(stokes_q_file)
    stokes_u_header = fits.getheader(stokes_u_file)
    stokes_v_header = (
        fits.getheader(stokes_v_file) if stokes_v_file is not None else None
    )

    # Check that the headers are the same
    if stokes_i_header != stokes_q_header:
        msg = "Stokes I and Q headers are not the same."
        raise ValueError(msg)
    if stokes_i_header != stokes_u_header:
        msg = "Stokes I and U headers are not the same."
        raise ValueError(msg)
    if stokes_v_file is not None and stokes_i_header != stokes_v_header:
        msg = "Stokes I and V headers are not the same."
        raise ValueError(msg)

    # Check that the data are the same shape
    if stokes_i.shape != stokes_q.shape:
        msg = "Stokes I and Q data are not the same shape."
        raise ValueError(msg)
    if stokes_i.shape != stokes_u.shape:
        msg = "Stokes I and U data are not the same shape."
        raise ValueError(msg)
    if stokes_v_file is not None and stokes_i.shape != stokes_v.shape:
        msg = "Stokes I and V data are not the same shape."
        raise ValueError(msg)

    datas = (
        (stokes_i, stokes_q, stokes_u)
        if stokes_v_file is None
        else (stokes_i, stokes_q, stokes_u, stokes_v)
    )

    # Check if Stokes axis is present
    # Create the output header
    output_header = stokes_i_header.copy()
    # Check if Stokes axis is already present
    wcs = WCS(output_header)
    has_stokes = "STOKES" in wcs.axis_type_names
    if has_stokes:
        stokes_idx = wcs.axis_type_names[::-1].index("STOKES")
    else:
        stokes_idx = output_header["NAXIS"] + 1

    # Create the output cube
    if has_stokes:
        output_cube = np.concatenate(datas, axis=stokes_idx)
    else:
        output_cube = np.array(datas)

    output_header[f"CTYPE{stokes_idx}"] = "STOKES"
    output_header[f"CRVAL{stokes_idx}"] = 1
    output_header[f"CDELT{stokes_idx}"] = 1
    output_header[f"CRPIX{stokes_idx}"] = 1

    # Write the output file
    hdu = fits.PrimaryHDU(output_cube, output_header)
    return fits.HDUList([hdu])


def cli():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stokes_i_file", type=Path, help="Stokes I file")
    parser.add_argument("stokes_q_file", type=Path, help="Stokes Q file")
    parser.add_argument("stokes_u_file", type=Path, help="Stokes U file")
    parser.add_argument("output_file", type=Path, help="Output file")
    parser.add_argument("-V", "--stokes_v_file", type=Path, help="Stokes V file")
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    parser.add_argument(
        "-v", "--verbosity", default=0, action="count", help="Increase output verbosity"
    )
    args = parser.parse_args()
    set_verbosity(
        logger=logger,
        verbosity=args.verbosity,
    )
    overwrite = args.overwrite
    output_file = Path(args.output_file)
    if not overwrite and output_file.exists():
        msg = f"Output file {output_file} already exists. Use --overwrite to overwrite."
        raise FileExistsError(msg)

    hdul = combine_stokes(
        stokes_i_file=args.stokes_i_file,
        stokes_q_file=args.stokes_q_file,
        stokes_u_file=args.stokes_u_file,
        stokes_v_file=args.stokes_v_file,
    )
    hdul.writeto(output_file, overwrite=overwrite)
    logger.info("Written cube to %s", output_file)


if __name__ == "__main__":
    cli()
