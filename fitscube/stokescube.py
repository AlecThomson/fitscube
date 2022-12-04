#!/usr/bin/env python3
"""Fitscube: Combine single-Stokes FITS files into a Stokes cube.

Assumes:
- All files have the same WCS
- All files have the same shape / pixel grid
- All the relevant information is in the first header of the first image

"""

import os
from typing import List, Tuple
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

def main(
    stokes_I_file: str,
    stokes_Q_file: str,
    stokes_U_file: str,
    stokes_V_file: str = None,
) -> fits.HDUList:

    # Read in the data
    stokes_I = fits.getdata(stokes_I_file)
    stokes_Q = fits.getdata(stokes_Q_file)
    stokes_U = fits.getdata(stokes_U_file)
    if stokes_V_file is not None:
        stokes_V = fits.getdata(stokes_V_file)

    # Get the header
    stokes_I_header = fits.getheader(stokes_I_file)
    stokes_Q_header = fits.getheader(stokes_Q_file)
    stokes_U_header = fits.getheader(stokes_U_file)
    if stokes_V_file is not None:
        stokes_V_header = fits.getheader(stokes_V_file)

    # Check that the headers are the same
    if stokes_I_header != stokes_Q_header:
        raise ValueError("Stokes I and Q headers are not the same.")
    if stokes_I_header != stokes_U_header:
        raise ValueError("Stokes I and U headers are not the same.")
    if stokes_V_file is not None:
        if stokes_I_header != stokes_V_header:
            raise ValueError("Stokes I and V headers are not the same.")

    # Check that the data are the same shape
    if stokes_I.shape != stokes_Q.shape:
        raise ValueError("Stokes I and Q data are not the same shape.")
    if stokes_I.shape != stokes_U.shape:
        raise ValueError("Stokes I and U data are not the same shape.")
    if stokes_V_file is not None:
        if stokes_I.shape != stokes_V.shape:
            raise ValueError("Stokes I and V data are not the same shape.")

    datas = (stokes_I, stokes_Q, stokes_U) if stokes_V_file is None else (stokes_I, stokes_Q, stokes_U, stokes_V)

    # Check if Stokes axis is present
    # Create the output header
    output_header = stokes_I_header.copy()
    # Check if Stokes axis is already present
    wcs = WCS(output_header)
    has_stokes = "STOKES" in wcs.axis_type_names
    if has_stokes:
        stokes_idx = wcs.axis_type_names[::-1].index("STOKES")
    else:
        stokes_idx = output_header["NAXIS"] + 1

    # Create the output cube
    if has_stokes:
        output_cube = np.concatenate(
            datas, axis=stokes_idx
        )
    else:
        output_cube = np.array(
            datas
        )

    output_header[f"CTYPE{stokes_idx}"] = "STOKES"
    output_header[f"CRVAL{stokes_idx}"] = 1
    output_header[f"CDELT{stokes_idx}"] = 1
    output_header[f"CRPIX{stokes_idx}"] = 1

    # Write the output file
    hdu = fits.PrimaryHDU(output_cube, output_header)
    hdul = fits.HDUList([hdu])

    return hdul

def cli():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stokes_I_file", type=str, help="Stokes I file")
    parser.add_argument("stokes_Q_file", type=str, help="Stokes Q file")
    parser.add_argument("stokes_U_file", type=str, help="Stokes U file")
    parser.add_argument("output_file", type=str, help="Output file")
    parser.add_argument("-v","--stokes_V_file", type=str, help="Stokes V file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")

    args = parser.parse_args()

    overwrite = args.overwrite
    output_file = args.output_file
    if not overwrite and os.path.exists(output_file):
        raise FileExistsError(
            f"Output file {output_file} already exists. Use --overwrite to overwrite."
        )

    hdul = main(
        stokes_I_file=args.stokes_I_file,
        stokes_Q_file=args.stokes_Q_file,
        stokes_U_file=args.stokes_U_file,
        stokes_V_file=args.stokes_V_file,
    )
    hdul.writeto(output_file, overwrite=overwrite)
    print(f"Written cube to {output_file}")

if __name__ == "__main__":
    cli()