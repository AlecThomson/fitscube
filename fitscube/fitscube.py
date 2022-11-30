#!/usr/bin/env python3
"""Fitscube: Combine FITS files into a cube."""

import os
from typing import List

from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import numpy as np
from tqdm.auto import tqdm


def main(
    file_list: List[str],
    out_cube: str,
    overwrite: bool = False,
):
    if not overwrite and os.path.exists(out_cube):
        raise FileExistsError(f"Output file {out_cube} already exists. Use --overwrite to overwrite.")
    freqs = []
    for chan, image in enumerate(
        tqdm(
            file_list,
            desc="Reading channel image",
        )
    ):
        # init cube
        if chan == 0:
            old_name = image
            old_header = fits.getheader(old_name)
            wcs = WCS(old_header)
            # Look for the frequency axis in wcs
            idx = 0
            for j, t in enumerate(
                wcs.axis_type_names[::-1]
            ):  # Reverse to match index order
                if t == "FREQ":
                    idx = j
                    break
            if idx == 0:
                raise ValueError(
                    "No frequency axis found in WCS. "
                )
            plane_0 = fits.getdata(old_name)
            plane_shape = list(plane_0.shape)
            cube_shape = plane_shape.copy()
            cube_shape[idx] = len(file_list)

            data_cube = np.zeros(cube_shape)

        plane = fits.getdata(image)
        data_cube[chan] = plane
        freq = WCS(image).spectral.pixel_to_world(0)
        freqs.append(freq.to(u.Hz).value)
    # Write out cubes
    freqs = np.array(freqs) * u.Hz
    even_freq = np.diff(freqs).std() < 1e-6 * u.Hz
    if not even_freq:
        print("WARNING: Frequencies are not evenly spaced")
        print("Writing out frequency axis as a separate file")
        freqs_file = out_cube.replace(".fits", "_freqs.txt")
        np.savetxt(freqs_file.to(u.Hz).value, freqs)
    new_header = old_header.copy()
    new_header["NAXIS"] = len(cube_shape)
    new_header[f"NAXIS{idx}"] = len(freqs)
    new_header[f"CRPIX{idx}"] = 1
    new_header[f"CRVAL{idx}"] = freqs[0].value
    new_header[f"CDELT{idx}"] = np.diff(freqs).mean().value
    new_header[f"CUNIT{idx}"] = "Hz"
    fits.writeto(out_cube, data_cube, new_header, overwrite=overwrite)


def cli():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_list", nargs="+", help="List of FITS files to combine")
    parser.add_argument("out_cube", help="Output FITS file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")
    args = parser.parse_args()
    main(
        file_list=args.file_list,
        out_cube=args.out_cube,
        overwrite=args.overwrite,
    )

if __name__ == '__main__':
    cli()