#!/usr/bin/env python3
"""Fitscube: Combine FITS files into a cube.

Assumes:
- Images are passed in frequency order
- All files have the same WCS
- All files have the same shape / pixel grid
- Frequency is either a WCS axis or in the REFFREQ header keyword
- All the relevant information is in the first header of the first image

"""

import os
from typing import List, Tuple

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from tqdm.auto import tqdm


def init_cube(
    old_name: str,
    n_chan: int,
) -> Tuple[np.ndarray, fits.Header, int, bool,]:
    """Initialize the data cube.

    Args:
        old_name (str): Old FITS file name
        n_chan (int): Number of channels

    Raises:
        KeyError: If 2D and REFFREQ is not in header
        ValueError: If not 2D and FREQ is not in header

    Returns:
        Tuple[np.ndarray, fits.Header, int, bool,]: Output data cube, header, index of frequency axis, and if 2D
    """
    old_header = fits.getheader(old_name)
    old_data = fits.getdata(old_name)
    is_2d = len(old_data.shape) == 2
    if is_2d:
        print("Input image is 2D. Looking for REFFREQ.")
        idx = 0
        # Look for REFREQ in header
        try:
            _ = old_header["REFFREQ"]
        except KeyError:
            raise KeyError("Images are 2D and REFFREQ not found in header")

    else:
        print("Input image is a cube. Looking for FREQ axis.")
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
            raise ValueError("No FREQ axis found in WCS. ")
        fits_idx = len(wcs.axis_type_names) - idx
        print(f"FREQ axis found at index {idx} (NAXIS{fits_idx})")

    plane_shape = list(old_data.shape)
    cube_shape = plane_shape.copy()
    if is_2d:
        cube_shape.insert(0, n_chan)
    else:
        cube_shape[idx] = n_chan

    data_cube = np.zeros(cube_shape)
    return data_cube, old_header, idx, is_2d


def main(
    file_list: List[str],
    out_cube: str,
    overwrite: bool = False,
) -> None:
    """Combine FITS files into a cube.

    Args:
        file_list (List[str]): List of FITS files to combine
        out_cube (str): Name of output FITS file
        overwrite (bool, optional): Whether to overwrite output cube. Defaults to False.

    Raises:
        FileExistsError: If output file exists and overwrite is False.
    """
    if not overwrite and os.path.exists(out_cube):
        raise FileExistsError(
            f"Output file {out_cube} already exists. Use --overwrite to overwrite."
        )

    # TODO: Check that all files have the same WCS
    # TODO: Check if PSF is in header, and then add it to the output header / beamtable

    freqs = []
    for chan, image in enumerate(
        tqdm(
            file_list,
            desc="Reading channel image",
        )
    ):
        # init cube
        if chan == 0:
            data_cube, old_header, idx, is_2d = init_cube(
                old_name=image,
                n_chan=len(file_list),
            )

        plane = fits.getdata(image)
        slicer = [slice(None)] * len(plane.shape)
        if is_2d:
            slicer.insert(0, chan)
        else:
            slicer[idx] = chan
        data_cube[tuple(slicer)] = plane
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
    if is_2d:
        fits_idx = 3
    else:
        wcs = WCS(old_header)
        fits_idx = fits_idx = len(wcs.axis_type_names) - idx

    new_header["NAXIS"] = len(data_cube.shape)
    new_header[f"NAXIS{fits_idx}"] = len(freqs)
    new_header[f"CRPIX{fits_idx}"] = 1
    new_header[f"CRVAL{fits_idx}"] = freqs[0].value
    new_header[f"CDELT{fits_idx}"] = np.diff(freqs).mean().value
    new_header[f"CUNIT{fits_idx}"] = "Hz"
    fits.writeto(out_cube, data_cube, new_header, overwrite=overwrite)


def cli():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_list", nargs="+", help="List of FITS files to combine (in frequency order)")
    parser.add_argument("out_cube", help="Output FITS file")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )
    args = parser.parse_args()
    main(
        file_list=args.file_list,
        out_cube=args.out_cube,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    cli()
