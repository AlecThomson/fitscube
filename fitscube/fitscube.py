#!/usr/bin/env python3
"""Fitscube: Combine single-frequency FITS files into a cube.

Assumes:
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
    ignore_freq: bool = False,
) -> Tuple[np.ndarray, fits.Header, int, bool,]:
    """Initialize the data cube.

    Args:
        old_name (str): Old FITS file name
        n_chan (int): Number of channels
        ignore_freq (bool, optional): Ignore frequency information. Defaults to False.

    Raises:
        KeyError: If 2D and REFFREQ is not in header
        ValueError: If not 2D and FREQ is not in header

    Returns:
        Tuple[np.ndarray, fits.Header, int, bool,]: Output data cube, header, index of frequency axis, and if 2D
    """
    old_header = fits.getheader(old_name)
    old_data = fits.getdata(old_name)
    is_2d = len(old_data.shape) == 2
    idx = 0
    if not is_2d:
        print("Input image is a cube. Looking for FREQ axis.")
        wcs = WCS(old_header)
        # Look for the frequency axis in wcs
        try:
            idx = wcs.axis_type_names[::-1].index("FREQ")
        except ValueError:
            raise ValueError("No FREQ axis found in WCS.")
        fits_idx = wcs.axis_type_names.index("FREQ")
        print(f"FREQ axis found at index {idx} (NAXIS{fits_idx})")

    plane_shape = list(old_data.shape)
    cube_shape = plane_shape.copy()
    if is_2d:
        cube_shape.insert(0, n_chan)
    else:
        cube_shape[idx] = n_chan

    data_cube = np.zeros(cube_shape)
    return data_cube, old_header, idx, is_2d


def parse_freqs(
    file_list: List[str],
    freq_file: str = None,
    freq_list: List[float] = None,
    ignore_freq: bool = False,
) -> u.Quantity:
    """Parse the frequency information.

    Args:
        freq_file (str, optional): File containing frequencies. Defaults to None.
        freqs (List[float], optional): List of frequencies. Defaults to None.
        ignore_freq (bool, optional): Ignore frequency information. Defaults to False.

    Raises:
        ValueError: If freq_file and freqs are both None
        ValueError: If freq_file and freqs are both not None

    Returns:
        List[float]: List of frequencies
    """
    if ignore_freq:
        print("Ignoring frequency information")
        return np.arange(len(file_list)) * u.Hz
    # if freq_file is None and freq_list is None:
    #     raise ValueError("Must specify either freq_file or freq_list")
    if freq_file is not None and freq_list is not None:
        raise ValueError("Must specify either freq_file or freq_list, not both")
    if freq_file is not None:
        print(f"Reading frequencies from {freq_file}")
        freqs = np.loadtxt(freq_file) * u.Hz
    elif freq_list is not None:
        print(f"Using list of specified frequencies")
        freqs = np.array(freq_list) * u.Hz
    else:
        print("Reading frequencies from FITS files")
        freqs = np.arange(len(file_list)) * u.Hz
        for chan, image in enumerate(
            tqdm(
                file_list,
                desc="Extracting frequencies",
            )
        ):
            plane = fits.getdata(image)
            is_2d = len(plane.shape) == 2
            header = fits.getheader(image)
            if is_2d:
                try:
                    freqs[chan] = header["REFFREQ"] * u.Hz
                except KeyError:
                    raise KeyError(
                        "REFFREQ not in header. Cannot combine 2D images without frequency information."
                    )
            else:
                try:
                    freq = WCS(image).spectral.pixel_to_world(0)
                    freqs[chan] = freq.to(u.Hz)
                except Exception as e:
                    raise ValueError(
                        "No FREQ axis found in WCS. Cannot combine ND images without frequency information."
                    ) from e

    return freqs


def main(
    file_list: List[str],
    freq_file: str = None,
    freq_list: List[float] = None,
    ignore_freq: bool = False,
) -> Tuple[fits.HDUList, u.Quantity]:
    """Combine FITS files into a cube.

    Args:
        file_list (List[str]): List of FITS files to combine
        overwrite (bool, optional): Whether to overwrite output cube. Defaults to False.

    Raises:
        FileExistsError: If output file exists and overwrite is False.
    """

    # TODO: Check that all files have the same WCS
    # TODO: Check if PSF is in header, and then add it to the output header / beamtable

    n_images = len(file_list)

    freqs = parse_freqs(
        freq_file=freq_file,
        freq_list=freq_list,
        ignore_freq=ignore_freq,
        file_list=file_list,
    )

    assert (
        len(freqs) == n_images
    ), "Number of frequencies does not match number of images"

    # Sort the frequencies
    freqs = np.sort(freqs)

    # Sort the files by frequency
    file_list = [x for _, x in sorted(zip(freqs, file_list))]

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
                ignore_freq=ignore_freq,
            )

        plane = fits.getdata(image)
        slicer = [slice(None)] * len(plane.shape)
        if is_2d:
            slicer.insert(0, chan)
        else:
            slicer[idx] = chan
        data_cube[tuple(slicer)] = plane
    # Write out cubes
    even_freq = np.diff(freqs).std() < 1e-6 * u.Hz
    if not even_freq:
        print("WARNING: Frequencies are not evenly spaced")
        print("Use the frequency file to specify the frequencies")

    new_header = old_header.copy()
    wcs = WCS(old_header)
    if is_2d:
        fits_idx = len(wcs.axis_type_names) + 1
    else:
        fits_idx = wcs.axis_type_names.index("FREQ")
    new_header["NAXIS"] = len(data_cube.shape)
    new_header[f"NAXIS{fits_idx}"] = len(freqs)
    new_header[f"CRPIX{fits_idx}"] = 1
    new_header[f"CRVAL{fits_idx}"] = freqs[0].value
    new_header[f"CDELT{fits_idx}"] = np.diff(freqs).mean().value
    new_header[f"CUNIT{fits_idx}"] = "Hz"
    new_header[f"CTYPE{fits_idx}"] = "FREQ"
    if ignore_freq:
        new_header["HISTORY"] = "Frequency axis is not meaningful"

    hdu = fits.PrimaryHDU(data_cube, header=new_header)
    hdul = fits.HDUList([hdu])

    return hdul, freqs


def cli():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "file_list",
        nargs="+",
        help="List of FITS files to combine (in frequency order)",
    )
    parser.add_argument("out_cube", help="Output FITS file")
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    # Add options for specifying frequencies
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--freq-file",
        help="File containing frequencies in Hz",
        type=str,
        default=None,
    )
    group.add_argument(
        "--freqs",
        nargs="+",
        help="List of frequencies in Hz",
        type=float,
        default=None,
    )
    group.add_argument(
        "--ignore-freq",
        action="store_true",
        help="Ignore frequency information and just stack (probably not what you want)",
    )

    args = parser.parse_args()

    overwrite = args.overwrite
    out_cube = args.out_cube
    if not overwrite and os.path.exists(out_cube):
        raise FileExistsError(
            f"Output file {out_cube} already exists. Use --overwrite to overwrite."
        )

    freqs_file = out_cube.replace(".fits", ".freqs_Hz.txt")
    if os.path.exists(freqs_file) and not overwrite:
        raise FileExistsError(
            f"Output file {freqs_file} already exists. Use --overwrite to overwrite."
        )

    if overwrite:
        print("Overwriting output files")

    hdul, freqs = main(
        file_list=args.file_list,
        out_cube=args.out_cube,
        freq_file=args.freq_file,
        freq_list=args.freqs,
        ignore_freq=args.ignore_freq,
    )

    hdul.writeto(out_cube, overwrite=overwrite)
    print(f"Written cube to {out_cube}")
    np.savetxt(freqs_file, freqs.to(u.Hz).value)
    print(f"Written frequencies to {freqs_file}")


if __name__ == "__main__":
    cli()
