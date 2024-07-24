#!/usr/bin/env python3
"""Fitscube: Combine single-frequency FITS files into a cube.

Assumes:
- All files have the same WCS
- All files have the same shape / pixel grid
- Frequency is either a WCS axis or in the REFFREQ header keyword
- All the relevant information is in the first header of the first image

"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from radio_beam import Beam, Beams
from radio_beam.beam import NoBeamException
from tqdm.auto import tqdm

from fitscube.logging import set_verbosity, setup_logger

logger = setup_logger()


class InitResult(NamedTuple):
    data_cube: np.ndarray
    """Output data cube"""
    header: fits.Header
    """Output header"""
    idx: int
    """Index of frequency axis"""
    fits_idx: int
    """FITS index of frequency axis"""
    is_2d: bool
    """Whether the input is 2D"""


def isin_close(element: np.ndarray, test_element: np.ndarray) -> np.ndarray:
    """Check if element is in test_element, within a tolerance.

    Args:
        element (np.ndarray): Element to check
        test_element (np.ndarray): Element to check against

    Returns:
        np.ndarray: Boolean array
    """
    return np.isclose(element[:, None], test_element).any(1)


def even_spacing(freqs: u.Quantity) -> tuple[u.Quantity, np.ndarray]:
    """Make the frequencies evenly spaced.

    Args:
        freqs (u.Quantity): Original frequencies

    Returns:
        Tuple[u.Quantity, np.ndarray]: Evenly spaced frequencies and missing channel indices
    """
    freqs_arr = freqs.value.astype(np.longdouble)
    diffs = np.diff(freqs_arr)
    min_diff = np.min(diffs)
    # Create a new array with the minimum difference
    new_freqs = np.arange(freqs_arr[0], freqs_arr[-1], min_diff)
    missing_chan_idx = ~isin_close(new_freqs, freqs_arr)

    return new_freqs * freqs.unit, missing_chan_idx


def create_blank_data(
    data_cube: np.ndarray,
    freqs: u.Quantity,
    idx: int,
) -> tuple[np.ndarray | None, u.Quantity]:
    """Create a new data cube with evenly spaced frequencies, and fill in the missing channels with NaNs.

    Args:
        data_cube (np.ndarray): Original data cube
        freqs (u.Quantity): Original frequencies
        idx: Index of frequency axis

    Returns:
        Tuple[Optional[np.ndarray], u.Quantity]: New data cube and frequencies
    """
    new_freqs, missing_chan_idx = even_spacing(freqs)
    # Check if all frequencies present
    all_there = isin_close(freqs, new_freqs).all()
    if not all_there:
        return None, new_freqs

    # Create a new data cube with the new frequencies
    new_shape = list(data_cube.shape)
    new_shape[idx] = len(new_freqs)
    new_data_cube = np.empty(new_shape) * np.nan
    for old_chan, freq in enumerate(freqs):
        new_chans = np.where(np.isclose(new_freqs, freq))[0]
        assert len(new_chans) == 1, "Too many matching channels"
        new_chan = new_chans[0]
        new_slice = [slice(None)] * len(new_shape)
        new_slice[idx] = new_chan
        old_slice = [slice(None)] * len(new_shape)
        old_slice[idx] = old_chan
        new_data_cube[tuple(new_slice)] = data_cube[tuple(old_slice)]

    return new_data_cube, new_freqs


def init_cube(
    old_name: Path,
    n_chan: int,
) -> InitResult:
    """Initialize the data cube.

    Args:
        old_name (str): Old FITS file name
        n_chan (int): Number of channels

    Raises:
        KeyError: If 2D and REFFREQ is not in header
        ValueError: If not 2D and FREQ is not in header

    Returns:
        InitResult: Output data cube, header, index of frequency axis, FITS index, and if 2D
    """
    old_header = fits.getheader(old_name)
    old_data = fits.getdata(old_name)
    is_2d = len(old_data.shape) == 2
    idx = 0
    if not is_2d:
        logger.info("Input image is a cube. Looking for FREQ axis.")
        wcs = WCS(old_header)
        # Look for the frequency axis in wcs
        try:
            idx = wcs.axis_type_names[::-1].index("FREQ")
        except ValueError as e:
            msg = "No FREQ axis found in WCS."
            raise ValueError(msg) from e
        fits_idx = wcs.axis_type_names.index("FREQ") + 1
        logger.info("FREQ axis found at index %s (NAXIS%s)", idx, fits_idx)

    plane_shape = list(old_data.shape)
    cube_shape = plane_shape.copy()
    if is_2d:
        cube_shape.insert(0, n_chan)
    else:
        cube_shape[idx] = n_chan

    data_cube = np.zeros(cube_shape) * np.nan
    return InitResult(data_cube, old_header, idx, fits_idx, is_2d)


def parse_freqs(
    file_list: list[str],
    freq_file: str | None = None,
    freq_list: list[float] | None = None,
    ignore_freq: bool | None = False,
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
        logger.info("Ignoring frequency information")
        return np.arange(len(file_list)) * u.Hz

    if freq_file is not None and freq_list is not None:
        msg = "Must specify either freq_file or freq_list, not both"
        raise ValueError(msg)

    if freq_file is not None:
        logger.info("Reading frequencies from %s", freq_file)
        return np.loadtxt(freq_file) * u.Hz

    logger.info("Reading frequencies from FITS files")
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
            except KeyError as e:
                msg = "REFFREQ not in header. Cannot combine 2D images without frequency information."
                raise KeyError(msg) from e
        else:
            try:
                freq = WCS(image).spectral.pixel_to_world(0)
                freqs[chan] = freq.to(u.Hz)
            except Exception as e:
                msg = "No FREQ axis found in WCS. Cannot combine N-D images without frequency information."
                raise ValueError(msg) from e

    return freqs


def parse_beams(
    file_list: list[str],
) -> Beams:
    """Parse the beam information.

    Args:
        file_list (List[str]): List of FITS files

    Returns:
        Beams: Beams object
    """
    beam_list: list[Beam] = []
    for image in tqdm(
        file_list,
        desc="Extracting beams",
    ):
        header = fits.getheader(image)
        try:
            beam = Beam.from_fits_header(header)
        except NoBeamException:
            beam = Beam(major=np.nan * u.deg, minor=np.nan * u.deg, pa=np.nan * u.deg)
        beam_list.append(beam)

    return Beams(
        major=[beam.major.to(u.deg).value for beam in beam_list] * u.deg,
        minor=[beam.minor.to(u.deg).value for beam in beam_list] * u.deg,
        pa=[beam.pa.to(u.deg).value for beam in beam_list] * u.deg,
    )


def get_polarisation(header: fits.Header) -> int:
    wcs = WCS(header)

    for i, (ctype, naxis) in enumerate(zip(wcs.axis_type_names, wcs.array_shape[::-1])):
        if ctype == "STOKES":
            assert (
                naxis <= 1
            ), f"Only one polarisation axis is supported - found {naxis}"
            return i
    return 0


def make_beam_table(
    beams: Beams, header: fits.Header
) -> tuple[fits.BinTableHDU, fits.Header]:
    header["CASAMBM"] = True
    header["COMMENT"] = "The PSF in each image plane varies."
    header["COMMENT"] = "Full beam information is stored in the second FITS extension."
    nchan = len(beams.major)
    chans = np.arange(nchan)
    pol = get_polarisation(header)
    pols = np.ones(nchan, dtype=int) * pol
    tiny = np.finfo(np.float32).tiny
    beam_table = Table(
        data=[
            # Replace NaNs with np.finfo(np.float32).tiny - this is the smallest
            # positive number that can be represented in float32
            # We use this to keep CASA happy
            np.nan_to_num(beams.major.to(u.arcsec), nan=tiny * u.arcsec),
            np.nan_to_num(beams.minor.to(u.arcsec), nan=tiny * u.arcsec),
            np.nan_to_num(beams.pa.to(u.deg), nan=tiny * u.deg),
            chans,
            pols,
        ],
        names=["BMAJ", "BMIN", "BPA", "CHAN", "POL"],
        dtype=["f4", "f4", "f4", "i4", "i4"],
    )
    header["COMMENT"] = f"The value '{tiny}' repsenents a NaN PSF in the beamtable."
    tab_hdu = fits.table_to_hdu(beam_table)
    tab_header = tab_hdu.header
    tab_header["EXTNAME"] = "BEAMS"
    tab_header["NCHAN"] = nchan
    tab_header["NPOL"] = 1  # Only one pol for now

    return tab_hdu, header


def combine_fits(
    file_list: list[Path],
    freq_file: Path | None = None,
    freq_list: list[float] | None = None,
    ignore_freq: bool = False,
    create_blanks: bool = False,
) -> tuple[fits.HDUList, u.Quantity]:
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
    sort_idx = np.argsort(freqs)
    freqs = freqs[sort_idx]

    # Sort the files by frequency
    file_list = np.array(file_list)[sort_idx].tolist()

    for chan, image in enumerate(
        tqdm(
            file_list,
            desc="Reading channel image",
        )
    ):
        # init cube
        if chan == 0:
            init_res = init_cube(
                old_name=image,
                n_chan=len(file_list),
            )
            data_cube = init_res.data_cube
            old_header = init_res.header
            idx = init_res.idx
            fits_idx = init_res.fits_idx
            is_2d = init_res.is_2d

        plane = fits.getdata(image)
        slicer = [slice(None)] * len(plane.shape)
        if is_2d:
            slicer.insert(0, chan)
        else:
            slicer[idx] = chan
        data_cube[tuple(slicer)] = plane

    # Write out cubes
    even_freq = np.diff(freqs).std() < 1e-6 * u.Hz
    if not even_freq and create_blanks:
        logger.info("Trying to create a blank cube with evenly spaced frequencies")
        new_data_cube, new_freqs = create_blank_data(
            data_cube=data_cube,
            freqs=freqs,
            idx=idx,
        )
        if new_data_cube is not None:
            even_freq = True
            data_cube = new_data_cube
            freqs = new_freqs

    if not even_freq:
        logger.warning("Frequencies are not evenly spaced")
        logger.info("Use the frequency file to specify the frequencies")

    new_header = old_header.copy()
    new_header["NAXIS"] = len(data_cube.shape)
    new_header[f"NAXIS{fits_idx}"] = len(freqs)
    new_header[f"CRPIX{fits_idx}"] = 1
    new_header[f"CRVAL{fits_idx}"] = freqs[0].value
    new_header[f"CDELT{fits_idx}"] = np.diff(freqs).mean().value
    new_header[f"CUNIT{fits_idx}"] = "Hz"
    new_header[f"CTYPE{fits_idx}"] = "FREQ"
    if ignore_freq or not even_freq:
        new_header[f"CDELT{fits_idx}"] = 1
        del new_header[f"CUNIT{fits_idx}"]
        new_header[f"CTYPE{fits_idx}"] = "CHAN"
        new_header[f"CRVAL{fits_idx}"] = 1

    # Handle beams
    has_beams = "BMAJ" in fits.getheader(file_list[0])
    if has_beams:
        beams = parse_beams(file_list)
        beam_table, new_header = make_beam_table(beams, new_header)

    hdu = fits.PrimaryHDU(data_cube, header=new_header)
    hdul = fits.HDUList([hdu])

    if has_beams:
        hdul.append(beam_table)

    return hdul, freqs


def cli():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "file_list",
        nargs="+",
        help="List of FITS files to combine (in frequency order)",
        type=Path,
    )
    parser.add_argument("out_cube", help="Output FITS file", type=Path)
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    parser.add_argument(
        "--create-blanks",
        action="store_true",
        help="Try to create a blank cube with evenly spaced frequencies",
    )
    # Add options for specifying frequencies
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--freq-file",
        help="File containing frequencies in Hz",
        type=Path,
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
    parser.add_argument(
        "-v", "--verbosity", default=0, action="count", help="Increase output verbosity"
    )
    args = parser.parse_args()

    set_verbosity(
        logger=logger,
        verbosity=args.verbosity,
    )
    overwrite = bool(args.overwrite)
    out_cube = Path(args.out_cube)
    if not overwrite and out_cube.exists():
        msg = f"Output file {out_cube} already exists. Use --overwrite to overwrite."
        raise FileExistsError(msg)

    freqs_file = out_cube.with_suffix(".freqs_Hz.txt")
    if freqs_file.exists() and not overwrite:
        msg = f"Output file {freqs_file} already exists. Use --overwrite to overwrite."
        raise FileExistsError(msg)

    if overwrite:
        logger.info("Overwriting output files")

    hdul, freqs = combine_fits(
        file_list=args.file_list,
        freq_file=args.freq_file,
        freq_list=args.freqs,
        ignore_freq=args.ignore_freq,
        create_blanks=args.create_blanks,
    )

    hdul.writeto(out_cube, overwrite=overwrite)
    logger.info("Written cube to %s", out_cube)
    np.savetxt(freqs_file, freqs.to(u.Hz).value)
    logger.info("Written frequencies to %s", freqs_file)


if __name__ == "__main__":
    cli()
