#!/usr/bin/env python3
"""Fitscube: Combine single-frequency FITS files into a cube.

Assumes:
- All files have the same WCS
- All files have the same shape / pixel grid
- Frequency is either a WCS axis or in the REFFREQ header keyword
- All the relevant information is in the first header of the first image

"""

from __future__ import annotations

import argparse
import asyncio
from io import BufferedRandom
from pathlib import Path
from typing import Awaitable, NamedTuple, TypeVar, cast

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from numpy.typing import ArrayLike
from radio_beam import Beam, Beams
from radio_beam.beam import NoBeamException
from tqdm.asyncio import tqdm

from fitscube.logging import set_verbosity, setup_logger

logger = setup_logger()

T = TypeVar("T")

# store the number of bytes per value in a dictionary
BIT_DICT = {
    64: 8,
    32: 4,
    16: 2,
    8: 1,
}


class InitResult(NamedTuple):
    """Initialization result."""

    header: fits.Header
    """Output header"""
    freq_idx: int
    """Index of frequency axis"""
    freq_fits_idx: int
    """FITS index of frequency axis"""
    is_2d: bool
    """Whether the input is 2D"""


class FrequencyInfo(NamedTuple):
    """Frequency information."""

    freqs: u.Quantity
    """Frequencies"""
    missing_chan_idx: ArrayLike
    """Missing channel indices"""


class FileFrequencyInfo(NamedTuple):
    """File frequency information."""

    file_freqs: u.Quantity
    """Frequencies matching each file"""
    freqs: u.Quantity
    """Frequencies"""
    missing_chan_idx: ArrayLike
    """Missing channel indices"""


async def write_channel_to_cube_coro(
    file_handle: BufferedRandom, plane_bytes: bytes, chan: int, header: fits.Header
) -> None:
    msg = f"Writing channel {chan} to cube"
    logger.info(msg)
    seek_length = len(header.tostring()) + (len(plane_bytes) * chan)
    file_handle.seek(seek_length)
    file_handle.write(plane_bytes)


def write_channel_to_cube(
    file_handle: BufferedRandom, plane_bytes: bytes, chan: int, header: fits.Header
) -> None:
    return asyncio.run(
        write_channel_to_cube_coro(file_handle, plane_bytes, chan, header)
    )


# Stolen from https://stackoverflow.com/a/61478547
async def gather_with_limit(
    limit: int | None, *coros: Awaitable[T], desc: str | None = None
) -> list[T]:
    """Gather with a limit on the number of coroutines running at once.

    Args:
        limit (int): The number of coroutines to run at once
        coros (Awaitable): The coroutines to run

    Returns:
        Awaitable: The result of the coroutines
    """
    if limit is None:
        return cast(list[T], await tqdm.gather(*coros, desc=desc))

    semaphore = asyncio.Semaphore(limit)

    async def sem_coro(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return cast(list[T], await tqdm.gather(*(sem_coro(c) for c in coros), desc=desc))


# https://stackoverflow.com/a/66082278
def np_arange_fix(start: float, stop: float, step: float) -> ArrayLike:
    n = (stop - start) / step + 1
    x = n - int(n)
    stop += step * max(0.1, x) if x < 0.5 else 0
    return np.arange(start, stop, step)


def isin_close(element: ArrayLike, test_element: ArrayLike) -> ArrayLike:
    """Check if element is in test_element, within a tolerance.

    Args:
        element (ArrayLike): Element to check
        test_element (ArrayLike): Element to check against

    Returns:
        ArrayLike: Boolean array
    """
    return np.isclose(element[:, None], test_element).any(1)


def even_spacing(freqs: u.Quantity) -> FrequencyInfo:
    """Make the frequencies evenly spaced.

    Args:
        freqs (u.Quantity): Original frequencies

    Returns:
        FrequencyInfo: freqs, missing_chan_idx
    """
    freqs_arr = freqs.value.astype(np.longdouble)
    diffs = np.diff(freqs_arr)
    min_diff: float = np.min(diffs)
    # Create a new array with the minimum difference
    new_freqs = np_arange_fix(freqs_arr[0], freqs_arr[-1], min_diff)
    missing_chan_idx = np.logical_not(isin_close(new_freqs, freqs_arr))

    return FrequencyInfo(new_freqs * freqs.unit, missing_chan_idx)


async def create_cube_from_scratch_coro(
    output_file: Path,
    output_header: fits.Header,
    overwrite: bool = False,
) -> fits.Header:
    if output_file.exists() and not overwrite:
        msg = f"Output file {output_file} already exists."
        raise FileExistsError(msg)

    if output_file.exists() and overwrite:
        output_file.unlink()

    output_wcs = WCS(output_header)
    output_shape = output_wcs.array_shape
    msg = f"Creating a new FITS file with shape {output_shape}"
    logger.info(msg)
    # If the output shape is less than 1801, we can create a blank array
    # in memory and write it to disk
    if np.prod(output_shape) < 1801:
        msg = "Output cube is small enough to create in memory"
        logger.warning(msg)
        out_arr = np.zeros(output_shape)
        fits.writeto(output_file, out_arr, output_header, overwrite=overwrite)
        with fits.open(output_file, mode="denywrite", memmap=True) as hdu_list:
            hdu = hdu_list[0]
            data = hdu.data
            on_disk_shape = data.shape
            assert (
                data.shape == output_shape
            ), f"Output shape {on_disk_shape} does not match header {output_shape}!"
        return fits.getheader(output_file)

    logger.info("Output cube is too large to create in memory. Creating a blank file.")

    small_size = [1 for _ in output_shape]
    data = np.zeros(small_size)
    hdu = fits.PrimaryHDU(data)
    header = hdu.header
    while len(header) < (36 * 4 - 1):
        header.append()  # Adds a blank card to the end

    for key, value in output_header.items():
        header[key] = value

    header.tofile(output_file, overwrite=overwrite)

    bytes_per_value = BIT_DICT.get(abs(output_header["BITPIX"]), None)
    if bytes_per_value is None:
        msg = f"BITPIX value {output_header['BITPIX']} not recognized"
        raise ValueError(msg)

    with output_file.open("rb+") as fobj:
        # Seek past the length of the header, plus the length of the
        # Data we want to write.
        # 8 is the number of bytes per value, i.e. abs(header['BITPIX'])/8
        # (this example is assuming a 64-bit float)
        file_length = len(output_header.tostring()) + (
            np.prod(output_shape) * np.abs(output_header["BITPIX"] // bytes_per_value)
        )
        # FITS files must be a multiple of 2880 bytes long; the final -1
        # is to account for the final byte that we are about to write.
        file_length = ((file_length + 2880 - 1) // 2880) * 2880 - 1
        fobj.seek(file_length)
        fobj.write(b"\0")

    with fits.open(output_file, mode="denywrite", memmap=True) as hdu_list:
        hdu = hdu_list[0]
        data = hdu.data
        on_disk_shape = data.shape
        assert (
            on_disk_shape == output_shape
        ), f"Output shape {on_disk_shape} does not match header {output_shape}!"

    return fits.getheader(output_file)


def create_output_cube(
    old_name: Path,
    out_cube: Path,
    freqs: u.Quantity,
    ignore_freq: bool = False,
    overwrite: bool = False,
) -> InitResult:
    return asyncio.run(
        create_output_cube_coro(old_name, out_cube, freqs, ignore_freq, overwrite)
    )


async def create_output_cube_coro(
    old_name: Path,
    out_cube: Path,
    freqs: u.Quantity,
    ignore_freq: bool = False,
    overwrite: bool = False,
) -> InitResult:
    """Initialize the data cube.

    Args:
        old_name (str): Old FITS file name
        n_chan (int): Number of channels

    Raises:
        KeyError: If 2D and REFFREQ is not in header
        ValueError: If not 2D and FREQ is not in header

    Returns:
        InitResult: header, freq_idx, freq_fits_idx, is_2d
    """
    old_data, old_header = fits.getdata(old_name, header=True)
    even_freq = np.diff(freqs).std() < (1e-4 * u.Hz)
    if not even_freq:
        logger.warning("Frequencies are not evenly spaced")

    n_chan = len(freqs)

    is_2d = len(old_data.shape) == 2
    idx = 0
    fits_idx = 3
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

    new_header = old_header.copy()
    new_header["NAXIS"] = 3 if is_2d else len(old_data.shape)
    new_header[f"NAXIS{fits_idx}"] = n_chan
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

    plane_shape = list(old_data.shape)
    cube_shape = plane_shape.copy()
    if is_2d:
        cube_shape.insert(0, n_chan)
    else:
        cube_shape[idx] = n_chan

    output_header = await create_cube_from_scratch_coro(
        output_file=out_cube, output_header=new_header, overwrite=overwrite
    )
    return InitResult(
        header=output_header, freq_idx=idx, freq_fits_idx=fits_idx, is_2d=is_2d
    )


def read_freq_from_header(image_path: Path) -> u.Quantity:
    return asyncio.run(read_freq_from_header_coro(image_path))


async def read_freq_from_header_coro(
    image_path: Path,
) -> u.Quantity:
    header = await asyncio.to_thread(fits.getheader, image_path)
    wcs = WCS(header)
    array_shape = wcs.array_shape
    if array_shape is None:
        msg = "WCS does not have an array shape"
        raise ValueError(msg)
    is_2d = len(array_shape) == 2
    if is_2d:
        try:
            freq = await asyncio.to_thread(header.get, "REFFREQ")
            return freq * u.Hz
        except KeyError as e:
            msg = "REFFREQ not in header. Cannot combine 2D images without frequency information."
            raise KeyError(msg) from e
    try:
        if "SPECSYS" not in header:
            header["SPECSYS"] = "TOPOCENT"
        wcs = WCS(header)
        return wcs.spectral.pixel_to_world(0).to(u.Hz)
    except Exception as e:
        msg = "No FREQ axis found in WCS. Cannot combine N-D images without frequency information."
        raise ValueError(msg) from e


def parse_freqs(
    file_list: list[Path],
    freq_file: Path | None = None,
    freq_list: list[float] | None = None,
    ignore_freq: bool = False,
    create_blanks: bool = False,
) -> FileFrequencyInfo:
    return asyncio.run(
        parse_freqs_coro(file_list, freq_file, freq_list, ignore_freq, create_blanks)
    )


async def parse_freqs_coro(
    file_list: list[Path],
    freq_file: Path | None = None,
    freq_list: list[float] | None = None,
    ignore_freq: bool = False,
    create_blanks: bool = False,
) -> FileFrequencyInfo:
    """Parse the frequency information.

    Args:
        file_list (list[str]): List of FITS files
        freq_file (str | None, optional): File containing frequnecies. Defaults to None.
        freq_list (list[float] | None, optional): List of frequencies. Defaults to None.
        ignore_freq (bool | None, optional): Ignore frequency information. Defaults to False.

    Raises:
        ValueError: If both freq_file and freq_list are specified
        KeyError: If 2D and REFFREQ is not in header
        ValueError: If not 2D and FREQ is not in header

    Returns:
        FileFrequencyInfo: file_freqs, freqs, missing_chan_idx
    """
    if ignore_freq:
        logger.info("Ignoring frequency information")
        return FileFrequencyInfo(
            file_freqs=np.arange(len(file_list)) * u.Hz,
            freqs=np.arange(len(file_list)) * u.Hz,
            missing_chan_idx=np.zeros(len(file_list)).astype(bool),
        )

    if freq_file is not None and freq_list is not None:
        msg = "Must specify either freq_file or freq_list, not both"
        raise ValueError(msg)

    if freq_file is not None:
        logger.info("Reading frequencies from %s", freq_file)
        file_freqs = np.loadtxt(freq_file) * u.Hz
        assert (
            len(file_freqs) == len(file_list)
        ), f"Number of frequencies in {freq_file} ({len({file_freqs})}) does not match number of images ({len(file_list)})"
        missing_chan_idx = np.zeros(len(file_list)).astype(bool)

    else:
        logger.info("Reading frequencies from FITS files")
        first_header = fits.getheader(file_list[0])
        if "SPECSYS" not in first_header:
            logger.warning("SPECSYS not in header(s). Will set to TOPOCENT")
        # file_freqs = np.arange(len(file_list)) * u.Hz
        missing_chan_idx = np.zeros(len(file_list)).astype(bool)
        coros = []
        for image_path in file_list:
            coro = read_freq_from_header_coro(image_path)
            coros.append(coro)

        list_of_freqs = await gather_with_limit(
            None, *coros, desc="Extracting frequencies"
        )
        file_freqs = np.array([f.to(u.Hz).value for f in list_of_freqs]) * u.Hz

        freqs = file_freqs.copy()

    if create_blanks:
        logger.info("Trying to create a blank cube with evenly spaced frequencies")
        freqs, missing_chan_idx = even_spacing(file_freqs)

    return FileFrequencyInfo(
        file_freqs=file_freqs,
        freqs=freqs,
        missing_chan_idx=missing_chan_idx,
    )


def parse_beams(
    file_list: list[Path],
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
    """Get the polarisation axis.

    Args:
        header (fits.Header): Primary header

    Returns:
        int: Polarisation axis (in FITS)
    """
    wcs = WCS(header)
    array_shape = wcs.array_shape
    if array_shape is None:
        msg = "WCS does not have an array shape"
        raise ValueError(msg)

    for _, (ctype, naxis, crpix) in enumerate(
        zip(wcs.axis_type_names, array_shape[::-1], wcs.wcs.crpix)
    ):
        if ctype == "STOKES":
            assert (
                naxis <= 1
            ), f"Only one polarisation axis is supported - found {naxis}"
            return int(crpix - 1)
    return 0


def make_beam_table(
    beams: Beams, header: fits.Header
) -> tuple[fits.BinTableHDU, fits.Header]:
    """Make a beam table.

    Args:
        beams (Beams): Beams object
        header (fits.Header): Primary header

    Returns:
        tuple[fits.BinTableHDU, fits.Header]: Beam table and updated header
    """
    header["CASAMBM"] = True
    header["COMMENT"] = "The PSF in each image plane varies."
    header["COMMENT"] = "Full beam information is stored in the second FITS extension."
    del header["BMAJ"], header["BMIN"], header["BPA"]
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
    out_cube: Path,
    freq_file: Path | None = None,
    freq_list: list[float] | None = None,
    ignore_freq: bool = False,
    create_blanks: bool = False,
    overwrite: bool = False,
    max_workers: int | None = None,
) -> u.Quantity:
    return asyncio.run(
        combine_fits_coro(
            file_list=file_list,
            out_cube=out_cube,
            freq_file=freq_file,
            freq_list=freq_list,
            ignore_freq=ignore_freq,
            create_blanks=create_blanks,
            overwrite=overwrite,
            max_workers=max_workers,
        )
    )


async def process_channel(
    file_handle: BufferedRandom,
    new_header: fits.Header,
    new_channel: int,
    old_channel: int,
    is_missing: bool,
    file_list: list[Path],
) -> None:
    msg = f"Processing channel {new_channel}"
    logger.info(msg)
    if is_missing:
        plane = await asyncio.to_thread(fits.getdata, file_list[0])
        plane *= np.nan
    else:
        plane = await asyncio.to_thread(fits.getdata, file_list[old_channel])

    plane_bytes = plane.tobytes()
    await write_channel_to_cube_coro(
        file_handle=file_handle,
        plane_bytes=plane_bytes,
        chan=new_channel,
        header=new_header,
    )
    del plane, plane_bytes


async def combine_fits_coro(
    file_list: list[Path],
    out_cube: Path,
    freq_file: Path | None = None,
    freq_list: list[float] | None = None,
    ignore_freq: bool = False,
    create_blanks: bool = False,
    overwrite: bool = False,
    max_workers: int | None = None,
) -> u.Quantity:
    """Combine FITS files into a cube.

    Args:
        file_list (list[Path]): List of FITS files to combine
        freq_file (Path | None, optional): Frequency file. Defaults to None.
        freq_list (list[float] | None, optional): List of frequencies. Defaults to None.
        ignore_freq (bool, optional): Ignore frequency information. Defaults to False.
        create_blanks (bool, optional): Attempt to create even frequency spacing. Defaults to False.

    Returns:
        tuple[fits.HDUList, u.Quantity]: The combined FITS cube and frequencies
    """
    # TODO: Check that all files have the same WCS

    file_freqs, freqs, missing_chan_idx = await parse_freqs_coro(
        freq_file=freq_file,
        freq_list=freq_list,
        ignore_freq=ignore_freq,
        file_list=file_list,
        create_blanks=create_blanks,
    )

    # Sort the files by frequency
    old_sort_idx = np.argsort(file_freqs)
    file_list = np.array(file_list)[old_sort_idx].tolist()
    new_sort_idx = np.argsort(freqs)
    freqs = freqs[new_sort_idx]
    missing_chan_idx = missing_chan_idx[new_sort_idx]

    # Initialize the data cube
    new_header, _, _, _ = await create_output_cube_coro(
        old_name=file_list[0],
        out_cube=out_cube,
        freqs=freqs,
        ignore_freq=ignore_freq,
        overwrite=overwrite,
    )

    new_channels = np.arange(len(freqs))
    old_channels = np.arange(len(file_freqs))

    new_to_old = dict(zip(new_channels[np.logical_not(missing_chan_idx)], old_channels))

    coros = []
    with out_cube.open("rb+") as file_handle:
        for new_channel in new_channels:
            is_missing = missing_chan_idx[new_channel]
            old_channel = new_to_old.get(new_channel)
            if is_missing:
                old_channel = 0
            if old_channel is None:
                msg = f"Missing channel {new_channel} in input files"
                raise ValueError(msg)

            coro = process_channel(
                file_handle=file_handle,
                new_header=new_header,
                new_channel=new_channel,
                old_channel=old_channel,
                is_missing=is_missing,
                file_list=file_list,
            )
            coros.append(coro)

        await gather_with_limit(max_workers, *coros, desc="Writing channels")

    # Handle beams
    if "BMAJ" in fits.getheader(file_list[0]):
        logger.info("Extracting beam information")
        beams = parse_beams(file_list)
        with fits.open(out_cube, memmap=True, mode="denywrite") as hdu_list:
            hdu = hdu_list[0]
            data = hdu.data
            header = hdu.header
        primary_hdu = fits.PrimaryHDU(data=data, header=header)
        logger.info("Adding beam table to primary header")
        beam_table_hdu, new_header = make_beam_table(beams, new_header)
        new_hdu_list = fits.HDUList([primary_hdu, beam_table_hdu])
        new_hdu_list.writeto(out_cube, overwrite=True)

    return freqs


def cli() -> None:
    """Command-line interface."""
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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of workers to use for concurrent processing",
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

    freqs = combine_fits(
        file_list=args.file_list,
        out_cube=out_cube,
        freq_file=args.freq_file,
        freq_list=args.freqs,
        ignore_freq=args.ignore_freq,
        create_blanks=args.create_blanks,
        overwrite=overwrite,
        max_workers=args.max_workers,
    )

    logger.info("Written cube to %s", out_cube)
    np.savetxt(freqs_file, freqs.to(u.Hz).value)
    logger.info("Written frequencies to %s", freqs_file)


if __name__ == "__main__":
    cli()
