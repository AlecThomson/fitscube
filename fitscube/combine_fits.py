#!/usr/bin/env python3
"""Fitscube: Combine single-frequency FITS files into a cube.

Assumes:
- All files have the same WCS
- All files have the same shape / pixel grid
- All the relevant information is in the first header of the first image
- Frequency is either a WCS axis or in the REFFREQ header keyword OR
- Time is present in the DATE-OBS header keyword for time-domain-mode
"""

from __future__ import annotations

import argparse
import asyncio
import warnings
from io import BufferedRandom
from pathlib import Path
from typing import (
    NamedTuple,
    TypeVar,
)

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from numpy.typing import ArrayLike
from radio_beam import Beam, Beams
from radio_beam.beam import NoBeamException
from tqdm.asyncio import tqdm

from fitscube.asyncio import gather_with_limit, sync_wrapper
from fitscube.logging import TQDM_OUT, logger, set_verbosity

T = TypeVar("T")

# store the number of bytes per value in a dictionary
BIT_DICT = {
    64: 8,
    32: 4,
    16: 2,
    8: 1,
}

warnings.filterwarnings("ignore", category=UserWarning, module="astropy.io.fits")
warnings.filterwarnings("ignore", category=VerifyWarning)


class InitResult(NamedTuple):
    """Initialization result."""

    header: fits.Header
    """Output header"""
    spec_idx: int
    """Index of Frequency/Time axis"""
    spec_fits_idx: int
    """FITS index of Frequency/Time axis"""
    is_2d: bool
    """Whether the input is 2D"""


# Note: spequency = generic term for time or frequency
class SpequencyInfo(NamedTuple):
    """Frequency/time information."""

    specs: u.Quantity
    """Frequencies/Times"""
    missing_chan_idx: ArrayLike
    """Missing channel indices"""


class FileSpequencyInfo(NamedTuple):
    """File frequency/time information."""

    file_specs: u.Quantity
    """Frequencies or times matching each file"""
    specs: u.Quantity
    """Frequency/time in Hz or s"""
    missing_chan_idx: ArrayLike
    """Missing channel/time indices"""


async def write_channel_to_cube_coro(
    file_handle: BufferedRandom, plane: ArrayLike, chan: int, header: fits.Header
) -> None:
    msg = f"Writing channel {chan} to cube"
    logger.info(msg)
    seek_length = len(header.tostring()) + (plane.nbytes * chan)
    file_handle.seek(seek_length)
    plane.tofile(file_handle)


write_channel_to_cube = sync_wrapper(write_channel_to_cube_coro)


# https://stackoverflow.com/a/66082278
def np_arange_fix(start: float, stop: float, step: float) -> ArrayLike:
    n = (stop - start) / step + 1
    x = n - int(n)
    stop += step * max(0.1, x) if x < 0.5 else 0
    return np.arange(start, stop, step)


def isin_close(
    element: ArrayLike, test_element: ArrayLike, time_domain_mode: bool = False
) -> ArrayLike:
    """Check if element is in test_element, within a tolerance.

    Args:
        element (ArrayLike): Element to check
        test_element (ArrayLike): Element to check against

    Returns:
        ArrayLike: Boolean array
    """
    if time_domain_mode:
        # the following should be sufficient to test integration times to ~5ms accuracy
        rtol = 1e-10
        atol = 1e-9
    else:
        # defaults for np.isclose
        rtol = 1e-5
        atol = 1e-9
    return np.isclose(element[:, None], test_element, atol, rtol).any(1)


def even_spacing(specs: u.Quantity, time_domain_mode: bool = False) -> SpequencyInfo:
    """Make the frequencies or times evenly spaced.

    Args:
        specs (u.Quantity): Original frequencies/times

    Returns:
        SpequencyInfo: specs, missing_chan_idx
    """
    specs_arr = specs.value.astype(np.longdouble)
    diffs = np.diff(specs_arr)
    min_diff: float = np.min(diffs)
    # Create a new array with the minimum difference
    new_specs = np_arange_fix(specs_arr[0], specs_arr[-1], min_diff)
    missing_chan_idx = np.logical_not(
        isin_close(new_specs, specs_arr, time_domain_mode)
    )
    return SpequencyInfo(new_specs * specs.unit, missing_chan_idx)


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
            assert data.shape == output_shape, (
                f"Output shape {on_disk_shape} does not match header {output_shape}!"
            )
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
        logger.debug(f"{key}={value}")

    header.tofile(output_file, overwrite=overwrite)

    bytes_per_value = BIT_DICT.get(abs(output_header["BITPIX"]), None)
    msg = f"Header BITPIX={output_header['BITPIX']}, bytes_per_value={bytes_per_value}"
    logger.info(msg)
    if bytes_per_value is None:
        msg = f"BITPIX value {output_header['BITPIX']} not recognized"
        raise ValueError(msg)

    with output_file.open("rb+") as fobj:
        # Seek past the length of the header, plus the length of the
        # Data we want to write.
        # 8 is the number of bytes per value, i.e. abs(header['BITPIX'])/8
        # (this example is assuming a 64-bit float)
        file_length = len(header.tostring()) + (np.prod(output_shape) * bytes_per_value)
        # FITS files must be a multiple of 2880 bytes long; the final -1
        # is to account for the final byte that we are about to write.
        file_length = ((file_length + 2880 - 1) // 2880) * 2880 - 1
        fobj.seek(file_length)
        fobj.write(b"\0")

    with fits.open(output_file, mode="denywrite", memmap=True) as hdu_list:
        hdu = hdu_list[0]
        data = hdu.data
        on_disk_shape = data.shape
        assert on_disk_shape == output_shape, (
            f"Output shape {on_disk_shape} does not match header {output_shape}!"
        )

    return fits.getheader(output_file)


async def create_output_cube_coro(
    old_name: Path,
    out_cube: Path,
    specs: u.Quantity,
    ignore_spec: bool = False,
    has_beams: bool = False,
    single_beam: bool = False,
    overwrite: bool = False,
    time_domain_mode: bool = False,
) -> InitResult:
    """Initialize the data cube.

    Args:
        old_name (str): Old FITS file name
        n_chan (int): Number of channels

    Raises:
        KeyError: If 2D and REFFREQ is not in header
        ValueError: If not 2D and FREQ is not in header

    Returns:
        InitResult: header, spec_idx, spec_fits_idx, is_2d
    """

    # define units if in time or freq domain
    unit = u.s if time_domain_mode else u.Hz
    ctype = "TIME" if time_domain_mode else "FREQ"

    old_data, old_header = fits.getdata(old_name, header=True, memmap=True)
    sorted_specs = np.sort(specs)
    if time_domain_mode:
        logger.info("Computing time-differences")

        # This attempts to constrain the deviation away from 'asbolute' time
        # as far as it can be. If some time steps are not regularly space
        # (differencce between the time of adjacent scans) can be positive or negative,
        # but so long as the assumulated total is close to 0 then we can say
        # it is close. This approach is catering to some strangeness in
        # ASKAP data. If the accumulated error is small enough we can assume
        # the FITS header can encoude the times as regular steps.
        diff_time = np.diff(sorted_specs)
        diff_diff_time = np.diff(diff_time)
        running_deviation_from_zero = np.abs(np.cumsum(diff_diff_time))
        even_spec = np.max(running_deviation_from_zero) < (np.mean(diff_time) * 0.02)

        # This is a simpler way where no attempt is made to ensure the total
        # error on the irregular steps accumulates and violates the regular
        # spacing we can encode in the fits header. I am less trustworthy of this,
        # if all deviations are negative they would accumulate. Individually
        # they may not fail but across the whole TIME dimension, as encoded by the
        # C-type header fields that define a regularly spaced interval, may.
        # even_spec = np.all(np.abs(np.diff(np.diff(sorted_specs))) < (0.15*u.s))
    else:
        even_spec = np.diff(sorted_specs).std() < (1e-4 * unit)

    logger.debug(f"{np.diff(sorted_specs).std()=}")
    if not even_spec:
        spequency = "Times" if time_domain_mode else "Frequencies"
        msg = f"{spequency} are not evenly spaced"
        logger.warning(msg)
        logger.debug(f"{np.max(np.diff(sorted_specs))=}")
        logger.debug(f"{np.min(np.diff(sorted_specs))=}")

    n_chan = len(specs)

    is_2d = len(old_data.shape) == 2
    idx = 0
    fits_idx = 3
    if not is_2d:
        logger.info("Input image is a cube. Looking for FREQ axis.")
        wcs = WCS(old_header)
        # Look for the frequency axis in wcs
        try:
            idx = wcs.axis_type_names[::-1].index(ctype)

            fits_idx = wcs.axis_type_names.index(ctype) + 1
            logger.info(f"{ctype} axis found at index {idx} (NAXIS{fits_idx})")

        except ValueError:
            msg = f"No {ctype} axis not found in WCS."
            logger.info(msg)
            fits_idx = len(old_data.shape) + 1

    new_header = old_header.copy()
    new_header[f"NAXIS{fits_idx}"] = n_chan
    new_header[f"CRPIX{fits_idx}"] = 1
    new_header[f"CRVAL{fits_idx}"] = specs[0].value
    new_header[f"CDELT{fits_idx}"] = np.median(np.diff(specs)).value
    new_header[f"CUNIT{fits_idx}"] = f"{unit:fits}"
    new_header[f"CTYPE{fits_idx}"] = ctype

    # Figure out the correct number of dimensions to use
    _no_of_naxis = [k for k in new_header if k.startswith("NAXIS") and k != "NAXIS"]
    new_header["NAXIS"] = len(_no_of_naxis)

    for k in ["CTYPE", "CUNIT", "CDELT"]:
        key = f"{k}{fits_idx}"
        logger.debug(f"{key}={new_header[key]}")

    if ("PV1_1" in new_header or "PC1_1" in new_header) and fits_idx != 1:
        logger.info("Inserting PV fields into header")
        pv1 = f"PC{fits_idx}_1"
        logger.info(f"Adding {pv1} to header")
        new_header[pv1] = 1.0
        pv2 = f"PC{fits_idx}_2"
        logger.info(f"Adding {pv2} to header")
        new_header[pv2] = 0.0

    if ignore_spec or not even_spec:
        logger.info(
            f"Ignore the specrency information, {ignore_spec=} or {not even_spec=}"
        )
        new_header[f"CDELT{fits_idx}"] = 1
        del new_header[f"CUNIT{fits_idx}"]
        new_header[f"CTYPE{fits_idx}"] = "CHAN"
        new_header[f"CRVAL{fits_idx}"] = 1

    if has_beams and not single_beam:
        tiny = np.finfo(np.float32).tiny
        new_header["CASAMBM"] = True
        new_header["COMMENT"] = "The PSF in each image plane varies."
        new_header["COMMENT"] = (
            "Full beam information is stored in the second FITS extension."
        )
        new_header["COMMENT"] = (
            f"The value '{tiny}' repsenents a NaN PSF in the beamtable."
        )
        del new_header["BMAJ"], new_header["BMIN"], new_header["BPA"]

    if time_domain_mode:
        new_header["COMMENT"] = "The frequency/chan axis in this cube represents TIME."

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
        header=output_header, spec_idx=idx, spec_fits_idx=fits_idx, is_2d=is_2d
    )


create_output_cube = sync_wrapper(create_output_cube_coro)


def utc_to_mjdsec(utc_time: str) -> float:
    """
    convert UTC time (in isot format as found in fits header)
    to MJD seconds to shunt into a freq
    """
    # logger.info(f"UTC time is {utc_time}")
    secperday = 86400.0
    return float((Time(utc_time, format="isot")).mjd) * secperday


async def read_spec_from_header_coro(
    image_path: Path,
    time_domain_mode: bool = False,
) -> u.Quantity:
    header = await asyncio.to_thread(fits.getheader, image_path)
    wcs = WCS(header)
    array_shape = wcs.array_shape
    unit = u.s if time_domain_mode else u.Hz
    quantity = "DATE-OBS" if time_domain_mode else "REFFREQ"
    spequency = "Time" if time_domain_mode else "Frequency"
    if array_shape is None:
        msg = "WCS does not have an array shape"
        raise ValueError(msg)
    is_2d = len(array_shape) == 2
    if is_2d:
        try:
            spec = await asyncio.to_thread(header.get, quantity)
            if time_domain_mode:
                spec = utc_to_mjdsec(spec)
            return spec * unit
        except KeyError as e:
            msg = f"{quantity} not in header. Cannot combine 2D images without {spequency} information."
            raise KeyError(msg) from e
    try:
        if "SPECSYS" not in header:
            header["SPECSYS"] = "TOPOCENT"
        wcs = WCS(header)
        if time_domain_mode:
            spec = await asyncio.to_thread(header.get, quantity)
            spec = utc_to_mjdsec(spec)
            return spec * unit

        return wcs.spectral.pixel_to_world(0).to(u.Hz)
    except Exception as e:
        # there should probably be better handling of other errors for time domain mode
        msg = "No FREQ axis found in WCS. Cannot combine N-D images without frequency information."
        raise ValueError(msg) from e


read_spec_from_header = sync_wrapper(read_spec_from_header_coro)


async def parse_specs_coro(
    file_list: list[Path],
    spec_file: Path | None = None,
    spec_list: list[float] | None = None,
    ignore_spec: bool = False,
    create_blanks: bool = False,
    time_domain_mode: bool = False,
) -> FileSpequencyInfo:
    """Parse the frequency/time information.

    Args:
        file_list (list[str]): List of FITS files
        spec_file (str | None, optional): File containing frequencies/times. Defaults to None.
        spec_list (list[float] | None, optional): List of frequencies/times. Defaults to None.
        ignore_spec (bool | None, optional): Ignore frequency/time information. Defaults to False.
        time_domain_mode (bool, optional): Whether these cubes dhould be formed over the time axis. Defaults to False.

    Raises:
        ValueError: If both spec_file and spec_list are specified
        KeyError: If 2D and (REFFREQ or DATE-OBS)  is not in header
        ValueError: If not 2D and FREQ is not in header

    Returns:
        FileSpequencyInfo: file_specs, specs, missing_chan_idx
    """
    unit = u.s if time_domain_mode else u.Hz
    spequencies = "times" if time_domain_mode else "frequencies"
    if ignore_spec:
        logger.info("Ignoring frequency information")
        return FileSpequencyInfo(
            file_specs=np.arange(len(file_list)) * unit,
            specs=np.arange(len(file_list)) * unit,
            missing_chan_idx=np.zeros(len(file_list)).astype(bool),
        )

    if spec_file is not None and spec_list is not None:
        msg = "Must specify either spec_file or spec_list, not both"
        raise ValueError(msg)

    if spec_file is not None:
        msg = f"Reading from {spec_file}"
        logger.info(msg)
        file_specs = np.loadtxt(spec_file) * unit
        assert len(file_specs) == len(file_list), (
            f"Number of {spequencies} in {spec_file} ({len({file_specs})}) does not match number of images ({len(file_list)})"
        )
        missing_chan_idx = np.zeros(len(file_list)).astype(bool)

    else:
        msg = f"Reading {spequencies} from list"
        logger.info(msg)
        first_header = fits.getheader(file_list[0])
        if "SPECSYS" not in first_header:
            logger.warning("SPECSYS not in header(s). Will set to TOPOCENT")
        # file_specs = np.arange(len(file_list)) * u.Hz
        missing_chan_idx = np.zeros(len(file_list)).astype(bool)
        coros = []
        for image_path in file_list:
            coro = read_spec_from_header_coro(
                image_path, time_domain_mode=time_domain_mode
            )
            coros.append(coro)

        list_of_specs = await gather_with_limit(
            None, *coros, desc=f"Extracting {spequencies}"
        )

        file_specs = np.array([f.to(unit).value for f in list_of_specs]) * unit

        specs = file_specs.copy()

    if create_blanks:
        msg = f"Trying to create a blank cube with evenly spaced {spequencies}"
        logger.info(msg)
        specs, missing_chan_idx = even_spacing(
            file_specs, time_domain_mode=time_domain_mode
        )

    return FileSpequencyInfo(
        file_specs=file_specs,
        specs=specs,
        missing_chan_idx=missing_chan_idx,
    )


parse_specs = sync_wrapper(parse_specs_coro)


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
        file=TQDM_OUT,
    ):
        header = fits.getheader(image)
        try:
            beam = Beam.from_fits_header(header)
        except NoBeamException:
            beam = Beam(major=np.nan * u.deg, minor=np.nan * u.deg, pa=np.nan * u.deg)
        logger.debug(f"{image.name=} {beam=}")
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
            assert naxis <= 1, (
                f"Only one polarisation axis is supported - found {naxis}"
            )
            return int(crpix - 1)
    return 0


def make_beam_table(beams: Beams, old_header: fits.Header) -> fits.BinTableHDU:
    """Make a beam table.

    Args:
        beams (Beams): Beams object
        header (fits.Header): Old header to infer polarisation

    Returns:
        fits.BinTableHDU: Beam table
    """
    nchan = len(beams.major)
    chans = np.arange(nchan)
    pol = get_polarisation(old_header)
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
    tab_hdu = fits.table_to_hdu(beam_table)
    tab_header = tab_hdu.header
    tab_header["EXTNAME"] = "BEAMS"
    tab_header["NCHAN"] = nchan
    tab_header["NPOL"] = 1  # Only one pol for now

    return tab_hdu


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
    # Use memmap=False to force the data to be read into memory - gives a speedup
    if is_missing:
        plane = await asyncio.to_thread(fits.getdata, file_list[0], memamp=False)
        plane *= np.nan
    else:
        plane = await asyncio.to_thread(
            fits.getdata, file_list[old_channel], memmap=False
        )

    await write_channel_to_cube_coro(
        file_handle=file_handle,
        plane=plane,
        chan=new_channel,
        header=new_header,
    )
    del plane


async def combine_fits_coro(
    file_list: list[Path],
    out_cube: Path,
    spec_file: Path | None = None,
    spec_list: list[float] | None = None,
    ignore_spec: bool = False,
    create_blanks: bool = False,
    overwrite: bool = False,
    max_workers: int | None = None,
    time_domain_mode: bool = False,
) -> u.Quantity:
    """Combine FITS files into a cube.
    Can handle either frequency or time dimensions agnostically

    Args:
        spec_file (Path | None, optional): Frequency/time file. Defaults to None.
        spec_list (list[float] | None, optional): List of frequencies/times. Defaults to None.
        ignore_spec (bool, optional): Ignore frequency/time information. Defaults to False.
        create_blanks (bool, optional): Attempt to create even frequency spacing. Defaults to False.
        time_domain_mode (bool, optional): Work in time domain mode - make a time-cube. Default = False.

    Returns:
        tuple[fits.HDUList, u.Quantity]: The combined FITS cube and frequencies
    """
    # TODO: Check that all files have the same WCS

    file_specs, specs, missing_chan_idx = await parse_specs_coro(
        spec_file=spec_file,
        spec_list=spec_list,
        ignore_spec=ignore_spec,
        file_list=file_list,
        create_blanks=create_blanks,
        time_domain_mode=time_domain_mode,
    )
    has_beams = "BMAJ" in fits.getheader(file_list[0])
    if has_beams:
        msg = f"Found beam in {file_list[0]} - assuming all files have beams"
        logger.info(msg)
        beams = parse_beams(file_list)
        for beam in beams:
            logger.info(f"{beams[0]==beam=}")

        # Be sure to match on beam shape, not on area as a
        # beam[0] == beam[1] would be checking.
        same_beam = (
            np.isclose(beams[0].major, beams.major)
            & np.isclose(beams[0].minor, beams.minor)
            & np.isclose(beams[0].pa, beams.pa)
        )
        single_beam = np.all(same_beam)

        if single_beam:
            logger.info("All beams are the same")
    else:
        beams = None
        single_beam = False

    # Sort the files by spequency
    old_sort_idx = np.argsort(file_specs)
    file_list = np.array(file_list)[old_sort_idx].tolist()
    new_sort_idx = np.argsort(specs)
    specs = specs[new_sort_idx]
    missing_chan_idx = missing_chan_idx[new_sort_idx]

    # Initialize the data cube
    new_header, _, _, _ = await create_output_cube_coro(
        old_name=file_list[0],
        out_cube=out_cube,
        specs=specs,
        ignore_spec=ignore_spec,
        has_beams=has_beams,
        single_beam=single_beam,
        overwrite=overwrite,
        time_domain_mode=time_domain_mode,
    )

    new_channels = np.arange(len(specs))
    old_channels = np.arange(len(file_specs))

    new_to_old = dict(zip(new_channels[np.logical_not(missing_chan_idx)], old_channels))

    coros = []
    with out_cube.open("rb+") as file_handle:
        for new_channel in new_channels:
            is_missing = missing_chan_idx[new_channel]
            msg = f"Channel {new_channel} missing == {is_missing}"
            logger.info(msg)
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
    if has_beams and not single_beam:
        old_header = fits.getheader(file_list[0])
        beam_table_hdu = make_beam_table(beams, old_header)
        msg = f"Appending beam table to {out_cube}"
        logger.info(msg)
        fits.append(
            out_cube,
            data=beam_table_hdu.data,
            header=beam_table_hdu.header,
        )

    return specs


combine_fits = sync_wrapper(combine_fits_coro)


def get_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Command-line interface."""

    parser = parser if parser else argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "file_list",
        nargs="+",
        help="List of FITS files to combine (in frequency or time order)",
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
    parser.add_argument(
        "--time-domain",
        action="store_true",
        help="Flag for constructing a time-domain cube",
        default=False,
    )
    # Add options for specifying frequencies
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--spec-file",
        help="File containing frequencies in Hz or times in MJD s (if --time-domain == True)",
        type=Path,
        default=None,
    )
    group.add_argument(
        "--specs",
        nargs="+",
        help="List of frequencies or times in Hz or MJD s respectively",
        type=float,
        default=None,
    )
    group.add_argument(
        "--ignore-spec",
        action="store_true",
        help="Ignore frequency or time information and just stack (probably not what you want)",
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

    return parser


def cli(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = get_parser()
        args = parser.parse_args()

    set_verbosity(
        verbosity=args.verbosity,
    )
    overwrite = bool(args.overwrite)
    out_cube = Path(args.out_cube)
    time_domain_mode = bool(args.time_domain)
    if not overwrite and out_cube.exists():
        msg = f"Output file {out_cube} already exists. Use --overwrite to overwrite."
        raise FileExistsError(msg)

    output_unit = u.s if time_domain_mode else u.Hz

    specs_file = out_cube.with_suffix(f".specs_{output_unit:fits}.txt")

    if specs_file.exists() and not overwrite:
        msg = f"Output file {specs_file} already exists. Use --overwrite to overwrite."
        raise FileExistsError(msg)

    if overwrite:
        logger.info("Overwriting output files")

    specs = combine_fits(
        file_list=args.file_list,
        out_cube=out_cube,
        spec_file=args.spec_file,
        spec_list=args.specs,
        ignore_spec=args.ignore_spec,
        create_blanks=args.create_blanks,
        overwrite=overwrite,
        max_workers=args.max_workers,
        time_domain_mode=time_domain_mode,
    )

    spequency = "times" if time_domain_mode else "frequencies"
    logger.info("Written cube to %s", out_cube)
    np.savetxt(specs_file, specs.to(output_unit).value)
    logger.info(f"Written {spequency} to %s", specs_file)


if __name__ == "__main__":
    cli()
