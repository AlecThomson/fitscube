"""Compress finished cubes, and turn compressed cubes back into plain FITS.

Two families of compression are offered:

- Whole-file gzip ('gzip', 'pgzip'): the cube becomes `<name>.fits.gz`. Readers
  have to decompress the stream from its start, so reading any part of the data is
  slow, and astropy's `.data` decompresses the whole cube into memory.
- FITS tile compression ('tile-gzip', 'tile-rice'): the cube becomes
  `<name>.fits.fz`, the format written by CFITSIO's `fpack`. Each tile (by default
  64 full-width rows of one plane) is compressed on its own, so planes and boxes can
  be read cheaply with `hdul[1].section[...]`. 'tile-gzip' (GZIP_2) is lossless. 'tile-rice' (RICE_1)
  rounds floats to `quantize_level` steps per noise sigma, which is lossy.

Tile compression streams one plane at a time across threads, so memory stays at a
few planes however large the cube is.
"""

from __future__ import annotations

import argparse
import gzip
import io
import itertools
import math
import os
import shutil
import zlib
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import IO, Any, Literal, NamedTuple, cast

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from fitscube.logging import logger, set_verbosity

try:
    import pgzip
except ImportError:
    pgzip = None

COMPRESS_METHOD = Literal["gzip", "pgzip", "tile-gzip", "tile-rice"]
TILE_COMPRESSION_TYPE = Literal["GZIP_2", "RICE_1"]


class CompressedPlane(NamedTuple):
    """The compressed tiles of one plane, as rows of the output table."""

    heap: dict[str, list[bytes]]
    """Bytes of each variable-length column, one entry per tile"""
    scales: list[float]
    """ZSCALE of each tile, empty unless quantized"""
    zeros: list[float]
    """ZZERO of each tile, empty unless quantized"""
    has_blanks: bool
    """Whether quantized tiles marked blank (NaN) pixels with ZBLANK"""


def compress_cube(
    out_cube: Path,
    method: COMPRESS_METHOD = "gzip",
    max_workers: int | None = None,
    quantize_level: float = 4.0,
    tile_shape: tuple[int, int] | None = None,
    allow_requantize: bool = False,
) -> Path:
    """Compress a finished cube in place, streaming to avoid loading it into memory.

    Args:
        out_cube (Path): Path of the already-written, uncompressed cube. Deleted on success.
        method (COMPRESS_METHOD): 'gzip' and 'pgzip' gzip the whole file to `.fits.gz`;
            'pgzip' needs the 'pgzip' extra. 'tile-gzip' (lossless) and 'tile-rice'
            (lossy) tile-compress to `.fits.fz`.
        max_workers (int | None): Threads used by 'pgzip' and the tile methods; ignored for 'gzip'.
        quantize_level (float): Quantization steps per noise sigma for 'tile-rice'.
        tile_shape (tuple[int, int] | None): (y, x) tile size for the tile methods. Defaults to bands of 64 full-width rows.
        allow_requantize (bool): Let 'tile-rice' quantize a cube that was already quantized once.

    Returns:
        Path: Path of the compressed output.
    """
    if method in ("tile-gzip", "tile-rice"):
        return tile_compress_cube(
            out_cube,
            compression_type="RICE_1" if method == "tile-rice" else "GZIP_2",
            quantize_level=quantize_level if method == "tile-rice" else 0.0,
            tile_shape=tile_shape,
            max_workers=max_workers,
            allow_requantize=allow_requantize,
        )

    compressed_path = out_cube.with_suffix(out_cube.suffix + ".gz")
    msg = f"Compressing {out_cube} to {compressed_path} via {method}"
    logger.info(msg)

    opener = gzip_opener(method=method, max_workers=max_workers)
    with out_cube.open("rb") as src, opener(compressed_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    out_cube.unlink()
    return compressed_path


def gzip_opener(method: str, max_workers: int | None) -> Callable[..., IO[bytes]]:
    """The gzip `open` to use: the stdlib's, or pgzip's for 'pgzip'."""
    if method != "pgzip":
        return cast("Callable[..., IO[bytes]]", gzip.open)
    if pgzip is None:
        msg = "method='pgzip' requires the 'pgzip' extra: pip install fitscube[pgzip]"
        raise ImportError(msg)
    return cast("Callable[..., IO[bytes]]", partial(pgzip.open, thread=max_workers))


def is_compressed(cube: Path) -> bool:
    """Whether a cube is gzipped or tile-compressed."""
    if cube.suffix in (".gz", ".fz"):
        return True
    with fits.open(cube, disable_image_compression=True) as hdul:
        return any(hdu.header.get("ZIMAGE", False) for hdu in hdul)


def dither_seed_for_row(dither_seed: int, row: int) -> int:
    """The seed that dithers a lone tile the way row `row` of a larger table is dithered.

    CFITSIO dithers row `row` (0-based) with random sequence `(row + ZDITHER0 - 1) % 10000`.
    """
    return (dither_seed - 1 + row) % 10000 + 1


def gzip2_tiles(
    plane: NDArray[Any], tile_shape: tuple[int, int], gzip_level: int
) -> list[bytes]:
    """GZIP_2 bytes of each tile of a plane, in FITS tile order.

    GZIP_2 gzips the big-endian bytes of the tile, shuffled so that all the first
    bytes of each value come first, then all the second bytes, and so on.
    """
    big_endian = plane.astype(plane.dtype.newbyteorder(">"), copy=False)
    itemsize = big_endian.dtype.itemsize
    tiles = []
    for y in range(0, plane.shape[0], tile_shape[0]):
        for x in range(0, plane.shape[1], tile_shape[1]):
            tile = np.ascontiguousarray(
                big_endian[y : y + tile_shape[0], x : x + tile_shape[1]]
            )
            shuffled = tile.view(np.uint8).reshape(-1, itemsize).T.tobytes()
            tiles.append(gzip.compress(shuffled, compresslevel=gzip_level, mtime=0))
    return tiles


def astropy_compress_plane(
    plane: NDArray[Any],
    header: fits.Header,
    compression_type: TILE_COMPRESSION_TYPE,
    quantize_level: float,
    tile_shape: tuple[int, ...],
    dither_seed: int,
) -> tuple[CompressedPlane, fits.Header]:
    """Tile-compress a plane with astropy, returning its tiles and the table header."""
    hdu = fits.CompImageHDU(
        plane,
        header,
        compression_type=compression_type,
        tile_shape=tile_shape,
        quantize_level=quantize_level,
        quantize_method=2 if quantize_level else -1,
        dither_seed=dither_seed,
    )
    buffer = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(buffer)
    buffer.seek(0)
    with fits.open(buffer, disable_image_compression=True) as hdul:
        table = hdul[1]
        names = table.columns.names
        compressed = CompressedPlane(
            heap={
                col.name: [
                    _without_gzip_mtime(np.asarray(cell).tobytes())
                    for cell in table.data[col.name]
                ]
                for col in table.columns
                if "P" in col.format
            },
            scales=list(table.data["ZSCALE"]) if "ZSCALE" in names else [],
            zeros=list(table.data["ZZERO"]) if "ZZERO" in names else [],
            has_blanks="ZBLANK" in table.header,
        )
        return compressed, table.header.copy()


def _without_gzip_mtime(data: bytes) -> bytes:
    """Zero the timestamp astropy's gzip puts in a tile, so output is reproducible."""
    if data[:2] == b"\x1f\x8b":
        return data[:4] + bytes(4) + data[8:]
    return data


def compress_plane(
    plane: NDArray[Any],
    compression_type: TILE_COMPRESSION_TYPE,
    quantize_level: float,
    tile_shape: tuple[int, int],
    dither_seed: int,
    gzip_level: int,
) -> CompressedPlane:
    """The compressed tiles of one plane."""
    if compression_type == "GZIP_2" and not quantize_level:
        # Done here rather than by astropy, which always gzips at the slowest level
        return CompressedPlane(
            heap={"COMPRESSED_DATA": gzip2_tiles(plane, tile_shape, gzip_level)},
            scales=[],
            zeros=[],
            has_blanks=False,
        )
    compressed, _ = astropy_compress_plane(
        plane, fits.Header(), compression_type, quantize_level, tile_shape, dither_seed
    )
    return compressed


def table_header(
    template: fits.Header,
    cube_shape: tuple[int, ...],
    nrows: int,
    heap_lengths: dict[str, int],
    heap_size: int,
    dither_seed: int,
    has_blanks: bool,
) -> fits.Header:
    """Header of the whole cube's compressed table, built from the first plane's."""
    header = template.copy()
    header["NAXIS2"] = nrows
    header["PCOUNT"] = heap_size
    for axis, size in enumerate(reversed(cube_shape), start=1):
        header[f"ZNAXIS{axis}"] = size
    row_width = 0
    for index in range(1, header["TFIELDS"] + 1):
        name = header[f"TTYPE{index}"]
        if name in heap_lengths:
            # 64-bit descriptors, so the heap may grow past 2 GB
            header[f"TFORM{index}"] = f"1QB({heap_lengths[name]})"
            row_width += 16
        else:
            row_width += 8
    header["NAXIS1"] = row_width
    if "ZDITHER0" in header:
        header["ZDITHER0"] = dither_seed
    if has_blanks:
        header["ZBLANK"] = np.iinfo(np.int32).min
    return header


def table_dtype(header: fits.Header, heap_columns: list[str]) -> np.dtype[Any]:
    """Row layout of the compressed table: descriptor pairs, or ZSCALE/ZZERO doubles."""
    return np.dtype(
        [
            (name, ">i8", (2,)) if name in heap_columns else (name, ">f8")
            for name in (header[f"TTYPE{i}"] for i in range(1, header["TFIELDS"] + 1))
        ]
    )


def compressed_planes(
    image: fits.PrimaryHDU,
    compression_type: TILE_COMPRESSION_TYPE,
    quantize_level: float,
    tile_shape: tuple[int, int],
    tiles_per_plane: int,
    dither_seed: int,
    gzip_level: int,
    max_workers: int,
) -> Iterator[CompressedPlane]:
    """Compress every plane of `image` across threads, yielding them in order."""
    with ThreadPoolExecutor(max_workers) as pool:
        pending: deque[Future[CompressedPlane]] = deque()
        for plane_index, index in enumerate(np.ndindex(image.shape[:-2])):
            plane = image.section[index]
            pending.append(
                pool.submit(
                    compress_plane,
                    plane.astype(plane.dtype.newbyteorder("="), copy=False),
                    compression_type,
                    quantize_level,
                    tile_shape,
                    dither_seed_for_row(dither_seed, plane_index * tiles_per_plane),
                    gzip_level,
                )
            )
            # Bounded so memory stays at a few planes
            if len(pending) > 2 * max_workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def tile_compress_cube(
    cube: Path,
    compression_type: TILE_COMPRESSION_TYPE = "GZIP_2",
    quantize_level: float = 0.0,
    tile_shape: tuple[int, int] | None = None,
    max_workers: int | None = None,
    dither_seed: int | None = None,
    gzip_level: int = 6,
    allow_requantize: bool = False,
) -> Path:
    """Tile-compress a cube to `<name>.fits.fz`, as fpack does, then delete the original.

    Args:
        cube (Path): Uncompressed float cube, image in the primary HDU.
        compression_type (TILE_COMPRESSION_TYPE): 'GZIP_2' (lossless when `quantize_level` is 0) or 'RICE_1'.
        quantize_level (float): Quantization steps per noise sigma. 0 keeps floats exact (GZIP_2 only).
        tile_shape (tuple[int, int] | None): (y, x) tile size. Defaults to bands of 64 full-width rows.
        max_workers (int | None): Compression threads. Defaults to the number of CPUs.
        dither_seed (int | None): ZDITHER0 for quantized compression, 1 to 10000. Defaults to one derived from the data.
        gzip_level (int): zlib level for lossless GZIP_2, 1 (fastest, as fpack) to 9 (smallest).
        allow_requantize (bool): Allow quantizing a cube that `decompress_cube` marked as already quantized.

    Returns:
        Path: The compressed cube.
    """
    if compression_type == "RICE_1" and not quantize_level:
        msg = "RICE_1 cannot store floats losslessly; use GZIP_2, or a quantize_level above 0"
        raise ValueError(msg)
    if is_compressed(cube):
        msg = f"{cube} is already compressed"
        raise ValueError(msg)

    output = cube.with_name(cube.name + ".fz")
    partial_path = output.with_name(output.name + ".partial")
    msg = (
        f"Tile-compressing {cube} to {output} via {compression_type}, {quantize_level=}"
    )
    logger.info(msg)

    with fits.open(cube, memmap=False, do_not_scale_image_data=True) as hdul:
        image = hdul[0]
        if image.header["BITPIX"] > 0:
            msg = f"{cube} holds integers (BITPIX={image.header['BITPIX']}); only float cubes are supported"
            raise ValueError(msg)
        if quantize_level and image.header.get("FCLOSSY") and not allow_requantize:
            msg = (
                f"{cube} was already quantized once (FCLOSSY=T), and quantizing again "
                "adds a new error each time. Use lossless 'tile-gzip', or set "
                "allow_requantize to accept the extra error."
            )
            raise ValueError(msg)

        cube_shape = image.shape
        # Full-width bands: CFITSIO's funpack (4.3.1) mis-decodes quantized tiles
        # narrower than a row when there are blanks, and bands still keep box reads cheap
        tile = tile_shape or (min(64, cube_shape[-2]), cube_shape[-1])
        tiles_per_plane = math.ceil(cube_shape[-2] / tile[0]) * math.ceil(
            cube_shape[-1] / tile[1]
        )
        nplanes = math.prod(cube_shape[:-2])
        nrows = nplanes * tiles_per_plane
        if dither_seed is None:
            # Seeded from the data, so output is reproducible, yet recompressing
            # changed (e.g. already quantized) data dithers afresh. A fixed seed
            # would line up the errors of repeated quantization.
            middle = image.section[np.unravel_index(nplanes // 2, cube_shape[:-2])]
            dither_seed = zlib.crc32(np.ascontiguousarray(middle).tobytes()) % 10000 + 1
        planes = compressed_planes(
            image,
            compression_type=compression_type,
            quantize_level=quantize_level,
            tile_shape=tile,
            tiles_per_plane=tiles_per_plane,
            dither_seed=dither_seed,
            gzip_level=gzip_level,
            max_workers=max_workers or os.cpu_count() or 1,
        )
        # The table header only depends on the settings, so one tile is enough
        leading_ones = (1,) * (len(cube_shape) - 2)
        first_tile = image.section[(*((0,) * len(leading_ones)), slice(0, tile[0]))]
        _, template = astropy_compress_plane(
            first_tile.astype(first_tile.dtype.newbyteorder("="), copy=False).reshape(
                leading_ones + first_tile.shape
            ),
            image.header,
            compression_type,
            quantize_level,
            leading_ones + tile,
            dither_seed,
        )

        with partial_path.open("wb") as handle:
            handle.write(fits.PrimaryHDU().header.tostring().encode())
            header_offset = handle.tell()

            first = next(planes)
            heap_lengths = dict.fromkeys(first.heap, 0)
            table = np.zeros(nrows, dtype=table_dtype(template, list(heap_lengths)))
            # Placeholders: the header and table are rewritten once the heap is known,
            # and stay the same size because only fixed-width card values change
            handle.write(
                table_header(
                    template, cube_shape, nrows, heap_lengths, 0, dither_seed, False
                )
                .tostring()
                .encode()
            )
            handle.write(table.tobytes())

            heap_size = 0
            has_blanks = False
            row = 0
            for plane in itertools.chain([first], planes):
                has_blanks |= plane.has_blanks
                for tile_index in range(tiles_per_plane):
                    for name in heap_lengths:
                        data = plane.heap[name][tile_index]
                        table[name][row] = (len(data), heap_size)
                        heap_lengths[name] = max(heap_lengths[name], len(data))
                        handle.write(data)
                        heap_size += len(data)
                    if plane.scales:
                        table["ZSCALE"][row] = plane.scales[tile_index]
                        table["ZZERO"][row] = plane.zeros[tile_index]
                    row += 1

            handle.write(b"\0" * (-handle.tell() % 2880))
            handle.seek(header_offset)
            handle.write(
                table_header(
                    template,
                    cube_shape,
                    nrows,
                    heap_lengths,
                    heap_size,
                    dither_seed,
                    has_blanks,
                )
                .tostring()
                .encode()
            )
            handle.write(table.tobytes())

        for hdu in hdul[1:]:
            fits.append(partial_path, hdu.data, hdu.header)

    partial_path.replace(output)
    cube.unlink()
    return output


def decompress_cube(
    cube: Path,
    output: Path | None = None,
    overwrite: bool = False,
    remove: bool = False,
) -> Path:
    """Write a gzipped or tile-compressed cube back out as plain FITS, a plane at a time.

    A cube that was quantized (lossy) is marked FCLOSSY=T, with FCQLEVEL and a HISTORY
    card, so that later compression knows not to quantize it again.

    Args:
        cube (Path): A `.fits.gz` or tile-compressed (`.fits.fz`) cube.
        output (Path | None): Where to write. Defaults to `cube` without its `.gz`/`.fz` suffix.
        overwrite (bool): Replace `output` if it exists.
        remove (bool): Delete `cube` once the output is written.

    Returns:
        Path: The uncompressed cube.
    """
    if output is None:
        if cube.suffix not in (".gz", ".fz"):
            msg = f"Cannot derive an output name for {cube}; pass output"
            raise ValueError(msg)
        output = cube.with_suffix("")
    if output.exists() and not overwrite:
        msg = f"{output} already exists. Use overwrite to replace it."
        raise FileExistsError(msg)
    partial_path = output.with_name(output.name + ".partial")

    if cube.suffix == ".gz":
        msg = f"Decompressing {cube} to {output}"
        logger.info(msg)
        with gzip.open(cube, "rb") as src, partial_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        untile_cube(cube, partial_path)

    partial_path.replace(output)
    if remove:
        cube.unlink()
    return output


def untile_cube(cube: Path, output: Path) -> None:
    """Write the tile-compressed image in `cube` to `output` as a plain primary image."""
    with fits.open(cube, disable_image_compression=True) as raw:
        compressed_index = next(
            (i for i, hdu in enumerate(raw) if hdu.header.get("ZIMAGE", False)), None
        )
        if compressed_index is None:
            msg = f"{cube} holds no tile-compressed image"
            raise ValueError(msg)
        raw_header = raw[compressed_index].header
        # Quantized tiles always carry a scale, whatever the dithering
        quantized = any(
            raw_header[f"TTYPE{i}"] == "ZSCALE"
            for i in range(1, raw_header["TFIELDS"] + 1)
        )
        noisebit = next(
            (
                raw_header[f"ZVAL{i}"]
                for i in range(1, 10)
                if raw_header.get(f"ZNAME{i}") == "NOISEBIT"
            ),
            None,
        )

    with fits.open(cube, memmap=False, do_not_scale_image_data=True) as hdul:
        image = hdul[compressed_index]
        header = image.header.copy()
        if "XTENSION" in header:
            header.remove("XTENSION")
            header.insert(0, ("SIMPLE", True, "conforms to FITS standard"))
        for key in ("PCOUNT", "GCOUNT", "EXTNAME"):
            header.remove(key, ignore_missing=True)
        header.set("EXTEND", True, after=f"NAXIS{header['NAXIS']}")
        if quantized:
            header["FCLOSSY"] = (True, "Values were quantized by lossy compression")
            if noisebit is not None:
                header["FCQLEVEL"] = (noisebit, "Quantization steps per noise sigma")
            header.add_history(
                f"fitscube: decompressed from lossy {raw_header['ZCMPTYPE']}, "
                f"quantize level {noisebit}. Values are not the originals."
            )

        dtype = np.dtype(
            f">{'f' if header['BITPIX'] < 0 else 'i'}{abs(header['BITPIX']) // 8}"
        )
        if header["BITPIX"] == 8:
            dtype = np.dtype("u1")
        msg = f"Decompressing {cube} to {output} ({image.size * dtype.itemsize / 1e9:.1f} GB)"
        logger.info(msg)
        with output.open("wb") as handle:
            handle.write(header.tostring().encode())
            for index in np.ndindex(image.shape[:-2]):
                handle.write(np.asarray(image.section[index], dtype=dtype).tobytes())
            handle.write(b"\0" * (-handle.tell() % 2880))

        for index, hdu in enumerate(hdul):
            if index not in (0, compressed_index):
                fits.append(output, hdu.data, hdu.header)


def get_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Command-line interface for decompressing cubes."""
    parser = (
        parser
        if parser
        else argparse.ArgumentParser(
            description="Decompress a .fits.gz or .fits.fz cube"
        )
    )
    parser.add_argument(
        "cube", type=Path, help="A .fits.gz or tile-compressed .fits.fz cube"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to the input without .gz/.fz",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace the output if it exists"
    )
    parser.add_argument(
        "--remove", action="store_true", help="Delete the compressed input once done"
    )
    parser.add_argument(
        "-v", "--verbosity", default=0, action="count", help="Increase output verbosity"
    )
    return parser


def cli(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = get_parser().parse_args()
    set_verbosity(verbosity=args.verbosity)
    output = decompress_cube(
        args.cube, output=args.output, overwrite=args.overwrite, remove=args.remove
    )
    logger.info("Written %s", output)


if __name__ == "__main__":
    cli()
