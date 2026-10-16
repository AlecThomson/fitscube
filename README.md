# FITSCUBE

[![Actions Status][actions-badge]][actions-link]
[![Codecov Status][codecov-badge]][codecov-link]

<!-- [![Documentation Status][rtd-badge]][rtd-link] -->

[![PyPI version][pypi-version]][pypi-link]

<!-- [![Conda-Forge][conda-badge]][conda-link] -->

[![PyPI platforms][pypi-platforms]][pypi-link]

<!-- [![GitHub Discussion][github-discussions-badge]][github-discussions-link] -->

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[codecov-link]:             https://codecov.io/gh/AlecThomson/fitscube
[codecov-badge]:            https://codecov.io/gh/AlecThomson/fitscube/graph/badge.svg?token=RNXELOOH1Z
[actions-badge]:            https://github.com/AlecThomson/fitscube/workflows/CI/badge.svg
[actions-link]:             https://github.com/AlecThomson/fitscube/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/fitscube
[conda-link]:               https://github.com/conda-forge/fitscube-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/AlecThomson/fitscube/discussions
[pypi-link]:                https://pypi.org/project/fitscube/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/fitscube
[pypi-version]:             https://img.shields.io/pypi/v/fitscube
[rtd-badge]:                https://readthedocs.org/projects/fitscube/badge/?version=latest
[rtd-link]:                 https://fitscube.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

From the [wsclean](https://wsclean.readthedocs.io/) docs:

> WSClean does not output these images in a normal “imaging cube” like CASA
> does, i.e., a single fits file with several images in it. For now I’ve decided
> not to implement this (one of the reasons for this is that information about
> the synthesized beam is not properly stored in a multi-frequency fits file).
> One has of course the option to combine the output manually, e.g. with a
> simple Python script.

This is a simple Python script to combine (single-frequency) FITS images
manually.

Current assumptions:

- All files have the same WCS
- All files have the same shape / pixel grid
- Frequency is either a WCS axis or in the REFFREQ header keyword
- All the relevant information is in the first header of the first image

## Installation

Install from PyPI (stable):

```
pip install fitscube
```

Or, install from this git repo (latest):

```
pip install git+https://github.com/AlecThomson/fitscube.git
```

To install and use the [uvloop](https://github.com/MagicStack/uvloop) Async
runner do:

```
pip install fitscube[uvloop]
```

## Usage

Command line:

```
❯ fitscube -h
usage: fitscube [-h] {combine,extract,decompress} ...

Tooling to create fitscubes

positional arguments:
  {combine,extract,decompress}
    combine             Combine FITS images together into a cube
    extract             Extract a plane from an existing cube
    decompress          Turn a .fits.gz or .fits.fz cube back into plain FITS

options:
  -h, --help            show this help message and exit
```

```
❯ fitscube combine -h
usage: fitscube combine [-h] [-o] [--create-blanks] [--time-domain]
                        [--spec-file SPEC_FILE | --specs SPECS [SPECS ...] | --ignore-spec] [-v]
                        [--max-workers MAX_WORKERS] [--bounding-box] [--invalidate-zeros]
                        [--no-blank-zero-beams] [--floating {8,16,32,64}] [--compress]
                        [--compress-method {gzip,pgzip,tile-gzip,tile-rice}]
                        [--quantize-level QUANTIZE_LEVEL]
                        file_list [file_list ...] out_cube

positional arguments:
  file_list             List of FITS files to combine (in frequency or time order)
  out_cube              Output FITS file

options:
  -h, --help            show this help message and exit
  -o, --overwrite       Overwrite output file if it exists
  --create-blanks       Try to create a blank cube with evenly spaced frequencies
  --time-domain         Flag for constructing a time-domain cube
  --spec-file SPEC_FILE
                        File containing frequencies in Hz or times in MJD s (if --time-domain ==
                        True)
  --specs SPECS [SPECS ...]
                        List of frequencies or times in Hz or MJD s respectively
  --ignore-spec         Ignore frequency or time information and just stack (probably not what you
                        want)
  -v, --verbosity       Increase output verbosity
  --max-workers MAX_WORKERS
                        Maximum number of workers to use for concurrent processing
  --bounding-box        Attempt to consider padded images when creating the cube. Requires an
                        extract read of the input data.
  --invalidate-zeros    Set pixels whose values are exactly zero to NaNs
  --no-blank-zero-beams
                        Keep images whose restoring beam is exactly zero. By default such images
                        are blanked with NaNs, as a zero beam means no PSF was fitted for that
                        plane (e.g. a wsclean model image planted by -fit-spectral-pol)
  --floating {8,16,32,64}
                        The number of floating point bits to use in the out cube. If None the
                        input data precision is used.
  --compress            Compress the output cube once written
  --compress-method {gzip,pgzip,tile-gzip,tile-rice}
                        How --compress compresses. 'gzip' and 'pgzip' gzip the whole file to
                        .fits.gz; 'pgzip' parallelizes across --max-workers threads but requires
                        the 'pgzip' extra. 'tile-gzip' (lossless) and 'tile-rice' (lossy, see
                        --quantize-level) tile-compress to .fits.fz, as fpack does, so single
                        planes can be read without decompressing the whole cube
  --quantize-level QUANTIZE_LEVEL
                        Quantization steps per noise sigma for 'tile-rice'. Higher keeps more
                        precision and compresses less
```

```
❯ fitscube extract -h
usage: fitscube extract [-h] [--channel-index CHANNEL_INDEX] [--hdu-index HDU_INDEX] [-v] [--overwrite] [--output-path OUTPUT_PATH] fits_cube

positional arguments:
  fits_cube             The cube to extract a plane from

options:
  -h, --help            show this help message and exit
  --channel-index CHANNEL_INDEX
                        The channel to extract
  --hdu-index HDU_INDEX
                        The HDU index of the data card containing the cube data
  -v, --verbosity       Increase output verbosity
  --overwrite           overwrite the output file, if it exists.
  --output-path OUTPUT_PATH
                        The name of the new output file. If not specified it is generated from the fits cube name
```

```
❯ fitscube decompress -h
usage: fitscube decompress [-h] [-o OUTPUT] [--overwrite] [--remove] [-v] cube

positional arguments:
  cube                 A .fits.gz or tile-compressed .fits.fz cube

options:
  -h, --help           show this help message and exit
  -o, --output OUTPUT  Output path. Defaults to the input without .gz/.fz
  --overwrite          Replace the output if it exists
  --remove             Delete the compressed input once done
  -v, --verbosity      Increase output verbosity
```

Python:

```python
from pathlib import Path

from fitscube import combine_fits

file_list = list(Path().glob("*.fits"))

frequencies = combine_fits(
    file_list
)
```

## Working with compressed cubes

`fitscube combine --compress` (or `fitscube.compress.compress_cube`) can write a
cube in one of two compressed forms. They behave very differently when you come to
read them.

| `--compress-method` | File | Lossless? | What it is |
|---|---|---|---|
| `gzip`, `pgzip` | `cube.fits.gz` | yes | The whole file gzipped. Any read decompresses the stream from its start. |
| `tile-gzip` | `cube.fits.fz` | yes | FITS tile compression (GZIP_2), the format CFITSIO's `fpack` writes. Bit-exact. |
| `tile-rice` | `cube.fits.fz` | **no** | FITS tile compression (RICE_1) after rounding each value to `--quantize-level` steps per noise sigma (default 4). |

Tile-compressed cubes are split into tiles of 64 full-width rows of one plane, and
each tile is compressed on its own, so reading a plane or a box only decompresses
the tiles it touches. NaNs (blanked pixels) are kept exactly by every method,
including `tile-rice`.

What they buy, on a real 155 GB ASKAP Stokes I cube that is mostly blanked:

| Method | Size | Ratio |
|---|---|---|
| uncompressed | 155.4 GB | 1× |
| `gzip -9` | 10.42 GB | 14.9× |
| tile GZIP_2 (lossless) | 10.23 GB | 15.2× |
| tile RICE, quantize level 4 | 4.62 GB | 33.6× (error 0.2–0.5% of the noise) |

Lossless tile compression is about the size of gzip on noise-dominated cubes, and
20–40% smaller on smooth ones (weights, backgrounds). Its gain is in reading, below.
RICE is where the big disk saving is, at the cost of being lossy. On smooth,
noiseless cubes such as weights RICE cannot quantize, and comes out *larger* than
lossless; always use `tile-gzip` for those.

### Which kind of file do I have?

```python
from astropy.io import fits

fits.info("cube.fits.fz")
# 0  PRIMARY           PrimaryHDU     ()           <- empty
# 1  COMPRESSED_IMAGE  CompImageHDU   (x, y, chan) <- the cube
# 2  BEAMS             BinTableHDU                <- per-channel beams, if any

# Lossy files carry ZQUANTIZ in the raw compressed header
raw = fits.getheader("cube.fits.fz", 1, disable_image_compression=True)
print(raw.get("ZQUANTIZ"))  # e.g. 'SUBTRACTIVE_DITHER_2' for tile-rice, None for tile-gzip
```

A `.fits.gz` file looks like a normal FITS file to astropy (image in HDU 0), but
see below for what reading it costs.

### Do I need to decompress?

| What you want to do | `.fits.gz` | `.fits.fz` |
|---|---|---|
| Read the header or WCS | no | no (use HDU 1) |
| Look at a few channels in astropy | no, but each read is slow | no |
| View in CARTA or DS9 | no, but CARTA unpacks the whole cube to RAM or temp disk first | no |
| Spectra or boxes through all channels (e.g. RM synthesis) | **yes** | small boxes are fine (see below); for heavy use, **yes** |
| Tools that memory-map the cube, or read the image from HDU 0 | **yes** | **yes** |
| CASA, casacore or ASKAPsoft linmos | **yes** (cannot open it) | **yes** (cannot open it) |

### How to decompress

You need free disk for the whole uncompressed cube (e.g. 155 GB for the cube
above). The compressed input is kept unless you ask for it to be removed.

```sh
fitscube decompress cube.fits.fz            # writes cube.fits; add --remove to delete the input
fitscube decompress cube.fits.gz            # works for gzip too
```

```python
from pathlib import Path
from fitscube.compress import decompress_cube

decompress_cube(Path("cube.fits.fz"))       # -> Path("cube.fits")
```

`fitscube decompress` streams a plane at a time, so it needs little memory, and it
puts the image back in the primary HDU with the beam table after it. Other routes:

- `.fits.gz`: `gunzip -k cube.fits.gz`, or `pigz -dk -p 8 cube.fits.gz` for speed.
- `.fits.fz`: `funpack cube.fits.fz` from CFITSIO (`conda install -c conda-forge
  cfitsio`, or `apt install libcfitsio-bin`). See the warning about `funpack` below.

Do **not** decompress with `fits.writeto("cube.fits", fits.getdata("cube.fits.fz"))`:
it loads the whole cube into memory.

### Reading without decompressing

```python
from astropy.io import fits

with fits.open("cube.fits.fz") as hdul:
    cube = hdul[1]                          # the image is in HDU 1, not 0
    header = cube.header                    # the image header (WCS etc.), as if uncompressed
    plane = cube.section[10]                # one channel: only its tiles are decompressed
    box = cube.section[:, 500:600, 500:600] # a box through all channels
    beams = hdul["BEAMS"].data
```

Use `.section`. `.data` (and `fits.getdata`) decompress the whole cube into memory,
and `memmap=True` has no effect on compressed data. For a `.fits.gz` file `.section`
stays small in memory but has to decompress the stream up to the plane you ask for,
every time.

Measured with astropy 8.0 on a 1.66 GB, 288-channel, 1200×1200 cube (astropy 6.0
behaves the same):

| Read | `.fits.gz` | `.fits.fz` (`tile-gzip`) |
|---|---|---|
| `section[30]`, one channel | 10 s, 68 MB | 0.06 s, 70 MB |
| `section[:, 500:600, 500:600]`, a 100×100 box through all channels | did not finish in 5 minutes | 2.7 s, 442 MB |
| `data[30]` (reads everything) | 26 s, **3.3 GB** | 19 s, **2.3 GB** |

A box read decompresses every tile it touches in every channel, so its memory grows
with the number of channels; very large spectral reads are better done on a
decompressed copy.

### Sharp edges

- **`.data`, `fits.getdata` and memory-mapping read the whole cube**, for both
  formats. For `.fits.gz` the peak is about twice the uncompressed size.
- **The image is in HDU 1** of a `.fits.fz` file. `hdul[0]`, `fits.getheader(path)`
  and `fitscube extract` with the default `--hdu-index 0` see the empty primary. Use
  `hdul[1]`, `fits.getheader(path, 1)` or `fitscube extract --hdu-index 1`.
  (`fits.getdata(path)` finds the image by itself, but reads all of it.)
- **Do not edit headers in place** (`fits.open(path, mode="update")`). It works, but
  astropy decompresses and recompresses the whole file to do it: on a 415 MB cube
  that took 54 s and 0.9 GB of memory for `.fz`, and 22 s and 0.9 GB for `.gz`. Fix
  headers before compressing, or decompress first.
- **`funpack` can silently corrupt RICE files with narrow tiles.** CFITSIO 4.3.1's
  `funpack` (fpack 1.7.0) gets the values and NaNs wrong when a quantized tile is
  narrower than a full row and the plane has blanks. CFITSIO's normal read path, as
  used by CARTA and DS9, and astropy are fine. fitscube's own tiles are always full
  width, so its files are safe; use `fitscube decompress` for `.fz` files from other
  tools.
- **CASA, casacore and linmos cannot open either format**; decompress first.
- **CARTA (3.0 and later) opens both**, but for `.fits.gz` it decompresses the whole
  cube into memory, or to the system temp directory when memory is short. That is
  the full uncompressed size of RAM or disk on the CARTA server.
- **Names and globs.** `*.fits` does not match `cube.fits.gz` or `cube.fits.fz`, and
  `Path("cube.fits.fz").suffix` is `.fz`. Use `*.fits*`, or check
  `path.name.endswith((".fits", ".fits.gz", ".fits.fz"))`.
- **Disk.** Compressing deletes the uncompressed cube once the compressed one is
  written. Decompressing needs room for the whole uncompressed cube.

### Lossy compression, and compressing more than once

`tile-rice` rounds every value to a grid with `--quantize-level` steps per noise
sigma, with a random offset ("dithering") so the rounding error is unbiased and
uncorrelated with the data. The noise is measured per tile, from differences
between neighbouring pixels. At the default level of 4 the added error is at most
about 7% of the noise (a step of sigma/4, divided by √12), which adds at most about
0.3% to the total noise in quadrature. Radio images have noise correlated over the
beam, which makes the neighbouring-pixel estimate low and the steps finer: on real
ASKAP and simulated beam-convolved cubes the error was 0.2–1% of the noise. Higher
levels keep more precision and compress less.

The rules:

- **Quantize once, as the final, archival step.** Keep a lossless copy if anything
  might be reprocessed.
- **Each decompress-and-recompress adds a fresh error.** Recompressing estimates
  the noise again from data that was already rounded, and rounds it again on a
  slightly different grid, so it is not a no-op: the error grows like the square root
  of the number of cycles. Measured on a 415 MB beam-convolved noise cube at quantize
  level 4 (the file size stays the same):

  | Times compressed with `tile-rice` | 1 | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|---|
  | Error, % of the noise | 1.20 | 1.70 | 2.09 | 2.42 | 2.71 |
  | Error relative to one pass | 1 | 1.42 | 1.74 | 2.01 | 2.25 |

- **Lossless after lossy is exact but wastes the saving.** `tile-gzip` of a
  decompressed RICE cube stores the rounded values exactly, but is about the size of
  ordinary lossless compression (147 MB against 43 MB for RICE on a 415 MB cube), not of the RICE file. Keep the
  `.fz` file rather than decompressing and recompressing it.
- **fitscube guards against quantizing twice.** `fitscube decompress` marks a cube it
  decompressed from a lossy file with `FCLOSSY = T`, `FCQLEVEL` and a `HISTORY` card,
  and `compress_cube(..., method="tile-rice")` refuses such a cube unless given
  `allow_requantize=True`. `funpack` and other tools do not leave this mark, so once
  a cube has been through them nothing in it records that it is lossy.
- **Derived products inherit the error.** Products made from a lossy cube (RM
  synthesis, moment maps, noise maps) carry the quantization error, and compressing
  them lossily again adds more. Quote noise levels from the original cube.
- **Not for checksums or regression tests.** A lossy file never reproduces the
  original bits, and two runs with a different dither seed or tile shape differ from
  each other.

## Zero-sized beams

wsclean writes `BMAJ = BMIN = 0` into an image when no PSF was fitted for that
plane. This happens with `-fit-spectral-pol`, where the channels that were not
imaged directly are populated with the model image instead. Those planes look
like data, but they are not comparable to the rest of the cube and will produce
nasty effects downstream.

By default `fitscube` blanks any such plane with NaNs and logs a warning. Pass
`--no-blank-zero-beams` (or `blank_zero_beams=False` in Python) to keep the
image data as it is.

Either way, a zero beam is never written into the beam table as a literal
zero - it gets the same `np.finfo(np.float32).tiny` sentinel that already marks
a NaN PSF, which is what keeps CASA happy. This is applied per beam, so a zero
`BPA` on an otherwise valid beam is left alone.

## Convolving to a common resolution

See [RACS-Tools](https://github.com/AlecThomson/RACS-tools).

## License

MIT

## Contributing

Contributions are welcome. Please open an issue or pull request.
