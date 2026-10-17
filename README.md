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
usage: fitscube [-h] {combine,extract,compress,decompress} ...

Tooling to create fitscubes

positional arguments:
  {combine,extract,compress,decompress}
    combine             Combine FITS images together into a cube
    extract             Extract a plane from an existing cube
    compress            Compress a cube to .fits.gz or .fits.fz
    decompress          Turn a .fits.gz or .fits.fz cube back into plain FITS

options:
  -h, --help            show this help message and exit
```

```
❯ fitscube combine -h
usage: fitscube combine [-h] [-o] [--create-blanks] [--time-domain]
                        [--spec-file SPEC_FILE | --specs SPECS [SPECS ...] | --ignore-spec] [-v]
                        [--max-workers MAX_WORKERS] [--bounding-box] [--invalidate-zeros]
                        [--no-blank-zero-beams] [--floating {32,64}] [--compress]
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
  --floating {32,64}
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
❯ fitscube compress -h
usage: fitscube compress [-h] [--method {gzip,pgzip,tile-gzip,tile-rice}]
                         [--quantize-level QUANTIZE_LEVEL] [--max-workers MAX_WORKERS]
                         [--allow-requantize] [--remove] [-v]
                         cube

positional arguments:
  cube                  An uncompressed FITS cube

options:
  -h, --help            show this help message and exit
  --method {gzip,pgzip,tile-gzip,tile-rice}
                        'gzip'/'pgzip' gzip the whole file to .fits.gz. 'tile-gzip' (lossless) and
                        'tile-rice' (lossy) tile-compress to .fits.fz, as fpack does
  --quantize-level QUANTIZE_LEVEL
                        Quantization steps per noise sigma for 'tile-rice'
  --max-workers MAX_WORKERS
                        Compression threads
  --allow-requantize    Let 'tile-rice' quantize a cube that was already quantized once
  --remove              Delete the input once done
  -v, --verbosity       Increase output verbosity
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

## Compressed cubes

`fitscube compress`, or `combine --compress`, writes the cube in one of three ways:

| Method | Output | |
|---|---|---|
| `gzip`, `pgzip` | `.fits.gz` | The whole file gzipped. Lossless. |
| `tile-gzip` | `.fits.fz` | [Tile compressed](https://heasarc.gsfc.nasa.gov/fitsio/fpack/) with GZIP_2. Lossless. |
| `tile-rice` | `.fits.fz` | Tile compressed with RICE_1, after quantising to `--quantize-level` steps per noise sigma. Lossy. |

`.fits.fz` files are what `fpack` writes, so DS9, CARTA, `funpack` and astropy can all
read them. The image is in HDU 1, and `.section` only decompresses the tiles it needs:

```python
from astropy.io import fits

with fits.open("cube.fits.fz") as hdul:
    plane = hdul[1].section[10]
```

`.data` and `fits.getdata` decompress the whole cube into memory, for both `.fits.gz`
and `.fits.fz`. A `.fits.gz` cube has no cheap way to read part of it. CARTA (6.0) also
reads the whole cube into memory when it opens either kind, so decompress large cubes
before viewing them there.

To compress an existing cube, and to get plain FITS back:

```
fitscube compress cube.fits --method tile-rice
fitscube decompress cube.fits.fz
```

`funpack` works too, though `funpack` from CFITSIO 4.3.1 mis-decodes RICE files whose
tiles are narrower than a row. fitscube always writes full-width tiles.

Quantise once. Every decompress and recompress with `tile-rice` adds a new error, so
`fitscube decompress` marks lossy cubes with `FCLOSSY`, and `tile-rice` refuses them
unless given `allow_requantize=True`. Smooth images such as weights do not shrink under
RICE; use `tile-gzip` for them.

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
