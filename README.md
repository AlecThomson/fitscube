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

```bash
pip install git+https://github.com/AlecThomson/fitscube.git
```

To install and use the (uvloop)[https://github.com/MagicStack/uvloop] Async
runner do:

```
pip install fitscube[uvloop]
```

## Usage

Command line:

```
❯ fitscube -h
usage: fitscube [-h] [-o] [--create-blanks] [--freq-file FREQ_FILE | --freqs FREQS [FREQS ...] | --ignore-freq] [-v] [--max-workers MAX_WORKERS] file_list [file_list ...] out_cube

Fitscube: Combine single-frequency FITS files into a cube. Assumes: - All files have the same WCS - All files have the same shape / pixel grid - Frequency is either a WCS axis or in the REFFREQ header keyword -
All the relevant information is in the first header of the first image

positional arguments:
  file_list             List of FITS files to combine (in frequency order)
  out_cube              Output FITS file

options:
  -h, --help            show this help message and exit
  -o, --overwrite       Overwrite output file if it exists
  --create-blanks       Try to create a blank cube with evenly spaced frequencies
  --spec-file SPEC_FILE
                        File containing frequencies in Hz or times in MJD s (if --time-domain == True)
  --specs SPECS [SPECS ...]
                        List of spequencies in Hz or MJD s
  --ignore-spec         Ignore frequency or time information and just stack (probably not what you want)
  -v, --verbosity       Increase output verbosity
  --max-workers MAX_WORKERS
                        Maximum number of workers to use for concurrent processing
  --time-domain         Flag for constructing a time-domain cube
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

## Convolving to a common resolution

See [RACS-Tools](https://github.com/AlecThomson/RACS-tools).

## License

MIT

## Contributing

Contributions are welcome. Please open an issue or pull request.

## TODO

- [ ] Add support for non-frequency axes
- [ ] Add convolution to a common resolution via RACS-Tools
