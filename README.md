# FITSCUBE

From the [wsclean](https://wsclean.readthedocs.io/) docs:

> WSClean does not output these images in a normal “imaging cube” like CASA
> does, i.e., a single fits file with several images in it. For now I’ve decided
> not to implement this (one of the reasons for this is that information about
> the synthesized beam is not properly stored in a multi-frequency fits file).
> One has of course the option to combine the output manually, e.g. with a
> simple Python script.

This is a simple Python script to combine (single-frequency or single-Stokes)
FITS images manually.

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

Or, onstall from this git repo (latest):

```bash
pip install git+https://github.com/AlecThomson/fitscube.git
```

## Usage

Command line:

```bash
fitscube -h
# usage: fitscube [-h] [-o] [--create-blanks] [--freq-file FREQ_FILE | --freqs FREQS [FREQS ...] | --ignore-freq] [-v] file_list [file_list ...] out_cube
#
# Fitscube: Combine single-frequency FITS files into a cube. Assumes: - All files have the same WCS - All files have the same shape / pixel grid - Frequency is either a WCS axis or  in the REFFREQ header
# keyword - All the relevant information is in the first header of the first image
#
# positional arguments:
#   file_list             List of FITS files to combine (in frequency order)
#   out_cube              Output FITS file
#
# options:
#   -h, --help            show this help message and exit
#   -o, --overwrite       Overwrite output file if it exists
#   --create-blanks       Try to create a blank cube with evenly spaced frequencies
#   --freq-file FREQ_FILE
#                         File containing frequencies in Hz
#   --freqs FREQS [FREQS ...]
#                         List of frequencies in Hz
#   --ignore-freq         Ignore frequency information and just stack (probably not what you want)
#   -v, --verbosity       Increase output verbosity

stokescube -h
# usage: stokescube [-h] [-V STOKES_V_FILE] [-o] [-v] stokes_I_file stokes_Q_file stokes_U_file output_file
#
# Fitscube: Combine single-Stokes FITS files into a Stokes cube. Assumes: - All files have the same WCS - All files have the same shape / pixel grid - All the relevant information # is in the first header of
# the first image
#
# positional arguments:
#   stokes_I_file         Stokes I file
#   stokes_Q_file         Stokes Q file
#   stokes_U_file         Stokes U file
#   output_file           Output file
#
# options:
#   -h, --help            show this help message and exit
#   -V STOKES_V_FILE, --stokes_V_file STOKES_V_FILE
#                         Stokes V file
#   -o, --overwrite       Overwrite output file if it exists
#   -v, --verbosity       Increase output verbosity
```

Python:

```python
from pathlib import Path

from fitscube import combine_fits, combine_stokes

file_list = list(Path().glob("*.fits"))

hdu_list, frequencies = combine_fits(
    file_list
)
hdu_stokes_list = combine_stokes(
    Path("stokes_I.fits"),
    Path("stokes_Q.fits"),
    Path("stokes_U.fits"),
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
