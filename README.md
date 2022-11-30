# FITSCUBE

From the [wsclean](https://wsclean.readthedocs.io/) docs:
> WSClean does not output these images in a normal “imaging cube” like CASA does, i.e., a single fits file with several images in it. For now I’ve decided not to implement this (one of the reasons for this is that information about the synthesized beam is not properly stored in a multi-frequency fits file). One has of course the option to combine the output manually, e.g. with a simple Python script.

This is a simple Python script to combine (single-frequency) FITS images manually.

Current assumptions:
- Images are passed in frequency order
- All files have the same WCS
- All files have the same shape / pixel grid
- Frequency is either a WCS axis or in the REFFREQ header keyword
- All the relevant information is in the first header of the first image

## Installation

Install from git:
```bash
pip install git+https://github.com/AlecThomson/fitscube.git
```

## Usage

Command line:
```bash
fitscube -h
# usage: fitscube [-h] [--overwrite] file_list [file_list ...] out_cube

# Fitscube: Combine FITS files into a cube. Assumes: - All files have the same WCS - All files have
# the same shape / pixel grid - Frequency is either a WCS axis or in the REFFREQ header keyword

# positional arguments:
#   file_list    List of FITS files to combine
#   out_cube     Output FITS file

# optional arguments:
#   -h, --help   show this help message and exit
#   --overwrite  Overwrite output file if it exists
```

Python:
```python
from fitscube import combine_fits

combine_fits(
    ['file1.fits', 'file2.fits', 'file3.fits'],
    'out.fits',
    overwrite=True
)
```

## Convolving to a common resolution
See [RACS-Tools](https://github.com/AlecThomson/RACS-tools).

## License
MIT

## Contributing
Contributions are welcome. Please open an issue or pull request.

## TODO
- [ ] Allow for images in any order
- [ ] Add support for non-frequency axes
- [ ] Add tracking of the PSF in header / beamtable
- [ ] Add convolution to a common resolution via RACS-Tools