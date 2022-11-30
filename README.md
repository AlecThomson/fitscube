# FITSCUBE

Combine FITS images into a cube.

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