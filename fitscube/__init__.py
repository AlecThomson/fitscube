"""Combine FITS cubes and Stokes cubes."""

from __future__ import annotations

from fitscube.combine_fits import combine_fits

from ._version import version as __version__

__all__ = ["__version__", "combine_fits"]
