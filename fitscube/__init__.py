"""Combine FITS cubes and Stokes cubes."""

from __future__ import annotations

from ._version import version as __version__

__all__ = ["__version__"]


from .fitscube import combine_fits  # noqa: F401
from .stokescube import combine_stokes  # noqa: F401
