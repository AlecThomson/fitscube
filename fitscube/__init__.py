"""Combine FITS cubes."""

from __future__ import annotations

from fitscube.bounding_box import BoundingBox, get_common_bounding_box
from fitscube.combine_fits import combine_fits

from ._version import version as __version__

__all__ = ["BoundingBox", "__version__", "combine_fits", "get_common_bounding_box"]
