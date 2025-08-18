"""Some basic utilities to help with the creating of
bounding boxes to use in fitscube"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from fitscube.logging import logger


@dataclass(frozen=True)
class BoundingBox:
    """Simple container to represent a bounding box"""

    xmin: int
    """Minimum x pixel"""
    xmax: int
    """Maximum x pixel"""
    ymin: int
    """Minimum y pixel"""
    ymax: int
    """Maximum y pixel"""
    original_shape: tuple[int, int]
    """The original shape of the image. If constructed against a cube this is the shape of a single plane."""
    y_span: int
    """The span between ymax and ymin"""
    x_span: int
    """The span between xmax and xmin"""


def create_bound_box_plane(image_data: np.ndarray) -> BoundingBox | None:
    """Create a bounding box around pixels in a 2D image. If all
    pixels are not valid, then ``None`` is returned.

    Args:
        image_data (np.ndarray): The 2D image to construct a bounding box around

    Returns:
        Optional[BoundingBox]: None if no valid pixels, a bounding box with the (xmin,xmax,ymin,ymax) of valid pixels
    """
    assert len(image_data.shape) == 2, (
        f"Only two-dimensional arrays supported, received {image_data.shape}"
    )

    # First convert to a boolean array
    image_valid = np.isfinite(image_data)

    if not any(image_valid.reshape(-1)):
        logger.info("No pixels to creating bounding box for")
        return None

    # Then make them 1D arrays
    x_valid = np.any(image_valid, axis=1)
    y_valid = np.any(image_valid, axis=0)

    # Now get the first and last index
    xmin, xmax = np.where(x_valid)[0][[0, -1]]
    ymin, ymax = np.where(y_valid)[0][[0, -1]]

    y_span = ymax - ymin
    x_span = xmax - xmin

    return BoundingBox(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        y_span=y_span,
        x_span=x_span,
        original_shape=image_data.shape[-2:],
    )


def extract_common_bounding_box(
    bounding_boxes: list[BoundingBox | None],
) -> BoundingBox:
    """Get the smallest bounding box that encompasses all bounding boxes

    Args:
        bounding_boxes (list[BoundingBox | None]): A list of bounding boxes. If None (returned for invalid images) skip it.

    Raises:
        ValueError: If all input bounding boxes are invalid
        ValueError: If there is an `original_shape` mismatch

    Returns:
        BoundingBox: The smallest bounding box
    """

    # Step 1: filter out all Nones
    valid_boxes: list[BoundingBox] = [bb for bb in bounding_boxes if bb is not None]

    if len(valid_boxes) == 0:
        msg = "No valid input boxes to consider"
        raise ValueError(msg)

    if not all(
        valid_boxes[0].original_shape == bb.original_shape for bb in valid_boxes
    ):
        msg = "Different shapes, and not sure this is really supported or meaningful"
        raise ValueError(msg)

    xmin = int(np.min([bb.xmin for bb in valid_boxes]))
    xmax = int(np.max([bb.xmax for bb in valid_boxes]))
    ymin = int(np.min([bb.ymin for bb in valid_boxes]))
    ymax = int(np.max([bb.ymax for bb in valid_boxes]))

    y_span = ymax - ymin
    x_span = xmax - xmin

    return BoundingBox(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        y_span=y_span,
        x_span=x_span,
        original_shape=valid_boxes[0].original_shape,
    )


async def get_bounding_box_for_fits_coro(
    fits_path: Path, invalidate_zeros: bool = False
) -> BoundingBox:
    """Create a bounding box for an image contained in a FITS file.

    The assumption is that the FITS file contains an image, not a cube.
    If the cube can bot be reshapped to an image without losing data
    the underlying bounding box creation will fail.

    Args:
        fits_path (Path): The fits image to call
        invalidate_zeros (bool, optional): Mark pixels that are exactly 0.0 as invalid (NaN them). Defaults to False.

    Returns:
        BoundingBox: The bounding box that describes the bounds of valid data
    """
    data = await asyncio.to_thread(fits.getdata, fits_path, memmap=False)
    data = np.squeeze(data)

    if invalidate_zeros:
        data[data == 0.0] = np.nan

    return await asyncio.to_thread(create_bound_box_plane, image_data=data)
