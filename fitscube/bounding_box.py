"""Some basic utilities to help with the creating of
bounding boxes to use in fitscube"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from fitscube.asyncio import gather_with_limit, sync_wrapper
from fitscube.logging import logger


@dataclass(frozen=True)
class BoundingBox:
    """Simple container to represent a bounding box.

    .. warning::
        ``x`` and ``y`` here are the *numpy* axes, which are the reverse of the
        FITS ``NAXIS`` convention:

        - ``x`` is axis ``-2`` -- image rows -- which is ``NAXIS2`` (declination)
        - ``y`` is axis ``-1`` -- image columns -- which is ``NAXIS1`` (right ascension)

        So a plane is sliced as ``data[..., xmin:xmax, ymin:ymax]`` and the
        trimmed header takes ``NAXIS1 = y_span``, ``NAXIS2 = x_span``.

    Minimum values are inclusive and maximum values are exclusive, so both can
    be used as is when slicing. Note that ``xmax``/``ymax`` were *inclusive*
    prior to ``v2.3.2``.
    """

    xmin: int
    """Minimum row pixel (numpy axis -2, FITS NAXIS2). Inclusive."""
    xmax: int
    """Maximum row pixel (numpy axis -2, FITS NAXIS2). Can be used as is in a slice (e.g. is exclusive)."""
    ymin: int
    """Minimum column pixel (numpy axis -1, FITS NAXIS1). Inclusive."""
    ymax: int
    """Maximum column pixel (numpy axis -1, FITS NAXIS1). Can be used as is in a slice (e.g. is exclusive)."""
    original_shape: tuple[int, int]
    """The original shape of the image. If constructed against a cube this is the shape of a single plane."""
    y_span: int
    """The span between ymax and ymin (i.e. the trimmed NAXIS1)"""
    x_span: int
    """The span between xmax and xmin (i.e. the trimmed NAXIS2)"""


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

    xmax += 1
    ymax += 1

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
) -> BoundingBox | None:
    """Create a bounding box for an image contained in a FITS file.

    The assumption is that the FITS file contains an image, not a cube.
    If the cube can bot be reshapped to an image without losing data
    the underlying bounding box creation will fail.

    Args:
        fits_path (Path): The fits image to call
        invalidate_zeros (bool, optional): Mark pixels that are exactly 0.0 as invalid (NaN them). Defaults to False.

    Returns:
        BoundingBox | None: The bounding box that describes the bounds of valid data. If all data are invalid (and not bounding box possible) None is returned.
    """
    data = await asyncio.to_thread(fits.getdata, fits_path, memmap=False)
    data = np.squeeze(data)

    if invalidate_zeros:
        data[data == 0.0] = np.nan

    return await asyncio.to_thread(create_bound_box_plane, image_data=data)


async def get_common_bounding_box_coro(
    file_list: list[Path],
    invalidate_zeros: bool = False,
    max_workers: int | None = None,
) -> BoundingBox:
    """Compute the single bounding box that encompasses the valid data of every
    image in ``file_list``.

    This is the box that ``combine_fits`` computes internally when
    ``bounding_box=True``. Compute it once with this function and pass the
    result to ``combine_fits(bounding_box=...)`` when several cubes (e.g. an
    image cube and its weights cube) must land on an identical pixel grid.

    Args:
        file_list (list[Path]): The FITS images to consider
        invalidate_zeros (bool, optional): Mark pixels that are exactly 0.0 as invalid (NaN them). Defaults to False.
        max_workers (int | None, optional): Maximum number of concurrent reads. Defaults to None.

    Returns:
        BoundingBox: The smallest bounding box that contains all valid data
    """
    boxes = await gather_with_limit(
        max_workers,
        *(
            get_bounding_box_for_fits_coro(
                fits_path=fits_path, invalidate_zeros=invalidate_zeros
            )
            for fits_path in file_list
        ),
        desc="Bounding boxes",
    )
    return extract_common_bounding_box(bounding_boxes=boxes)


get_common_bounding_box = sync_wrapper(get_common_bounding_box_coro)
