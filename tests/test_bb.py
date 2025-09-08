# Tests to ensure the bounding box functionality is working
from __future__ import annotations

import numpy as np
import pytest
from fitscube.bounding_box import (
    BoundingBox,
    create_bound_box_plane,
    extract_common_bounding_box,
)


def test_bound_box_plane() -> None:
    """See if we can make the bounding box correctly"""

    image = np.zeros((100, 100))

    bb = create_bound_box_plane(image_data=image)
    assert isinstance(bb, BoundingBox)
    assert bb.xmin == 0
    assert bb.ymin == 0
    assert bb.xmax == 100
    assert bb.ymax == 100
    assert bb.x_span == 100
    assert bb.y_span == 100

    image[:, :4] = np.nan
    image[96:, :] = np.nan

    bb = create_bound_box_plane(image_data=image)
    assert isinstance(bb, BoundingBox)
    assert bb.xmin == 0
    assert bb.ymin == 4
    assert bb.xmax == 96
    assert bb.ymax == 100
    assert bb.x_span == 96
    assert bb.y_span == 96


def test_extract_common_bounding_box() -> None:
    """Construct a set of bounding boxes to ensure a largest fits all
    one can be created"""

    bbs = []
    for i in range(2, 8, 1):
        image = np.zeros((100, 100))

        print(i)

        image[:i, :] = np.nan
        image[:, -i:] = np.nan
        assert np.sum(np.isnan(image)) > 1
        bbs.append(create_bound_box_plane(image_data=image))

    assert len(bbs) == 6

    bb = extract_common_bounding_box(bounding_boxes=bbs)
    assert bb.xmin == 2
    assert bb.ymin == 0
    assert bb.xmax == 100
    assert bb.ymax == 98


def test_extract_common_bounding_box_error() -> None:
    """See if the correct errors are raised"""

    with pytest.raises(ValueError, match="No valid"):
        extract_common_bounding_box(bounding_boxes=[None, None, None])

    bbs = []
    for i in range(2, 8, 1):
        image = np.zeros((100 + i, 100))
        bbs.append(create_bound_box_plane(image_data=image))

    with pytest.raises(ValueError, match="Different shapes"):
        extract_common_bounding_box(bounding_boxes=bbs)


# @pytest.mark.asyncio
# async def test_get_bounding_box_from_fits(time_image_paths) -> None:
#     futures = [
#         await get_bounding_box_for_fits_coro(fits_path=fits_path)
#         for fits_path in time_image_paths
#     ]
#     assert all(isinstance(bb, BoundingBox) for bb in futures)

#     common_bb = extract_common_bounding_box(bounding_boxes=futures)
#     assert common_bb.xmin == 0
#     assert common_bb.ymin == 0
#     assert common_bb.xmax == 100
#     assert common_bb.ymax == 100
