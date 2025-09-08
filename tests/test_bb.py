# Tests to ensure the bounding box functionality is working
from __future__ import annotations

import numpy as np
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
    assert bb.xmax == 99
    assert bb.ymax == 99
    assert bb.x_span == 99
    assert bb.y_span == 99

    image[:, :4] = np.nan
    image[96:, :] = np.nan

    bb = create_bound_box_plane(image_data=image)
    assert isinstance(bb, BoundingBox)
    assert bb.xmin == 0
    assert bb.ymin == 4
    assert bb.xmax == 95
    assert bb.ymax == 99
    assert bb.x_span == 95
    assert bb.y_span == 95


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
    assert bb.xmax == 99
    assert bb.ymax == 97
