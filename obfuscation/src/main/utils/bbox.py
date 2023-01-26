# Ref: https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
# voc: [x_min, y_min, x_max, y_max]
# yolo: [x_center, y_center, width, height] normalized
# fo: [x_min, y_min, width, height] normalized
from itertools import permutations

import numpy as np


def voc_to_fo(box, w, h):
    x1, y1, x2, y2 = box
    return [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]


def fo_to_voc(box, w, h):
    x1, y1, w1, h1 = box
    return [x1 * w, y1 * h, (w1 + x1) * w, (h1 + y1) * h]


def fo_to_yolo(box):
    x1, y1, w1, h1 = box
    return [x1 + w1 / 2, y1 + h1 / 2, w1, h1]


def not_intersect(self, other):
    # self and other: x1, y1, x2, y2 based on https://stackoverflow.com/questions/40795709/checking-whether-two
    # -rectangles-overlap-in-python-using-two-bottom-left-corners return not (self.top_right.x < other.bottom_left.x
    # or self.bottom_left.x > other.top_right.x or self.top_right.y < other.bottom_left.y or self.bottom_left.y >
    # other.top_right.y)

    return (
        self[2] < other[0]
        or self[0] > other[2]
        or self[1] > other[3]
        or self[3] < other[1]
    )  # change sign as y increases downwards


def non_overlapping(ground_truth, data, rng, **kwargs):
    """Get two non_overlapping bboxes idxs or return None"""
    pairs = list(permutations(ground_truth["tp_idxs"], 2))
    rng.shuffle(pairs)  # in_place

    for perturb_idx, target_idx in pairs:
        if not_intersect(
            ground_truth["boxes"][perturb_idx], ground_truth["boxes"][target_idx]
        ):
            return perturb_idx, target_idx, data["gt_bboxes"][0][perturb_idx].tolist()

    return None


def arbitrary(ground_truth, data, rng, **perturb_kwargs):
    """Create arbitrary bbox or return None

    Inputs: arbitrary_bbox_length and boundary_distance in perturb_kwargs are original image units
    Returns: [start_x, start_y, end_x, end_y] are scaled image units
    """

    # original image units ------------------------------
    arbitrary_bbox_rad = perturb_kwargs["arbitrary_bbox_length"] / 2
    boundary_distance = perturb_kwargs["boundary_distance"]

    # resized image units ------------------------------
    target_idx = rng.choice(ground_truth["tp_idxs"])
    target_bbox = data["gt_bboxes"][0][target_idx]

    start_x, start_y, end_x, end_y = target_bbox.tolist()
    target_center_x, target_center_y = (end_x + start_x) / 2, (end_y + start_y) / 2
    target_rad_x, target_rad_y = (end_x - start_x) / 2, (end_y - start_y) / 2

    # img_metas: https://mmdetection.readthedocs.io/en/latest/tutorials/data_pipeline.html
    # scale_factor is scale_x, scale_y, scale_x, scale_y
    # mmdetection/mmdet/datasets/pipelines/transforms.py:240
    scale_x, scale_y, *_ = data["img_metas"][0]["scale_factor"]

    # img_shape is (h, w, c): mmdetection/mmdet/datasets/pipelines/formatting.py:289
    # Image is padded bottom and right (mmcv/image/geometric.py:495).
    # We don't use pad_shape since we would like arbitrary bbox to be within image excluding paddind
    img_y, img_x, *_ = data["img_metas"][0]["img_shape"]

    # scale is based on img_shape/ori_shape
    ori_y, ori_x, *_ = data["img_metas"][0]["ori_shape"]
    assert np.allclose(img_y, ori_y * scale_y)
    assert np.allclose(img_x, ori_x * scale_x)

    # try to construct an arbitrary bbox in every perturb_direction
    # and returns the 1st arbitrary bbox within image bounds,
    perturb_directions = ["left", "right", "top", "bottom"]
    rng.shuffle(perturb_directions)

    for perturb_dir in perturb_directions:
        if perturb_dir == "left" or perturb_dir == "right":
            # distance in width between arbitrary and target bbox centers in resized image units
            distance_x = (
                target_rad_x + (boundary_distance + arbitrary_bbox_rad) * scale_x
            )
            if perturb_dir == "left":
                center_x = target_center_x - distance_x
            else:
                center_x = target_center_x + distance_x

            start_x = center_x - arbitrary_bbox_rad * scale_x
            end_x = center_x + arbitrary_bbox_rad * scale_x

            # arbitrary bbox center is aligned to target center in height
            start_y = target_center_y - arbitrary_bbox_rad * scale_y
            end_y = target_center_y + arbitrary_bbox_rad * scale_y

        else:  # similar to "left" and "right" besides switching width and height
            distance_y = (
                target_rad_y + (boundary_distance + arbitrary_bbox_rad) * scale_y
            )
            if perturb_dir == "top":
                # y increases downwards
                center_y = target_center_y - distance_y
            else:
                center_y = target_center_y + distance_y

            start_y = center_y - arbitrary_bbox_rad * scale_y
            end_y = center_y + arbitrary_bbox_rad * scale_y

            start_x = target_center_x - arbitrary_bbox_rad * scale_x
            end_x = target_center_x + arbitrary_bbox_rad * scale_x

        within_img = (0 <= start_x <= end_x <= img_x) and (
            0 <= start_y <= end_y <= img_y
        )
        if within_img:
            return perturb_dir, target_idx, [start_x, start_y, end_x, end_y]

    return None
