import torch


def vanish(data, target_bbox_idx, *args):
    """
    data:
        gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
            shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

        gt_labels (list[Tensor]): class indices corresponding to each box
    """

    idx = torch.arange(len(data["gt_bboxes"][0])) != target_bbox_idx

    data["gt_bboxes"] = data["gt_bboxes"][:, idx]
    data["gt_labels"] = data["gt_labels"][:, idx]

    return data


def mislabel(data, target_bbox_idx, target_probas, *args):
    """
    data:
        gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
            shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

        gt_labels (list[Tensor]): class indices corresponding to each box

        target_probas(Tensor): predicted probas
    """

    target_probas[data["gt_labels"][0][target_bbox_idx]] = -float(
        "inf"
    )  # remove correct label

    target = torch.max(target_probas, dim=0)
    target_class, proba = target.indices, target.values

    data["gt_labels"][0][
        target_bbox_idx
    ] = target_class  # change target label to most likely
    return data, target_class, proba


def untarget(data, *args):
    """untargeted attack"""
    return data
