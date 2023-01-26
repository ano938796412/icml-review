import warnings

import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import TwoStageDetector, SingleStageDetector
from mmdet.models import build_detector
from torch import nn


def init_train_detector(config, checkpoint=None, device="cuda:0", cfg_options=None):
    """Initialize a detector from config file with train cfg.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError(
            "config must be a filename or Config object, " f"but got {type(config)}"
        )
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.pretrained = None

    model = build_detector(
        config.model, test_cfg=config.get("test_cfg"), train_cfg=config.get("train_cfg")
    )
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            warnings.simplefilter("once")
            warnings.warn(
                "Class names are not saved in the checkpoint's "
                "meta data, use COCO classes by default."
            )
            model.CLASSES = get_classes("coco")
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def get_train_data(img, gt_bboxes, gt_labels, config, device):
    # data is preprocessed according to mmdetection.mmdet.datasets.pipelines.formatting.Collect
    config = config.copy()

    # set loading pipeline type
    config.data.train.pipeline[0].type = "LoadImageFromWebcam"
    # manually pass in annotations
    if config.data.train.pipeline[1].type == "LoadAnnotations":
        config.data.train.pipeline.pop(1)
    config.data.train.pipeline = replace_ImageToTensor(config.data.train.pipeline)

    data = dict(
        img=img, gt_labels=gt_labels, gt_bboxes=gt_bboxes, bbox_fields=["gt_bboxes"]
    )
    train_pipeline = Compose(config.data.train.pipeline)
    # build the data pipeline
    data = train_pipeline(data)

    return {
        k: v.data.unsqueeze(0).to(device)
        if isinstance(v.data, torch.Tensor)
        else [v.data]
        for k, v in data.items()
    }


def get_boxes_labels_scores(bbox_result):
    labels = np.concatenate(
        [np.full(bbox.shape[0], i, dtype=np.int_) for i, bbox in enumerate(bbox_result)]
    )
    bboxes_and_confs = np.vstack(bbox_result)  # xyxy + c
    boxes = bboxes_and_confs[:, :4]
    scores = bboxes_and_confs[:, -1]
    return boxes, labels, scores


def get_probas(model):
    # get class probability stored in model.cfg
    if isinstance(model, TwoStageDetector):
        results = model.test_cfg.rcnn.nms.results
    elif isinstance(model, SingleStageDetector):
        results = model.test_cfg.nms.results
    else:
        raise RuntimeError

    if results:
        # mimic mmdet.core.bbox.transforms.bbox2result in ordering results
        bbox_inds = np.argsort(results["labels"], kind="stable")
        return results["logits"][bbox_inds]
    else:
        return results  # [] if bboxes.numel() == 0 in mmdetection/mmdet/core/post_processing/bbox_nms.py:79


def get_norm_params(model_config):
    img_norm_cfg = model_config["img_norm_cfg"]

    # mimic mmdetection.mmdet.datasets.pipelines.transforms.Normalize
    norm_mean = np.array(img_norm_cfg["mean"], dtype=np.float32) / 255
    norm_std = np.array(img_norm_cfg["std"], dtype=np.float32) / 255

    return norm_mean, norm_std


def get_bounds(model_config, device):
    norm_mean, norm_std = get_norm_params(model_config)

    lower_bound = (np.zeros_like(norm_mean) - norm_mean) / norm_std
    upper_bound = (np.ones_like(norm_mean) - norm_mean) / norm_std

    return torch.tensor(lower_bound).reshape(3, 1, 1).to(device), torch.tensor(
        upper_bound
    ).reshape(3, 1, 1).to(device)


def unnormalize(img_, model_config):
    """Unnormalize ImageNet image according to Joel Simon at
    https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3"""
    mean, std = get_norm_params(model_config)

    img = img_.detach().clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img
