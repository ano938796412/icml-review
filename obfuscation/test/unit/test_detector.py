import numpy as np
import pytest
import torch
from mmcv import Config, imread
from mmdet.apis import inference_detector, show_result_pyplot, init_detector
from mmdet.models import TwoStageDetector, SingleStageDetector

from main.utils.detector import get_boxes_labels_scores, get_probas

device = torch.device("cpu")

cfg_paths = [
    "./configs/models/ssd_512.py",
    "./configs/models/cascade_rcnn.py",
    "./configs/models/retinanet.py",
    "./configs/models/yolo_v3.py",
    "./configs/models/faster_rcnn.py",
]

cfgs = [Config.fromfile(path) for path in cfg_paths]


@pytest.mark.parametrize("cfg", cfgs)
def test_nms_logits(cfg):
    """check nms class probas equate to mmdet.core.bbox.transforms.bbox2result in ordering results"""
    model = init_detector(cfg.model_config, cfg.model_checkpoint, device=device)
    img = imread("../mmdetection/demo/demo.jpg", "color", "bgr")

    bbox_result = inference_detector(model, img)
    show_result_pyplot(model, img, bbox_result, score_thr=0)

    boxes, labels, scores = get_boxes_labels_scores(bbox_result)
    probas = get_probas(model)

    assert np.all((probas <= 1) & (probas >= 0))

    if isinstance(model, TwoStageDetector):
        results = model.test_cfg.rcnn.nms.results
    elif isinstance(model, SingleStageDetector):
        results = model.test_cfg.nms.results
    else:
        raise RuntimeError

    if results:
        bbox_inds = np.argsort(results["labels"], kind="stable")

        assert np.array_equal(results["labels"][bbox_inds], labels)
        assert np.array_equal(results["dets"][bbox_inds][:, :4], boxes)
        assert np.array_equal(results["dets"][bbox_inds][:, -1], scores)

    else:
        # [] if bboxes.numel() == 0 in mmdetection/mmdet/core/post_processing/bbox_nms.py:79
        assert labels.size == 0
        assert boxes.size == 0
        assert scores.size == 0
