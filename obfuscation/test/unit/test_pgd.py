import numpy as np
import torch
from mmcv import imread

from main.utils.detector import get_train_data, init_train_detector
from main.utils.loss import get_yolo_v3_vanish_loss
from main.utils.pgd import pgd

device = torch.device("cpu")

model = init_train_detector(
    "../mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py",
    "../mmdetection/checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth",
    device=device,
)

img = imread("../mmdetection/demo/demo.jpg", "color", "bgr")

data = get_train_data(img, np.float32([[0, 1, 3, 4]]), np.int_([0]), model.cfg, device)

original_img_tensor = data["img"].clone()


def test_pgd_perturb_inside_zero_bbox():
    """check pgd with no perturbation does not change image"""
    pgd_data = pgd(
        model_=model,
        data_=data,
        backward_loss=get_yolo_v3_vanish_loss,
        perturb_bbox=[0, 0, 0, 0],
        itr=2,
        lr=1.0,
        lower_bound=0,
        upper_bound=1,  # yolo don't normalize
        **dict(perturb_fun="perturb_inside")
    )

    assert torch.equal(original_img_tensor, pgd_data["img"])
