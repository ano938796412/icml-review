from datetime import datetime

from mmcv import Config

from main import attack, setup


def test_success_ct():
    """check manual counting equals mongodb select"""
    cfg = Config.fromfile("./configs/vanish_yolo_v3_test.py")

    cfg.gt_samples = 100
    cfg.attack_samples = 10

    cfg.itrs = 10
    cfg.log_name = datetime.now().strftime("%y%m%d_%H%M%S")

    setup.setup(**cfg)
    dataset, attack_data = attack.attack(**cfg)

    success_ct = 0
    attack_ct = 0

    for sample, target_bbox_idx in zip(
        attack_data["attack_min_conf_slice"], attack_data["target_bboxes_idx"]
    ):
        if (
            sample[cfg.attack_bbox]["detections"][target_bbox_idx][
                f"pgd_{cfg.itrs}_eval"
            ]
            == "fn"
        ):
            success_ct += 1
        attack_ct += 1

    assert attack_ct == len(attack_data["attack_min_conf_slice"])
    assert success_ct == len(attack_data["success_slice"])
