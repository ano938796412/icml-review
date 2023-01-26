import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from mmcv import Config

from main import attack, setup

cfg_dir = Path("./configs")

cfg_paths = itertools.chain(cfg_dir.glob("*arbitrary*"), cfg_dir.glob("*bbox*"))

cfgs = [Config.fromfile(path) for path in cfg_paths]


@pytest.mark.parametrize("cfg", cfgs)
def test_no_pgd(cfg):
    """check pgd with no perturbation does not change predictions"""
    cfg.gt_samples = 10
    cfg.attack_samples = 1

    cfg.itrs = 0
    cfg.compute_map = True

    cfg.log_name = datetime.now().strftime("%H%M%S")
    cfg.log_dir = "./data/run_test/logs"

    cfg.result_dir = "./data/run_test/results"
    cfg.dataset_name = f"{cfg.dataset_name}_{cfg.log_name}"

    setup.setup(**cfg)
    dataset, attack_data = attack.attack(**cfg)

    original_results = attack_data["attack_min_conf_slice"].evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key="original_eval",
        compute_mAP=True,
        iou=cfg.min_iou,
    )

    # predictions == pgd predictions
    assert np.allclose(original_results.mAP(), attack_data["attack_results"].mAP())
