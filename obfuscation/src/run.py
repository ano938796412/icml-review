""" Intent obfuscating attack

In order, run
    1. setup.py
    2. attack.py
    3. analyze.py
and export attacked dataset
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import fiftyone as fo
from mmcv import Config, DictAction

from main.analyze import analyze
from main.attack import attack
from main.setup import setup
from main.utils.misc import get_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intent obfuscating attack")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/mislabel_yolo_v3_test.py",
        help="attack config in ./configs directory",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=datetime.now().strftime("%y%m%d_%H%M%S"),
        help="log name",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="modify config according to https://mmdetection.readthedocs.io/en/latest/tutorials/config.html",
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.log_name = f"{args.log}_{Path(args.config).stem}"
    logger = get_logger(cfg.log_dir, cfg.log_name, cfg.log_level)

    cfg.dataset_name = f"{cfg.log_name}_{cfg.dataset_name}"

    logger.info(cfg)
    logger.debug(fo.config)

    try:
        setup(**cfg)
        dataset = attack(**cfg)[0]

        # save config
        os.makedirs(cfg.dataset_dir, exist_ok=True)
        save_name = os.path.join(cfg.dataset_dir, f"{cfg.dataset_name}")

        cfg.dump(f"{save_name}.py")

        # export dataset
        # https://voxel51.com/docs/fiftyone/user_guide/export_datasets.html#fiftyonedataset
        dataset.export(
            export_dir=save_name,
            dataset_type=fo.types.FiftyOneDataset,
            export_media=False,
            rel_dir=cfg.images_path,
        )

        analyze(**cfg)

        if cfg.launch_app:
            # ssh -N -L 5151:127.0.0.1:5151 [<username>@]<hostname>
            session = fo.launch_app(dataset, remote=True)
            session.wait()

    except Exception as e:
        logger.exception(e)
