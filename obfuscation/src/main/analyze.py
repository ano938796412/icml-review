""" Extract statistics in attacked dataset

Save table as .parquet. Every bbox is a row
 and (parent) sample attributes are duplicated across (children) bboxes.

 Table columns includes:
1. sample data: id, width, height etc.
2. sample tags: present or absent
3. bbox data: id, width, height etc.
4. bbox evaluation: TP/FP/FN + iou
4. bbox tags: present or absent
"""

import logging
import os
from pathlib import Path

import fiftyone as fo
import numpy as np
import pandas as pd
from mmcv import Config
from tqdm import tqdm

from main.utils.misc import get_logger, chain_remove_none


def analyze(
    model_config,
    dataset_name,
    itrs,
    adversarial_target,
    attack_bbox,
    perturb_kwargs,
    result_dir,
    log_name,
    **kwargs,
):
    logger = logging.getLogger(log_name)

    os.makedirs(result_dir, exist_ok=True)

    if isinstance(itrs, int):
        itrs = (itrs,)

    dataset = fo.load_dataset(dataset_name)

    logger.debug(dataset)
    logger.info(dataset.name)

    attack_slice = dataset.match_tags("attack")

    # sample data
    sample_ids = attack_slice.values("id")
    sample_df = pd.DataFrame(
        dict(
            sample_id=sample_ids,
            sample_path=attack_slice.values("filepath"),
            sample_width=attack_slice.values(f"metadata.width"),
            sample_height=attack_slice.values(f"metadata.height"),
        )
    )

    if adversarial_target == "mislabel":
        for itr in itrs:
            sample_df[f"sample_mislabel_class_{itr}"] = attack_slice.values(
                f"mislabel_target_class_{itr}"
            )
            sample_df[f"sample_mislabel_proba_{itr}"] = attack_slice.values(
                f"mislabel_target_proba_{itr}"
            )

    # sample tags
    # grouped by samples
    sample_tags = attack_slice.values("tags")

    unique_st = set(chain_remove_none(sample_tags))
    logger.debug(f"Sample tags: {list(unique_st)}")

    for tag in unique_st:
        sample_df[f"sample_{tag}"] = [tag in s for s in sample_tags]

    bbox_dfs = []

    for bbox in ["ground_truth", "predictions"] + [f"pgd_{itr}" for itr in itrs]:
        # bbox data
        # grouped by samples
        bbox_ids = attack_slice.values(f"{bbox}.detections.id")
        bboxes_per_sample = [len(s) if s is not None else 0 for s in bbox_ids]

        bbox_data = dict(
            bbox_id=bbox_ids,
            bbox_class=attack_slice.values(f"{bbox}.detections.label"),
            bbox_xywhn=attack_slice.values(
                f"{bbox}.detections.bounding_box"
            ),  # normalized x1, y1, w, h
            bbox_conf=attack_slice.values(
                f"{bbox}.detections.confidence"
            ),  # bbox == "ground_truth" => NA
        )

        # bbox evaluations
        # whenever an evaluation doesn't apply to bbox => NA
        # (e.g. pgd evaluated against ground_truth doesn't apply to predictions)
        for evl in attack_slice.list_evaluations():
            bbox_data[f"bbox_res_{evl}"] = attack_slice.values(
                f"{bbox}.detections.{evl}"
            )
            bbox_data[f"bbox_iou_{evl}"] = attack_slice.values(
                f"{bbox}.detections.{evl}_iou"
            )

        # bbox tags
        # grouped by samples then bboxes
        # tags are [] grouped by bboxes
        bbox_tags = attack_slice.values(f"{bbox}.detections.tags")

        unique_bt = set(chain_remove_none(chain_remove_none(bbox_tags)))
        logger.debug(f"{bbox} tags: {list(unique_bt)}")

        for tag in unique_bt:
            bbox_data[f"bbox_{tag}"] = [
                [tag in b for b in s] for s in bbox_tags if s is not None
            ]

        bbox_data = {k: list(chain_remove_none(v)) for k, v in bbox_data.items()}
        bbox_df = pd.DataFrame(bbox_data)

        bbox_df["bbox_type"] = bbox
        bbox_df["sample_id"] = np.repeat(sample_ids, bboxes_per_sample)

        bbox_dfs.append(bbox_df)

    df = pd.concat(bbox_dfs)

    # duplicate sample attributes across children bboxes
    df = sample_df.join(df.set_index("sample_id"), on="sample_id")

    df["model_name"] = model_config
    df["loss_target"] = adversarial_target
    df["attack_bbox"] = attack_bbox

    for k, v in perturb_kwargs.items():
        df[k] = v

    df_path = os.path.join(result_dir, f"{log_name}_bboxes.parquet")

    logger.info(f"Saving to: {df_path}")
    df.to_parquet(df_path, index=False)


if __name__ == "__main__":
    """Reanalyse dataset in `dataset_dir` and launch app to visualize dataset"""

    import sys

    sys.path.extend(
        [
            "$PROJECT_DIR/",
            "$PROJECT_DIR/obfuscation",
            "$PROJECT_DIR/obfuscation/src",
        ]
    )

    dataset_dir = "./data/random/dataset"
    coco_dir = "./coco/val2017"

    for cfg_name in tqdm(sorted(Path(dataset_dir).glob("*.py"))):
        cfg = Config.fromfile(cfg_name)
        logger = get_logger(cfg.log_dir, cfg.log_name, cfg.log_level)

        # https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#fiftyonedataset
        if not cfg.dataset_name in fo.list_datasets():
            cfg_dir = Path(dataset_dir) / Path(cfg_name).stem
            logger.info(cfg_dir)

            dataset = fo.Dataset.from_dir(
                dataset_dir=cfg_dir,
                dataset_type=fo.types.FiftyOneDataset,
                rel_dir=coco_dir,
                name=cfg.dataset_name,
            )

        logger.debug(cfg)
        logger.info(cfg.log_name)

        logger.debug(fo.config)
        logger.info(fo.config.database_dir)

        analyze(**cfg)

    # ssh -N -L 5151:127.0.0.1:5151 [<username>@]<hostname>
    # fiftyone app connect --destination [<username>@]<hostname> --port 5151
    session = fo.launch_app(remote=True)
    session.wait()
