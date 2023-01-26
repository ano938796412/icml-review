""" Attack images and evaluate success

1. Prepare the experiment including loading dataset and model
2. Evaluate the dataset according to success criteria and restrict samples to minimum correct detections
3. For every pgd iteration, iterate through samples
    a. prepare training data
    b. sample target and perturb bboxes (or else discard image)
    c. tag dataset and get class probabilities
    d. obtain optimization ground-truth
    e. perturb image and get model predictions
    f. stop attacking upon getting enough samples
4. Evaluate model predictions on attacked images, compute success breakdowns, tag  dataset and save to .csv
"""

import logging
import os
from dataclasses import make_dataclass
from pathlib import Path

import fiftyone as fo
import numpy as np
import pandas as pd
import torch
from fiftyone import ViewField as F
from mmcv import imread
from torchvision.utils import save_image
from tqdm import tqdm

import main.utils.bbox
import main.utils.loss
import main.utils.target
from main.utils.bbox import voc_to_fo, fo_to_voc
from main.utils.detector import (
    init_train_detector,
    get_train_data,
    get_boxes_labels_scores,
    get_bounds,
    unnormalize,
)
from main.utils.pgd import pgd


def attack(
    model_config,
    model_checkpoint,
    dataset_name,
    attack_samples,
    seed,
    compute_map,
    cuda,
    itrs,
    min_iou,
    min_score,
    adversarial_target,
    bbox_sampler,
    attack_bbox,
    perturb_kwargs,
    backward_loss,
    img_dir,
    viz_pgd,
    result_dir,
    log_name,
    **kwargs,
):
    logger = logging.getLogger(log_name)

    os.makedirs(result_dir, exist_ok=True)

    if isinstance(itrs, int):
        itrs = (itrs,)

    if img_dir is None:
        if viz_pgd:
            logger.warning(
                f"{viz_pgd=} will be ignore since {img_dir=}. To be visualized, image has to be saved "
            )

    else:
        os.makedirs(img_dir, exist_ok=True)

        if viz_pgd and len(itrs) > 1:
            logger.warning(
                f"Subsequent iterations will use attacked images in previous iterations (rather than original images)"
            )

    device = torch.device(f"cuda:{cuda}")

    assert attack_bbox in ["ground_truth", "predictions"]
    alt_attack_bbox = "predictions" if attack_bbox == "ground_truth" else "ground_truth"

    dataset = fo.load_dataset(dataset_name)

    logger.debug(dataset)
    logger.info(f"{dataset.name=}")

    adversarial_target_fun = getattr(main.utils.target, adversarial_target)
    bbox_sampler_fun = getattr(main.utils.bbox, bbox_sampler)
    backward_loss_fun = getattr(main.utils.loss, backward_loss)

    model = init_train_detector(model_config, model_checkpoint, device=device)
    classes = [
        cls for cls in dataset.default_classes if not cls.isnumeric()
    ]  # 80 rather than 91

    logger.debug(model)
    logger.info(f"{classes=}")

    lower_bound, upper_bound = get_bounds(model.cfg, device)
    normalized_range = upper_bound - lower_bound

    logger.debug(
        f"{lower_bound.squeeze()=}, {upper_bound.squeeze()=}, {normalized_range.squeeze()=}"
    )

    # only include predicted bboxes with confidence > min_score
    min_conf_slice = dataset.filter_labels("predictions", F("confidence") > min_score)
    min_conf_slice.save()

    eval_key = "predictions_eval"
    min_conf_slice.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key=eval_key,
        iou=min_iou,
        compute_mAP=compute_map,
    )

    min_correct = 1 if bbox_sampler == "arbitrary" else 2
    min_correct_slice = min_conf_slice.match(F(f"{eval_key}_tp") >= min_correct)

    logger.debug(min_correct_slice)
    logger.info(f"{len(min_correct_slice)=}")

    datapoint = make_dataclass(
        "datapoint",
        [
            ("num_iteration", int),
            ("attack_count", int),
            ("success_count", int),
            ("vanish_count", int),
            ("mislabel_count", int),
            ("mislabel_intended_count", int),
            ("sample_count", int),
        ],
    )

    datalist = []

    for itr in itrs:
        if itr > 0:
            lr = normalized_range / itr
        else:
            lr = torch.tensor([0, 0, 0]).to(device)
            logger.warning(f"{lr} is <= 0, setting lr to {lr}")

        logger.info(f"Starting attack {itr=}, {lr.squeeze()=}")

        attack_sample_idx = []
        attack_sample_path = []

        target_bboxes_idx = []
        perturb_bboxes_idx = []

        min_correct_slice_itr = iter(min_correct_slice)
        pbar = tqdm(total=attack_samples)

        sample_count = 0
        while len(attack_sample_idx) < attack_samples:
            sample = next(min_correct_slice_itr)
            sample_count += 1

            # uses 'bgr' because Normalize converts to rgb in data pipeline
            img = imread(sample.filepath, "color", "bgr")
            h, w, c = img.shape

            # sample identical bboxes per image to better compare between iterations and attacks
            # (and not between models since samples and predictions are not identical)
            if seed is not None:
                combined_seed = [seed, int(Path(sample.filepath).stem)]
                rng = np.random.default_rng(combined_seed)
            else:
                rng = np.random.default_rng(seed)

            ground_truth = {"boxes": [], "labels": [], "tp_idxs": []}

            for idx, dt in enumerate(sample[attack_bbox]["detections"]):
                ground_truth["boxes"].append(fo_to_voc(dt["bounding_box"], w, h))
                ground_truth["labels"].append(classes.index(dt["label"]))

                if dt[eval_key] == "tp":
                    if (
                        attack_bbox == "predictions" or dt["iscrowd"] == 0
                    ):  # ignore ground-truth 'iscrowd' is 1
                        ground_truth["tp_idxs"].append(idx)

            data = get_train_data(
                img,
                np.float32(ground_truth["boxes"]),
                np.int_(ground_truth["labels"]),
                model.cfg,
                device,
            )

            selected_bboxes = bbox_sampler_fun(
                ground_truth, data, rng, **perturb_kwargs
            )

            # no bboxes eligible
            if selected_bboxes is None:
                continue

            # perturb_bbox is List, perturb = arbitrary => perturb_bbox_idx = "[direction]"
            # and perturb_bbox is network input scale
            perturb_bbox_idx, target_bbox_idx, perturb_bbox = selected_bboxes

            target_bboxes_idx.append(target_bbox_idx)
            perturb_bboxes_idx.append(perturb_bbox_idx)

            sample[attack_bbox]["detections"][target_bbox_idx].tags.append(
                f"target_{itr}"
            )

            # create an arbitrary bbox label or else tag perturb bbox
            if bbox_sampler == "arbitrary":
                sample[f"arbitrary_{itr}"] = fo.Detections(
                    detections=[
                        fo.Detection(
                            label="arbitrary",
                            bounding_box=voc_to_fo(
                                perturb_bbox,
                                data["img_metas"][0]["img_shape"][1],  # width
                                data["img_metas"][0]["img_shape"][0],  # height
                            ),
                            tags=[f"perturb_{itr}"],
                        )
                    ]
                )
            else:
                sample[attack_bbox]["detections"][perturb_bbox_idx].tags.append(
                    f"perturb_{itr}"
                )

            # retrieve target probas and tag perturb and target `attack_bbox` and corresponding `alt_attack_bbox` bboxes
            if attack_bbox == "predictions":
                target_probas = torch.tensor(
                    sample[attack_bbox]["detections"][target_bbox_idx]["proba"]
                )

            for dt in sample[alt_attack_bbox]["detections"]:
                if (
                    isinstance(perturb_bbox_idx, int)
                    and dt.id
                    == sample[attack_bbox]["detections"][perturb_bbox_idx][
                        f"{eval_key}_id"
                    ]
                ):
                    dt.tags.append(f"perturb_{itr}")

                    # alt_attack_bbox == "predictions" only: or else predicted bbox may not match 'iscrowd' ground-truth
                    if alt_attack_bbox == "predictions":
                        assert (
                            dt[f"{eval_key}_id"]
                            == sample[attack_bbox]["detections"][perturb_bbox_idx].id
                        )

                if (
                    dt.id
                    == sample[attack_bbox]["detections"][target_bbox_idx][
                        f"{eval_key}_id"
                    ]
                ):
                    dt.tags.append(f"target_{itr}")

                    if alt_attack_bbox == "predictions":
                        target_probas = torch.tensor(dt["proba"])
                        assert (
                            dt[f"{eval_key}_id"]
                            == sample[attack_bbox]["detections"][target_bbox_idx].id
                        )

            attack_sample_idx.append(sample.id)
            attack_sample_path.append(sample.filepath)

            if "attack" not in sample.tags:
                sample.tags.append("attack")

            data = adversarial_target_fun(data, target_bbox_idx, target_probas)

            # mislabel returns data + target class
            if adversarial_target == "mislabel":
                data, target_class, proba = data
                sample[f"mislabel_target_class_{itr}"] = classes[target_class]
                sample[f"mislabel_target_proba_{itr}"] = proba.item()

            pgd_data = pgd(
                model_=model,
                data_=data,
                backward_loss=backward_loss_fun,
                perturb_bbox=perturb_bbox,
                itr=itr,
                lr=lr,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                **perturb_kwargs,
            )

            if img_dir is not None:
                save_dir = os.path.join(img_dir, log_name, f"iteration_{itr}")
                os.makedirs(save_dir, exist_ok=True)

                save_name = os.path.join(save_dir, os.path.basename(sample.filepath))
                save_image(unnormalize(pgd_data["img"], model.cfg), save_name)

                if viz_pgd:
                    # images will be normalized and bboxes are rescaled to the original image
                    # and won't exactly align to the saved image because the saved image is padded
                    sample.filepath = os.path.abspath(save_name)

            # model.forward uses double nested inputs, i.e. imgs (List[Tensor]) and img_metas (List[List[dict]])
            # and len(bbox_result) = 80, i.e. one array per class containing bboxes n * (xyxy + c)
            # passing rescale=True ensures we are using the original image scale
            with torch.no_grad():
                bbox_result = model(
                    return_loss=False,
                    rescale=True,
                    **{"img_metas": [pgd_data["img_metas"]], "img": [pgd_data["img"]]},
                )[0]

            boxes, labels, scores = get_boxes_labels_scores(bbox_result)

            detections = []
            for label, score, box in zip(labels, scores, boxes):
                detections.append(
                    fo.Detection(
                        label=classes[int(label)],
                        bounding_box=voc_to_fo(box, w, h),
                        confidence=score,
                    )
                )

            sample[f"pgd_{itr}"] = fo.Detections(detections=detections)
            sample.save()

            pbar.update(1)
        pbar.close()

        if viz_pgd:
            # since the image is resized
            min_correct_slice.compute_metadata(overwrite=True)

        logger.debug(f"{attack_sample_idx=}")
        logger.debug(f"{attack_sample_path=}")
        logger.debug(f"{perturb_bboxes_idx=}")
        logger.debug(f"{target_bboxes_idx=}")

        logger.debug(f"Sampled {sample_count}")
        logger.info("Finished attack")

        attack_slice = min_correct_slice.match_tags("attack")

        logger.debug(attack_slice)
        logger.info(f"{len(attack_slice)=}")

        assert (
            len(attack_slice)
            == len(attack_sample_idx)
            == len(attack_sample_path)
            == len(perturb_bboxes_idx)
            == len(target_bboxes_idx)
        )

        # set `only_matches=False` to retain all samples,
        # even those with `pgd_{itr}` empty
        attack_min_conf_slice = attack_slice.filter_labels(
            f"pgd_{itr}", F("confidence") > min_score, only_matches=False
        )
        attack_min_conf_slice.save()

        attack_results = attack_min_conf_slice.evaluate_detections(
            f"pgd_{itr}",
            gt_field=attack_bbox,
            eval_key=f"pgd_{itr}_eval",
            iou=min_iou,
            compute_mAP=compute_map,
        )

        # only include images with target bbox correctly predicted (but retain all bboxes in those samples)
        success_slice = attack_min_conf_slice.match_labels(
            fields=attack_bbox,
            filter=(F(f"pgd_{itr}_eval") == "fn") & F("tags").contains(f"target_{itr}"),
        )

        logger.debug(success_slice)
        logger.info(
            f"Success/Attack: {len(success_slice)}/{len(attack_min_conf_slice)}"
            f" ({len(success_slice) / len(attack_min_conf_slice) * 100:.2f}%)"
        )

        success_slice.tag_samples(f"success_{itr}")

        # class mislabel with classwise=False
        success_slice.evaluate_detections(
            f"pgd_{itr}",
            gt_field=attack_bbox,
            eval_key=f"pgd_{itr}_mislabel_eval",
            iou=min_iou,
            classwise=False,
        )

        # get mislabeled samples
        mislabel_criteria = (
            (F(f"pgd_{itr}_mislabel_eval") == "fn")
            & (F(f"pgd_{itr}_mislabel_eval_iou") >= min_score)
            & (F("tags").contains(f"target_{itr}"))
        )

        mislabel_slice = success_slice.match_labels(
            fields=attack_bbox, filter=mislabel_criteria
        )
        mislabel_slice.tag_samples(f"mislabel_{itr}")
        logger.info(f"Mislabeled: {len(mislabel_slice)}/{len(success_slice)}")

        # check mislabeled to desired class and tag mislabeled bboxes in ground_truth, predictions and pgd
        mislabel_intended_slice = []

        if len(mislabel_slice) > 0:
            mislabel_slice = mislabel_slice.filter_labels(
                attack_bbox, F("tags").contains(f"target_{itr}")
            )
            mislabel_slice = mislabel_slice.filter_labels(
                alt_attack_bbox, F("tags").contains(f"target_{itr}")
            )
            mislabel_slice = mislabel_slice.filter_labels(
                f"pgd_{itr}",
                F(f"pgd_{itr}_mislabel_eval_id").is_in(
                    np.ravel(mislabel_slice.values(f"{attack_bbox}.detections.id"))
                ),
            )

            mislabel_slice.tag_labels(tags=f"mislabel_{itr}", label_fields=f"pgd_{itr}")

            # check mislabeled class matches desired class in mislabeling attack only
            if adversarial_target == "mislabel":
                mislabel_intended_slice = mislabel_slice.match(
                    F(f"pgd_{itr}.detections.label")
                    == [F(f"mislabel_target_class_{itr}")]
                )
                mislabel_intended_slice.tag_samples(f"mislabel_intended_{itr}")

                assert len(mislabel_slice) >= len(mislabel_intended_slice)
                logger.info(
                    f"Mislabeled to desired class/Mislabeled: {len(mislabel_intended_slice)} / {len(mislabel_slice)}"
                    f" ({len(mislabel_intended_slice) / len(mislabel_slice) * 100:.2f}%)"
                )

        # tag non-mislabeled success as vanished
        vanish_slice = success_slice.match_tags(f"mislabel_{itr}", bool=False)
        vanish_slice.tag_samples(f"vanish_{itr}")
        logger.info(f"Vanished: {len(vanish_slice)}/{len(success_slice)}")

        assert len(success_slice) == (len(mislabel_slice) + len(vanish_slice))

        datalist.append(
            datapoint(
                num_iteration=itr,
                attack_count=len(attack_min_conf_slice),
                success_count=len(success_slice),
                vanish_count=len(vanish_slice),
                mislabel_count=len(mislabel_slice),
                mislabel_intended_count=len(mislabel_intended_slice),
                sample_count=sample_count,
            )
        )

        # save every itr
        df = pd.DataFrame(datalist)

        df["model_name"] = model_config
        df["loss_target"] = adversarial_target
        df["attack_bbox"] = attack_bbox

        for k, v in perturb_kwargs.items():
            df[k] = v

        df_path = os.path.join(result_dir, f"{log_name}_trend.csv")
        logger.info(f"Saving to: {df_path}")

        df.to_csv(df_path, index=False)

        attack_data = dict(
            attack_min_conf_slice=attack_min_conf_slice,
            success_slice=success_slice,
            attack_results=attack_results,
            attack_sample_idx=attack_sample_idx,
            attack_sample_path=attack_sample_path,
            perturb_bboxes_idx=perturb_bboxes_idx,
            target_bboxes_idx=target_bboxes_idx,
        )

    return dataset, attack_data
