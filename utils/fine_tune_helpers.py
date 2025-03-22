import os
import unicodedata
import numpy as np
import torch
import json

import wandb
import warnings

from models.CarinaNet.CarinaNetModel import CarinaNetModel
from models.ETTModel import ETTModel
from utils.constants import *
from utils.utils import get_annotation_file_path, get_annotations_by_image_id, get_center_coordinates_from_bbox, normalize_hospital_name
from utils.AnnotationLoader import AnnotationLoader
from utils.MAIDA_Dataset import MAIDA_Dataset
import kwcoco
import pandas as pd

from torch.utils.data import DataLoader, ConcatDataset, random_split, RandomSampler


def calculate_loss_and_error(
    predictions_dict: dict, annotation_loader: AnnotationLoader
) -> dict:
    loss_err = {
        CLASSIFICATION_LOSS: [],
        REGRESSION_LOSS: [],
        f"{ANNO_CAT_TIP}{ERROR_SUFFIX}": [],
        f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}": [],
        f"{ANNO_CAT_TIP}{RECALL_SUFFIX}": [],
        f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}": [],
    }

    coco_annotations = annotation_loader.coco_annotations

    for image_id, prediction in predictions_dict.items():
        loss_err[CLASSIFICATION_LOSS].append(prediction[CLASSIFICATION_LOSS] if isinstance(prediction[CLASSIFICATION_LOSS], int) else prediction[CLASSIFICATION_LOSS].cpu())
        loss_err[REGRESSION_LOSS].append(prediction[REGRESSION_LOSS] if isinstance(prediction[REGRESSION_LOSS], int) else prediction[REGRESSION_LOSS].cpu())
        
        for class_id, scores in prediction.items():
            if not class_id in COCO_LABELS:
                continue

            coco_category_id = COCO_LABELS[class_id]
            coco_category_name = coco_annotations.cats[coco_category_id]["name"]

            coco_annots = get_annotations_by_image_id(coco_annotations, image_id)
            # select annotation whose category_id is coco_category_id
            for coco_annot in coco_annots:
                has_prediction = False
                if coco_annot["category_id"] == coco_category_id:
                    gPoint = get_center_coordinates_from_bbox(coco_annot["bbox"])
                    pPoint = scores["pred"]
                    error = np.sqrt(
                        (gPoint[0] - pPoint[0]) ** 2 + (gPoint[1] - pPoint[1]) ** 2
                    )
                    loss_err[f"{coco_category_name}{ERROR_SUFFIX}"].append(error)

                    loss_err[f"{coco_category_name}{RECALL_SUFFIX}"].append(1)
                    has_prediction = True
                    break
            if not has_prediction:
                loss_err[f"{coco_category_name}{RECALL_SUFFIX}"].append(0)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        try:
            mean_loss_err = {k: 1 if len(v) == 0 else np.mean(v) for k, v in loss_err.items()}
        except:
            breakpoint()

    return mean_loss_err


def validate(
    carinaNet_model: ETTModel,
    dataloader: DataLoader,
    annotation_loader: AnnotationLoader,
    val_epoch_step: int = None,
) -> float:
    predictions_dict = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, image_ids = batch["image"], batch["image_id"].tolist()
            images_and_ids = zip(images, image_ids)

            predictions = carinaNet_model.predict(images_and_ids, annotation_loader)
            predictions_dict.update(predictions)

    mean_loss_err = calculate_loss_and_error(predictions_dict, annotation_loader)
    classification_loss = mean_loss_err[CLASSIFICATION_LOSS]
    regression_loss = mean_loss_err[REGRESSION_LOSS]

    loss = classification_loss + regression_loss

    if not WANDB_OFF:
        wandb.run.log(
            {
                "val_classification_loss": classification_loss,
                "val_regression_loss": regression_loss,
                f"{ANNO_CAT_TIP}{ERROR_SUFFIX}": mean_loss_err[f"{ANNO_CAT_TIP}{ERROR_SUFFIX}"],
                f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}": mean_loss_err[f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}"],
                f"{ANNO_CAT_TIP}{RECALL_SUFFIX}": mean_loss_err[f"{ANNO_CAT_TIP}{RECALL_SUFFIX}"],
                f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}": mean_loss_err[f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}"],
                "val_loss": loss,
                "val_epoch_step": val_epoch_step,
            },
            commit=False,
        )           

    return loss 

def get_data_loader(
    config: dict, use_public_data: bool, use_hospitals_data: bool
) -> tuple[DataLoader, DataLoader]:
    """
    Return train and validation data loaders (test set will be target hospital data)
    """
    batch_size = config["batch_size"]
    data_path = config["data_path"]
    default_train_ratio = 0.8
    train_dataset = None
    val_dataset = None

    datasets = {TRAIN_DATA_SOURCE: [], VAL_DATA_SOURCE: [], TEST_DATA_SOURCE: []}
    if use_public_data:
        print("Use MIMIC + RANZCR data...")
        for src in [TRAIN_DATA_SOURCE, VAL_DATA_SOURCE, TEST_DATA_SOURCE]:
            annotation_file_path = get_annotation_file_path(
                data_path, f"{src}-annotations.json"
            )

            with open(annotation_file_path) as f:
                annotation_file = json.load(f)
                image_meta = pd.DataFrame(annotation_file[ANNO_IMAGES_FIELD])

            datasets[src] = MAIDA_Dataset(data_path, src, image_meta)

        train_dataset = ConcatDataset(
            [datasets[VAL_DATA_SOURCE], datasets[TRAIN_DATA_SOURCE]]
        )
        val_dataset = datasets[TEST_DATA_SOURCE]

        # Check that dataset size is correct
        assert len(train_dataset) == (
            len(datasets[VAL_DATA_SOURCE]) + len(datasets[TRAIN_DATA_SOURCE])
        )
        train_ratio = len(train_dataset) / (len(train_dataset) + len(val_dataset))

    if (not config["is_hyperparameter_tuning"]) and (
        config["target_hospital"] != PUBLIC_ONLY_FINETUNED
    ):
        print("Use all but target hospital data...")

        # Combine MIMIC, RANZCR, and all but the target hospital from the hospital dataset
        hospital_to_exclude = config["target_hospital"]
        annotation_file_path = get_annotation_file_path(
            data_path, f"{HOSPITAL_DATA_SOURCE}-annotations.json"
        )
        with open(annotation_file_path) as f:
            annotation_file = json.load(f)
            image_meta = pd.DataFrame(annotation_file[ANNO_IMAGES_FIELD])
            image_meta[ANNO_HOSPITAL_NAME_FIELD] = image_meta[
                ANNO_HOSPITAL_NAME_FIELD
            ].apply(lambda x: normalize_hospital_name(x))

            total_hospital_data_num = len(image_meta)
            total_hospital_num = len(image_meta[ANNO_HOSPITAL_NAME_FIELD].unique())
            image_meta = image_meta[
                image_meta[ANNO_HOSPITAL_NAME_FIELD] != hospital_to_exclude
            ]
            training_hospital_data_num = len(image_meta)
            training_hospital_num = len(image_meta[ANNO_HOSPITAL_NAME_FIELD].unique())

            assert training_hospital_data_num < total_hospital_data_num
            assert training_hospital_num == (total_hospital_num - 1)

        hospital_datasets = MAIDA_Dataset(data_path, HOSPITAL_DATA_SOURCE, image_meta)

        hospital_train_size = int(train_ratio * len(hospital_datasets))
        hospital_val_size = len(hospital_datasets) - hospital_train_size
        hospital_train_dataset, hospital_val_dataset = random_split(
            hospital_datasets, [hospital_train_size, hospital_val_size]
        )

        train_dataset = (
            ConcatDataset([train_dataset, hospital_train_dataset])
            if train_dataset is not None
            else hospital_train_dataset
        )
        val_dataset = (
            ConcatDataset([val_dataset, hospital_val_dataset])
            if val_dataset is not None
            else hospital_val_dataset
        )

        # Check that dataset size is correct
        assert len(train_dataset) == (
            len(datasets[VAL_DATA_SOURCE])
            + len(datasets[TRAIN_DATA_SOURCE])
            + hospital_train_size
        )
        assert len(val_dataset) == (len(datasets[TEST_DATA_SOURCE]) + hospital_val_size)

    val_loader = DataLoader(val_dataset,
                            num_workers=WORKER_NUM,
                            batch_size=batch_size, 
                            shuffle=True)
    train_loader = DataLoader(train_dataset, 
                              num_workers=WORKER_NUM,
                              batch_size=batch_size, 
                              shuffle=True)

    print(f" - Train size: {len(train_loader.dataset)}")
    print(f" - Val size: {len(val_loader.dataset)}")

    return train_loader, val_loader
