import json
import os
import kwcoco
import pandas as pd
import wandb
from torch.utils.data import DataLoader, RandomSampler

from models.CarinaNet.CarinaNetModel import CarinaNetModel
from models.ETTModel import ETTModel
from utils.AnnotationLoader import AnnotationLoader
from utils.MAIDA_Dataset import MAIDA_Dataset
from utils.constants import *
from utils.utils import (
    get_annotation_file_path,
    get_annotations_by_image_id,
    get_center_coordinates_from_bbox,
    get_image_filename_by_image_id,
    normalize_hospital_name,
)

def wandb_setup_metrics():
    wandb.define_metric("train_batch_step")
    wandb.define_metric("train_classification_loss", step_metric="train_batch_step", step_sync=True, goal="minimize")
    wandb.define_metric("train_regression_loss", step_metric="train_batch_step", step_sync=True, goal="minimize")
    wandb.define_metric("weighted_tip_loss", step_metric="train_batch_step", step_sync=True, goal="minimize")
    wandb.define_metric("weighted_carina_loss", step_metric="train_batch_step", step_sync=True, goal="minimize")
    wandb.define_metric("train_loss", step_metric="train_batch_step", step_sync=True, goal="minimize")
    wandb.define_metric("actual_learning_rate", step_metric="train_batch_step", step_sync=True)

    wandb.define_metric("val_epoch_step")
    wandb.define_metric("val_classification_loss", step_metric="val_epoch_step", step_sync=True, goal="minimize")
    wandb.define_metric("val_regression_loss", step_metric="val_epoch_step", step_sync=True, goal="minimize")
    wandb.define_metric("val_loss", step_metric="val_epoch_step", step_sync=True, goal="minimize")
    wandb.define_metric(f"{ANNO_CAT_TIP}{ERROR_SUFFIX}", step_metric="val_epoch_step", step_sync=True, goal="minimize")
    wandb.define_metric(f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}", step_metric="val_epoch_step", step_sync=True, goal="minimize")
    wandb.define_metric(f"{ANNO_CAT_TIP}{RECALL_SUFFIX}", step_metric="val_epoch_step", step_sync=True, goal="maximize")
    wandb.define_metric(f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}", step_metric="val_epoch_step", step_sync=True, goal="maximize")
    wandb.define_metric("final_val_loss")
    wandb.define_metric("best_batch")
    
def get_image_metadata(annotation_file_path):
    with open(annotation_file_path) as f:
        annotation_file = json.load(f)
    image_meta = pd.DataFrame(annotation_file[ANNO_IMAGES_FIELD])

    if ANNO_HOSPITAL_NAME_FIELD in image_meta.columns:
        image_meta[ANNO_HOSPITAL_NAME_FIELD] = image_meta[
            ANNO_HOSPITAL_NAME_FIELD
        ].apply(lambda x: normalize_hospital_name(x) if x is not None else None)
    
    return image_meta

def get_model(
    config: dict,
    pretrained_model_path: str = None,
    update_method: str = None,
) -> ETTModel:
    """
    Return a model that inherits ETTModel. For now we only return CarinaNetModel.
    Future implementation can update this method to return other kind of models.
    If pretrained_model_path is provided, load from pretrained_model_path. Otherwise, load
    from config.
    """

    if pretrained_model_path:
        model_path = pretrained_model_path
    else:
        model_path = config["model_path"]

    # If model_path is empty, load the default CarinaNetModel
    if model_path == "":
        model_path = os.path.join(CARINA_NET_OTS_MODEL_DIR, DEFAULT_MODEL_NAME)

    model = CarinaNetModel(model_path, update_method)

    return model

def get_annotation_loader(config: dict) -> AnnotationLoader:
    coco_annotations_list = []

    for src in [
        TRAIN_DATA_SOURCE,
        VAL_DATA_SOURCE,
        TEST_DATA_SOURCE,
        HOSPITAL_DATA_SOURCE,
    ]:
        annotation_file_path = get_annotation_file_path(
            config["data_path"], f"{src}-annotations.json"
        )
        coco_annotations = kwcoco.CocoDataset(annotation_file_path)
        coco_annotations_list.append(coco_annotations)

    final_coco_annotations = kwcoco.CocoDataset.union(*coco_annotations_list)

    # Check that annotation size is correct
    assert len(final_coco_annotations.imgs) == sum(
        [len(coco.imgs) for coco in coco_annotations_list]
    )
    assert len(final_coco_annotations.anns) == sum(
        [len(coco.anns) for coco in coco_annotations_list]
    )

    return AnnotationLoader(final_coco_annotations)


def format_results(
    predictions: dict,
    annotation_loader: AnnotationLoader,
    update_dict: dict,
) -> list[dict]:
    """
    Format batch predictions to a list of dictionaries, where each dictionary contains
    the prediction results and ground truths for a single image.
    """
    ett_prediction_count = 0
    carina_prediction_count = 0
    all_predictions = []
    coco_annotations = annotation_loader.coco_annotations
    patient_idx = update_dict[PATIENT_IDX_INIT]

    # convert classification id to coco category name
    for image_id, prediction in predictions.items():
        file_name = get_image_filename_by_image_id(coco_annotations, image_id)
        patient_idx += 1
        for class_id, scores in prediction.items():
            if not class_id in COCO_LABELS:
                continue
            
            scores["file_name"] = file_name

            coco_category_id = COCO_LABELS[class_id]
            coco_category_name = coco_annotations.cats[coco_category_id]["name"]
            scores["category"] = coco_category_name

            coco_annots = get_annotations_by_image_id(coco_annotations, image_id)
            # select annotation whose category_id is coco_category_id
            for coco_annot in coco_annots:
                if coco_annot["category_id"] == coco_category_id:
                    scores["gPoint"] = get_center_coordinates_from_bbox(
                        coco_annot["bbox"]
                    )
                    break

            scores["iteration"] = update_dict[ITERATION_IDX_INIT] + update_dict[BATCH_IDX]
            scores["simulation"] = update_dict[SIMULATION_IDX]
            scores["batch_size"] = update_dict[BATCH_SIZE]
            scores["patient_idx"] = patient_idx

            all_predictions.append(scores)

            if coco_category_name == ANNO_CAT_TIP:
                ett_prediction_count += 1
            if coco_category_name == ANNO_CAT_CARINA:
                carina_prediction_count += 1

    return all_predictions


def get_hospital_data_annos_loaders(config):
    annos_dir = config['annos_dir']
    loaders = {}
    
    # go through every file in annos_dir
    for file in os.listdir(annos_dir):
        if file in ['hospital-test-annotations.json', 'hospital-train-annotations.json']:
            continue

        if TRAIN_DATA_SOURCE in file:
            split_type = TRAIN_DATA_SOURCE
        elif TEST_DATA_SOURCE in file:
            split_type = TEST_DATA_SOURCE
        else:
            print(f"Invalid file name: {file}")

        # get the hospital name from the file name
        hospital_name = file.split(f"-{split_type}-annotations.json")[0]
        if not hospital_name in loaders:
            loaders[hospital_name] = {}

        annos_loader = AnnotationLoader(kwcoco.CocoDataset(os.path.join(annos_dir, file)))
        dataloader = get_dataloader_from_annoloader(config, HOSPITAL_DATA_SOURCE, annos_loader)
        
        loaders[hospital_name][split_type] = {
            ANNOS_LOADER_KEY: annos_loader,
            DATA_LOADERS_KEY: dataloader
        }

    loaders[ALL_KEY] = {}
    
    all_train_annos_loader = AnnotationLoader(kwcoco.CocoDataset(config['train_annos_path']))
    all_train_dataloader = get_dataloader_from_annoloader(config, HOSPITAL_DATA_SOURCE, all_train_annos_loader)
    loaders[ALL_KEY][TRAIN_DATA_SOURCE] = {
        ANNOS_LOADER_KEY: all_train_annos_loader,
        DATA_LOADERS_KEY: all_train_dataloader
    }
    
    all_test_annos_loader = AnnotationLoader(kwcoco.CocoDataset(config['test_annos_path']))
    all_test_dataloader = get_dataloader_from_annoloader(config, HOSPITAL_DATA_SOURCE, all_test_annos_loader)
    loaders[ALL_KEY][TEST_DATA_SOURCE] = {
        ANNOS_LOADER_KEY: all_test_annos_loader,
        DATA_LOADERS_KEY: all_test_dataloader
    }

    return loaders

def get_dataloader_from_annoloader(config, data_source, annos_loader):
    dataset = MAIDA_Dataset(config[DATA_PATH], data_source, annos_loader.get_all_image_meta())
    dataloader = DataLoader(dataset, 
                            num_workers=WORKER_NUM,
                            batch_size=config[BATCH_SIZE], 
                            shuffle=True)
    return dataloader


def validate_model_type(config):
    if not "model_type" in config:
        raise ValueError("model_type is not specified in the config file")

    if not config["model_type"] in [
        OFF_THE_SHELF,
        PUBLIC_ONLY_FINETUNED,
        ALL_BUT_TARGET_HOSPITALS_ONLY_FINETUNED,
        PUBLIC_HOSPITALS_FINETUNED,
    ]:
        raise ValueError(f"Invalid model type: {config['model_type']}")