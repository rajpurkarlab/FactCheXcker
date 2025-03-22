import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import wandb

import numpy as np
import warnings


from models.ETTModel import ETTModel
from utils.AnnotationLoader import AnnotationLoader
from utils.constants import *
from utils.model_helpers import UPDATE_ON_BATCH
from utils.utils import get_annotations_by_image_id, get_center_coordinates_from_bbox

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    annotation_loader: AnnotationLoader,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_batch_step: int,
    epoch_idx: int,
) -> int:
    iters = len(dataloader)

    model.train()
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        images, image_ids = batch["image"], batch["image_id"].tolist()
        annotations = [
            annotation_loader.get_annotations_by_image_id(image_id)
            for image_id in image_ids
        ]
        optimizer.zero_grad()

        if CUDA_AVAILABLE:
            classification_loss, regression_loss = model(
                [images.cuda().float(), annotations]
            )
        else:
            classification_loss, regression_loss = model([images.float(), annotations])

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            try:
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
            except:
                breakpoint()
            
        loss = classification_loss + regression_loss
        wandb.run.log(
            {
                "train_batch_step": train_batch_step,
                "train_classification_loss": classification_loss.item(),
                "train_regression_loss": regression_loss.item(),
                "train_loss": loss.item(),
                "actual_learning_rate": scheduler.get_last_lr()[0],
            }
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()
        
        if UPDATE_ON_BATCH:
            scheduler.step()

        train_batch_step += 1

        del classification_loss
        del regression_loss
        
        # Finish a batch
        
    
    # Finish an epoch    
    if not UPDATE_ON_BATCH:
        scheduler.step()

    return train_batch_step


def validate(
    carinaNet_model: ETTModel,
    dataloader: DataLoader,
    annotation_loader: AnnotationLoader,
    val_epoch_step: int,
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

    wandb.run.log(
        {   
            "val_epoch_step": val_epoch_step,
            "val_classification_loss": classification_loss,
            "val_regression_loss": regression_loss,
            f"{ANNO_CAT_TIP}{ERROR_SUFFIX}": mean_loss_err[f"{ANNO_CAT_TIP}{ERROR_SUFFIX}"],
            f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}": mean_loss_err[f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}"],
            f"{ANNO_CAT_TIP}{RECALL_SUFFIX}": mean_loss_err[f"{ANNO_CAT_TIP}{RECALL_SUFFIX}"],
            f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}": mean_loss_err[f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}"],
            "val_loss": loss,  
        },
        commit=False
    )

    return loss

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
        loss_err[CLASSIFICATION_LOSS].append(
            prediction[CLASSIFICATION_LOSS] if isinstance(prediction[CLASSIFICATION_LOSS], int) else prediction[CLASSIFICATION_LOSS].item())
            
        loss_err[REGRESSION_LOSS].append(
            prediction[REGRESSION_LOSS] if isinstance(prediction[REGRESSION_LOSS], int) else prediction[REGRESSION_LOSS].item())
       
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
