from tqdm import tqdm
import pandas as pd
import os

import wandb
from models.CarinaNet.CarinaNetModel import CarinaNetModel
from utils.AnnotationLoader import AnnotationLoader
from torch.utils.data import DataLoader
from utils.common_helpers import format_results
from utils.constants import *
from utils.fine_tune_helpers import calculate_loss_and_error

def inference(
    carinaNet_model: CarinaNetModel,
    dataloader: DataLoader,
    annotation_loader: AnnotationLoader,
):
    all_predictions = []
    patient_idx = -1
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        images, image_ids = batch["image"], batch["image_id"].tolist()
        images_and_ids = zip(images, image_ids)
        batch_size = len(images)

        ### Prediction
        update_dict = {
            PATIENT_IDX_INIT: patient_idx,
            BATCH_IDX: batch_idx,
            BATCH_SIZE: batch_size,
            SIMULATION_IDX: 0,
            ITERATION_IDX_INIT: 0,
        }
        predictions = carinaNet_model.predict(images_and_ids)
        mean_loss_err = calculate_loss_and_error(predictions, annotation_loader)
        
        if not WANDB_OFF:
            wandb.run.log(
                {
                    f"{ANNO_CAT_TIP}{ERROR_SUFFIX}": mean_loss_err[f"{ANNO_CAT_TIP}{ERROR_SUFFIX}"],
                    f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}": mean_loss_err[f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}"],
                    f"{ANNO_CAT_TIP}{RECALL_SUFFIX}": mean_loss_err[f"{ANNO_CAT_TIP}{RECALL_SUFFIX}"],
                    f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}": mean_loss_err[f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}"],
                },
            )
            
        all_predictions += format_results(
            predictions,
            annotation_loader,
            update_dict)
        
        patient_idx += batch_size

    # convert all_predictions to a dataframe
    all_predictions = pd.DataFrame(all_predictions)

    return all_predictions

def save_predictions(all_predictions, config, output_path, split=None):
    # Save predictions
    if config[INFERENCE_DATASET] == USE_HOSPITLAS_DATASET_FOR_INFERENCE:
        # all_predictions["hospital"] = config["target_hospital"]
        split = "" if None else f"{split}_"
        file_path = os.path.join(
            output_path,
            f"{config["model_type"]}_hospitals-{split}{INFERENCE_OUTPUT_FILENAME}",
        )
    else:
        file_path = os.path.join(
            output_path,
            f"{config["model_type"]}_{USE_PUBLIC_DATASET_FOR_INFERENCE}-{INFERENCE_OUTPUT_FILENAME}",
        )

    all_predictions.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")
