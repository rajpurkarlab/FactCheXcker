import copy
import pandas as pd
from tqdm import tqdm
import wandb

from models.CarinaNet.CarinaNetModel import CarinaNetModel
from models.ETTModel import ETTModel
from utils.AnnotationLoader import AnnotationLoader
from utils.MAIDA_Dataset import MAIDA_Dataset
from utils.common_helpers import format_results
from torch.utils.data import DataLoader, RandomSampler
from utils.constants import *
from utils.fine_tune_helpers import validate
from utils.hyper_tune_helpers import calculate_loss_and_error
from utils.model_helpers import UPDATE_ON_BATCH
from utils.training_helper import find_epoch


def perform_continual_learning(
    model: ETTModel,
    loaders: dict,
    update_dict: dict ,
) -> list[dict]:
    """
    Perform continual learning (first update then inference) with
    the given dataloader.
    """
    all_predictions = []
    patient_idx = -1
    
    ### perform training
    if not PREV_DATA in update_dict:

        # Use cross validation to find the optimal number of epochs
        best_epoch = find_epoch(model, loaders, update_dict)
        
        iters = len(loaders[TRAIN_DATA_SOURCE][DATA_LOADERS_KEY])
        for epoch in range(best_epoch):
            for batch_idx, batch in enumerate(loaders[TRAIN_DATA_SOURCE][DATA_LOADERS_KEY]):
                images, image_ids = batch["image"], batch["image_id"].tolist()

                model.update_weight(
                    images,
                    image_ids,
                    loaders[TRAIN_DATA_SOURCE][ANNOS_LOADER_KEY],
                    update_dict,
                )
                
                if not WANDB_OFF:
                    wandb.run.log(
                        {
                            "actual_learning_rate": model.scheduler.get_last_lr()[0],
                        }
                    )
                    
                if UPDATE_ON_BATCH:       
                    model.scheduler.step()
            if not UPDATE_ON_BATCH:
                model.scheduler.step()

    else:
        # mix current and previous data
        curr_train_dataset=loaders[TRAIN_DATA_SOURCE][DATA_LOADERS_KEY].dataset
        prev_train_dataset = update_dict[PREV_DATA][DATA_LOADERS_KEY].dataset
        dataloader_batch_size = loaders[TRAIN_DATA_SOURCE][DATA_LOADERS_KEY].batch_size
        mixed_image_meta = pd.concat([curr_train_dataset.get_image_meta(),
                                     prev_train_dataset.get_image_meta()])
        
        # create a new train-validation split
        mixed_image_meta = mixed_image_meta.sample(frac=1).reset_index(drop=True)
        new_train_image_meta = mixed_image_meta.iloc[:len(mixed_image_meta)//2]
        new_val_image_meta = mixed_image_meta.iloc[len(new_train_image_meta):]

        new_train_dataset = MAIDA_Dataset(dataset=curr_train_dataset)
        new_train_dataset.reset_image_meta(new_train_image_meta)
        train_dataloader = DataLoader(new_train_dataset, 
                                      num_workers=WORKER_NUM,
                                      batch_size=dataloader_batch_size, 
                                      shuffle=True)
        
        new_val_dataset = MAIDA_Dataset(dataset=curr_train_dataset)
        new_val_dataset.reset_image_meta(new_val_image_meta)
        val_dataloader = DataLoader(new_val_dataset, 
                                    num_workers=WORKER_NUM,
                                    batch_size=dataloader_batch_size, 
                                    shuffle=True)

        annotation_loader = loaders[TRAIN_DATA_SOURCE][ANNOS_LOADER_KEY] + update_dict[PREV_DATA][ANNOS_LOADER_KEY]

        best_loss = float("inf")
        early_stopping_counter = 0

        for epoch_idx in tqdm(range(50)):
            for batch_idx, batch in enumerate(train_dataloader):
                images, image_ids = batch["image"], batch["image_id"].tolist()

                # update_dict[IS_FINAL_UPDATE] = batch_idx == (loaders[TRAIN_DATA_SOURCE][DATA_LOADERS_KEY] - 1)
                model.update_weight(
                    images,
                    image_ids,
                    annotation_loader,
                    update_dict,
                )
                if UPDATE_ON_BATCH:
                    print("Update on batch")
                    model.scheduler.step()
                    
            if not UPDATE_ON_BATCH:
                print("Update on epoch")
                model.scheduler.step()

            #stop training when validation loss is not decreasing
            val_loss = validate(
                model, val_dataloader, annotation_loader, epoch_idx
            )

            # check if validation loss has improved
            if val_loss < best_loss:
                best_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # check if early stopping criteria is met
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch_idx + 1} epochs")
                break
            

    ### Perform inference 
    for batch_idx, batch in tqdm(enumerate(loaders[TEST_DATA_SOURCE][DATA_LOADERS_KEY])):
        images, image_ids = batch["image"], batch["image_id"].tolist()
        batch_size = len(images)

        ### Predict
        predictions = model.predict(zip(images, image_ids))
        
        mean_loss_err = calculate_loss_and_error(predictions, loaders[TEST_DATA_SOURCE][ANNOS_LOADER_KEY])
        
        print(mean_loss_err)
        print(f'carina error: {mean_loss_err[f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}"]} | tip error: {mean_loss_err[f"{ANNO_CAT_TIP}{ERROR_SUFFIX}"]}')
        print(f'carina recall: {mean_loss_err[f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}"]} | tip recall: {mean_loss_err[f"{ANNO_CAT_TIP}{RECALL_SUFFIX}"]}')
        
        if not WANDB_OFF:
            wandb.run.log(
                {
                    f"{ANNO_CAT_TIP}{ERROR_SUFFIX}": mean_loss_err[f"{ANNO_CAT_TIP}{ERROR_SUFFIX}"],
                    f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}": mean_loss_err[f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}"],
                    f"{ANNO_CAT_TIP}{RECALL_SUFFIX}": mean_loss_err[f"{ANNO_CAT_TIP}{RECALL_SUFFIX}"],
                    f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}": mean_loss_err[f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}"],
                },
                commit=False
            )

        update_dict[PATIENT_IDX_INIT] = patient_idx
        update_dict[BATCH_IDX] = batch_idx
        update_dict[BATCH_SIZE] = batch_size

        all_predictions += format_results(
            predictions,
            loaders[TEST_DATA_SOURCE][ANNOS_LOADER_KEY],
            update_dict,
        )
        patient_idx += batch_size
    
    if PREV_DATA in update_dict:
        del update_dict[PREV_DATA]
    if model.update_method == REHEARSAL_UPDATE:
        # If rehearsal, store current images
        update_dict[PREV_DATA] = loaders[TRAIN_DATA_SOURCE]

    return all_predictions

def perform_continual_learning_on_single_batch(
    model: ETTModel,
    annotation_loader: AnnotationLoader,
    update_dict: dict ,
    batch: dict,) -> dict:
    """
    Perform inference then update on a given batch, and returns a dictionary of the predictions.
    """

    images, image_ids = batch["image"], batch["image_id"].tolist()
    
    ### Predict
    predictions = model.predict(zip(images, image_ids))
    batch_predictions = format_results(
        predictions,
        annotation_loader,
        update_dict,
    )

    ### Update
    update_dict[IS_FINAL_UPDATE] = True
    model.update_weight(
        images,
        image_ids,
        annotation_loader,
        update_dict,
    )

    return batch_predictions