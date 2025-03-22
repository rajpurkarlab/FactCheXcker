import copy
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
import wandb

from models.CarinaNet.CarinaNetModel import CarinaNetModel
from utils.MAIDA_Dataset import MAIDA_Dataset
from utils.constants import ANNOS_LOADER_KEY, CLASSIFICATION_LOSS, CUDA_AVAILABLE, DATA_LOADERS_KEY, OUTPUT_PATH, REGRESSION_LOSS, TRAIN_DATA_SOURCE, WANDB_OFF, WORKER_NUM
from utils.fine_tune_helpers import calculate_loss_and_error
from utils.model_helpers import UPDATE_ON_BATCH


def find_epoch(model, loaders, update_dict = {}):    
    
    best_loss = float("inf")
    best_epoch = 0
    early_stopping_counter = 0
    early_stopping_patience = 3
    
    for epoch_idx in tqdm(range(100)):
        # Use hold-1-out cross validation to get the model performance after training fixed number of epochs
        total_loss = train_and_evaluate_for_single_epoch(model, loaders, update_dict, epoch_idx)
        print(f"Epoch {epoch_idx}: total_loss = {total_loss}")
        
        if total_loss < best_loss:
            best_loss = total_loss
            best_epoch = epoch_idx
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        print(f'epoch {epoch_idx} best_epoch: {best_epoch} early_stopping_counter: {early_stopping_counter}')
        if early_stopping_counter >= early_stopping_patience:
            break
        
    if not WANDB_OFF:
        wandb.run.log(
            {
                "total_loss": total_loss,
                "best_epoch": best_epoch,
            },
            commit=False
        )

    print(f"Best epoch index: {best_epoch}")

    return best_epoch + 1  # returning the total number of epochs not the epoch index!

def train_and_evaluate_for_single_epoch(model_orig, loaders, update_dict, current_epoch):
    train_dataset=loaders[TRAIN_DATA_SOURCE][DATA_LOADERS_KEY].dataset
    train_image_meta = train_dataset.get_image_meta()
    total_loss = 0
    
    temp_model_dir = os.path.join(update_dict[OUTPUT_PATH], "temp_models")
    if not os.path.exists(temp_model_dir):
        os.makedirs(temp_model_dir)
        
    for fold_idx in range(len(train_image_meta)):
        new_train_dataloader, new_val_dataset = create_new_folds(loaders, train_dataset, train_image_meta, fold_idx)
        
        # let val image id be the model id
        new_val_image_meta = new_val_dataset.get_image_meta()
        temp_model_id = new_val_image_meta.iloc[0]["id"]
        temp_model_path = os.path.join(temp_model_dir, f"{temp_model_id}.pth")

        model_copy, checkpoint = get_model_and_checkpoint(temp_model_path, model_orig)
                   
        model_updated = False            
        for batch_idx, batch in enumerate(new_train_dataloader):
            images, image_ids = batch["image"], batch["image_id"].tolist()
            
            # if current_epoch is smaller or equal to checkpoint['epoch_idx'], we will skip the update
            # but keep the scheduler step to make sure the learning rate is updated
            if (checkpoint is None) or (current_epoch > checkpoint['epoch_idx']):
                model_copy.update_weight(
                        images,
                        image_ids,
                        loaders[TRAIN_DATA_SOURCE][ANNOS_LOADER_KEY],
                        update_dict,
                    )
                model_updated = True
            else:
                print(f"Skip updating the model for epoch {current_epoch}")

            # Finish a single batch
            if UPDATE_ON_BATCH:
                model_copy.scheduler.step()
                
        # Finish a single epoch
        if not UPDATE_ON_BATCH:
            model_copy.scheduler.step()
            
        # save the model
        if model_updated:
            checkpoint = {
                'model_state_dict': model_copy.model.module.state_dict(),
                'epoch_idx': current_epoch
            }
            model_copy.save_model(temp_model_path, checkpoint)
            print(f"Model saved at {temp_model_path}")
                                
        ### Perform inference on the held-out fold
        val_data = new_val_dataset.__getitem__(0)
        images, image_ids = [val_data["image"]], [val_data["image_id"]]
        predictions = model_copy.predict(zip(images, image_ids), loaders[TRAIN_DATA_SOURCE][ANNOS_LOADER_KEY])

        mean_loss_err = calculate_loss_and_error(predictions, loaders[TRAIN_DATA_SOURCE][ANNOS_LOADER_KEY])
        classification_loss = mean_loss_err[CLASSIFICATION_LOSS]
        regression_loss = mean_loss_err[REGRESSION_LOSS]
        
        loss = classification_loss + regression_loss
        total_loss += loss

        del model_copy
                
    return total_loss

def create_new_folds(loaders, train_dataset, train_image_meta, fold_idx):
    new_val_image_meta = train_image_meta.iloc[fold_idx:fold_idx+1]
    new_train_image_meta = pd.concat((train_image_meta.iloc[:fold_idx], train_image_meta.iloc[fold_idx+1:]), ignore_index=True, axis=0)

    new_train_dataset = MAIDA_Dataset(dataset=train_dataset)
    new_train_dataset.reset_image_meta(new_train_image_meta)
    new_train_dataloader = DataLoader(new_train_dataset, 
                                              num_workers=WORKER_NUM,
                                              batch_size=loaders[TRAIN_DATA_SOURCE][DATA_LOADERS_KEY].batch_size, 
                                              shuffle=True)

    new_val_dataset = MAIDA_Dataset(dataset=train_dataset)
    new_val_dataset.reset_image_meta(new_val_image_meta)
    
    return new_train_dataloader, new_val_dataset

def get_model_and_checkpoint(temp_model_path, model_orig):
    '''
    If temp_model_id is not in temp_model_dir, create one; otherwise, load the model
    '''
    checkpoint = None
    
    if os.path.exists(temp_model_path):
        checkpoint = torch.load(temp_model_path)

        model_copy = CarinaNetModel(temp_model_path, model_orig.update_method, copy.deepcopy(model_orig.initial_model_weights),  checkpoint)
    else:
        # Make sure to use a copy of the model to avoid overwriting the weights
        model_copy = copy.deepcopy(model_orig)
        model_copy.reset_optimizer() # avoid accidentally modifying the original model's optimizer
        model_copy.reset_scheduler() # avoid accidentally modifying the original model's scheduler
    
    return model_copy, checkpoint
