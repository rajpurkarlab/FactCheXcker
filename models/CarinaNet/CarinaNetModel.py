from copy import deepcopy
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import warnings
from models import CarinaNet
from models.CarinaNet import retinanet
from models.ETTModel import ETTModel
from utils.AnnotationLoader import AnnotationLoader
from utils.model_helpers import get_optimizer, get_scheduler
from utils.constants import *
import wandb

L2_FLAG_PRINTED = False

class CarinaNetModel(ETTModel):
    def __init__(
        self,
        model_path: str,
        update_method: str,
        initial_model_weights: dict = None,
        checkpoint: dict = None,
    ):
        super().__init__(update_method)

        if checkpoint is None:
            model_path = os.path.join(CARINA_NET_OTS_MODEL_DIR, DEFAULT_MODEL_NAME) if model_path == "" else model_path
            print(f"Loading checkpoint from {model_path} ...") 
            checkpoint = torch.load(model_path) # GPU
        
        state_dict = checkpoint.state_dict() if isinstance(checkpoint, torch.nn.Module) else checkpoint
        state_dict = state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict
        
        # Have to reload the default CarinaNet model to get the model architecture
        if CUDA_AVAILABLE:
            model = torch.load(model_path).cuda().module # GPU
        else:
            model = torch.load(model_path, map_location=torch.device("cpu")).module
        
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
        model = torch.nn.DataParallel(model)                
        print("Finished loading")

        # save the initial model weights
        self.initial_model_weights = deepcopy(model.state_dict()) if initial_model_weights is None else initial_model_weights

        if update_method == EWC_UPDATE:
            # initiate the dictionary to store the fisher information matrix and optimal parameters
            self.fisher_dict = {}
            self.optparam_dict = {}
        
        self.model = model
        self.optimizer = get_optimizer(self.model, LEARNING_RATE, WEIGHT_DECAY)
        self.scheduler = get_scheduler(self.optimizer, MAX_LR, PCT_START)

    def reset_optimizer(self) -> None:
        self.optimizer = get_optimizer(self.model, LEARNING_RATE, WEIGHT_DECAY)
        
    def reset_scheduler(self) -> None:
        self.scheduler = get_scheduler(self.optimizer, MAX_LR, PCT_START)

    def save_model(self, model_path: str, checkpoint: dict = None) -> None:
        if checkpoint:
            torch.save(
                checkpoint,
                model_path,
            )
            print(f"Checkpoint saved to {model_path}")
        else:
            torch.save(self.model, model_path)
            print(f"Model saved to {model_path}")

    def predict(
        self, images_and_ids: zip, annotation_loader: AnnotationLoader = None
    ) -> dict:
        """
        Perform inference on a batch of images. If annotation_loader is provided,
        the model will calculate the loss; otherwise, it will only perform inference.
        Return a dictionary of predictions per image.
        """
        predictions = {}

        self.model.eval()
        with torch.no_grad():
            for image, image_id in images_and_ids:
                predictions[image_id] = {}
                image = image.unsqueeze(0)
                image_input = image.cuda().float() if CUDA_AVAILABLE else image.float()

                if annotation_loader is not None:
                    annotation = annotation_loader.get_annotations_by_image_id(image_id)
                    inputs = [image_input, [annotation]]
                else:
                    inputs = image_input

                scores, classifications, transformed_anchors, focal_loss = self.model(
                    inputs
                )

                if focal_loss is not None:
                    classification_loss, regression_loss = focal_loss
                else:
                    classification_loss, regression_loss = -1, -1
                    
                predictions[image_id] = {
                        CLASSIFICATION_LOSS: classification_loss,
                        REGRESSION_LOSS: regression_loss,
                }

                scores = scores.cpu().numpy()
                classifications = classifications.cpu().numpy()

                idxs = np.array(
                    [
                        np.argmax(scores * (classifications == c))
                        for c in np.unique(classifications)
                    ]
                )  # Max detection of each class.
                
                for idx in idxs:
                    bbox = transformed_anchors[idx, :]
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    pred = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

                    predictions[image_id][classifications[idx]] = {
                        "confidence": float(scores[idx]),
                        "pred": pred,
                    }

        return predictions

    def get_loss(self, images, annotations, use_l2init = False):
        """
        Helper function for the update functions. Return the loss in train mode
        on a batch of images and annotations.
        """
        self.model.train()
        # Note: self.model.module.freeze_bn() is called within the forward function instead

        if CUDA_AVAILABLE:
            classification_loss, regression_loss = self.model(
                [images.cuda().float(), annotations]
            )
        else:
            classification_loss, regression_loss = self.model(
                [images.float(), annotations]
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            try:
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
            except:
                breakpoint()
                
        loss = classification_loss + regression_loss

        if use_l2init:
            global L2_FLAG_PRINTED
            if not L2_FLAG_PRINTED:
                print("Using L2 init !!!")
                L2_FLAG_PRINTED = True

            # calculate the L2 regularization loss towards the initial model weights
            L2_init_loss = [(v - self.initial_model_weights[k]) ** 2 for k, v in self.model.named_parameters() if v.requires_grad]
            L2_init_loss = sum([l.sum() for l in L2_init_loss])
            loss+= L2_init_loss * L2_INIT_LAMBDA
            
        if not WANDB_OFF:
            wandb.run.log(
                {
                    CLASSIFICATION_LOSS: classification_loss.item(),
                    REGRESSION_LOSS: regression_loss.item(),
                    TOTAL_LOSS: loss.item(),
                },
                commit=False,
            )
            
        return classification_loss, regression_loss, loss

    def naive_update(
        self, images: torch.Tensor, 
        annotations: list[torch.Tensor],
        naive_update_dict: dict,
    ) -> None:
        """
        Perform naive update on a batch of images and annotations.
        """
        self.optimizer.zero_grad()

        classification_loss, regression_loss, loss = self.get_loss(images, annotations, naive_update_dict.get(HAS_L2INIT, False))

        if not WANDB_OFF:
            wandb.run.log(
                {
                    CLASSIFICATION_LOSS: classification_loss.item(),
                    REGRESSION_LOSS: regression_loss.item(),
                    TOTAL_LOSS: loss.item(),
                },
                commit=False,
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

        self.optimizer.step()

        del classification_loss
        del regression_loss

    def rehearsal_update(
        self,
        images: torch.Tensor,
        annotations: list[torch.Tensor],
        rehearsal_update_dict: dict,  # expect prev_data: list[tuple[torch.Tensor, list[torch.Tensor]]],
    ) -> None:
        """
        Perform rehearsal update on a batch of images and annotations. Image from the previous task
        should be provided. Note that the model is trained on twice the batch size given that
        we randomly sample the same amount of images from the previous task.
        """
        self.optimizer.zero_grad()

        classification_loss, regression_loss, loss = self.get_loss(images, annotations, rehearsal_update_dict.get(HAS_L2INIT, False))

        prev_data = rehearsal_update_dict[PREV_DATA] if PREV_DATA in rehearsal_update_dict else []
        if len(prev_data) != 0:
            # Randomly sample from the previous images by generating random integers
            random_indices = random.sample(range(len(prev_data)), len(images))
            prev_images = [prev_data[i][0] for i in random_indices]
            prev_images = torch.stack(prev_images)

            prev_annotations = [prev_data[i][1] for i in random_indices]

            # **accumulate** gradients on the data from the previous task
            addtional_classification_loss, addtional_regression_loss, addtional_loss = (
                self.get_loss(prev_images, prev_annotations, rehearsal_update_dict.get(HAS_L2INIT, False))
            )

            classification_loss += addtional_classification_loss
            regression_loss += addtional_regression_loss
            loss += addtional_loss

        if not WANDB_OFF:
            wandb.run.log(
                {
                    CLASSIFICATION_LOSS: classification_loss.item(),
                    REGRESSION_LOSS: regression_loss.item(),
                    TOTAL_LOSS: loss.item(),
                },
                commit=False,
            )


        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

        self.optimizer.step()

        del classification_loss
        del regression_loss

    def EWC_update(
        self,
        images: torch.Tensor,
        annotations: list[torch.Tensor],
        EWC_update_dict: dict,
    ) -> None:
        """
        Perform EWC update on a batch of images and annotations. The EWC_dict should contain
        curr_domain_data (list of batches of images and annotations), is_final_update (bool),
        prev_domain_name (optional), and curr_domain_name (required)

        reference: https://pub.towardsai.net/overcoming-catastrophic-forgetting-a-simple-guide-to-elastic-weight-consolidation-122d7ac54328
        https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb
        The Fisher information matrix explicitly calculates the importance of each weight for a given task
        The Fisher information matrix diagonal, approximated by the squared expectation of the gradients, represents the importance of each weight
        This importance value quantifies the weight’s contribution to the performance on previously learned tasks.
        Weights with higher importance values have a greater impact on the performance of the previous tasks, and thus, their updates should
        be constrained more during the learning of new tasks.
        """

        def on_task_update():

            self.fisher_dict[EWC_update_dict[CURR_DOMAIN_NAME]] = {}
            self.optparam_dict[EWC_update_dict[CURR_DOMAIN_NAME]] = {}

            # accumulate gradient of the current task
            self.optimizer.zero_grad()
            for images, annotations in EWC_update_dict[CURR_DOMAIN_DATA]:
                _, _, loss = self.get_loss(images, annotations, EWC_update_dict.get(HAS_L2INIT, False))
                loss.backward()

            # Update Fisher diagonal for each weight for the current task
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                self.optparam_dict[EWC_update_dict[CURR_DOMAIN_NAME]][
                    name
                ] = param.data.clone()
                self.fisher_dict[EWC_update_dict[CURR_DOMAIN_NAME]][
                    name
                ] = param.grad.data.clone().pow(2)

        self.optimizer.zero_grad()

        classification_loss, regression_loss, loss = self.get_loss(images, annotations, EWC_update_dict.get(HAS_L2INIT, False))

        if EWC_update_dict[PREV_DOMAIN_NAME] in self.optparam_dict:
            prev_task_fisher_diag_dict = self.fisher_dict[
                EWC_update_dict[PREV_DOMAIN_NAME]
            ]
            prev_task_optparam_dict = self.optparam_dict[
                EWC_update_dict[PREV_DOMAIN_NAME]
            ]

            EWC_loss = 0
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                fisher = prev_task_fisher_diag_dict[name]
                optparam = prev_task_optparam_dict[name]
                EWC_loss += (fisher * (optparam - param).pow(2)).sum() * EWC_LAMBDA
            loss += EWC_loss
            if not WANDB_OFF:
                wandb.run.log({"EWC_loss": EWC_loss.item()}, commit=False)
        
        if not WANDB_OFF:
            wandb.run.log(
                {
                    CLASSIFICATION_LOSS: classification_loss.item(),
                    REGRESSION_LOSS: regression_loss.item(),
                    TOTAL_LOSS: loss.item(),
                },
                commit=False,
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        self.optimizer.step()

        if EWC_update_dict[IS_FINAL_UPDATE]:
            on_task_update()

        del classification_loss
        del regression_loss
