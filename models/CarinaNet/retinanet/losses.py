import numpy as np
import torch
import torch.nn as nn
import wandb
import warnings

from utils.constants import CAT_ID_CARINA, CAT_ID_TIP, COCO_LABELS_INVERSE, CUDA_AVAILABLE, WANDB_OFF
from utils.utils import convert_coco_annot_to_tensors


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(
        torch.unsqueeze(a[:, 0], 1), b[:, 0]
    )
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(
        torch.unsqueeze(a[:, 1], 1), b[:, 1]
    )

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = (
        torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1)
        + area
        - iw * ih
    )

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    # def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        weight_delta = 0.2

        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        tip_regression_losses = 0
        carina_regression_losses = 0
        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j]  # [j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if CUDA_AVAILABLE:
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1.0 - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().cuda())

                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1.0 - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue

            IoU = calc_iou(
                anchors[0, :, :], bbox_annotation[:, :4]
            )  # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1

            if CUDA_AVAILABLE:
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0

            targets[
                positive_indices, assigned_annotations[positive_indices, 4].long()
            ] = 1

            if CUDA_AVAILABLE:
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(
                torch.eq(targets, 1.0), alpha_factor, 1.0 - alpha_factor
            )
            focal_weight = torch.where(
                torch.eq(targets, 1.0), 1.0 - classification, classification
            )
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(
                targets * torch.log(classification)
                + (1.0 - targets) * torch.log(1.0 - classification)
            )

            cls_loss = focal_weight * bce

            if CUDA_AVAILABLE:
                cls_loss = torch.where(
                    torch.ne(targets, -1.0),
                    cls_loss,
                    torch.zeros(cls_loss.shape).cuda(),
                )
            else:
                cls_loss = torch.where(
                    torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape)
                )

            classification_losses.append(
                cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0)
            )

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if CUDA_AVAILABLE:
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                # calculate the distance between the groundtruth and predicted boxes
                # predicted_x = regression[positive_indices, 0]
                # predicted_y = regression[positive_indices, 1]
                # predicted_w = regression[positive_indices, 2]
                # predicted_h = regression[positive_indices, 3]
                # predicted_x_ctr = predicted_x + 0.5 * predicted_w
                # predicted_y_ctr = predicted_y + 0.5 * predicted_h
                # regression_diff = torch.sqrt(
                #     torch.pow(predicted_x_ctr - gt_ctr_x, 2)
                #     + torch.pow(predicted_y_ctr - gt_ctr_y, 2)
                # )
                
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0,
                )

                # increase the weight for predictions of the tip
                tip_index = assigned_annotations[:, 4] == COCO_LABELS_INVERSE[CAT_ID_TIP]
                carina_index = assigned_annotations[:, 4] == COCO_LABELS_INVERSE[CAT_ID_CARINA]
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    try:
                        regression_loss[tip_index] *= (1 + weight_delta)
                        tip_regression_losses += regression_loss[tip_index].mean()

                        regression_loss[carina_index] *= (1 - weight_delta)
                        carina_regression_losses += regression_loss[carina_index].mean()

                        regression_losses.append(regression_loss.mean())
                    except:
                        breakpoint()
            else:
                if CUDA_AVAILABLE:
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        if not WANDB_OFF:
            wandb.run.log(
                {
                    "weighted_tip_loss": tip_regression_losses / batch_size,
                    "weighted_carina_loss": carina_regression_losses / batch_size,
                },
                commit=False,
            )
            
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            try:
                return torch.stack(classification_losses).mean(
                    dim=0, keepdim=True
                ), torch.stack(regression_losses).mean(dim=0, keepdim=True)
            except:
                breakpoint()