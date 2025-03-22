import torch
import torch.nn as nn
from torch.optim import Optimizer

from utils.constants import TOTAL_STEP

UPDATE_ON_BATCH = None


def get_optimizer(
    model: nn.Module, learning_rate: float, weight_decay: float
) -> Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def get_scheduler(
    optimizer: Optimizer, max_lr: float, pct_start: float
) -> torch.optim.lr_scheduler:
    global UPDATE_ON_BATCH

    UPDATE_ON_BATCH = False
    return torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1, total_iters=TOTAL_STEP
    )
