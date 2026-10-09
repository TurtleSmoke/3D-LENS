# Warmup scheduler adapted from fast-reid (https://github.com/JDAI-CV/fast-reid), by liaoxingyu.
import math
from typing import List

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_factor: float = 0.1,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_epoch(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [base_lr * warmup_factor for base_lr in self.base_lrs]

    def _compute_values(self) -> List[float]:
        return self.get_lr()


def _get_warmup_factor_at_epoch(method: str, iter: int, warmup_iters: int, warmup_factor: float) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.
    Args:
        method (str): warmup method, "constant", "linear" or "exp".
        iter (int): iter at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    elif method == "exp":
        return warmup_factor ** (1 - iter / warmup_iters)
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


def build_scheduler(optimizer, cfg, iters_per_epoch):
    max_epoch = cfg.MAX_EPOCHS
    max_epoch = max_epoch - max(math.ceil(cfg.WARMUP_STEPS / iters_per_epoch), 0)

    scheduler_dict = {}
    scheduler_dict["lr_sched"] = CosineAnnealingLR(
        optimizer,
        T_max=max_epoch,
        eta_min=1.6e-06,
    )

    warmup_args = {
        "optimizer": optimizer,
        "warmup_factor": 0.01,
        "warmup_iters": cfg.WARMUP_STEPS,
        "warmup_method": "linear",
    }

    scheduler_dict["warmup_sched"] = WarmupLR(**warmup_args)

    return scheduler_dict
