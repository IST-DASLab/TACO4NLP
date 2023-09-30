import torch

from typing import List, Union
from types import MethodType
from torch.optim import Optimizer
from fsml.compression.pruners.base import BasePruner


__all__ = [
    "wrap_optimizer",
    "unwrap_optimizer"
]


def wrap_optimizer(optimizer: Optimizer, pruners: Union[BasePruner, List[BasePruner]]) -> Optimizer:
    optimizer.__step = optimizer.step

    if not isinstance(pruners, list):
        pruners = [pruners]

    def step(self, *args, **kwargs):
        # 1) prune gradient before step is applied
        with torch.no_grad():
            for pruner in pruners:
                for param, mask in zip(pruner.params.values(), pruner.param_masks.values()):
                    param.grad.mul_(mask)
        # 2) apply original optimizer step
        self.__step(*args, **kwargs)
        # 3) zero params after optimizer step
        with torch.no_grad():
            for pruner in pruners:
                for param, mask in zip(pruner.params.values(), pruner.param_masks.values()):
                    param.mul_(mask)

    optimizer.step = MethodType(step, optimizer)
    return optimizer


def unwrap_optimizer(optimizer: Optimizer) -> Optimizer:
    optimizer.step = optimizer.__step
    return optimizer
    