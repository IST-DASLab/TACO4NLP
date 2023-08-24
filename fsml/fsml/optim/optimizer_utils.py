import torch

from types import MethodType
from torch.optim import Optimizer
from fsml.compression.pruners.base import BasePruner


__all__ = [
    "wrap_optimizer",
    "unwrap_optimizer"
]


def wrap_optimizer(optimizer: Optimizer, pruner: BasePruner) -> Optimizer:
    optimizer.__step = optimizer.step
    params = pruner.params.values()
    param_masks = pruner.param_masks.values()

    def step(self, *args, **kwargs):
        # 1) prune gradient before step is applied
        with torch.no_grad():
            for param, mask in zip(params, param_masks):
                param.grad.mul_(mask)
        # 2) apply original optimizer step
        self.__step(*args, **kwargs)
        # 3) zero params after optimizer step
        with torch.no_grad():
            for param, mask in zip(params, param_masks):
                param.mul_(mask)

    optimizer.step = MethodType(step, optimizer)
    return optimizer


def unwrap_optimizer(optimizer: Optimizer) -> Optimizer:
    optimizer.step = optimizer.__step
    return optimizer
    