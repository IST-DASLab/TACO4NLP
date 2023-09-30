from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor

from .base import BasePruner
from ..defs import Sparsity


__all__ = [
    "ConstantPruner"
]


class ConstantPruner(BasePruner):

    _supported_sparsity_distributions = ("uniform", "global")
    _supported_sparsity_types = ("unstructured", "blocked", "n:m", "structured")
    _required_kwargs = ()

    def __init__(
        self,
        model: nn.Module,
        sparsity_distribution: str = 'uniform',
        sparsity_type: str = 'unstructured',
        is_module_pruner: bool = False,
        target_params: Union[str, List[str]] = '.*',
        target_modules: Union[str, List[str]] = '.*',
        prune_biases: bool = False,
        **pruner_kwargs
    ) -> None:
        super().__init__(
            model, 
            sparsity_distribution,
            sparsity_type,
            is_module_pruner,
            target_params,
            target_modules,
            prune_biases
        )
        self._get_masks_from_weight()

    __init__.__doc__ = BasePruner.__init__.__doc__


    @torch.no_grad()
    def _get_masks_from_weight(self) -> None:
        for param_name in self.params:
            self.param_masks[param_name] = self.params[param_name].ne(0)


    @torch.no_grad()
    def prune(self, sparsity: Sparsity) -> List[Tensor]:
        """
        ConstantPruner doesn't change param masks.
        """
        pass
