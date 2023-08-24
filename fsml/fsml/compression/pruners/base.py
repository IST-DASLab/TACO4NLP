import re
from abc import ABC, abstractmethod
from typing import List, Union, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.conv import _ConvNd

from ..defs import Sparsity


__all__ = [
    "BasePruner"
]


class BasePruner(ABC):
    """Base class for pruning algorithms.
    """
    _supported_sparsity_distributions = ()
    _supported_sparsity_types = ()
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
        sparsity_block_size: int = 4,
        dim: int = 1, # TODO sanity check that only tensors with ndim > 1 are selected
        **pruner_kwargs
    ) -> None:
        """
        Args:
            model: model to prune
            sparsity_distribution: how the sparsity is distributed across the model (`uniform`, `global`, e.t.c)
            sparsity_type: sparsity mask structure (i.e `unstructured`, `block`, `n:m`, e.t.c)
            is_module_pruner: select modules to prune instead of parameters
            target_params: list or regexp to select parameters pruned
            target_modules: list or regexp to select modules pruned
            prune_biases: whether to prune biases (in case is_module_pruner=True)
            sparsity_block_size: block size for block sparsity_type=`blocked`
            dim: dimension pruned for block sparsity_type=`structured`
        """
        self._validate_sparsity_distribution_and_type(
            sparsity_distribution, 
            sparsity_type,
        )
        super().__init__()
        self.model = model
        # basic pruner properties
        self.sparsity_distribution = sparsity_distribution
        self.sparsity_type = sparsity_type
        self.is_module_pruner = is_module_pruner
        self.target_params = target_params
        self.target_modules = target_modules
        self.prune_biases = prune_biases
        self.sparsity_block_size = sparsity_block_size
        self.dim = dim
        # dict with model parameters
        self.params: Dict[str, Tensor] = {}
        # dict with parameter masks
        self.param_masks: Dict[str, Tensor] = {}
        # dict with module type indicators (active if is_module pruner on)
        self.is_conv_weight: Dict[str, bool] = {}
        self._init_params()

    @abstractmethod
    def prune(self, sparsity: Sparsity) -> None:
        """
        Prune the model to target sparsity

        Args:
            sparsity - the target sparsity configuration
        """
        self._validate_sparsity(sparsity)

    @torch.no_grad()
    def mask_params(self) -> None:
        """
        Multiply parameters by the pruning mask
        """
        if self.param_masks is None:
            raise ValueError("Masks not initialized.")
        for param_name in self.params:
            self.params[param_name].data.mul_(self.param_masks[param_name])

    def _init_params(self) -> None:
        """
        Prepare dicts with params to be pruned.
        """
        if self.is_module_pruner:
            for module_name, module in self.model.named_modules():
                is_target_module = False
                # check if module is in target modules
                if isinstance(self.target_modules, str):
                    if re.search(self.target_modules, module_name):
                        is_target_module = True
                elif isinstance(self.target_modules, list):
                    if module_name in self.target_modules:
                        is_target_module = True
                # add to dict if found
                if is_target_module:
                    for module_param_name, module_param in module.named_parameters():
                        if not 'bias' in module_param_name or self.prune_biases:
                            if module_name:
                                joint_name = f'{module_name}.{module_param_name}'
                            else:
                                joint_name = module_param_name
                            self.params[joint_name] = module_param
                            self.param_masks[joint_name] = torch.ones_like(module_param, dtype=torch.bool)
                            if isinstance(module, _ConvNd) and not 'bias' in module_param_name:
                                self.is_conv_weight[joint_name] = True
        else:
            for param_name, param in self.model.named_parameters():
                is_target_param = False
                # check if param is in target modules
                if isinstance(self.target_params, str):
                    if re.search(self.target_params, param_name):
                        is_target_param = True
                elif isinstance(self.target_params, list):
                    if param_name in self.target_params:
                        is_target_param = True
                # add to dict if found
                if is_target_param:
                    self.params[param_name] = param
                    self.param_masks[param_name] = torch.ones_like(param, dtype=torch.bool)

        return self.params

    def _validate_sparsity(self, sparsity: Sparsity) -> None:
        """
        Check if target sparsity is consistent with the pruner.
        """
        if self.sparsity_type in ['unstructured', 'blocked', 'structured']:
            assert isinstance(sparsity, float), "sparsity has to be a float"
            assert 0 <= sparsity <= 1, "sparsity has to be in [0, 1]"
        elif self.sparsity_type == 'n:m':
            assert isinstance(sparsity, str), "sparsity has to be a str"
            assert len(sparsity.split(':')) == 2, "Sparsity has to be in format n:m"

    def _validate_sparsity_distribution_and_type(
        self, 
        sparsity_distribution: str, 
        sparsity_type: str
    ) -> None:
        """
        Check whether the target sparsity and type are consistent with the pruner.
        """
        assert sparsity_distribution in self._supported_sparsity_distributions
        assert sparsity_type in self._supported_sparsity_types
        if sparsity_type in ['n:m', 'structured']:
            assert sparsity_distribution == 'uniform', f"{sparsity_type} sparsity supports only uniform sparsity distribution"
