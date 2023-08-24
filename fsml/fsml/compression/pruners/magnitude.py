from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor

from .base import BasePruner
from ..defs import Sparsity


__all__ = [
    "MagnitudePruner"
]


class MagnitudePruner(BasePruner):

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
        sparsity_block_size: int = 4,
        **pruner_kwargs
    ) -> None:
        super().__init__(
            model, 
            sparsity_distribution,
            sparsity_type,
            is_module_pruner,
            target_params,
            target_modules,
            prune_biases,
            sparsity_block_size,
        )

    __init__.__doc__ = BasePruner.__init__.__doc__

    def get_score(self, param_name: str) -> Tensor:
        """
        """
        score = self.params[param_name].abs()
        if self.sparsity_type == "unstructured":
            return score.reshape(-1)
        elif self.sparsity_type == "blocked":
            return score.reshape(-1, self.sparsity_block_size).sum(dim=-1)
        elif self.sparsity_type == "n:m":
            return score.reshape(-1)
        # TODO conv handling
        elif self.sparsity_type == "structured":
            # move the pruned dim to the end
            score = score.movedim(self.dim, -1)
            # sum scores across other dimensions
            return score.sum(dim=tuple(range(score.ndim - 1)))

    def get_threshold(self, score: Tensor, sparsity: Sparsity) -> Union[float, Tensor]:
        if sparsity == 0.0:
            return 0.0
        # kthvalue is implemented only for float
        score = score.float()
        if self.sparsity_type == "n:m":
            n, m = [int(x) for x in sparsity.split(':')]
            threshold, _ = torch.kthvalue(score.reshape(-1, m), k=n, dim=-1)
        else:
            threshold, _ = torch.kthvalue(score, k=int(sparsity * score.numel()))
        return threshold

    def get_mask(self, param_name: str, score: Tensor, threshold: Union[float, Tensor]) -> Tensor:
        param_shape = self.params[param_name].shape
        if self.sparsity_type == "unstructured":
            mask = (score > threshold).reshape(param_shape)
        elif self.sparsity_type == "blocked":
            mask = (score > threshold).repeat_interleave(self.sparsity_block_size).reshape(param_shape)
        elif self.sparsity_type == "n:m":
            m = score.numel() // threshold.numel()
            mask = (score > threshold.repeat_interleave(m)).reshape(param_shape)
        elif self.sparsity_type == "structured":
            expanded_shape = [1,] * (len(param_shape) - 1) + [len(score)]
            mask = (score > threshold).view(expanded_shape).movedim(-1, self.dim).expand(param_shape)
        return mask

    @torch.no_grad()
    def prune(self, sparsity: Sparsity) -> List[Tensor]:
        if self.sparsity_distribution == 'uniform':
            for param_name, param in self.params.items():
                score = self.get_score(param_name)
                threshold = self.get_threshold(score, sparsity)
                mask = self.get_mask(param_name, score, threshold)
                # mask param
                param.data.mul_(mask)
                # update mask
                self.param_masks[param_name] = mask
        elif self.sparsity_distribution == 'global':
            score_dict = {}
            # collect scores
            for param_name, param in self.params.items():
                score_dict[param_name] = self.get_score(param_name)
            aggregated_scores = torch.cat([score.view(-1) for _, score in score_dict.items()])
            threshold = self.get_threshold(aggregated_scores, sparsity)
            for param_name, param in self.params.items():
                score = score_dict[param_name]
                mask = self.get_mask(param_name, score, threshold)
                # mask param
                param.data.mul_(mask)
                # update mask
                self.param_masks[param_name] = mask
            del score_dict
            del aggregated_scores
            torch.cuda.empty_cache()
