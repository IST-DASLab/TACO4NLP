from typing import Callable, List, Union, Iterable, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor

from .base import BasePruner
from .woodfisher import WoodFisherPruner
from ..defs import Sparsity
from ..utils.fisher import CAPUtil
from ..utils.fisher import EmpiricalBlockFisherReduced, FastCAPUtil


__all__ = ["CorrelationAwarePruner", "FastCorrelationAwarePruner"]


class CorrelationAwarePruner(WoodFisherPruner):

    _supported_sparsity_distributions = ("uniform", "global")
    _supported_sparsity_types = ("unstructured", "blocked", "n:m")
    _required_kwargs = ('data_loader',)

    def __init__(
        self,
        # general args
        model: nn.Module,
        # woodfisher requires data_loader
        data_loader: Iterable,
        # in most cases one has to provide loss_fn, but HF models compute loss inside
        loss_fn: Optional[Callable] = None,
        sparsity_distribution: str = 'uniform',
        sparsity_type: str = 'unstructured',
        is_module_pruner: bool = False,
        target_params: Union[str, List[str]] = '.*',
        target_modules: Union[str, List[str]] = '.*',
        prune_biases: bool = False,
        sparsity_block_size: int = 4,
        # woodfisher specific args
        num_grads: int = 1024,
        fisher_block_size: int = 64,
        damp: float = 1e-6,
        storage_devices: Union[str, list[str]] = None,
        eps: float = 1e-9,
        rows_in_parallel: Optional[int] = None,
    ) -> None:
        super().__init__(
            model, 
            data_loader,
            loss_fn,
            sparsity_distribution,
            sparsity_type,
            is_module_pruner=is_module_pruner,
            target_params=target_params,
            target_modules=target_modules,
            prune_biases=prune_biases,
            sparsity_block_size=sparsity_block_size,
            num_grads=num_grads,
            fisher_block_size=fisher_block_size,
            damp=damp,
            storage_devices=storage_devices,
            eps=eps,
        )
        # CAP specific params
        self.rows_in_parallel = rows_in_parallel

    @torch.no_grad()
    def prune(self, sparsity: Sparsity) -> List[Tensor]:
        # prepare Fisher inverses
        self.get_fisher_inverses()
        # prune
        if self.sparsity_distribution == 'uniform':
            for param_name, param in self.params.items():
                cap_util = CAPUtil(
                    param, 
                    self.fisher_inverse_dict[param_name].F_inv, 
                    self.rows_in_parallel,
                )
                param_sparse = cap_util.prune(self.sparsity_type, sparsity)[sparsity]
                mask = param_sparse.ne(0)
                # update param
                param.data = param_sparse.to(param.device)
                # update mask
                self.param_masks[param_name] = mask
        elif self.sparsity_distribution == 'global':
            score_dict = {}
            cap_util_dict = {}
            # collect scores
            for param_name, param in self.params.items():
                cap_util = CAPUtil(
                    param, 
                    self.fisher_inverse_dict[param_name].F_inv, 
                    self.rows_in_parallel,
                )
                # update dict with cap_utils
                cap_util_dict[param_name] = cap_util
                # prepare losses and traces
                cap_util.prepare(self.sparsity_type, block_size=self.sparsity_block_size)
                score_dict[param_name] = cap_util.get_score()
            aggregated_scores = torch.cat([score.cpu().view(-1) for _, score in score_dict.items()])
            threshold = self.get_threshold(aggregated_scores, sparsity)
            for param_name, param in self.params.items():
                score = score_dict[param_name]
                cap_util = cap_util_dict[param_name]
                mask = self.get_mask(param_name, score, threshold)
                # get sparsity
                sparsity = 1 - mask.sum().item() / mask.numel()
                # extract param
                param_sparse, _ = cap_util._extract_from_traces(sparsity)
                # update param
                param.data = param_sparse.to(param.device)
                # update mask
                self.param_masks[param_name] = mask
            del score_dict
            del aggregated_scores
            torch.cuda.empty_cache()
        # cleanup
        self.fisher_inverse_dict = {}
        self.fisher_inverse_w_dict = {}
        torch.cuda.empty_cache()


class FastCorrelationAwarePruner(BasePruner):

    _supported_sparsity_distributions = ("uniform",)
    _supported_sparsity_types = ("unstructured",)
    _required_kwargs = ('data_loader',)

    def __init__(
        self,
        # general args
        model: nn.Module,
        # woodfisher requires data_loader
        data_loader: Iterable,
        # in most cases one has to provide loss_fn, but HF models compute loss inside
        loss_fn: Optional[Callable] = None,
        sparsity_distribution: str = 'uniform',
        sparsity_type: str = 'unstructured',
        is_module_pruner: bool = False,
        target_params: Union[str, List[str]] = '.*',
        target_modules: Union[str, List[str]] = '.*',
        prune_biases: bool = False,
        sparsity_block_size: int = 4,
        # woodfisher specific args
        num_grads: int = 1024,
        fisher_block_size: int = 64,
        rel_damp: float = 1e-6,
        storage_devices: Union[str, list[str]] = None,
        eps: float = 1e-9,
        # cap args
        block_size: Optional[int] = None
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
        #
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        # WoodFisher specific params
        self.num_grads = num_grads
        self.fisher_block_size = fisher_block_size
        self.rel_damp = rel_damp
        self.eps = eps
        self.block_size = block_size
        # dict with Fisher blocks
        self.fisher_dict = {}

    # inherit methods from CAP
    prepare_grad = CorrelationAwarePruner.prepare_grad
    zero_grad = CorrelationAwarePruner.zero_grad

    def get_fisher_blocks(self) -> None:
        # TODO unsafe - better suggestions?
        device = next(self.model.parameters()).device
        # allocate Fisher inverses
        for param_name, param in self.params.items():
            self.fisher_dict[param_name] = EmpiricalBlockFisherReduced(
                block_size=self.fisher_block_size, device=device
            )
        # TODO add verbose argument?
        pbar = tqdm(enumerate(self.data_loader), total=self.num_grads, desc='Fisher inverse estimation')
        # accumulate Fisher inverses
        for batch_id, batch in pbar:
            if batch_id == self.num_grads:
                break
            # prepare grad
            self.prepare_grad(batch, device)
            # update Fisher inverses
            for param_name, finv in self.fisher_dict.items():
                param = self.params[param_name]
                mask = self.param_masks.get(param_name, None)
                if mask:
                    param.grad.mul_(mask)
                # update
                finv.update(param.grad.reshape(-1))
            # zero grads
            self.zero_grad()

    @torch.no_grad()
    def prune(self, sparsity: Sparsity) -> List[Tensor]:
        # prepare Fisher inverses
        self.get_fisher_blocks()
        # prune
        if self.sparsity_distribution == 'uniform':
            for param_name, param in self.params.items():
                cap_util = FastCAPUtil(
                    param, 
                    self.fisher_dict[param_name].F, 
                    self.rel_damp,
                    self.block_size
                )
                param_sparse = cap_util.prune(self.sparsity_type, sparsity)[sparsity]
                mask = param_sparse.ne(0)
                # update param
                param.data = param_sparse.to(param.device)
                # update mask
                self.param_masks[param_name] = mask
        # cleanup
        self.fisher_dict = {}
        torch.cuda.empty_cache()
