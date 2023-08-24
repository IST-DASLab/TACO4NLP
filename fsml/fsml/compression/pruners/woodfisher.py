from typing import Callable, List, Dict, Any, Union, Iterable, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor

from .base import BasePruner
from ..defs import Sparsity
from ..utils.fisher import EmpiricalBlockFisherInverse
from ...utils.common import to, as_list


__all__ = [
    "WoodFisherPruner"
]


class WoodFisherPruner(BasePruner):

    _supported_sparsity_distributions = ("uniform", "global")
    _supported_sparsity_types = ("unstructured", "blocked", "n:m", "structured")
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
        eps: float = 1e-9
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
        self.damp = damp
        if storage_devices is None:
            self.storage_devices = None
        else:
            self.storage_devices = as_list(storage_devices)
        self.eps = eps
        # TODO suggest better naming?
        self.fisher_storage_dict = self.distribute_fisher_inverses()
        # dict with Fisher inverses
        self.fisher_inverse_dict = {}
        # dict with Fisher inverses * w (required for `blocked` and `structured pruning`)
        self.fisher_inverse_w_dict = {}

    def distribute_fisher_inverses(self) -> Dict[str, str]:
        storage_dict = {}
        # get devices
        if self.storage_devices:
            storage_devices = [torch.device(device) for device in self.storage_devices]
        else:
            num_devices = torch.cuda.device_count()
            if num_devices == 0:
                storage_devices = [torch.device("cpu")]
            else:
                storage_devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]
        for i, param_name in enumerate(self.params):
            storage_dict[param_name] = storage_devices[i % len(storage_devices)]
        return storage_dict

    @torch.enable_grad()
    def prepare_grad(self, batch: Any, device: str) -> None:
        """
        """
        input_args, input_kwargs, targets = batch
        input_args = to(input_args, device=device)
        input_kwargs = to(input_kwargs, device=device)
        targets = to(targets, device=device)
        outputs = self.model(*input_args, **input_kwargs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

    def zero_grad(self) -> None:
        for param in self.model.parameters():
            param.grad = None

    def get_fisher_inverses(self) -> None:
        # allocate Fisher inverses
        for param_name, param in self.params.items():
            self.fisher_inverse_dict[param_name] = EmpiricalBlockFisherInverse(
                self.num_grads,
                block_size=self.fisher_block_size,
                num_weights=param.numel(),
                damp=self.damp,
                device=self.fisher_storage_dict[param_name],
            )
        # TODO unsafe - better suggestions?
        device = next(self.model.parameters()).device
        # TODO add verbose argument?
        pbar = tqdm(enumerate(self.data_loader), total=self.num_grads, desc='Fisher inverse estimation')
        # accumulate Fisher inverses
        for batch_id, batch in pbar:
            if batch_id == self.num_grads:
                break
            # prepare grad
            self.prepare_grad(batch, device)
            # update Fisher inverses
            for param_name, finv in self.fisher_inverse_dict.items():
                param = self.params[param_name]
                mask = self.param_masks.get(param_name, None)
                if mask is not None:
                    param.grad.mul_(mask)
                # update
                finv.update(param.grad.to(self.fisher_storage_dict[param_name], dtype=torch.float32))
            # zero grads
            self.zero_grad()

    def get_score(self, param_name: str) -> Tensor:
        """
        """
        param = self.params[param_name]
        F_inv = self.fisher_inverse_dict[param_name]
        if self.sparsity_type in ["unstructured", "n:m"]:
            score = torch.pow(param.data.reshape(-1), 2).to(F_inv.device).div(2 * F_inv.diag() + self.eps)
        elif self.sparsity_type in ["blocked", "structured"]:
            # structured sparsity is blocked sparsity with block_size param.shape[dim]
            if self.sparsity_type == 'blocked':
                block_size = self.sparsity_block_size
            else:
                block_size = param.shape[self.dim]
            # truncated block size
            B = min(F_inv.block_size, block_size) 
            NFB = F_inv.block_size // B
            NB = block_size // B
             # (d / B, B)
            w_blk = param.reshape(-1, B).to(F_inv.device) 
            F_inv_blk = F_inv.F_inv.reshape(-1, NFB, B, NFB, B).diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2).reshape(-1, B, B)
            # TODO try cholesky solve?
            # (d / B, B)
            F_inv_w_blk = torch.linalg.solve(F_inv_blk, w_blk)  
            score = torch.einsum("bi,bi->b", w_blk, F_inv_w_blk).mul_(0.5).view(-1, NB)
            # cache product block_F_inv_w for mask update
            self.fisher_inverse_w_dict[param_name] = F_inv_w_blk

        if self.sparsity_type == "unstructured":
            return score.reshape(-1)
        elif self.sparsity_type == "blocked":
            return score.sum(dim=-1).reshape(-1)
        elif self.sparsity_type == "n:m":
            return score.reshape(-1)
        # TODO conv handling
        elif self.sparsity_type == "structured":
            return score.sum(dim=-1).reshape(-1)

    def get_threshold(self, score: Tensor, sparsity: Sparsity) -> Union[float, Tensor]:
        if sparsity == 0.0:
            return 0.0
        if self.sparsity_type == "n:m":
            n, m = [int(x) for x in sparsity.split(':')]
            threshold, _ = torch.kthvalue(score.reshape(-1, m), k=n, dim=-1)
        else:
            threshold, _ = torch.kthvalue(score, k=int(sparsity * score.numel()))
        return threshold

    def get_mask(self, param_name: str, score: Tensor, threshold: Union[float, Tensor]) -> Tensor:
        param_shape = self.params[param_name].shape
        param_device = self.params[param_name].device
        if self.sparsity_type == "unstructured":
            mask = (score > threshold).reshape(param_shape)
        elif self.sparsity_type == "blocked":
            mask = (score > threshold).repeat_interleave(self.sparsity_block_size).reshape(param_shape)
        elif self.sparsity_type == "n:m":
            m = score.numel() // threshold.numel()
            mask = (score > threshold.repeat_interleave(m)).reshape(param_shape)
        elif self.sparsity_type == "structured":
            mask = (score > threshold).reshape(
                *param_shape[:self.dim], 
                *param_shape[self.dim + 1:]
            ).unsqueeze(self.dim).expand(param_shape)
        # transfer the mask to param device (in case it is on Fisher inverse device)
        return mask.to(param_device)

    @torch.no_grad()
    def update_param(self, param_name: str, mask_new: Tensor):
        param = self.params[param_name]
        mask_old = self.param_masks.get(
            param_name, 
            torch.ones_like(param, dtype=torch.bool)
        )
        F_inv = self.fisher_inverse_dict[param_name]
        F_inv_w = self.fisher_inverse_w_dict.get(param_name, None)
        mask_diff = ~mask_new & mask_old
        if self.sparsity_type in ["unstructured", "n:m"]:
            param_update = F_inv.mul((param * mask_diff).reshape(-1).to(F_inv.device).div(F_inv.diag() + self.eps)).reshape(param.shape).to(param.device)
        elif self.sparsity_type in ["blocked", "structured"]:
            param_update =  F_inv.mul(F_inv_w.reshape(-1) * mask_diff.reshape(-1).to(F_inv.device)).reshape(param.data.shape).to(param.device)
        param.data.add_(param_update)

    @torch.no_grad()
    def prune(self, sparsity: Sparsity) -> List[Tensor]:
        # prepare Fisher inverses
        self.get_fisher_inverses()
        # prune
        if self.sparsity_distribution == 'uniform':
            for param_name, param in self.params.items():
                score = self.get_score(param_name)
                threshold = self.get_threshold(score, sparsity)
                mask = self.get_mask(param_name, score, threshold)
                self.update_param(param_name, mask)
                # mask param
                param.data.mul_(mask)
                # update mask
                self.param_masks[param_name] = mask
        elif self.sparsity_distribution == 'global':
            score_dict = {}
            # collect scores
            for param_name, param in self.params.items():
                score_dict[param_name] = self.get_score(param_name)
            aggregated_scores = torch.cat([score.cpu().view(-1) for _, score in score_dict.items()])
            threshold = self.get_threshold(aggregated_scores, sparsity)
            for param_name, param in self.params.items():
                score = score_dict[param_name]
                mask = self.get_mask(param_name, score, threshold)
                self.update_param(param_name, mask)
                # mask param
                param.data.mul_(mask)
                # update mask
                self.param_masks[param_name] = mask
            del score_dict
            del aggregated_scores
            torch.cuda.empty_cache()
        # cleanup
        self.fisher_inverse_dict = {}
        self.fisher_inverse_w_dict = {}
        torch.cuda.empty_cache()
