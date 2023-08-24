import numpy as np
from abc import ABC
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.conv import _ConvNd

from ...defs import Sparsity, Bitwidth, SparsityQuery, BitwidthQuery
from ....utils.common import as_list


__all__ = ["BaseOBCUtil"]


class BaseOBCUtil(ABC):
    """Base class for OBC family of methods
    """

    _supported_sparsity_types = ()

    def __init__(self, layer: nn.Module, rel_damp: float = 0.0, **kwargs) -> None:
        """Class constructor.

        Args:
            layer: a linear layer to be compressed. Can be either nn.Linear or nn.ConvNd.
            rel_damp: a fraction of the average hessian diagonal used for regularization of Hessian matrix.
        """
        self.layer = layer
        self.rel_damp = rel_damp
        self.weight: Optional[Tensor] = None
        self.H: Optional[Tensor] = None
        self.losses: Optional[Tensor] = None
        self.num_samples: int = 0
        # pre & post step indicators
        self.pre_step_completed = False
        self.post_step_completed = False
        self.d_row, self.d_col = self._get_number_of_rows_and_cols()

        self._validate_layer()

    def _get_number_of_rows_and_cols(self) -> Tuple[int, int]:
        return self.layer.weight.shape[0], np.prod(self.layer.weight.shape[1:])

    def _validate_layer(self) -> None:
        # 1) check layer type
        assert isinstance(
            self.layer, (nn.Linear, _ConvNd)
        ), "OBCUtil supports only linear and convolutional layers."


    # preparatory methods
    @torch.no_grad()
    def update(self, input: Tensor, output: Optional[Tensor] = None) -> None:
        """
        Update the estimate of Hessian matrix from a batch of data.

        Args:
            input: batch of layer inputs
            outputs: batch of layer outputs
        """
        # get batch size
        batch_size = input.shape[0]
        # init hessian
        if self.H is None:
            # TODO support different precision options?
            self.H = torch.zeros(
                (self.d_col, self.d_col), device=input.device, dtype=torch.float32
            )
        # input reshaping
        if isinstance(self.layer, nn.Linear):
            input = input.reshape(-1, input.shape[-1])
        else:
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            ) 
            # output size (batch_size, channels * \prod kernel_size, num_patches)
            input = unfold(input)
            input = input.transpose(1, 2).flatten(0, 1)
        # cast input to float32 before addition
        input = input.float()
        # hessian update
        beta = self.num_samples / (self.num_samples + batch_size)
        alpha = 2.0 / (self.num_samples + batch_size)
        self.H.addmm_(input.T, input, beta=beta, alpha=alpha)
        # update number of collected samples
        self.num_samples += batch_size
        
 
    @torch.no_grad()
    def pruning_pre_step(self, sparsity_type: str = 'unstructured') -> None:
        """
        Preparatory step with hessian regularization and weight reshaping.
        """
        # move input channel dimension to end for N:M pruning of conv layers
        channels_last = isinstance(self.layer, _ConvNd) and sparsity_type == 'n:m'
        # 1) Hessian preparation
        assert self.H is not None, \
            "One has to process at least one sample of calibration data to run pruning"
        # reorder hessian rows and columns
        if channels_last:
            ks_prod = np.prod(self.layer.kernel_size)
            self.H = self.H.reshape(
                self.d_col // ks_prod, 
                ks_prod, 
                self.d_col // ks_prod, 
                ks_prod
            ).permute(1, 0, 3, 2).reshape(self.d_col, self.d_col)
        # get ids of pruned channels
        pruned_ids = torch.diag(self.H) == 0
        self.H[pruned_ids, pruned_ids] = 1
        # Hessian regularization
        damp = self.rel_damp * torch.diag(self.H).mean()
        self.H.add_(torch.eye(self.H.shape[0], device=self.H.device), alpha=damp)
        # 2) Weight preparation
        # copy weight, flatten and convert to float
        self.weight = self.layer.weight.data.clone().float()
        if isinstance(self.layer, _ConvNd):
            if channels_last:
                self.weight = self.weight.movedim(1, -1)
            self.weight = self.weight.flatten(1, -1)
        self.weight[:, pruned_ids] = 0
        # flag pre step as completed
        self.pre_step_completed = True

    # post compression methods
    @torch.no_grad()
    def pruning_post_step(self) -> None:
        """
        """
        self.reset()
        # flag post step as completed
        self.post_step_completed = True

    # compression methods
    @torch.no_grad()
    def prune(self, sparsity_type: str, sparsities: Sparsity, **prune_kw) -> Dict[float, Tensor]:
        """
        """
        self._validate_sparsity(sparsity_type, sparsities)
        self.pruning_pre_step(sparsity_type)
        # convert to single element list
        if isinstance(sparsities, (float, str)):
            sparsities = [sparsities]
        if sparsity_type == "unstructured":
            sparse_weights = self.prune_unstructured(sparsities)
        elif sparsity_type == "blocked":
            assert "block_size" in prune_kw, "One has to specify block size"
            sparse_weights = self.prune_blocked(sparsities, block_size=prune_kw["block_size"])
        elif sparsity_type == "n:m":
            sparse_weights = self.prune_semistructured(sparsities)
        elif sparsity_type == "structured":
            sparse_weights = self.prune_structured(sparsities)
        else:
            raise ValueError(f"Unknown `sparsity_type` {sparsity_type}")
        self.pruning_post_step()
        return sparse_weights

    @torch.no_grad()
    def prune_unstructured(self, sparsities: SparsityQuery) -> Dict[float, Tensor]:
        raise NotImplementedError(f"`prune_unstructured not implemented for class {self.__class__.__name__}`")

    @torch.no_grad()
    def prune_semistructured(self, sparsities: SparsityQuery) -> Dict[float, Tensor]:
        raise NotImplementedError(f"`prune_semistructured not implemented for class {self.__class__.__name__}`")

    @torch.no_grad()
    def prune_blocked(self, sparsities: SparsityQuery, block_size: int) -> Dict[float, Tensor]:
        raise NotImplementedError(f"`prune_blocked not implemented for class {self.__class__.__name__}`")

    @torch.no_grad()
    def prune_structured(self, sparsities: SparsityQuery) -> Dict[float, Tensor]:
        raise NotImplementedError(f"`prune_structured not implemented for class {self.__class__.__name__}`")

    @torch.no_grad()
    def quantize(self, bits: BitwidthQuery, **quant_kw) -> Dict[int, Tensor]:
        raise NotImplementedError(f"`quantize not implemented for class {self.__class__.__name__}`")

    # TODO add different sparsity and bitwidth options
    @torch.no_grad()
    def prune_and_quantize(self, sparsity_type: str, sparsity: Sparsity, bits: Bitwidth, **quant_kw) -> Dict[float, Tensor]:
        raise NotImplementedError(f"`prune_and_quantize not implemented for class {self.__class__.__name__}`")

    # utility methods
    def _validate_sparsity(self, sparsity_type: str, sparsities: SparsityQuery) -> None:
        assert sparsity_type in self._supported_sparsity_types
        sparsities = as_list(sparsities)
        for sparsity in sparsities:
            if sparsity_type == 'n:m':
                assert isinstance(sparsity, str)
                self._validate_semistructured(sparsity)
            else:
                assert isinstance(sparsity, float)
                self._validate_float(sparsity)      

    def _validate_bitwidth(self, bits: BitwidthQuery):
        if isinstance(bits, (tuple, list)):
            assert isinstance(bits, Bitwidth)
        else:
            assert isinstance(bits, Bitwidth)

    def _validate_semistructured(self, sparsity: str) -> None:
        assert len(sparsity.split(':')), "Sparsity should be of form`N:M`"
        n, m = sparsity.split(':')
        assert n.isdigit() and m.isdigit(), "`N:M` have to be integer numbers"

    def _validate_float(self, sparsity: float) -> None:
        assert 0 <= sparsity <= 1, "Sparsity has to be in [0, 1] range"

    def _reshape_to_orig_shape(self, weight: Tensor, sparsity_type: str = 'unstructured') -> None:
        dtype = self.layer.weight.dtype
        # move input channel dimension to 1st place after N:M pruning of conv layers
        channels_last = isinstance(self.layer, _ConvNd) and sparsity_type == 'n:m'
        if isinstance(self.layer, _ConvNd):
            if channels_last:
                return weight.reshape(self.d_row, *self.layer.kernel_size, -1).movedim(-1, 1).to(dtype)
            return weight.reshape(self.d_row, -1, *self.layer.kernel_size).to(dtype)
        else:
            return weight.reshape(self.d_row, self.d_col).to(dtype)

    def reset(self) -> None:
        """Free all allocated data.
        """
        self.H = None
        self.weight = None
        self.num_samples = 0
        self.losses = None
        self.pre_step_completed = False
        self.post_step_completed = False
        torch.cuda.empty_cache()
