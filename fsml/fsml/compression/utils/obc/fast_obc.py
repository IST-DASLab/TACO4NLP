from typing import Dict, List, Union, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseOBCUtil
from ...defs import Bitwidth, BitwidthQuery, Sparsity, SparsityQuery
from ...utils.quantization import QuantUtil
from ....utils.linalg import inv_sym


__all__ = ["FastOBCUtil"]


class FastOBCUtil(BaseOBCUtil):

    _supported_sparsity_types = ("unstructured", "n:m")

    def __init__(
        self,
        layer: nn.Module,
        rel_damp: float = 0.0,
        block_size: int = None,
    ) -> None:
        super().__init__(layer, rel_damp)
        # by default process all rows at once
        self.block_size = block_size or self.d_col

    @torch.no_grad()
    def prepare_data(self):
        w = self.weight.clone()
        # get columns with all zeros
        zero_cols = torch.nonzero(w.eq(0).all(dim=0))
        # get number of zero structures
        num_zeros = len(zero_cols)
        H = self.H
        # mask rows with zero input channels
        H[zero_cols, :] = 0
        H[:, zero_cols] = 0
        H[zero_cols, zero_cols] = 1
        # invert
        H = inv_sym(H)
        H_inv_cho = torch.linalg.cholesky(H, upper=True)
        return w, H_inv_cho, num_zeros
    
    @torch.no_grad()
    def prune_unstructured(self, sparsities: SparsityQuery) -> Dict[str, Tensor]:
        assert self.pre_step_completed
        d_col, block_size = self.d_col, self.block_size

        sparse_weights = {}
        for sparsity in sparsities:
            # prepare weight and Cholesky of H^{-1}
            w, H_inv_cho, nzeros = self.prepare_data()
            # iterate over columns
            for c1 in range(nzeros, d_col, block_size):
                c2 = min(c1 + block_size, d_col)
                ncols = c2 - c1 # number of columns
                w_blk = w[:, c1:c2].clone() # column-wise weight slice
                res = torch.zeros_like(w_blk)
                errs = torch.zeros_like(w_blk)
                losses_blk = torch.zeros_like(w_blk)
                H_inv_cho_blk = H_inv_cho[c1:c2, c1:c2]
                # 1) score computation
                scores = w_blk ** 2 / H_inv_cho_blk.diag().reshape(1, -1) ** 2
                thr, _ = torch.kthvalue(scores.view(-1), round(w_blk.numel() * sparsity))
                mask = scores > thr
                # 2) iterate over block
                for i in range(ncols):
                    w_ci = w_blk[:, i]
                    d = H_inv_cho_blk[i, i]

                    q = w_ci.clone()
                    q[~mask[:, i]] = 0

                    res[:, i] = q
                    err = (w_ci - q) / d
                    losses_blk[:, i] = err ** 2
                    
                    w_blk[:, i:].addr_(err, H_inv_cho_blk[i, i:], alpha=-1)
                    errs[:, i] = err
                # 3) update the weights after block
                w[:, c1:c2] = res
                w[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)

            sparse_weights[sparsity] = self._reshape_to_orig_shape(w)
        
        return sparse_weights

    @torch.no_grad()
    # TODO make quant_kw a dataclass or specify quant_kw explicitly?
    def quantize(
        self,
        bits: BitwidthQuery, 
        dim: Optional[int] = None,
        groupsize: int = -1,
        sym: bool = False,
        flatten: bool = True,
        optimize_grid: bool = False,
        min_scale: float = 0.8,
        optimization_iters: int = 100,
    ) -> Dict[int, Tensor]:
        self.pruning_pre_step()
        d_col, block_size = self.d_col, self.block_size

        quant_kw = dict(
            sym=sym,
            flatten=flatten,
            optimize_grid=optimize_grid,
            min_scale=min_scale,
            optimization_iters=optimization_iters,
            groupsize=groupsize,
            dim=dim
        )

        quantized_weights = {}
        for bit in bits:
            # prepare weight and Cholesky of H^{-1}
            w, H_inv_cho, num_zeros = self.prepare_data()
            # create one quantizer for the whole weight in case of single group for the last dimension
            # otherwise create separate quantizer for each group 
            if 'dim' != 1 or 'groupsize' == -1:
                quantizer = QuantUtil(w, bits=bit, **quant_kw)
            else:
                quantizer = None
             # iterate over columns
            for c1 in range(num_zeros, d_col, block_size):
                c2 = min(c1 + block_size, d_col)
                ncols = c2 - c1
                w_blk = w[:, c1:c2].clone()
                res = torch.zeros_like(w_blk)
                errs = torch.zeros_like(w_blk)
                losses_b = torch.zeros_like(w_blk)
                H_inv_cho_blk = H_inv_cho[c1:c2, c1:c2]
                # 1) iterate over block
                for i in range(ncols):
                    # init new quantizer
                    if dim == 1 and groupsize != -1:
                        quantizer_idx = (c1 + i) % groupsize
                        if (c1 + i) % groupsize == 0:
                            quantizer = QuantUtil(w[:, (c1 + i):(c1 + i + groupsize)], bits=bit, **quant_kw)
                    else:
                        quantizer_idx = c1 + i

                    w_ci = w_blk[:, i]
                    d = H_inv_cho_blk[i, i]

                    q = quantizer(w_ci, idx=quantizer_idx, dim=1)

                    res[:, i] = q
                    err = (w_ci - q) / d
                    losses_b[:, i] = err ** 2
                    
                    w_blk[:, i:].addr_(err, H_inv_cho_blk[i, i:], alpha=-1)
                    errs[:, i] = err

                w[:, c1:c2] = res
                w[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)

            quantized_weights[bit] = self._reshape_to_orig_shape(w)
        
        return quantized_weights

    @torch.no_grad()
    # TODO code duplication with prune and quantize. Unify somehow?
    def prune_and_quantize(
        self, 
        sparsity: Sparsity,
        bits: Bitwidth, 
        dim: Optional[int] = None,
        groupsize: int = -1,
        sym: bool = False,
        flatten: bool = True,
        optimize_grid: bool = False,
        min_scale: float = 0.8,
        optimization_iters: int = 100,
    ) -> Dict[int, Tensor]:
        self.pruning_pre_step()
        d_col, block_size = self.d_col, self.block_size
        # prepare weight and Cholesky of H^{-1}
        w, H_inv_cho, num_zeros = self.prepare_data()

        quant_kw = dict(
            sym=sym,
            flatten=flatten,
            optimize_grid=optimize_grid,
            min_scale=min_scale,
            optimization_iters=optimization_iters,
            groupsize=groupsize,
            dim=dim
        )

        # create one quantizer for the whole weight in case of single group for the last dimension
        # otherwise create separate quantizer for each group 
        if 'dim' != 1 or 'groupsize' == -1:
            quantizer = QuantUtil(w, bits=bits, **quant_kw)
        else:
            quantizer = None
            # iterate over columns
        for c1 in range(num_zeros, d_col, block_size):
            c2 = min(c1 + block_size, d_col)
            ncols = c2 - c1
            w_blk = w[:, c1:c2].clone()
            res = torch.zeros_like(w_blk)
            errs = torch.zeros_like(w_blk)
            losses_b = torch.zeros_like(w_blk)
            H_inv_cho_blk = H_inv_cho[c1:c2, c1:c2]
            # 1) score computation
            scores = w_blk ** 2 / H_inv_cho_blk.diag().reshape(1, -1) ** 2
            thr, _ = torch.kthvalue(scores.view(-1), round(w_blk.numel() * sparsity))
            mask = scores > thr
            # 2) iterate over block
            for i in range(ncols):
                # init new quantizer
                if dim == 1 and groupsize != -1:
                    quantizer_idx = (c1 + i) % groupsize
                    if (c1 + i) % groupsize == 0:
                        quantizer = QuantUtil(w[:, (c1 + i):(c1 + i + groupsize)], bits=bits, **quant_kw)
                else:
                    quantizer_idx = c1 + i

                w_ci = w_blk[:, i]
                d = H_inv_cho_blk[i, i]

                q = quantizer(w_ci, idx=quantizer_idx, dim=1)
                q[~mask[:, i]] = 0

                res[:, i] = q
                err = (w_ci - q) / d
                losses_b[:, i] = err ** 2
                
                w_blk[:, i:].addr_(err, H_inv_cho_blk[i, i:], alpha=-1)
                errs[:, i] = err

            w[:, c1:c2] = res
            w[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)

        pruned_and_quantized_weights = self._reshape_to_orig_shape(w)
        
        return pruned_and_quantized_weights

    def reset(self):
        super().reset()
        self.losses = None
