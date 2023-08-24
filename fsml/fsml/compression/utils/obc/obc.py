import warnings
import numpy as np
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseOBCUtil
from ...defs import Sparsity
from ....utils.linalg import inv_sym


__all__ = ["OBCUtil"]


class OBCUtil(BaseOBCUtil):

    _supported_sparsity_types = ("unstructured", "blocked", "n:m")

    def __init__(
        self,
        layer: nn.Module,
        rel_damp: float = 0.0,
        rows_in_parallel: Optional[int] = None,
    ) -> None:
        super().__init__(layer, rel_damp)
        # by default process all rows at once
        self.rows_in_parallel = rows_in_parallel or self.d_row
        # weight pruning traces
        self.weight_traces: Optional[Tensor] = None

    @torch.no_grad()
    def _prepare_row_slice(
        self, r1: int, r2: int, block_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor, int, int, Tensor]:
        nr = r2 - r1
        # get a slice of rows
        w = self.weight[r1:r2].clone()
        # create mask of already pruned weights
        if block_size is not None:
            mask = w.reshape(w.shape[0], -1, block_size).ne(0).any(dim=-1)
            weight_mask = mask.repeat_interleave(block_size, dim=1)
        else:
            mask = w.ne(0)
            weight_mask = mask
        # get minimal number of zeros in a slice
        min_zeros = (~mask).sum(dim=1).min().item()
        # get nonzero ids
        row_ids, col_ids = torch.nonzero(~weight_mask).T
        # create N copies (d_row, d_col) -> (nr, d_col, d_col)
        H_inv = self.H.clone().expand(r2 - r1, self.d_col, self.d_col)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # mask rows with zeroed weights
            H_inv[row_ids, col_ids, :] = 0
            H_inv[row_ids, :, col_ids] = 0
            H_inv[row_ids, col_ids, col_ids] = 1
        # invert
        H_inv = inv_sym(H_inv)

        return w, mask, H_inv, min_zeros, nr, torch.arange(nr)

    # preparation
    @torch.no_grad()
    def prepare_unstructured(self) -> None:
        d_row, d_col, device, dtype = (
            self.d_row,
            self.d_col,
            self.weight.device,
            self.weight.dtype,
        )
        # prepare losses & traces
        self.losses = torch.zeros((d_row, d_col), dtype=dtype, device=device)
        self.weight_traces = torch.zeros(
            (d_col + 1, d_row, d_col), dtype=dtype, device="cpu"
        )
        # prune batch of rows
        for r1 in range(0, d_row, self.rows_in_parallel):
            r2 = min(r1 + self.rows_in_parallel, d_row)
          # prepare weight, mask and hessian inverse
            w, mask, H_inv, min_zeros, nr, row_ids = self._prepare_row_slice(r1, r2)
            # prepare pruning traces for slice of rows
            traces = torch.zeros(
                (self.d_col + 1, nr, self.d_col), device=device, dtype=dtype
            )
            traces[:(min_zeros + 1)] = w
            # accumulated losses for a given slice of rows
            accum_losses = torch.zeros(nr, device=device, dtype=dtype)
            # prune iteratively columns
            for col in range(min_zeros + 1, d_col + 1):
                # 1) compure scores
                H_inv_d = H_inv.diagonal(dim1=-2, dim2=-1)
                scores = w ** 2 / H_inv_d
                scores[~mask] = torch.inf
                # 2) mask selection
                p_ids = scores.argmin(dim=-1)
                mask[row_ids, p_ids] = False
                # 3) update losses
                accum_losses.add_(scores[row_ids, p_ids], alpha=0.5)
                self.losses[r1 + row_ids, p_ids] = accum_losses
                # 4) weight update
                H_inv_pr = H_inv[row_ids, p_ids]
                H_inv_pd = H_inv_d[row_ids, p_ids]
                w.add_(H_inv_pr * (w[row_ids, p_ids] / H_inv_pd).unsqueeze(1), alpha=-1)
                w[~mask] = 0
                # update pruning traces
                traces[col] = w
                # do not update H_inv on the last iteration
                if col == self.d_col:
                    break
                # update hessian
                H_inv_pr.div_(torch.sqrt(H_inv_pd).unsqueeze(1))
                H_inv.baddbmm_(H_inv_pr.unsqueeze(2), H_inv_pr.unsqueeze(1), alpha=-1)
                H_inv[row_ids, p_ids, p_ids] = 1.0

            self.weight_traces[:, r1:r2, :] = traces.cpu()

    @torch.no_grad()
    def prepare_blocked(self, block_size: int = 4) -> None:
        assert self.d_col % block_size == 0, "block size has to divide d_col"
        bs, d_row, d_col, device, dtype = (
            block_size,
            self.d_row,
            self.d_col,
            self.weight.device,
            self.weight.dtype,
        )
        # get number of blocks per row
        nbr = d_col // block_size
        b_ids = torch.arange(nbr)
        # prepare losses & traces
        self.losses = torch.zeros((d_row, nbr), dtype=dtype, device=device)
        self.weight_traces = torch.zeros((nbr + 1, d_row, d_col), dtype=dtype, device="cpu")

        for r1 in range(0, d_row, self.rows_in_parallel):
            r2 = min(r1 + self.rows_in_parallel, d_row)
            # prepare weight, mask and hessian inverse
            w, mask, H_inv, min_zeros, nr, row_ids = self._prepare_row_slice(
                r1, r2, block_size
            )
            # prepare pruning traces for slice of rows
            traces = torch.zeros((nbr + 1, nr, d_col), device=device, dtype=dtype)
            traces[: (min_zeros + 1)] = w
            # reshape w and H_inv  to convenient shape
            H_inv = H_inv.view(-1, nbr, bs, nbr, bs).transpose(-3, -2)
            w = w.view(-1, nbr, bs, 1)
            # 
            accum_losses = torch.zeros(nr, device=device, dtype=dtype)
            # pruning iterations
            for block_id in range(min_zeros + 1, nbr + 1):
                # 1) score computation
                H_inv_db = H_inv[:, b_ids, b_ids]  # shape (*, nb, bs, bs)
                inv_H_inv_db = inv_sym(H_inv_db) # shape (*, nb, bs, bs)
                inv_H_inv_db_w = inv_H_inv_db @ w 
                scores = (w * inv_H_inv_db_w).sum(dim=(-2, -1))
                scores[~mask] = torch.inf
                # 2) mask selection
                pb_ids = scores.argmin(dim=-1)
                mask[row_ids, pb_ids] = False
                # 3) update losses
                accum_losses.add_(scores[row_ids, pb_ids], alpha=0.5)
                self.losses[r1 + row_ids, pb_ids] = accum_losses
                # 4) weight update
                H_inv_pr = H_inv[row_ids, pb_ids]
                inv_H_inv_pdb = inv_H_inv_db[row_ids, pb_ids]
                    
                w -= H_inv_pr @ inv_H_inv_db_w[row_ids, pb_ids, None]
                w[~mask] = 0
                traces[block_id] = w.reshape(nr, d_col)
                # 5) hessian update
                if block_id == nbr:
                    break
                H_inv_pr1 = H_inv_pr.unsqueeze(1)
                H_inv_pr2 = H_inv_pr.unsqueeze(2)
                inv_H_inv_pdb = inv_H_inv_pdb.view(-1, 1, 1, bs, bs)
                H_inv -= H_inv_pr1 @ inv_H_inv_pdb @ H_inv_pr2
                # isolate pruned columns
                H_inv[row_ids, pb_ids, :] = 0.0
                H_inv[row_ids, :, pb_ids] = 0.0
                H_inv[row_ids, pb_ids, pb_ids] = torch.eye(bs, device=device, dtype=dtype)

            self.weight_traces[:, r1:r2, :] = traces.cpu()

    @torch.no_grad()
    def prune_unstructured(self, sparsities: Sparsity):
        assert self.pre_step_completed
        self.prepare_unstructured()
        sparse_weights = {}
        for sparsity in sparsities:
            sparse_weights[sparsity], _ = self._extract_from_traces(sparsity)
        return sparse_weights

    # TODO fixbugs
    @torch.no_grad()
    def prune_semistructured(self, sparsities: Sparsity) -> Dict[str, Tensor]:
        assert self.pre_step_completed
        d_row, d_col, device, dtype = (
            self.d_row,
            self.d_col,
            self.weight.device,
            self.weight.dtype,
        )

        sparse_weights = {}
        # TODO reuse iterations?
        for sparsity in sparsities:
            # get n, m
            n, m = map(int, sparsity.split(':'))

            sparse_weight = torch.empty(d_row, d_col, device=device, dtype=dtype)
            # prepare losses
            self.losses = torch.zeros((d_row, d_col), dtype=dtype, device=device)

            for r1 in range(0, d_row, self.rows_in_parallel):
                r2 = min(r1 + self.rows_in_parallel, d_row)
                # prepare weight, mask and hessian inverse
                w, mask, H_inv, min_zeros, nr, row_ids = self._prepare_row_slice(r1, r2)

                buckets = torch.zeros((nr, d_col // m, 1), device=device)

                accum_losses = torch.zeros(nr, device=device, dtype=dtype)
                for col in range(min_zeros + 1, d_col + 1):
                    # 1) compute  scores
                    diag = torch.diagonal(H_inv, dim1=1, dim2=2)
                    scores = w ** 2 / diag 
                    # 2) select mask
                    bucket_mask = (buckets >= n).repeat(1, 1, m).flatten(1)
                    scores[~mask | ~bucket_mask] = float('inf')
                    p_id = scores.argmin(dim=1)
                    mask[row_ids, p_id] = False
                    # 3) update losses
                    accum_losses.add_(scores[row_ids, p_id], alpha=0.5)
                    self.losses[r1:r2, col] = accum_losses
                    # 4) weight update
                    H_inv_pr = H_inv[row_ids, p_id, :]
                    H_inv_pd = diag[row_ids, p_id]
                    w -= H_inv_pr * (w[row_ids, p_id] / H_inv_pd).unsqueeze(1)
                    
                    buckets[row_ids, torch.div(p_id, m, rounding_mode='floor'), :] += 1
                    if col == round(d_col * n / m):
                        break
                    H_inv_pr /= H_inv_pd.sqrt().unsqueeze(1)
                    H_inv.baddbmm_(H_inv_pr.unsqueeze(2), H_inv_pr.unsqueeze(1), alpha=-1)

                w[~mask] = 0
                sparse_weight[r1: r2, :] = w

            sparse_weights[sparsity] = self._reshape_to_orig_shape(sparse_weight, 'n:m')
            del sparse_weight
            torch.cuda.empty_cache()
        return sparse_weights
        
    @torch.no_grad()
    def prune_blocked(self, sparsities: Sparsity, block_size: int = 4):
        assert self.pre_step_completed
        self.prepare_blocked(block_size)
        sparse_weights = {}
        for sparsity in sparsities:
            sparse_weights[sparsity], _ = self._extract_from_traces(sparsity)
        return sparse_weights

    def _extract_from_traces(self, sparsity: float) -> Tuple[Tensor, Tensor]:
        _, topk_indices = torch.topk(
            self.losses.reshape(-1), k=int((1 - sparsity) * self.losses.numel())
        )
        # mask with 0 for pruned weights and 1 elsewhere
        sparsity_mask = torch.zeros(np.prod(self.losses.shape), dtype=torch.bool)
        # in presence of nonzero weights
        if len(topk_indices) > 0:
            sparsity_mask[topk_indices] = 1
        # reshape mask to the weight shape
        sparsity_mask = sparsity_mask.reshape(self.losses.shape)
        # count number of zeros per row
        zeros_per_row = (~sparsity_mask).sum(dim=1)
        return (
            # weight for the given sparsity level
            self._reshape_to_orig_shape(
                self.weight_traces[zeros_per_row, torch.arange(self.d_row)],
            ),
            # reconstruction loss || w x - \hat{w} x ||_2^2
            (~sparsity_mask * self.losses.cpu()).max(dim=1)[0].sum().item(),
        )

    def reset(self):
        super().reset()
        self.losses = None
        self.weight_traces = None
        torch.cuda.empty_cache()
