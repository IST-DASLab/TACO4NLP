import numpy as np
from typing import Tuple, Optional, Dict, List

import torch
from torch import Tensor
from torch.nn.functional import pad

from ...defs import SparsityQuery
from ....utils.linalg import inv_sym
from ....utils.common import as_list


__all__ = ["CAPUtil", "FastCAPUtil"]


class CAPUtil:

    # TODO - add N:M (requires channel reordering in weight and fisher matrix)
    _supported_sparsity_types = ("unstructured", "blocked")

    def __init__(
        self,
        weight: Tensor,
        F_inv: Tensor,
        rows_in_parallel: Optional[int] = None,
        channels_last: bool = False,
    ) -> None:
        # validate Fisher inverse
        self._validate_fisher_inverse(F_inv)
        self.F_inv = F_inv
        # F_inv params
        self.n_blocks = F_inv.shape[0]
        self.fisher_block_size = F_inv.shape[1]
        # by default process all rows at once
        self.rows_in_parallel = rows_in_parallel or self.n_blocks
        # whether the weight is conv weight
        self.channels_last = channels_last
        # backup dtype
        self.dtype = weight.dtype
        # backup shape
        self.shape = weight.shape
         # reshape weight to (-1, block_size)
        self.padding = 0
        self.weight = self._prepare_weight(weight)
        # weight pruning traces
        self.weight_traces: Optional[Tensor] = None
        # store losses
        self.losses: Optional[Tensor] = None
   
    @staticmethod
    def _validate_fisher_inverse(F_inv: Tensor) -> None:
        assert F_inv.ndim == 3, "F_inv has to be 3-dimensional tensor"
        assert F_inv.shape[1] == F_inv.shape[2]

    def _prepare_weight(self, weight: Tensor) -> Tensor:
        if self.channels_last:
            assert weight.ndim > 2, "conv weight has to be at least 3-dimensional"
            # move input channel dimension to the end and reshape to (num_blocks, fisher_block_size )
            weight = weight.movedim(1, -1)
            # reorder Fisher TODO
        # pad weight if needed
        if weight.numel() % self.fisher_block_size != 0:
            self.padding = self.fisher_block_size - weight.numel() % self.fisher_block_size
            weight = pad(weight.view(-1), (0, self.padding))
        # reshape to (n_blocks, self.fisher_block_size) and return
        return weight.reshape(self.n_blocks, self.fisher_block_size)

    def _reshape_to_orig_shape(self, weight: Tensor) -> Tensor:
        # unpad
        if self.padding > 0:
            weight = weight.view(-1)[:-self.padding]
        # move last channel to beginning
        if self.channels_last:
            return weight.reshape(self.shape[0], -1, self.shape[1]).movedim(-1, 1)

        return weight.reshape(self.shape).to(self.dtype)

    # TODO distinguish block pruning and Fisher block size
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
        # get slice of F_inv
        F_inv_slice = self.F_inv[r1:r2]
        # mask rows with zeroed weights
        F_inv_slice[row_ids, col_ids, :] = 0
        F_inv_slice[row_ids, :, col_ids] = 0
        F_inv_slice[row_ids, col_ids, col_ids] = 1

        return w, mask, F_inv_slice, min_zeros, nr, torch.arange(nr)

    # preparation
    @torch.no_grad()
    def prepare_unstructured(self) -> None:
        n_blocks, fbs, device, dtype = (
            self.n_blocks,
            self.fisher_block_size, # fisher block size
            self.weight.device,
            self.weight.dtype,
        )
        # prepare losses & traces
        self.losses = torch.zeros((n_blocks, fbs), dtype=dtype, device=device)
        self.weight_traces = torch.zeros(
            (fbs + 1, n_blocks, fbs), dtype=dtype, device="cpu"
        )
        # prune batch of rows
        for r1 in range(0, n_blocks, self.rows_in_parallel):
            r2 = min(r1 + self.rows_in_parallel, n_blocks)
          # prepare weight, mask and hessian inverse
            w, mask, H_inv, min_zeros, nr, row_ids = self._prepare_row_slice(r1, r2)
            # prepare pruning traces for slice of rows
            traces = torch.zeros(
                (fbs + 1, nr, fbs), device=device, dtype=dtype
            )
            traces[:(min_zeros + 1)] = w
            # accumulated losses for a given slice of rows
            accum_losses = torch.zeros(nr, device=device, dtype=dtype)
            # prune iteratively columns
            for col in range(min_zeros + 1, fbs + 1):
                # 1) compure scores
                H_inv_d = H_inv.diagonal(dim1=-2, dim2=-1)
                scores = w ** 2 / H_inv_d
                scores[~mask] = torch.inf
                # 2) mask selection
                p_ids = scores.argmin(dim=-1)
                mask[row_ids, p_ids] = False
                # 3) update losses
                accum_losses += 0.5 * scores[row_ids, p_ids]
                self.losses[r1 + row_ids, p_ids] = accum_losses
                # 4) weight update
                H_inv_pr = H_inv[row_ids, p_ids]
                H_inv_pd = H_inv_d[row_ids, p_ids]
                w -= H_inv_pr * (w[row_ids, p_ids] / H_inv_pd).unsqueeze(1)
                w[~mask] = 0
                # update pruning traces
                traces[col] = w
                # do not update H_inv on the last iteration
                if col == fbs:
                    break
                # update hessian
                H_inv_pr /= torch.sqrt(H_inv_pd).unsqueeze(1)
                H_inv.baddbmm_(H_inv_pr.unsqueeze(2), H_inv_pr.unsqueeze(1), alpha=-1)
                H_inv[row_ids, p_ids, p_ids] = 1.0

            self.weight_traces[:, r1:r2, :] = traces.cpu()

    @torch.no_grad()
    def prepare_blocked(self, block_size: int = 4) -> None:
        bs, n_blocks, fbs, device, dtype = (
            block_size,
            self.n_blocks,
            self.fisher_block_size,
            self.weight.device,
            self.weight.dtype,
        )
        assert fbs % block_size == 0, "block size has to divide fbs"
        # get number of pruning blocks per fisher block
        nbr = fbs // block_size
        b_ids = torch.arange(nbr)
        # prepare losses & traces
        self.losses = torch.zeros((n_blocks, nbr), dtype=dtype, device=device)
        self.weight_traces = torch.zeros((nbr + 1, n_blocks, fbs), dtype=dtype, device="cpu")

        for r1 in range(0, n_blocks, self.rows_in_parallel):
            r2 = min(r1 + self.rows_in_parallel, n_blocks)
            # prepare weight, mask and hessian inverse
            w, mask, H_inv, min_zeros, nr, row_ids = self._prepare_row_slice(
                r1, r2, block_size
            )
            # prepare pruning traces for slice of rows
            traces = torch.zeros((nbr + 1, nr, fbs), device=device, dtype=dtype)
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
                accum_losses += 0.5 * scores[row_ids, pb_ids]
                self.losses[r1 + row_ids, pb_ids] = accum_losses
                # 4) weight update
                H_inv_pr = H_inv[row_ids, pb_ids]
                inv_H_inv_pdb = inv_H_inv_db[row_ids, pb_ids]
                    
                w -= H_inv_pr @ inv_H_inv_db_w[row_ids, pb_ids, None]
                w[~mask] = 0
                traces[block_id] = w.reshape(nr, fbs)
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

    def prepare(self, sparsity_type: str, **prune_kw):
        self.pruning_pre_step()
        if sparsity_type == 'unstructured':
            self.prepare_unstructured()
        elif sparsity_type == 'blocked':
            assert "block_size" in prune_kw, "One has to specify block size"
            self.prepare_blocked(block_size=prune_kw['block_size'])
        else:
            raise ValueError("Unsupported `sparsity_type`")

    @torch.no_grad()
    def prune_unstructured(self, sparsities: List[float]):
        self.prepare_unstructured()
        sparse_weights = {}
        for sparsity in sparsities:
            sparse_weights[sparsity], _ = self._extract_from_traces(sparsity)
        return sparse_weights

    # TODO fixbugs
    @torch.no_grad()
    def prune_semistructured(self, sparsities: List[str]):
        n_blocks, fbs, device, dtype = (
            self.n_blocks,
            self.fisher_block_size,
            self.weight.device,
            self.weight.dtype,
        )

        sparse_weights = {}
        # TODO reuse iterations?
        for sparsity in sparsities:
            # get n, m
            n, m = map(int, sparsity.split(':'))

            sparse_weight = torch.empty(n_blocks, fbs, device=device, dtype=dtype)
            # prepare losses
            self.losses = torch.zeros((n_blocks, fbs), dtype=dtype, device=device)

            for r1 in range(0, n_blocks, self.rows_in_parallel):
                r2 = min(r1 + self.rows_in_parallel, n_blocks)
                # prepare weight, mask and hessian inverse
                w, mask, H_inv, min_zeros, nr, row_ids = self._prepare_row_slice(r1, r2)

                buckets = torch.zeros((nr, fbs // m, 1), device=device)

                accum_losses = torch.zeros(nr, device=device, dtype=dtype)
                for col in range(min_zeros + 1, fbs + 1):
                    # 1) compute  scores
                    diag = torch.diagonal(H_inv, dim1=1, dim2=2)
                    scores = w ** 2 / diag 
                    # 2) select mask
                    bucket_mask = (buckets >= n).repeat(1, 1, m).flatten(1)
                    scores[~mask | ~bucket_mask] = float('inf')
                    p_id = scores.argmin(dim=1)
                    mask[row_ids, p_id] = False
                    # 3) update losses
                    accum_losses += 0.5 * scores[row_ids, p_id]
                    self.losses[r1:r2, col] = accum_losses
                    # 4) weight update
                    H_inv_pr = H_inv[row_ids, p_id, :]
                    H_inv_pd = diag[row_ids, p_id]
                    w -= H_inv_pr * (w[row_ids, p_id] / H_inv_pd).unsqueeze(1)
                    
                    buckets[row_ids, torch.div(p_id, m, rounding_mode='floor'), :] += 1
                    if col == round(fbs * n / m):
                        break
                    H_inv_pr /= H_inv_pd.sqrt().unsqueeze(1)
                    H_inv.baddbmm_(H_inv_pr.unsqueeze(2), H_inv_pr.unsqueeze(1), alpha=-1)

                w[~mask] = 0
                sparse_weight[r1: r2, :] = w

            sparse_weights[sparsity] = self._reshape_to_orig_shape(sparse_weight)
            del sparse_weight
            torch.cuda.empty_cache()
        return sparse_weights
        
    @torch.no_grad()
    def prune_blocked(self, sparsities: float, block_size: int = 4):
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
                self.weight_traces[zeros_per_row, torch.arange(self.n_blocks)]
            ),
            # reconstruction loss || w x - \hat{w} x ||_2^2
            (~sparsity_mask * self.losses.cpu()).max(dim=1)[0].sum().item(),
        )

    def pruning_pre_step(self):
        self.weight = self.weight.to(self.F_inv.device)

    def pruning_post_step(self):
        """Free all allocated data.
        """
        self.F_inv = None
        self.weight = None
        self.reset()
        torch.cuda.empty_cache()

    # compression methods
    @torch.no_grad()
    def prune(self, sparsity_type: str, sparsities: SparsityQuery, **prune_kw) -> Dict[float, Tensor]:
        """
        """
        self._validate_sparsity(sparsity_type, sparsities)
        self.pruning_pre_step()
        # convert to single element list
        if isinstance(sparsities, (float, str)):
            sparsities = [sparsities]
        if sparsity_type == "unstructured":
            sparse_weights = self.prune_unstructured(sparsities)
        elif sparsity_type == "blocked":
            assert "block_size" in prune_kw, "One has to specify block size"
            sparse_weights = self.prune_blocked(sparsities, block_size=prune_kw["block_size"])
        # TODO
        elif sparsity_type == "n:m":
            sparse_weights = self.prune_semistructured(sparsities)
        # TODO
        elif sparsity_type == "structured":
            sparse_weights = self.prune_structured(sparsities)
        else:
            raise ValueError(f"Unknown `sparsity_type` {sparsity_type}")
        self.pruning_post_step()
        return sparse_weights

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

    def _validate_semistructured(self, sparsity: str) -> None:
        assert len(sparsity.split(':')), "Sparsity should be of form`N:M`"
        n, m = sparsity.split(':')
        assert n.isdigit() and m.isdigit(), "`N:M` have to be integer numbers"

    def _validate_float(self, sparsity: float) -> None:
        assert 0 <= sparsity <= 1, "Sparsity has to be in [0, 1] range"

    def get_score(self) -> Tensor:
        assert self.losses is not None, "one has to run prepare to compute scores"
        return self.losses

    def reset(self):
        self.losses = None
        self.weight_traces = None


class FastCAPUtil:

    _supported_sparsity_types = ("unstructured",)

    def __init__(
        self,
        weight: Tensor,
        F: Tensor,
        rel_damp: float = 0.0,
        block_size: int = None,
    ) -> None:
        # validate Fisher inverse
        self._validate_fisher(F)
        self.F = F
        self.rel_damp = rel_damp
        self.fisher_block_size = F.shape[0]
        # by default process all rows at once
        self.block_size = block_size or self.fisher_block_size
        # backup dtype
        self.dtype = weight.dtype
        # backup shape
        self.shape = weight.shape
        # set default padding value
        self.padding = 0
         # reshape weight to (-1, fisher_block_size)
        self.weight = self._prepare_weight(weight)
        # flag if pre step was performed
        self.pre_step_completed = False

    @staticmethod
    def _validate_fisher(F: Tensor) -> None:
        assert F.ndim == 2, "F has to be 2-dimensional tensor"
        assert F.shape[0] == F.shape[1]

    def _prepare_weight(self, weight: Tensor) -> Tensor:
        if weight.numel() % self.fisher_block_size != 0:
            self.padding = self.fisher_block_size - weight.numel() % self.fisher_block_size
            weight = pad(weight.view(-1), (0, self.padding))
        return weight.reshape(-1, self.fisher_block_size)

    def _reshape_to_orig_shape(self, weight: Tensor) -> Tensor:
        # remove padding
        if self.padding != 0:
            weight = weight.view(-1)[:-self.padding]
        return weight.reshape(self.shape).to(self.dtype)

    @torch.no_grad()
    def prepare_data(self):
        w = self.weight.clone()
        # get columns with all zeros
        zero_cols = torch.nonzero(w.eq(0).all(dim=0))
        # get number of zero structures
        num_zeros = len(zero_cols)
        F = self.F
        # mask rows with zero input channels
        F[zero_cols, :] = 0
        F[:, zero_cols] = 0
        F[zero_cols, zero_cols] = 1
        # invert
        F = inv_sym(F)
        F_inv_cho = torch.linalg.cholesky(F, upper=True)
        return w, F_inv_cho, num_zeros

    @torch.no_grad()
    def pruning_pre_step(self) -> None:
        """
        Preparatory step with hessian regularization and weight reshaping.
        """
        assert self.F is not None, \
            "One has to process at least one sample of calibration data to run pruning"
        # get ids of pruned channels
        pruned_ids = torch.diag(self.F) == 0
        self.F[pruned_ids, pruned_ids] = 1
        # Hessian regularization
        damp = self.rel_damp * torch.diag(self.F).mean()
        self.F.add_(torch.eye(self.F.shape[0], device=self.F.device), alpha=damp)
        self.weight[:, pruned_ids] = 0
        # flag pre step as completed
        self.pre_step_completed = True

    @torch.no_grad()
    def prune_unstructured(self, sparsities: SparsityQuery) -> Dict[str, Tensor]:
        fisher_block_size, block_size = self.fisher_block_size, self.block_size

        sparse_weights = {}
        for sparsity in sparsities:
            # prepare weight and Cholesky of F^{-1}
            w, F_inv_cho, nzeros = self.prepare_data()
            # iterate over columns
            for c1 in range(nzeros, fisher_block_size, block_size):
                c2 = min(c1 + block_size, fisher_block_size)
                ncols = c2 - c1 # number of columns
                w_blk = w[:, c1:c2].clone() # column-wise weight slice
                res = torch.zeros_like(w_blk)
                errs = torch.zeros_like(w_blk)
                losses_blk = torch.zeros_like(w_blk)
                F_inv_cho_blk = F_inv_cho[c1:c2, c1:c2]
                # 1) score computation
                scores = w_blk ** 2 / F_inv_cho_blk.diag().reshape(1, -1) ** 2
                thr, _ = torch.kthvalue(scores.view(-1), round(w_blk.numel() * sparsity))
                mask = scores > thr
                # 2) iterate over block
                for i in range(ncols):
                    w_ci = w_blk[:, i]
                    d = F_inv_cho_blk[i, i]

                    q = w_ci.clone()
                    q[~mask[:, i]] = 0

                    res[:, i] = q
                    err = (w_ci - q) / d
                    losses_blk[:, i] = err ** 2
                    
                    w_blk[:, i:].addr_(err, F_inv_cho_blk[i, i:], alpha=-1)
                    errs[:, i] = err
                # 3) update the weights after block
                w[:, c1:c2] = res
                w[:, c2:].addmm_(errs, F_inv_cho[c1:c2, c2:], alpha=-1)

            sparse_weights[sparsity] = self._reshape_to_orig_shape(w)
        
        return sparse_weights

    def pruning_post_step(self):
        """
        Free all allocated data.
        """
        self.F = None
        self.weight = None
        self.reset()
        torch.cuda.empty_cache()

    # compression methods
    @torch.no_grad()
    def prune(self, sparsity_type: str, sparsities: SparsityQuery, **prune_kw) -> Dict[float, Tensor]:
        """
        """
        self._validate_sparsity(sparsity_type, sparsities)
        self.pruning_pre_step()
        # convert to single element list
        if isinstance(sparsities, (float, str)):
            sparsities = [sparsities]
        if sparsity_type == "unstructured":
            sparse_weights = self.prune_unstructured(sparsities)
        else:
            raise ValueError(f"Unknown `sparsity_type` {sparsity_type}")
        self.pruning_post_step()
        return sparse_weights

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

    def _validate_semistructured(self, sparsity: str) -> None:
        assert len(sparsity.split(':')), "Sparsity should be of form`N:M`"
        n, m = sparsity.split(':')
        assert n.isdigit() and m.isdigit(), "`N:M` have to be integer numbers"

    def _validate_float(self, sparsity: float) -> None:
        assert 0 <= sparsity <= 1, "Sparsity has to be in [0, 1] range"

    def reset(self):
        pass
