import numpy as np
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseOBCUtil
from ....utils.linalg import inv_sym


__all__ = ["StructOBCUtil"]

    
class StructOBCUtil(BaseOBCUtil):

    _supported_sparsity_types = ("structured")

    def __init__(
        self, 
        layer: nn.Module,
        rel_damp: float = 0.0,
        struct_size: Optional[int] = None,
    ) -> None:
        super().__init__(layer, rel_damp)
        self.struct_size = struct_size
        if not struct_size:
            self._set_default_struct_size()
        self.sparse_weights: Dict[float, Tensor] = {}

    def _set_default_struct_size(self):
        if isinstance(self.layer, nn.Linear):
            self.struct_size = 1
        else:
            # for convolutional layer default struct size is the product of kernel sizes
            self.struct_size = np.prod(self.layer.kernel_size)

    @torch.no_grad()
    def prepare_data(self, struct_size = 1):
        w = self.weight.clone()
        # create mask of already pruned weights
        if struct_size > 1:
            mask = w[0].reshape(self.d_col // struct_size, struct_size).ne(0).any(dim=-1)
            col_mask = mask.repeat_interleave(struct_size)
        else:
            mask = w.ne(0).any(dim=0)
            col_mask = mask
        # get number of zero structures
        num_zeros = (~mask).sum().item()
        H_inv = self.H.clone()
        # mask rows with zeroed weights
        H_inv.masked_fill_(~col_mask.unsqueeze(-1), 0)
        H_inv.masked_fill_(~col_mask.unsqueeze(-2), 0)
        H_inv.masked_fill_((~col_mask).diag_embed(dim1=-2, dim2=-1), 1)
        # invert
        H_inv = inv_sym(H_inv)
        return w, mask, H_inv, num_zeros
    
    def prepare_structured(self, sparsities: List[float]):
        if self.struct_size == 1:
            self.prepare_structured_col_single(sparsities)
        else:
            self.prepare_structured_col_multi(sparsities)

    @torch.no_grad()
    def prepare_structured_col_single(self, sparsities: List[float]):
        d_col, device, dtype = (
            self.d_col,
            self.weight.device,
            self.weight.dtype,
        )

        max_zero_col = round(max(sparsities) * d_col)
        zeros_to_sparsities = {
            round(sparsity * d_col): sparsity for sparsity in sparsities
        }

        w, mask, H_inv, num_zeros = self.prepare_data()
        # prepare losses TODO make useful
        self.losses = torch.zeros(len(sparsities), device=device, dtype=dtype)

        for col in range(num_zeros + 1, max_zero_col + 1):
            # 1) compure scores
            H_inv_d = torch.diag(H_inv)
            scores = (w ** 2 / H_inv_d).sum(dim=0)
            scores[~mask] = torch.inf
             # 2) mask selection
            p_id = scores.argmin(dim=0)
            mask[p_id] = False
            # 3) loss update
            self.losses += scores[p_id]
            # 4) weight update
            H_inv_pr = H_inv[p_id, :]
            H_inv_pd = H_inv_d[p_id]
            w -= H_inv_pr * (w[:, p_id] / H_inv_pd).unsqueeze(1)
            w[:, ~mask] = False
            # update weight database
            db_id = zeros_to_sparsities.get(col, None)
            if db_id:
                self.sparse_weights[db_id] = self._reshape_to_orig_shape(w.clone(), 'structured')               
            # 5) hessian update
            H_inv_pr.div_(H_inv_pd.sqrt())
            H_inv.addr_(H_inv_pr, H_inv_pr, alpha=-1)
        
    @torch.no_grad()
    def prepare_structured_col_multi(self, sparsities: List[float]):
        d_row, d_col, ss, device, dtype = (
            self.d_row,
            self.d_col,
            self.struct_size,
            self.weight.device,
            self.weight.dtype,
        )

        ns = d_col // ss
        s_ids = torch.arange(ns)
        row_ids = torch.arange(d_row)
        max_zero_str = round(max(sparsities) * ns)
        zeros_to_sparsities = {
            round(sparsity * ns): sparsity for sparsity in sparsities
        }

        w, mask, H_inv, num_zeros = self.prepare_data(ss)
        # prepare losses TODO make useful
        self.losses = torch.zeros(len(sparsities), device=device, dtype=dtype)
        # reshape w and H_inv  to convenient shape
        H_inv = H_inv.view(ns, ss, ns, ss).transpose(1, 2)
        w = w.view(-1, ns, ss, 1)

        for col in range(num_zeros + 1, max_zero_str + 1):
            # 1) compure scores
            H_inv_db = H_inv[None, s_ids, s_ids]  # shape (1, ns, ss, ss)
            inv_H_inv_db = inv_sym(H_inv_db) # shape (1, ns, ss, ss)
            inv_H_inv_db_w = inv_H_inv_db @ w # shape (d_row, ns, ss, 1)
            scores = (w * inv_H_inv_db_w).sum(dim=(0, 2, 3))
            scores[~mask] = torch.inf
             # 2) mask selection
            p_id = scores.argmin(dim=0)
            mask[p_id] = False
            # 3) loss update
            self.losses += scores[p_id]
            # 4) weight update
            H_inv_pr = H_inv[:, p_id] # shape (ns, ss, ss)
            inv_H_inv_pdb = inv_H_inv_db[:, p_id] # shape (ss, ss)
            w -= H_inv_pr[None, ...] @ inv_H_inv_db_w[row_ids, None, p_id]
            w[:, ~mask] = 0
            # update weight database
            db_id = zeros_to_sparsities.get(col, None)
            if db_id:
                self.sparse_weights[db_id] = self._reshape_to_orig_shape(w.clone(), 'structured')
            # 5) hessian update
            H_inv_pr1 = H_inv_pr.unsqueeze(0)
            H_inv_pr2 = H_inv_pr.unsqueeze(1)
            inv_H_inv_pdb = inv_H_inv_pdb.view(1, 1, ss, ss)
            H_inv -= H_inv_pr1 @ inv_H_inv_pdb @ H_inv_pr2
            # isolate pruned columns
            H_inv[p_id, :] = 0
            H_inv[:, p_id] = 0 
            H_inv[p_id, p_id] = torch.eye(ss, device=device, dtype=dtype)

    @torch.no_grad()
    def prune(self, sparsity_type: str, sparsities: List[float], **prune_kw) -> Dict[float, Tensor]:
        return super().prune(sparsity_type, sparsities, **prune_kw)

    def prune_structured(self, sparsities: List[float]):
        assert self.pre_step_completed
        self.prepare_structured(sparsities)
        return self.sparse_weights
    
    def reset(self):
        super().reset()
        self.sparse_weights = {}
    