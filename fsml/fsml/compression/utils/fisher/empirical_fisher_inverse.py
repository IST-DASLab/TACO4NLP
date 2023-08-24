# adopted from https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/sparsification/pruning/modifier_pruning_obs.py#L563
import math
import torch
import torch.nn.functional as F
from torch import Tensor


__all__ = [
    "EmpiricalBlockFisherInverse",
    "EmpiricalBlockFisherReduced"
]


class EmpiricalBlockFisherInverse:

    def __init__(
        self,
        num_grads: int,
        block_size: int,
        num_weights: int,
        damp: float,
        device: torch.device
    ):
        self.num_grads = num_grads
        self.block_size = block_size
        self.d = num_weights
        self.damp = damp
        self.device = device

        self.num_blocks = math.ceil(self.d / self.block_size)
        self.F_inv = (
            (1.0 / self.damp * torch.eye(n=self.block_size, device=self.device))
            .unsqueeze(0)
            .repeat(self.num_blocks, 1, 1)
        )  # O(d x B) memory

    def update(self, g: Tensor) -> None:
        """
        Rank-1 update of the Fisher inverse.

        Args:
            g: gradient for the given parameter
        """
        # flatten grad
        g = g.flatten()
        # if 'd / B' is not integer, pad with zeros for batch calculations
        if g.numel() < self.num_blocks * self.block_size:
            g = F.pad(g, (0, self.num_blocks * self.block_size - g.numel()))
        # prepare grad for batch calculations
        g = g.view(self.num_blocks, self.block_size)
        # batched Fs_inv x g: (n_B, B, B) x (n_B, B) -> (n_B, B)
        F_inv_g = torch.bmm(self.F_inv, g.unsqueeze(-1)).squeeze(-1)
        # scalar denominator for each block (n_B)
        denom = (self.num_grads + (g * F_inv_g).sum(dim=-1)).sqrt().unsqueeze(1)
        F_inv_g.div_(denom)
        # update inv_blocks with new outer product: (n_B, B) x (n_B, B) -> (n_B, B, B)
        self.F_inv.baddbmm_(F_inv_g.unsqueeze(2), F_inv_g.unsqueeze(1), alpha=-1)

    def diag(self) -> Tensor:
        """
        :return: diagonal of the Fisher inverse matrix
        """
        return self.F_inv.diagonal(dim1=1, dim2=2).flatten()[:self.d]

    def mul(self, v: Tensor) -> Tensor:
        """
        Computes matrix-vector product of the Fisher inverse matrix and a vector

        Args:
            v: a vector to compute matrix-vector product with

        Returns: 
            Result of the matrix-vector multiplication
        """
        if v.numel() < self.num_blocks * self.block_size:
            v = F.pad(v, (0, self.num_blocks * self.block_size - v.numel()))
        return torch.bmm(
            self.F_inv, v.view(self.num_blocks, self.block_size).unsqueeze_(2)
        ).flatten()[:self.d]


class EmpiricalBlockFisherReduced:

    def __init__(
        self,
        block_size: int,
        device: torch.device
    ):
        self.block_size = block_size
        self.device = device
        self.F = torch.zeros(
            (self.block_size, self.block_size), dtype=torch.float32, device=self.device
        ) # O(B x B) memory
        self.grads_collected = 0

    def update(self, g: Tensor) -> None:
        """
        Rank-1 update of the Fisher inverse.

        Args:
            g: gradient for the given parameter
        """
        # truncate if 'd / B' is not integer
        if g.numel() % self.block_size != 0:
            g = g[:-(g.numel() % self.block_size)]
        # reshape grad and cast to float32 before addition
        g = g.view(-1, self.block_size).float()
        # hessian update
        beta = self.grads_collected / (self.grads_collected + 1)
        alpha = 2.0 / (self.grads_collected + 1)
        self.F.addmm_(g.T, g, beta=beta, alpha=alpha)
        # update number of collected samples
        self.grads_collected += 1
