import torch

from torch import Tensor


__all__ = [
    "extract_diagonal_blocks",
    "split_along_dim"
]


def extract_diagonal_blocks(X: Tensor, block_size: int) -> Tensor:
    assert len(X.shape) >= 2, "X has to be at least 2-dimensional tensor"
    assert X.shape[-2] == X.shape[-1], "last two dimensions have to be equal"
    assert X.shape[-1] % block_size == 0, "block size has to divide last dimension"
    nbr = X.shape[-1] // block_size
    diag_ids = torch.arange(nbr, device=X.device)
    X = X.view(*X.shape[:-2], nbr, block_size, nbr, block_size)
    return X[..., diag_ids, :, diag_ids, :].reshape(-1, block_size, block_size)


def split_along_dim(x: Tensor, dim: int, groupsize: int):
    assert x.shape[dim] >= groupsize
    assert x.shape[dim] % groupsize == 0
    return x.view(*x.shape[:dim], -1, groupsize, *x.shape[dim + 1:])
    