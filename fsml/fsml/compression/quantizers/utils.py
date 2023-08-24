from typing import Optional

import torch
from torch import Tensor

# TODO add activation quantization


__all__ = [
    "QuantUtil",
    "quantize_fn",
    "get_scale_and_zero"
]


def quantize_fn(x: Tensor, scale: Tensor, zero: Tensor, qrange: int):
    q = torch.clamp(torch.round(x / scale) + zero, 0, qrange)
    return scale * (q - zero)


def get_scale_and_zero(w_max: Tensor, w_min: Tensor, qrange: int, sym: bool = False):
    scale = (w_max - w_min) / qrange
    if sym:
        zero = torch.full_like(scale, (qrange + 1) / 2)
    else:
        zero = torch.round(-w_min / scale)
    return scale, zero


def split_along_dim(x: Tensor, dim: int, groupsize: int):
    assert x.shape[dim] >= groupsize
    assert x.shape[dim] % groupsize == 0
    return x.view(*x.shape[:dim], -1, groupsize, *x.shape[dim + 1:])


class QuantUtil:

    def __init__(
        self, 
        weight: Tensor,
        bits: int,
        # TODO can we need groups spanning several dimension?
        dim: Optional[int] = None,
        # for this purpose len(grousize) == len(dim)
        groupsize: int = -1,
        sym: bool = False,
        flatten: bool = True,
        optimize_grid: bool = False,
        # TODO suggest better name
        min_scale: float = 0.8,
        optimization_iters: int = 100,
        # TODO add permutation
    ):
        if dim:
            assert 0 <= dim < weight.ndim
        self.weight = weight
        self.bits = bits
        self.dim = dim
        self.groupsize = groupsize
        self.sym = sym
        self.flatten = flatten
        # grid optimization params
        self.optimize_grid = optimize_grid
        self.min_scale = min_scale
        self.optimization_iters = optimization_iters
        # quantization params
        self.qrange = 2 ** bits - 1
        self.scale = None
        self.zero = None
        # init params
        self._init_params()

    def _init_params(self):
        w = self.weight
        # if weight is conv kernel reshape to (d_out, d_in \prod_i k_i)
        if self.flatten and w.ndim > 2:
            w = w.flatten(start_dim=1, end_dim=-1)
        if self.dim is None:
            w = w.flatten()
        # split among dim if groupsize != -1
        else:
            dim = self.dim
            if self.groupsize != -1:
                assert self.groupsize <= w.shape[self.dim]
                w = split_along_dim(w, self.dim, self.groupsize)
                dim += 1

        if self.dim is not None: 
            w_min = w.min(dim=dim, keepdim=True).values
            w_max = w.max(dim=dim, keepdim=True).values
        else:
            w_min = w.min()
            w_max = w.max()

        # symmetrize bounds
        if self.sym:
            w_max = torch.maximum(w_min.abs(), w_max)
            w_min = torch.where(w_min < 0, -w_max, w_min)

        scale, zero = get_scale_and_zero(w_max, w_min, self.qrange, self.sym)

        # search for optimal grid
        if self.optimize_grid:
            best_err = torch.full_like(w_max, float('inf'), device=w.device)
            for scale_mult in torch.linspace(self.min_scale, 1, self.optimization_iters):
                new_scale, new_zero = get_scale_and_zero(
                    scale_mult * w_max, 
                    scale_mult * w_min, self.qrange, self.sym
                )
                q = quantize_fn(w, new_scale, new_scale, self.qrange)
                err = torch.sum((w - q) ** 2, dim=dim, keepdim=True)

                best_err_mask = err < best_err
                if best_err.any():
                    best_err[best_err_mask] = err[best_err_mask]
                    scale[best_err_mask] = new_scale[best_err_mask]
                    zero[best_err_mask] = new_zero[best_err_mask]

        # expand scale and zero to the shape of weight
        if self.dim is None:
            scale = torch.full_like(self.weight, scale)
            zero = torch.full_like(self.weight, zero)
        else:
            num_repeats = self.groupsize if self.groupsize > 0 else w.shape[self.dim]
            scale = scale.repeat_interleave(num_repeats, dim).reshape_as(self.weight)
            zero = zero.repeat_interleave(num_repeats, dim).reshape_as(self.weight)
        self.scale, self.zero = scale, zero
            
    # TODO separate into quantize and dequantize fn
    def __call__(self, x: Tensor, idx: int = None, dim: int = None):
        if idx is None:
            return quantize_fn(x, self.scale, self.zero, self.qrange)
        else:
            assert dim is not None
            return quantize_fn(
                x,
                self.scale.select(dim, idx),
                self.zero.select(dim, idx),
                self.qrange
            )
