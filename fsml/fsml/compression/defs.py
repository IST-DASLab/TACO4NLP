from typing import Union, List

__all__ = ["Sparsity", "Bitwidth"]

Sparsity = Union[float, str]
Bitwidth = int

SparsityQuery = Union[Sparsity, List[Sparsity]]
BitwidthQuery = Union[Bitwidth, List[Bitwidth]]
