from typing import List
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


__all__ = [
    "BaseQuantizer"
]


class BaseQuantizer(ABC):

    def __init__(
        self,
        model: nn.Module,
        param_regex: str = '.*',
        **quantizer_kwargs
    ) -> None:
        super().__init__()
        self.model = model
        self.param_regex = param_regex

    @abstractmethod
    @torch.no_grad()
    def quantize(self, bits: int) -> None:
        pass
    