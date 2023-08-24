from typing import Any

import torch

from ..woodfisher import WoodFisherPruner
from ....utils.common import to

__all__ = [
    "WoodFisherPrunerForTransformers"
]

class WoodFisherPrunerForTransformers(WoodFisherPruner):

    @torch.enable_grad()
    def prepare_grad(self, batch: Any, device: str) -> None:
        """
        """
        input_args, input_kwargs = batch
        input_args = to(input_args, device=device)
        input_kwargs = to(input_kwargs, device=device)
        loss = self.model(*input_args, **input_kwargs).loss
        loss.backward()