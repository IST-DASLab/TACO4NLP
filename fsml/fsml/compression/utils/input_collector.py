import torch.nn as nn


__all__ = [
    "ForwardInterrupt", 
    "InputCollector"
]


class ForwardInterrupt(Exception):
    pass


class InputCollector(nn.Module):
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.input_args = []
        self.input_kwargs = []

    def forward(self, *input_args, **input_kwargs):
        """
        Assumes that the wrapped module has a single 
        input that can reside in inputs or input_kwargs.
        """
        self.input_args.append(input_args)
        self.input_kwargs.append(input_kwargs)
        raise ForwardInterrupt
        