import dataclasses
from typing import Any, Union, List

from torch import Tensor


__all__ = [
    "to", 
    "as_list"
]


def to(data: Any, *args, **kwargs):
    '''
    # adopted from https://github.com/Yura52/delu/blob/main/delu/_tensor_ops.py
    TODO
    '''
    def _to(x):
        return to(x, *args, **kwargs)

    if isinstance(data, Tensor):
        return data.to(*args, **kwargs)
    elif isinstance(data, (tuple, list, set)):
        return type(data)(_to(x) for x in data)
    elif isinstance(data, dict):
        return type(data)((k, _to(v)) for k, v in data.items())
    elif dataclasses.is_dataclass(data):
        return type(data)(**{k: _to(v) for k, v in vars(data).items()}) 
    # do nothing if provided value is not tensor or collection of tensors
    else:
        return data

def as_list(x: Union[str, List[str]]):
    if x is None:
        return x
    if isinstance(x, (list, tuple)):
        return x
    return [x]
