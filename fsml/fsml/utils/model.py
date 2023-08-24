from typing import List, Union, Dict, Optional

import re
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd


__all__ = [
    'LINEAR_LAYERS',
    'select_layers'
]


LINEAR_LAYERS = (nn.Linear, _ConvNd)


def select_layers(
    model: nn.Module, 
    layer_prefix: Optional[str] = '',
    layer_regex: str = '.*', 
    layer_classes: Union[nn.Module, List[nn.Module]] = nn.Module
) -> Dict[str, nn.Module]:
    layers = {}
    for layer_name, layer in model.named_modules():
        if isinstance(layer, layer_classes) and re.search(layer_regex, layer_name) and layer_name.startswith(layer_prefix):
            layers[layer_name] = layer
    return layers
