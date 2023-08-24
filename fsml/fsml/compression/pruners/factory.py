import yaml
import inspect
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Type

from .base import *
from .magnitude import *
from .obc import *
from .woodfisher import *
from .cap import *
from .transformers import *
from .diffusers import *


__all__ = [
    'parse_pruner_config',
    'create_pruner',
    'create_pruner_from_config'
]


PRUNER_REGISTRY = {
    'MagnitudePruner': MagnitudePruner,
    'CorrelationAwarePruner': CorrelationAwarePruner,
    'CorrelationAwarePrunerForTransformers': CorrelationAwarePrunerForTransformers,
    'OBCPruner': OBCPruner,
    'OBCPrunerForCausalLM': OBCPrunerForCausalLM,
    'OBCPrunerForMaskedLM': OBCPrunerForMaskedLM,
    'OBCPrunerForSeq2SeqLM': OBCPrunerForSeq2SeqLM,
    'OBCPrunerForUNet2D': OBCPrunerForUNet2D,
    "FastCorrelationAwarePruner": FastCorrelationAwarePruner,
    'FastOBCPruner': FastOBCPruner,
    'FastOBCPrunerForCausalLM': FastOBCPrunerForCausalLM,
    'FastOBCPrunerForMaskedLM': FastOBCPrunerForMaskedLM,
    'FastOBCPrunerForSeq2SeqLM': FastOBCPrunerForSeq2SeqLM,
    'FastOBCPrunerForUNet2D': FastOBCPrunerForUNet2D,
    'StructOBCPruner': StructOBCPruner,
    'StructOBCPrunerForCausalLM': StructOBCPrunerForCausalLM,
    'StructOBCPrunerForMaskedLM': StructOBCPrunerForMaskedLM,
    'StructOBCPrunerForSeq2SeqLM': StructOBCPrunerForSeq2SeqLM,
    'StructOBCPrunerForUNet2D': StructOBCPrunerForUNet2D,
    'WoodFisherPruner': WoodFisherPruner,
    'WoodFisherPrunerForTransformers': WoodFisherPrunerForTransformers,
}


def parse_pruner_config(config_path: str) -> List[Tuple[Type, Dict[str, Any]]]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    pruner_data = []
    for pruner_class_name, pruner_kwargs in config.items():
        assert PRUNER_REGISTRY.get(pruner_class_name), f"Unkwown pruner class {pruner_class_name}"
        pruner_data.append((PRUNER_REGISTRY[pruner_class_name], pruner_kwargs))
    return pruner_data


def create_pruner(model: nn.Module, pruner_class: Type, pruner_kwargs: Dict[str, Any] = {}) -> BasePruner:
    # check kwargs
    for kwarg_name in pruner_class._required_kwargs:
        assert pruner_kwargs.get(kwarg_name), f"{kwarg_name} is required to create {pruner_class.__name__}"
    constructor_signature = inspect.signature(pruner_class.__init__).parameters
    pruner_kwargs = {k: v for k, v in pruner_kwargs.items() if constructor_signature.get(k)}
    return pruner_class(model, **pruner_kwargs)


def create_pruner_from_config(
    model: nn.Module, 
    config_path: str, 
    required_pruner_kwargs: Dict[str, Any] = {}
) -> BasePruner:
    """
    Function to create pruner for a given model given config and pruner_kwargs.
    
    Args:
        model: model to prune
        config_path: path to pruner config
        required_pruner_kwargs: dict with arguments to init pruner
    """
    pruner_class, pruner_kwargs = parse_pruner_config(config_path)[0]
    pruner_kwargs = {**pruner_kwargs, **required_pruner_kwargs}
    return create_pruner(model, pruner_class, pruner_kwargs)
