import yaml
import inspect
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Type, Union

from .base import *
from .constant import *
from .magnitude import *
from .obc import *
from .woodfisher import *
from .cap import *
from .transformers import *
from .diffusers import *


__all__ = [
    'parse_pruner_config',
    'create_pruner',
    'create_pruners_from_config',
    'create_pruners_from_yaml'
]


PRUNER_REGISTRY = {
    'ConstantPruner': ConstantPruner,
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


def parse_pruner_config(config: List[dict]) -> List[Tuple[Type, Dict[str, Any]]]:
    pruner_data = []
    for pruner_config in config:
        assert len(pruner_config) == 1, 'Pruner config has a format `pruner_name: {**params}`'
        pruner_class_name, pruner_kwargs = next(iter(pruner_config.items()))
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


def create_pruners_from_config(
    model: nn.Module, 
    config: dict, 
    required_pruner_kwargs: Dict[str, Any] = {},
    override_kwargs: Dict[str, Any] = {},
) -> List[BasePruner]:
    """
    Function to create pruner for a given model given config and pruner_kwargs.
    
    Args:
        model: model to prune
        config_path: path to pruner config
        required_pruner_kwargs: dict with arguments to init pruner
        override_kwargs: dict of kwargs to override if provided
    """
    parsed_config = parse_pruner_config(config)
    pruners = []
    for pruner_class, pruner_kwargs in parsed_config:
        # override some kwargs, if desided
        for key, value in override_kwargs.items():
            pruner_kwargs[key] = value
        pruner_kwargs = {**pruner_kwargs, **required_pruner_kwargs}
        pruners.append(create_pruner(model, pruner_class, pruner_kwargs))
    return pruners

def create_pruners_from_yaml(
    model: nn.Module, 
    config_path: str, 
    required_pruner_kwargs: Dict[str, Any] = {},
    override_kwargs: Dict[str, Any] = {},
) -> List[BasePruner]:
    """
    Function to create pruner for a given model given config and pruner_kwargs.
    
    Args:
        model: model to prune
        config_path: path to pruner config
        required_pruner_kwargs: dict with arguments to init pruner
        override_kwargs: dict of kwargs to override if provided
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # handle case of single pruner
    if not isinstance(config, (tuple, list)):
        config = [config]
    return create_pruners_from_config(model, config, required_pruner_kwargs, override_kwargs)
