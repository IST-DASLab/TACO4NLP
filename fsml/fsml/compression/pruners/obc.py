from typing import List, Dict, Optional, Iterable, Union, Tuple, Any

import torch
import torch.nn as nn


from .base import BasePruner
from ..utils.obc import OBCUtil, FastOBCUtil, StructOBCUtil
from ...utils.model import select_layers, LINEAR_LAYERS
from ...utils.common import to


__all__ = [
    "OBCPruner", 
    "FastOBCPruner", 
    "StructOBCPruner"
]

class BaseOBCPruner(BasePruner):

    _required_kwargs = ('data_loader',)

    _obc_util_class = None

    def __init__(
        self,
        model: nn.Module,
        # obc requires data_loader
        data_loader: Iterable,
        sparsity_distribution: str = 'uniform',
        sparsity_type: str = 'unstructured',
        target_modules: Union[str, List[str]] = '.*',
        sparsity_block_size: int = 4,
        rel_damp: float = 1e-2,
        sequential: bool = False,
        device: Optional[str] = None,
        prune_kwargs: Dict[str, Any] = {},
        obc_util_kwargs: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            model, 
            sparsity_distribution,
            sparsity_type,
            is_module_pruner=True,
            target_params='.*',
            target_modules=target_modules,
            prune_biases=False,
            sparsity_block_size=sparsity_block_size,
        )
        self.data_loader = data_loader
        # OBC specific params
        self.rel_damp = rel_damp
        self.sequential = sequential
        # device to run pruning procedure on
        self.device = device
        # keyword arguments for util.prune()
        self.prune_kwargs = prune_kwargs
        # keyword arguments for util constructor
        self.obc_util_kwargs = obc_util_kwargs

    def _prepare_hooks_and_handles(self, layers) -> Tuple[Dict[str, Any]]:
        handles = {}
        hooks = {}
        for layer_name, layer in layers.items():

            def update_handle_hook(name):
                def _hook(_, inp, out):
                    handles[name].update(inp[0].data, out.data)

                return _hook

            handles[layer_name] = self._obc_util_class(
                layer, rel_damp=self.rel_damp, **self.obc_util_kwargs
            )
            hooks[layer_name] = layer.register_forward_hook(
                update_handle_hook(layer_name)
            )
        return handles, hooks

    def _prune_group(self, handles: List[Any], sparsity: float, device: torch.device) -> None:
        for handle_name, handle in handles.items():
            # TODO add error estimate
            sparse_weight = handle.prune(self.sparsity_type, [sparsity], **self.prune_kwargs)[sparsity].to(device)
            # update param
            self.params[f'{handle_name}.weight'].data = sparse_weight
            # update masks
            self.param_masks[f'{handle_name}.weight'] = sparse_weight.ne(0)
            handle.reset()

    @torch.no_grad()
    def prune_sequential(self, sparsity: float) -> None:
        raise NotImplementedError("This option is available only in subclasses")

    @torch.no_grad()
    def prune_parallel(self, sparsity: float) -> None:
        device = self.device or next(self.model.parameters()).device
        self.model = self.model.to(device)

        # find layers
        layers = select_layers(self.model, '', self.target_modules, LINEAR_LAYERS)
        handles, hooks = self._prepare_hooks_and_handles(layers)

        for (inp_args, inp_kwargs) in self.data_loader:
            self.model(*to(inp_args, device=device), **to(inp_kwargs, device=device))

        for _, h in hooks.items():
            h.remove()

        self._prune_group(handles, sparsity, device)

    @torch.no_grad()
    def prune(self, sparsity: float) -> None:
        if self.sequential:
            self.prune_sequential(sparsity)
        else:
            self.prune_parallel(sparsity)


class FastOBCPruner(BaseOBCPruner):

    _supported_sparsity_distributions = ("uniform",)
    _supported_sparsity_types = ("unstructured",)

    _obc_util_class = FastOBCUtil

    def __init__(
        self,
        model: nn.Module,
        # obc requires data_loader
        data_loader: Iterable,
        sparsity_distribution: str = 'uniform',
        sparsity_type: str = 'unstructured',
        target_modules: Union[str, List[str]] = '.*',
        rel_damp: float = 1e-2,
        sequential: bool = False,
        device: Optional[str] = None,
        block_size: Optional[int] = None
    ) -> None:
        super().__init__(
            model,
            data_loader,
            sparsity_distribution=sparsity_distribution,
            sparsity_type=sparsity_type,
            target_modules=target_modules,
            rel_damp=rel_damp,
            sequential=sequential,
            device=device,
            obc_util_kwargs={'block_size': block_size}
        )


class OBCPruner(BaseOBCPruner):

    _supported_sparsity_distributions = ("uniform",)
    _supported_sparsity_types = ("unstructured", "blocked")

    _obc_util_class = OBCUtil

    def __init__(
        self,
        model: nn.Module,
        # obc requires data_loader
        data_loader: Iterable,
        sparsity_distribution: str = 'uniform',
        sparsity_type: str = 'unstructured',
        target_modules: Union[str, List[str]] = '.*',
        sparsity_block_size: int = 4,
        rel_damp: float = 1e-2,
        sequential: bool = False,
        device: Optional[str] = None,
        rows_in_parallel: Optional[int] = None,
    ) -> None:
        super().__init__(
            model,
            data_loader,
            sparsity_distribution=sparsity_distribution,
            sparsity_type=sparsity_type,
            target_modules=target_modules,
            sparsity_block_size=sparsity_block_size,
            rel_damp=rel_damp,
            sequential=sequential,
            device=device,
            obc_util_kwargs={'rows_in_parallel': rows_in_parallel},
            prune_kwargs={'block_size': sparsity_block_size}
        )


# TODO add custom prune config option
class StructOBCPruner(BaseOBCPruner):

    _supported_sparsity_distributions = ("uniform",)
    _supported_sparsity_types = ("structured",)

    _obc_util_class = StructOBCUtil

    def __init__(
        self,
        model: nn.Module,
        # obc requires data_loader
        data_loader: Iterable,
        sparsity_distribution: str = 'uniform',
        sparsity_type: str = 'unstructured',
        target_modules: Union[str, List[str]] = '.*',
        sparsity_block_size: int = 4,
        rel_damp: float = 1e-2,
        sequential: bool = False,
        device: Optional[str] = None,
        struct_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            model,
            data_loader,
            sparsity_distribution=sparsity_distribution,
            sparsity_type=sparsity_type,
            target_modules=target_modules,
            sparsity_block_size=sparsity_block_size,
            rel_damp=rel_damp,
            sequential=sequential,
            device=device,
            obc_util_kwargs={'struct_size': struct_size},
        )
