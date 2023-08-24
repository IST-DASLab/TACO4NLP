from typing import List, Type, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseQuantizer
from ..utils import InputCollector, ForwardInterrupt
from ..utils.obc import OBCUtil, FastOBCUtil
from ...utils.model import select_layers, LINEAR_LAYERS
from ...utils.common import as_list, to, as_list


__all__ = [
    'OBCQuantizer',
    'FastOBCQuantizer',
]


# NOTE doesn't quantize embedding layers and lm head
class BaseOBCQuantizer(BaseQuantizer):

    def __init__(
        self,
        model: nn.Module,
        data: List[Tensor],
        obc_wrapper_class: Type,
        layer_regex: str = '.*',
        rel_damp: float = 1e-2,
        dim: Optional[int] = None,
        groupsize: int = -1,
        sym: bool = False,
        flatten: bool = True,
        optimize_grid: bool = False,
        min_scale: float = 0.8,
        optimization_iters: int = 100,
        sequential: bool = False,
        encoder_blocks: Union[str, List[str]] = None,
        pre_encoder_modules: Union[str, List[str]] = None,
        post_encoder_modules: Union[str, List[str]] = None,
        decoder_blocks: Union[str, List[str]] = None,
        pre_decoder_modules: Union[str, List[str]] = None,
        cpu_offload: bool = False,
        device: str = None,
        **obc_wrapper_kwargs
    ) -> None:
        # we select weights in layerwise fashions
        super().__init__(model, '.*')
        self.data = data
        # set obc wrapper class
        self.obc_wrapper_class = obc_wrapper_class
        # generic obc kwargs
        self.rel_damp = rel_damp
        # method specific kwargs
        self.obc_wrapper_kwargs = obc_wrapper_kwargs
        # quantization kwargs
        self.quantizer_kwargs = dict(
            dim=dim,
            sym=sym,
            groupsize=groupsize,
            flatten=flatten,
            optimize_grid=optimize_grid,
            min_scale=min_scale,
            optimization_iters=optimization_iters,
        )
        # whether to quantize in parallel or sequentially
        self.sequential = sequential
        # name of the sequential module for sequential pruning
        self.encoder_blocks = encoder_blocks
        self.pre_encoder_modules = as_list(pre_encoder_modules)
        self.post_encoder_modules =  as_list(post_encoder_modules)
        self.decoder_blocks = decoder_blocks
        self.pre_decoder_modules =  as_list(pre_decoder_modules)
        self.cpu_offload = cpu_offload
        self.layer_regex = layer_regex
        self.device = device

    def _get_blocks(self, block_spec: Union[str, List[str]]) -> List[nn.Module]:
        if isinstance(block_spec, list):
            return [self.model.get_submodule(block_str) for block_str in block_spec]
        else:
            return self.model.get_submodule(block_spec)

    @torch.no_grad()
    def quantize_sequential(self, bits: int) -> None:
        assert self.encoder_blocks is not None
        device = self.device or next(self.model.parameters()).device
           
        blocks = self._get_blocks(self.encoder_blocks)
        blocks[0] = blocks[0].to(device)
        if self.cpu_offload:
            assert self.pre_encoder_modules is not None
            # load input embeddings or any other preprocessing step
            for module_name in self.pre_encoder_modules:
                module = self.model.get_submodule(module_name)
                module.to(device)

        ### Input preparation ### 
        
        blocks[0] = InputCollector(blocks[0])
        for batch in self.data:
            try:
                self.model(to(batch, device=device))
            # TODO create special exception
            except ForwardInterrupt:
                pass
        input_args = blocks[0].input_args
        input_kwargs = blocks[0].input_kwargs
        blocks[0] = blocks[0].module

        if self.cpu_offload:
            # offload input embeddings or any other preprocessing step
            for module_name in self.pre_encoder_modules:
                module = self.model.get_submodule(module_name)
                module.cpu()

        ### Encoder pruning ### 

        # TODO add logging
        for block_id, block in enumerate(blocks):
            print(f"Processing encoder block {block_id}.")
            block = block.to(device)
            layers = select_layers(block, self.layer_regex, LINEAR_LAYERS)
            handles = {}
            hooks = {}
            for layer_name, layer in layers.items():

                def update_handle_hook(name):
                    def _hook(_, inp, out):
                        handles[name].update(inp[0].data, out.data)
                    return _hook

                handles[layer_name] = self.obc_wrapper_class(
                    layer, rel_damp=self.rel_damp, **self.obc_wrapper_kwargs
                )
                hooks[layer_name] = layer.register_forward_hook(
                    update_handle_hook(layer_name)
                )

            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                block(*inp_args, **inp_kwargs)

            for _, h in hooks.items():
                h.remove()

            for _, handle in handles.items():
                # TODO add error estimate
                quantized_weights = handle.quantize(
                    [bits], **self.quantizer_kwargs
                )
                handle.layer.weight.data = quantized_weights[bits].to(device)
                handle.reset()

            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                out = block(*inp_args, **inp_kwargs)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                # change only first input argument
                inp_args[0].data = out

            if self.cpu_offload:
                block = block.cpu()

            del handles
            del hooks
            torch.cuda.empty_cache()

        
        for block_id, block in enumerate(blocks):
            print(f"Processing decoder block {block_id}.")
            inputs = [inp_args[0] for inp_args in input_args]
            ### Hidden state extraction
            for module_name in self.post_encoder_modules:
                module = self.model.get_submodule(module_name)
                if self.cpu_offload:
                    module = module.to(device)
                for inp in inputs:
                    inp.data = module(inp)
                if self.cpu_offload:
                    module = module.cpu()

            encoder_hidden_states = inputs
            blocks = self._get_blocks(self.decoder_blocks)

            if self.cpu_offload:
                assert self.pre_decoder_modules is not None
                # load input embeddings or any other preprocessing step
                for module_name in self.pre_decoder_modules:
                    module = self.model.get_submodule(module_name)
                    module.to(device)

            blocks[0] = InputCollector(blocks[0])
            blocks[0] = blocks[0].to(device)
            # extract decoder TODO add regex?
            decoder = self.model.get_submodule(self.decoder_blocks.rsplit('.', 1)[0])
            for batch_id, batch in enumerate(self.data):
                try:
                    if isinstance(batch, Tensor):
                        decoder_input_ids = batch.to(device)
                    elif isinstance(batch, dict):
                        decoder_input_ids = batch['decoder_input_ids'].to(device)
                    else:
                        raise RuntimeError("Unsupported input type.")

                    decoder(
                        input_ids=decoder_input_ids, 
                        encoder_hidden_states=encoder_hidden_states[batch_id],
                    )
                except ValueError:
                    pass
            input_args = blocks[0].input_args
            input_kwargs = blocks[0].input_kwargs
            blocks[0] = blocks[0].module

            if self.cpu_offload:
                # offload input embeddings or any other preprocessing step
                for module_name in self.pre_decoder_modules:
                    module = self.model.get_submodule(module_name)
                    module.cpu()

            ### Decoder pruning ### 
            for block in blocks:
                block = block.to(device)
                layers = select_layers(block, self.layer_regex, LINEAR_LAYERS)
                handles = {}
                hooks = {}
                for layer_name, layer in layers.items():

                    def update_handle_hook(name):
                        def _hook(_, inp, out):
                            handles[name].update(inp[0].data, out.data)
                        return _hook

                    handles[layer_name] = self.obc_wrapper_class(
                        layer, rel_damp=self.rel_damp, **self.obc_wrapper_kwargs
                    )
                    hooks[layer_name] = layer.register_forward_hook(
                        update_handle_hook(layer_name)
                    )

                for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                    block(*inp_args, **inp_kwargs)

                for _, h in hooks.items():
                    h.remove()

                for _, handle in handles.items():
                    # TODO add error estimate
                    quantized_weights = handle.quantize(
                        [bits], **self.quantizer_kwargs
                    )
                    handle.layer.weight.data = quantized_weights[bits].to(device)
                    handle.reset()
                    handle.reset()

                for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                    out = block(*inp_args, **inp_kwargs)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    # change only first input argument
                    inp_args[0].data = out

                if self.cpu_offload:
                    block = block.cpu()

                del handles
                del hooks
                torch.cuda.empty_cache()

    @torch.no_grad()
    def quantize_parallel(self, bits: int) -> None:
        device = self.device or next(self.model.parameters()).device
        self.model = self.model.to(device)

        # find layers
        layers = select_layers(self.model, self.layer_regex, LINEAR_LAYERS)
        handles = {}
        hooks = {}
        
        for layer_name, layer in layers.items():

            def update_handle_hook(name):
                def _hook(_, inp, out):
                    handles[name].update(inp[0].data, out.data)
                return _hook

            handles[layer_name] = self.obc_wrapper_class(
                layer, rel_damp=self.rel_damp, **self.obc_wrapper_kwargs
            )
            hooks[layer_name] = layer.register_forward_hook(
                update_handle_hook(layer_name)
            )

        for batch in self.data:
            self.model(batch.to(device))

        for _, h in hooks.items():
            h.remove()

        for _, handle in handles.items():
            quantized_weights = handle.quantize(
                [bits], **self.quantizer_kwargs
            )
            handle.layer.weight.data = quantized_weights[bits].to(device)
            handle.reset()

    @torch.no_grad()
    def quantize(self, bits: int) -> None:
        if self.sequential:
            self.quantize_sequential(bits)
        else:
            self.quantize_parallel(bits)

class FastOBCQuantizer(BaseOBCQuantizer):

    def __init__(
        self,
        model: nn.Module,
        data: List[Tensor],
        layer_regex: str = '.*',
        dim: Optional[int] = None,
        groupsize: int = -1,
        sym: bool = False,
        flatten: bool = True,
        optimize_grid: bool = False,
        min_scale: float = 0.8,
        optimization_iters: int = 100,
        rel_damp: float = 1e-2,
        sequential: bool = False,
        encoder_blocks: Union[str, List[str]] = None,
        pre_encoder_modules: Union[str, List[str]] = None,
        post_encoder_modules: Union[str, List[str]] = None,
        decoder_blocks: Union[str, List[str]] = None,
        pre_decoder_modules: Union[str, List[str]] = None,
        cpu_offload: bool = False,
        device: str = None,
        block_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            model=model,
            data=data,
            obc_wrapper_class=FastOBCUtil,
            layer_regex=layer_regex,
            dim=dim,
            groupsize=groupsize,
            sym=sym,
            flatten=flatten,
            optimize_grid=optimize_grid,
            min_scale=min_scale,
            optimization_iters=optimization_iters,
            rel_damp=rel_damp,
            sequential=sequential,
            encoder_blocks=encoder_blocks,
            pre_encoder_modules=pre_encoder_modules,
            post_encoder_modules=post_encoder_modules,
            decoder_blocks=decoder_blocks,
            pre_decoder_modules=pre_decoder_modules,
            cpu_offload=cpu_offload,
            device=device,
            block_size=block_size
        )


class OBCQuantizer(BaseOBCQuantizer):

    def __init__(
        self,
        model: nn.Module,
        data: List[Tensor],
        layer_regex: str = '.*',
        dim: Optional[int] = None,
        groupsize: int = -1,
        sym: bool = False,
        flatten: bool = True,
        optimize_grid: bool = False,
        min_scale: float = 0.8,
        optimization_iters: int = 100,
        rel_damp: float = 1e-2,
        sequential: bool = False,
        encoder_blocks: str = None,
        pre_encoder_modules: str = None,
        post_encoder_modules: Union[str, List[str]] = None,
        decoder_blocks: str = None,
        pre_decoder_modules: str = None,
        cpu_offload: bool = False,
        device: str = None,
        rows_in_parallel: Optional[int] = None,
    ) -> None:
        super().__init__(
            model=model,
            data=data,
            obc_wrapper_class=OBCUtil,
            layer_regex=layer_regex,
            dim=dim,
            groupsize=groupsize,
            sym=sym,
            flatten=flatten,
            optimize_grid=optimize_grid,
            min_scale=min_scale,
            optimization_iters=optimization_iters,
            rel_damp=rel_damp,
            sequential=sequential,
            encoder_blocks=encoder_blocks,
            pre_encoder_modules=pre_encoder_modules,
            post_encoder_modules=post_encoder_modules,
            decoder_blocks=decoder_blocks,
            pre_decoder_modules=pre_decoder_modules,
            cpu_offload=cpu_offload,
            device=device,
            rows_in_parallel=rows_in_parallel
        )
