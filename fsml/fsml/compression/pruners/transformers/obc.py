from typing import List, Dict, Optional, Iterable, Union, Any

import torch
import torch.nn as nn

from ..obc import BaseOBCPruner
from ...utils import InputCollector, ForwardInterrupt
from ...utils.obc import OBCUtil, FastOBCUtil, StructOBCUtil
from ....utils.model import select_layers, LINEAR_LAYERS
from ....utils.common import to, as_list


__all__ = [
    "OBCPrunerForCausalLM", 
    "OBCPrunerForMaskedLM", 
    "OBCPrunerForSeq2SeqLM", 
    "FastOBCPrunerForCausalLM", 
    "FastOBCPrunerForMaskedLM", 
    "FastOBCPrunerForSeq2SeqLM", 
    "StructOBCPrunerForCausalLM",
    "StructOBCPrunerForMaskedLM",
    "StructOBCPrunerForSeq2SeqLM",
]


class BaseOBCPrunerForAnyLM(BaseOBCPruner):

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
        cpu_offload: bool = False,
        # specification for sequential pruning
        encoder_blocks: str = None,
        pre_encoder_modules: Union[str, List[str]] = None,
        post_encoder_modules: Union[str, List[str]] = None,
        decoder_blocks: str = None,
        pre_decoder_modules: Union[str, List[str]] = None,
        post_decoder_modules: Union[str, List[str]] = None,
    ) -> None:
        super().__init__(
            model,
            data_loader,
            sparsity_distribution,
            sparsity_type,
            target_modules,
            sparsity_block_size,
            rel_damp,
            sequential,
            device,
            prune_kwargs,
            obc_util_kwargs
        )
        self.cpu_offload = cpu_offload
        self.encoder_blocks = encoder_blocks
        self.pre_encoder_modules = as_list(pre_encoder_modules)
        self.post_encoder_modules = as_list(post_encoder_modules)
        self.decoder_blocks = decoder_blocks
        self.pre_decoder_modules = as_list(pre_decoder_modules)
        self.post_decoder_modules = as_list(post_decoder_modules)

    def _get_blocks(self, blocks: str):
        return self.model.get_submodule(blocks)

    def _validate_blocks(self):
        raise NotImplementedError("This method is expected to be overriden in subclasses")

    @torch.no_grad()
    def prune_sequential(self, sparsity: float) -> None:
        self._validate_blocks()
        device = self.device or next(self.model.parameters()).device
        # get first stage blocks (either encoder or decoder)
        blocks_name = self.encoder_blocks or self.decoder_blocks
        pre_modules = self.pre_encoder_modules or self.pre_decoder_modules
        # prepare pre blocks modules
        blocks = self._get_blocks(blocks_name)
        blocks[0] = blocks[0].to(device)
        if self.cpu_offload:
            assert pre_modules is not None
            # load input embeddings or any other preprocessing step
            for module_name in pre_modules:
                module = self.model.get_submodule(module_name)
                module.to(device)

        ### Input preparation ###
        blocks[0] = InputCollector(blocks[0])
        # TODO make namedtuple
        for (inp_args, inp_kwargs) in self.data_loader:
            try:    
                self.model(
                    *to(inp_args, device=device),
                    **to(inp_kwargs, device=device),
                )
            except ForwardInterrupt:
                pass
        input_args = blocks[0].input_args
        input_kwargs = blocks[0].input_kwargs
        blocks[0] = blocks[0].module

        if self.cpu_offload:
            # offload input embeddings or any other preprocessing step
            for module_name in pre_modules:
                module = self.model.get_submodule(module_name)
                module.cpu()

        ### Encoder/Decoder pruning ###
        for block_id, block in enumerate(blocks):
            # TODO change to logging
            print(f"Processing {blocks_name} {block_id}/{len(blocks)}.")
            block = block.to(device)
            # get layer prefix to select layers only within the block
            layer_prefix = f'{blocks_name}.{block_id}.'
            layers = select_layers(self.model, layer_prefix, self.target_modules, LINEAR_LAYERS)
            handles, hooks = self._prepare_hooks_and_handles(layers)

            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                block(*inp_args, **inp_kwargs)

            for _, h in hooks.items():
                h.remove()

            self._prune_group(handles, sparsity, device)

            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                out = block(*inp_args, **inp_kwargs)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                # change only first input argument
                if len(inp_args) > 0:
                    inp_args[0].data = out
                elif 'hidden_states' in inp_kwargs:
                    inp_kwargs['hidden_states'] = out
                else:
                    raise ValueError("Unsupported block input format.")

            if self.cpu_offload:
                block = block.cpu()

            del handles
            del hooks
            torch.cuda.empty_cache()

        ### This branch is entered for encoder-decoder model ###
        # TODO use utility functions for check
        if self.encoder_blocks is not None and self.decoder_blocks is not None:
            # collect inputs for encoder
            inputs = [inp_args[0] for inp_args in input_args]
            # Hidden state extraction
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
            for batch_id, (inp_args, inp_kwargs) in enumerate(self.data_loader):
                try:
                    if inp_kwargs.get('decoder_input_ids') is not None:
                        decoder_input_ids = inp_kwargs['decoder_input_ids'].to(device)
                    elif inp_kwargs.get('labels') is not None:
                        # Copied from transformers.models.bart.modeling_bart.shift_tokens_right
                        def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
                            """
                            Shift input ids one token to the right.
                            """
                            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
                            shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
                            shifted_input_ids[:, 0] = decoder_start_token_id

                            if pad_token_id is None:
                                raise ValueError("self.model.config.pad_token_id has to be defined.")
                            # replace possible -100 values in labels by `pad_token_id`
                            shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

                            return shifted_input_ids

                        decoder_input_ids = shift_tokens_right(
                            inp_kwargs['labels'].to(device), 
                            self.model.config.pad_token_id, 
                            self.model.config.decoder_start_token_id
                        )
                    
                    decoder(
                        input_ids=decoder_input_ids, 
                        encoder_hidden_states=encoder_hidden_states[batch_id],
                    )
                except ForwardInterrupt:
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
            for block_id, block in enumerate(blocks):
                # TODO change to logging
                print(f"Processing {self.decoder_blocks} {block_id}/{len(blocks)}.")
                block = block.to(device)
                # get layer prefix to select layers only within the block
                layer_prefix = f'{self.decoder_blocks}.{block_id}.'
                layers = select_layers(self.model, layer_prefix, self.target_modules, LINEAR_LAYERS)
                handles, hooks = self._prepare_hooks_and_handles(layers)

                for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                    block(*inp_args, **inp_kwargs)

                for _, h in hooks.items():
                    h.remove()

                self._prune_group(handles, sparsity, device)

                for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                    out = block(*inp_args, **inp_kwargs)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    # change only first input argument
                    if len(inp_args) > 0:
                        inp_args[0].data = out
                    elif 'hidden_states' in inp_kwargs:
                        inp_kwargs['hidden_states'] = out
                    else:
                        raise ValueError("Unsupported block input format.")

                if self.cpu_offload:
                    block = block.cpu()

                del handles
                del hooks
                torch.cuda.empty_cache()


class BaseOBCPrunerForCausalLM(BaseOBCPrunerForAnyLM):

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
        cpu_offload: bool = False,
        # specification for sequential pruning
        decoder_blocks: str = None,
        pre_decoder_modules: Union[str, List[str]] = None,
        post_decoder_modules: Union[str, List[str]] = None,
    ) -> None:
        super().__init__(
            model,
            data_loader,
            sparsity_distribution,
            sparsity_type,
            target_modules,
            sparsity_block_size,
            rel_damp,
            sequential,
            device,
            prune_kwargs,
            obc_util_kwargs,
            cpu_offload,
            decoder_blocks=decoder_blocks,
            pre_decoder_modules=pre_decoder_modules,
            post_decoder_modules=post_decoder_modules
        )

    def _validate_blocks(self):
        assert self.decoder_blocks is not None


class BaseOBCPrunerForMaskedLM(BaseOBCPrunerForAnyLM):

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
        cpu_offload: bool = False,
        # specification for sequential pruning
        encoder_blocks: str = None,
        pre_encoder_modules: Union[str, List[str]] = None,
        post_encoder_modules: Union[str, List[str]] = None,
    ) -> None:
        super().__init__(
            model,
            data_loader,
            sparsity_distribution,
            sparsity_type,
            target_modules,
            sparsity_block_size,
            rel_damp,
            sequential,
            device,
            prune_kwargs,
            obc_util_kwargs,
            cpu_offload,
            encoder_blocks=encoder_blocks,
            pre_encoder_modules=pre_encoder_modules,
            post_encoder_modules=post_encoder_modules
        )

    def _validate_blocks(self):
        assert self.encoder_blocks is not None


class BaseOBCPrunerForSeq2SeqLM(BaseOBCPrunerForAnyLM):

    def _validate_blocks(self):
        assert self.encoder_blocks is not None and self.decoder_blocks is not None


class FastOBCPrunerForCausalLM(BaseOBCPrunerForCausalLM):

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
        block_size: Optional[int] = None,
        cpu_offload: bool = False,
        # specification for sequential pruning
        decoder_blocks: str = None,
        pre_decoder_modules: Union[str, List[str]] = None,
        post_decoder_modules: Union[str, List[str]] = None,
    ) -> None:
        super().__init__(
            model,
            data_loader,
            sparsity_distribution=sparsity_distribution,
            sparsity_type=sparsity_type,
            target_modules=target_modules,
            sparsity_block_size=None,
            rel_damp=rel_damp,
            sequential=sequential,
            device=device,
            cpu_offload=cpu_offload,
            obc_util_kwargs={'block_size': block_size},
            decoder_blocks=decoder_blocks,
            pre_decoder_modules=pre_decoder_modules,
            post_decoder_modules=post_decoder_modules
        )


class OBCPrunerForCausalLM(BaseOBCPrunerForCausalLM):

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
        cpu_offload: bool = False,
        # specification for sequential pruning
        decoder_blocks: str = None,
        pre_decoder_modules: Union[str, List[str]] = None,
        post_decoder_modules: Union[str, List[str]] = None,
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
            cpu_offload=cpu_offload,
            obc_util_kwargs={'rows_in_parallel': rows_in_parallel},
            prune_kwargs={'block_size': sparsity_block_size},
            decoder_blocks=decoder_blocks,
            pre_decoder_modules=pre_decoder_modules,
            post_decoder_modules=post_decoder_modules,
        )


class StructOBCPrunerForCausalLM(BaseOBCPrunerForCausalLM):

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
        cpu_offload: bool = False,
        # specification for sequential pruning
        decoder_blocks: str = None,
        pre_decoder_modules: Union[str, List[str]] = None,
        post_decoder_modules: Union[str, List[str]] = None,
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
            cpu_offload=cpu_offload,
            obc_util_kwargs={'struct_size': struct_size},
            decoder_blocks=decoder_blocks,
            pre_decoder_modules=pre_decoder_modules,
            post_decoder_modules=post_decoder_modules,
        )


class FastOBCPrunerForMaskedLM(BaseOBCPrunerForMaskedLM):

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
        block_size: Optional[int] = None,
        cpu_offload: bool = False,
        # specification for sequential pruning
        encoder_blocks: str = None,
        pre_encoder_modules: Union[str, List[str]] = None,
        post_encoder_modules: Union[str, List[str]] = None,
    ) -> None:
        super().__init__(
            model,
            data_loader,
            sparsity_distribution=sparsity_distribution,
            sparsity_type=sparsity_type,
            target_modules=target_modules,
            sparsity_block_size=None,
            rel_damp=rel_damp,
            sequential=sequential,
            device=device,
            cpu_offload=cpu_offload,
            obc_util_kwargs={'block_size': block_size},
            encoder_blocks=encoder_blocks,
            pre_encoder_modules=pre_encoder_modules,
            post_encoder_modules=post_encoder_modules
        )


class OBCPrunerForMaskedLM(BaseOBCPrunerForMaskedLM):

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
        cpu_offload: bool = False,
        # specification for sequential pruning
        encoder_blocks: str = None,
        pre_encoder_modules: Union[str, List[str]] = None,
        post_encoder_modules: Union[str, List[str]] = None,
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
            cpu_offload=cpu_offload,
            obc_util_kwargs={'rows_in_parallel': rows_in_parallel},
            prune_kwargs={'block_size': sparsity_block_size},
            encoder_blocks=encoder_blocks,
            pre_encoder_modules=pre_encoder_modules,
            post_encoder_modules=post_encoder_modules,
        )


class StructOBCPrunerForMaskedLM(BaseOBCPrunerForMaskedLM):

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
        cpu_offload: bool = False,
        # specification for sequential pruning
        encoder_blocks: str = None,
        pre_encoder_modules: Union[str, List[str]] = None,
        post_encoder_modules: Union[str, List[str]] = None,
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
            cpu_offload=cpu_offload,
            obc_util_kwargs={'struct_size': struct_size},
            encoder_blocks=encoder_blocks,
            pre_encoder_modules=pre_encoder_modules,
            post_encoder_modules=post_encoder_modules,
        )


class FastOBCPrunerForSeq2SeqLM(BaseOBCPrunerForSeq2SeqLM):

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
        block_size: Optional[int] = None,
        cpu_offload: bool = False,
        # specification for sequential pruning
        encoder_blocks: str = None,
        pre_encoder_modules: Union[str, List[str]] = None,
        post_encoder_modules: Union[str, List[str]] = None,
        decoder_blocks: str = None,
        pre_decoder_modules: Union[str, List[str]] = None,
        post_decoder_modules: Union[str, List[str]] = None,
    ) -> None:
        super().__init__(
            model,
            data_loader,
            sparsity_distribution=sparsity_distribution,
            sparsity_type=sparsity_type,
            target_modules=target_modules,
            sparsity_block_size=None,
            rel_damp=rel_damp,
            sequential=sequential,
            device=device,
            cpu_offload=cpu_offload,
            obc_util_kwargs={'block_size': block_size},
            encoder_blocks=encoder_blocks,
            pre_encoder_modules=pre_encoder_modules,
            post_encoder_modules=post_encoder_modules,
            decoder_blocks=decoder_blocks,
            pre_decoder_modules=pre_decoder_modules,
            post_decoder_modules=post_decoder_modules,
        )


class OBCPrunerForSeq2SeqLM(BaseOBCPrunerForSeq2SeqLM):

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
        cpu_offload: bool = False,
        # specification for sequential pruning
        encoder_blocks: str = None,
        pre_encoder_modules: Union[str, List[str]] = None,
        post_encoder_modules: Union[str, List[str]] = None,
        decoder_blocks: str = None,
        pre_decoder_modules: Union[str, List[str]] = None,
        post_decoder_modules: Union[str, List[str]] = None,
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
            cpu_offload=cpu_offload,
            obc_util_kwargs={'rows_in_parallel': rows_in_parallel},
            prune_kwargs={'block_size': sparsity_block_size},
            encoder_blocks=encoder_blocks,
            pre_encoder_modules=pre_encoder_modules,
            post_encoder_modules=post_encoder_modules,
            decoder_blocks=decoder_blocks,
            pre_decoder_modules=pre_decoder_modules,
            post_decoder_modules=post_decoder_modules,
        )


class StructOBCPrunerForSeq2SeqLM(BaseOBCPrunerForSeq2SeqLM):

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
        cpu_offload: bool = False,
        # specification for sequential pruning
        encoder_blocks: str = None,
        pre_encoder_modules: Union[str, List[str]] = None,
        post_encoder_modules: Union[str, List[str]] = None,
        decoder_blocks: str = None,
        pre_decoder_modules: Union[str, List[str]] = None,
        post_decoder_modules: Union[str, List[str]] = None,
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
            cpu_offload=cpu_offload,
            obc_util_kwargs={'struct_size': struct_size},
            encoder_blocks=encoder_blocks,
            pre_encoder_modules=pre_encoder_modules,
            post_encoder_modules=post_encoder_modules,
            decoder_blocks=decoder_blocks,
            pre_decoder_modules=pre_decoder_modules,
            post_decoder_modules=post_decoder_modules,
        )
