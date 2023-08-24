from typing import List, Dict, Optional, Iterable, Union, Any
from tqdm import tqdm

import torch
import torch.nn as nn

from ..obc import BaseOBCPruner
from ...utils.obc import OBCUtil, FastOBCUtil, StructOBCUtil
from ....utils.model import select_layers, LINEAR_LAYERS
from ....utils.common import to


__all__ = [
    "OBCPrunerForUNet2D", 
    "FastOBCPrunerForUNet2D", 
    "StructOBCPrunerForUNet2D"
]

class BaseOBCPrunerForUNet2D(BaseOBCPruner):

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
        # TODO may be helpful for large models
        cpu_offload: bool = False,
    ) -> None:
        assert cpu_offload is False
        # TODO add check that the model is UNet2D
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

    @torch.no_grad()
    def prune_sequential(self, sparsity: float) -> None:
        device = self.device or next(self.model.parameters()).device
        unet = self.model

        samples = []
        embs = []
        # preprocessing steps
        # TODO add option of preprocessing steps compression
        for input_args, input_kwargs in self.data_loader:
            input_kwargs = to(input_kwargs, device=device)
            # center input if necessary
            sample = input_kwargs['sample']
            if unet.config.center_input_sample:
                sample = 2 * sample - 1.0
            samples.append(sample)
            # timestep embedding
            timestep = input_kwargs['timestep']
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

            t_emb = unet.time_proj(timesteps)

            # timesteps does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=unet.dtype)
            emb = unet.time_embedding(t_emb)

            # class coniditioning
            class_labels = input_kwargs.get('class_labels', None)
            if unet.class_embedding is not None:
                if class_labels is None:
                    raise ValueError("class_labels should be provided when doing class conditioning")

                if unet.config.class_embed_type == "timestep":
                    class_labels = unet.time_proj(class_labels)

                class_emb = unet.class_embedding(class_labels).to(dtype=unet.dtype)
                emb = emb + class_emb

            embs.append(emb)

        # 2. pre-process
        skip_samples = []
        for i, sample in enumerate(samples):
            skip_samples.append(sample)
            samples[i] = unet.conv_in(sample)

        # 3. down
        down_block_res_samples = [(sample,) for sample in samples]
        for block_id, downsample_block in tqdm(
            enumerate(unet.down_blocks), total=len(unet.down_blocks), desc='Processing down blocks'
        ):
            # get layer prefix to select layers only within the block
            layer_prefix = f'down_blocks.{block_id}'
            layers = select_layers(unet, layer_prefix, self.target_modules, LINEAR_LAYERS)
            handles, hooks = self._prepare_hooks_and_handles(layers)
            
            # propagate inputs
            for i, (sample, emb, skip_sample) in enumerate(zip(samples, embs, skip_samples)):
                if hasattr(downsample_block, "skip_conv"):
                    downsample_block(hidden_states=sample, temb=emb, skip_sample=skip_sample)
                else:
                    downsample_block(hidden_states=sample, temb=emb)

            for _, h in hooks.items():
                h.remove()

            self._prune_group(handles, sparsity, device)

            # update outputs
            for i, (sample, emb, skip_sample) in enumerate(zip(samples, embs, skip_samples)):
                if hasattr(downsample_block, "skip_conv"):
                    samples[i], res_samples, skip_samples[i] = downsample_block(
                        hidden_states=sample, temb=emb, skip_sample=skip_sample
                    )
                else:
                    samples[i], res_samples = downsample_block(hidden_states=sample, temb=emb)

                down_block_res_samples[i] += res_samples
            torch.cuda.empty_cache()

        # 4. mid
        # get layer prefix to select layers only within the block
        layer_prefix = f'middle_block.{block_id}'  
        layers = select_layers(unet, layer_prefix, self.target_modules, LINEAR_LAYERS)
        handles, hooks = self._prepare_hooks_and_handles(layers) 

        for sample, emb in zip(samples, embs):
            unet.mid_block(sample, emb)

        for _, h in hooks.items():
            h.remove()

        self._prune_group(handles, sparsity, device)

        for sample, emb in zip(samples, embs):
            samples[i] = unet.mid_block(sample, emb)
            skip_samples[i] = None
        torch.cuda.empty_cache()

        # 5. up
        for block_id, upsample_block in tqdm(enumerate(unet.up_blocks), desc='Processing up blocks'):
            # get layer prefix to select layers only within the block
            layer_prefix = f'up_blocks.{block_id}'
            layers = select_layers(unet, layer_prefix, self.target_modules, LINEAR_LAYERS)
            handles, hooks = self._prepare_hooks_and_handles(layers) 
            
            # propagate inputs
            for i, (sample, emb, skip_sample) in enumerate(zip(samples, embs, skip_samples)):
                res_samples = down_block_res_samples[i][-len(upsample_block.resnets) :]
                if hasattr(upsample_block, "skip_conv"):
                    upsample_block(sample, res_samples, emb, skip_sample)
                else:
                    upsample_block(sample, res_samples, emb)  

            for _, h in hooks.items():
                h.remove()   

            self._prune_group(handles, sparsity, device)

            # update outputs
            for i, (sample, emb, skip_sample) in enumerate(zip(samples, embs, skip_samples)):
                res_samples = down_block_res_samples[i][-len(upsample_block.resnets) :]
                down_block_res_samples[i] = down_block_res_samples[i][: -len(upsample_block.resnets)]
                if hasattr(upsample_block, "skip_conv"):
                    samples[i], skip_samples[i] = upsample_block(sample, res_samples, emb, skip_sample)
                else:
                    samples[i] = upsample_block(sample, res_samples, emb)
            torch.cuda.empty_cache()

        # 6. post-process
        # TODO add option pruning of output conv
        # sample = unet.conv_norm_out(sample)
        # sample = unet.conv_act(sample)
        # sample = unet.conv_out(sample)


class FastOBCPrunerForUNet2D(BaseOBCPrunerForUNet2D):

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
        cpu_offload: bool = False
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
            obc_util_kwargs={'block_size': block_size},
            cpu_offload=cpu_offload
        )


class OBCPrunerForUNet2D(BaseOBCPrunerForUNet2D):

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
        cpu_offload: bool = False
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
            prune_kwargs={'block_size': sparsity_block_size},
            cpu_offload=cpu_offload
        )


# TODO add custom prune config option
class StructOBCPrunerForUNet2D(BaseOBCPrunerForUNet2D):

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
        cpu_offload: bool = False
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
            cpu_offload=cpu_offload
        )
