#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import re
import sys
from copy import deepcopy

import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed, broadcast
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import transformers
from transformers import (
    MODEL_MAPPING,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from transformers.utils.versions import require_version

from apps_dataset import get_apps_dataset
# import fsml
sys.path.append(os.path.join(os.pardir, 'fsml'))
from fsml.compression import create_pruner_from_config
from fsml.schedules import SparsitySchedule, CyclicLinearLR
from fsml.optim import wrap_optimizer

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=0.05,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="\beta_1 in Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="\beta_2 in Adam optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "cyclic_linear"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--cycle_steps", type=int, default=None, help="Number of steps in cycles for cyclic linear scheduler."
    )
    parser.add_argument(
        "--cycle_epochs", type=int, default=None, help="Number of epochs in cycles for cyclic linear scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--checkpoint_before_pruning",
        action="store_true",
        help="Whether to checkpoint before pruning.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Interval between two logging steps.",
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        choices=['step', 'epoch', 'no'],
        default='epoch',
        help="Evaluation strategy.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Interval between two evaluation step. If provided, overrides eval_epochs.",
    )
    parser.add_argument(
        "--eval_epochs",
        type=int,
        default=1,
        help="Interval between two evaluation epochs.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    # Sparsification arguments
    parser.add_argument(
        "--sparsification_config",
        type=str,
        default=None,
        help='Path to sparsification config. If not given run usual training.'
    )
    parser.add_argument(
        "--init_sparsity",
        type=float,
        default=0.0,
        help="Initial sparsity level"
    )
    parser.add_argument(
        "--final_sparsity",
        type=float,
        default=0.0,
        help="Final sparsity level"
    )
    parser.add_argument(
        "--pruning_epochs",
        type=int,
        default=None,
        help="Interval between pruning steps in epochs.",
    )
    parser.add_argument(
        "--pruning_steps",
        type=int,
        default=None,
        help="Interval between pruning steps. Overrides pruning_epochs if set.",
    )
    parser.add_argument(
        "--cooldown_fraction",
        type=float,
        default=0.0,
        help="Fraction of training with constant sparsity"
    )
    parser.add_argument(
        "--sparsity_inter_func",
        type=str,
        default="cubic",
        choices=["linear", "cubic"],
        help="Sparsity interpolation function."
    )
    # Calibration data params
    parser.add_argument(
        "--use_calibration_data",
        action="store_true",
        help="Whether to use calibration data.",
    )
    parser.add_argument(
        '--calibration_dataset_size',
        default=512,
        type=int,
        help="Size of calibration dataset."
    )
    parser.add_argument(
        '--calibration_batch_size',
        default=1,
        type=int,
        help="Batch size in calibration loader."
    )
    # Distillation params
    parser.add_argument(
        "--distillation",
        action="store_true",
        help="Whether to use knowledge distillation.",
    )
    parser.add_argument(
        '--orig_loss_weight',
        default=1.0,
        type=float,
        help="Weight of the original loss in total loss."
    )
    parser.add_argument(
        '--dist_loss_weight',
        default=1.0,
        type=float,
        help="Weight of the output distillation loss in total loss."
    )
    parser.add_argument(
        '--feat_loss_weight',
        default=0.0,
        type=float,
        help="Weight of the feature distillation loss in total loss."
    )
    parser.add_argument(
        '--feat_names',
        default='',
        type=str,
        help="Regular expression for features used in knowledge distillation."
    )
    parser.add_argument(
        '--feat_loss',
        default='l2',
        type=str,
        choices=['l2', 'l2_norm'],
        help="Loss type."
    )

    args = parser.parse_args()

    return args


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def masked_norm_mse(x1, x2, mask, eps=1e-9):
    return (mask * (x1 - x2) ** 2).mean() / ((mask * x2 ** 2).mean() + eps)


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()


    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # prepare data
    raw_dataset = load_dataset("codeparrot/apps", split="train").shuffle(seed=args.seed)
    dataset = get_apps_dataset(raw_dataset, args, verbose=accelerator.is_main_process)
    train_size = int((1 - args.validation_split_percentage) * len(dataset))
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, len(dataset) - train_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        use_auth_token=False, 
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # we use default data collator
    data_collator = default_data_collator

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / (args.gradient_accumulation_steps * accelerator.num_processes))
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        # override number of epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # setup eval steps
    if args.eval_steps is None:
        args.eval_steps = num_update_steps_per_epoch * args.eval_epochs

    # multiply number of steps by number of workers
    if args.lr_scheduler_type == 'cyclic_linear':
        if args.cycle_steps:
            cycle_steps = args.cycle_steps
        elif args.cycle_epochs:
            cycle_steps = args.cycle_epochs * num_update_steps_per_epoch
        else:
            cycle_steps = args.max_train_steps
        lr_scheduler = CyclicLinearLR(
            optimizer=optimizer,
            cycle_steps=cycle_steps,
            num_warmup_steps=args.num_warmup_steps,
        )
    else:
        lr_scheduler = get_scheduler(
            name=SchedulerType(args.lr_scheduler_type),
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

    # Setup for pruning (if not pruning_steps do one-shot + finetune)
    if args.pruning_steps:
        pruning_steps = args.pruning_steps
    elif args.pruning_epochs:
        pruning_steps = args.pruning_epochs * num_update_steps_per_epoch
    else:
        pruning_steps = args.max_train_steps
    pruner_kwargs = {}
    # prepare calibration data
    if args.use_calibration_data:
        pruner_kwargs['data_loader'] = 'placeholder'
        if accelerator.is_main_process:
            # make a subset of training dataset
            calibration_dataset = Subset(
                train_dataset, 
                torch.randperm(len(train_dataset))[:args.calibration_dataset_size]
            )
            aux_dataloader = DataLoader(
                calibration_dataset, 
                shuffle=True, 
                collate_fn=data_collator, 
                batch_size=args.calibration_batch_size
            )
            calibration_dataloader = [([], {k: v for k, v in batch.items()}) for batch in aux_dataloader]
            pruner_kwargs['data_loader'] = calibration_dataloader
        accelerator.wait_for_everyone()

    # prepare teacher (if using distillation) # TODO try stronger model?
    feat_distillation = False
    if args.distillation:
        teacher_model = deepcopy(model)
        if args.feat_loss_weight > 0:
            feat_distillation = True

            def register_cache_output_hooks(model, feat_names, features):
                hooks = {}
                for module_name, module in model.named_modules():
                    if re.search(feat_names, module_name):
                        def cache_output_hook(name, features):
                            def hook(module, inputs, outputs):
                                features[name] = outputs
                            return hook
                        hooks[module_name] = module.register_forward_hook(
                            cache_output_hook(module_name, features)
                        )
                return hooks

            def remove_hooks(hooks):
                for _, hook in hooks.items():
                    hook.remove()

            # assign cache output hooks
            teacher_features = {}
            teacher_hooks = register_cache_output_hooks(teacher_model, args.feat_names, teacher_features)
            student_features = {}
            student_hooks = register_cache_output_hooks(model, args.feat_names, student_features)
                
        teacher_model = accelerator.prepare(teacher_model)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # create pruner
    pruner = None
    if args.sparsification_config:
        pruner = create_pruner_from_config(
            accelerator.unwrap_model(model), 
            args.sparsification_config, 
            pruner_kwargs
        )
        # init sparsity scheduler
        sparsity_schedule = SparsitySchedule(
            args.init_sparsity, 
            args.final_sparsity, 
            args.max_train_steps, 
            args.cooldown_fraction,
            args.sparsity_inter_func
        )
        # prepare optimizer
        optimizer = wrap_optimizer(optimizer, pruner)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = args.lr_scheduler_type
            if args.report_to in ['wandb', 'all']:
                project_name = os.environ.get("WANDB_PROJECT", "apps_finetuning")
            accelerator.init_trackers(project_name, experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # by default sparsity is zero
    last_applied_sparsity = 0
    # indicator whether to evaluate in the end
    evaluate_in_the_end = True

    # function for evaluation
    def evaluation_loop():
        # set to evaluation mode
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
        # return model back to train mode
        model.train()
        return eval_loss, perplexity

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    # init running loss
    if args.with_tracking:
        loss_m = AverageMeter()
        orig_loss_m = AverageMeter()
        dist_loss_m = AverageMeter()
        feat_loss_m = AverageMeter()

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):

            # if pruning step run pruning
            if args.sparsification_config and completed_steps % pruning_steps == 0 and step % args.gradient_accumulation_steps == 0:
                # save before pruning, if requested
                if args.checkpoint_before_pruning and args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, f'sparsity={last_applied_sparsity:.3f}')
                    accelerator.save_state(output_dir)

                sparsity = sparsity_schedule.get_sparsity_for_step(completed_steps)
                last_applied_sparsity = sparsity
                accelerator.print(
                    f'Pruning on step {completed_steps}. Current sparsity {(100 * sparsity):4.2f}%.'
                )
                # remove hooks before pruning
                if feat_distillation:
                    remove_hooks(student_hooks)
                if accelerator.is_main_process:
                    pruner.prune(sparsity)
                # synchronize masks across workers
                broadcast(pruner.params)
                broadcast(pruner.param_masks)
                # re-enable hooks after pruning
                if feat_distillation:
                    student_hooks = register_cache_output_hooks(
                        accelerator.unwrap_model(model), 
                        args.feat_names,
                        student_features
                    )

            outputs = model(**batch)
            orig_loss = outputs.loss
            loss = args.orig_loss_weight * orig_loss
            # add other losses
            if args.distillation:
                # make teacher forward pass
                with torch.no_grad():
                    teacher_outputs = teacher_model(**batch)
                decoder_mask = batch['labels'].ne(-100).unsqueeze(-1)
                dist_loss = F.cross_entropy(
                    decoder_mask * outputs.logits, 
                    decoder_mask * teacher_outputs.logits.softmax(dim=-1)
                )
                loss += args.dist_loss_weight * dist_loss
                if args.feat_loss_weight > 0:
                    feat_loss = 0
                    for feat_name in teacher_features:
                        x_teacher = teacher_features[feat_name]
                        x_student = student_features[feat_name]
                        if isinstance(x_teacher, (list, tuple)):
                            x_teacher = x_teacher[0]
                            x_student = x_student[0]
                        # get corresponding mask
                        mask = decoder_mask
                        # option 1 - MSE
                        if args.feat_loss == 'l2':
                            feat_loss += F.mse_loss(x_student * mask, x_teacher * mask) / mask.numel()
                        # option 2 - normalized MSE
                        else:
                            feat_loss += masked_norm_mse(x_student, x_teacher, mask)
                    loss += args.feat_loss_weight * feat_loss

            if args.with_tracking:
                loss_m.update(loss.item())
                orig_loss_m.update(orig_loss.item())
                if args.distillation:
                    dist_loss_m.update(dist_loss.item())
                    if args.feat_loss_weight > 0:
                        feat_loss_m.update(feat_loss.item())

            accelerator.backward(loss / args.gradient_accumulation_steps)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            # log current loss
            if args.with_tracking and completed_steps % args.logging_steps == 0:
                log_dict = {
                    'loss': loss_m.avg, 
                    'orig_loss': orig_loss_m.avg, 
                    'sparsity': last_applied_sparsity
                }
                if args.distillation:
                    log_dict['dist_loss'] = dist_loss_m.avg
                    if args.feat_loss_weight > 0:
                        log_dict['feat_loss'] = feat_loss_m.avg
                log_dict['lr'] = optimizer.param_groups[0]['lr']
                accelerator.log(log_dict, step=completed_steps)
                # reset all meters
                loss_m.reset()
                orig_loss_m.reset()
                dist_loss_m.reset()
                feat_loss_m.reset()

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            # run evaluation (if requested)
            if max(completed_steps - 1, 0) % args.eval_steps == 0 and step % args.gradient_accumulation_steps == 0 and args.eval_strategy != 'no':
                # remove student hooks before evaluation
                if feat_distillation:
                    remove_hooks(student_hooks)
                eval_loss, perplexity = evaluation_loop()
                accelerator.wait_for_everyone()
                # re-assign hooks after evaluation
                if feat_distillation:
                    student_hooks = register_cache_output_hooks(
                        accelerator.unwrap_model(model),
                        args.feat_names,
                        student_features
                    )
                logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")
                if args.with_tracking:
                    accelerator.log(
                        {
                            "perplexity": perplexity,
                            "eval_loss": eval_loss,
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )

                # do not evaluate in the end if evaluated on the last step
                if completed_steps >= args.max_train_steps:
                    evaluate_in_the_end = False

            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    # evaluate in the end
    if evaluate_in_the_end:
        # remove student hooks before evaluation
        if feat_distillation:
            remove_hooks(student_hooks)
        eval_loss, perplexity = evaluation_loop()
        accelerator.wait_for_everyone()
        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")
        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)

if __name__ == "__main__":
    main()
