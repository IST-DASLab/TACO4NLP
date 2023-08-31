#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning a 🤗 Transformers model on text translation.
"""
# You can also adapt this script on your own text translation task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import re
import sys
import random
from copy import deepcopy

import datasets
import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, broadcast
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    MBartTokenizer,
    MBartTokenizerFast,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    GenerationConfig
)

from transformers.utils.versions import require_version
# import fsml
sys.path.append(os.path.join(os.pardir, 'fsml'))
from fsml.compression import create_pruner_from_config
from fsml.schedules import SparsitySchedule, CyclicLinearLR
from fsml.optim import wrap_optimizer

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# Parsing input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--predict_with_generate",
        type=bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=False,
        help=(
            "Whether to pad all samples to model maximum sentence "
            "length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More"
            "efficient on GPU but very bad for TPU."
        ),
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument("--source_lang", type=str, default=None, help="Source language id for translation.")
    parser.add_argument("--target_lang", type=str, default=None, help="Target language id for translation.")
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
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
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="\beta_1 in Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="\beta_2 in Adam optimizer.")
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
        "--cycle_epochs", type=int, default=None, help="Number of epochs in cycles for cyclic linear scheduler."
    )
    parser.add_argument(
        "--cycle_steps", type=int, default=None, help="Number of steps in cycles for cyclic linear scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
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
        "--teacher_model_name_or_path",
        default=None,
        type=str,
        help="Path or name of teacher model.",
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
    parser.add_argument(
        '--reset_optimizer',
        action="store_true",
        help="Whether to reset optimizer on pruning step.",
    )

    args = parser.parse_args()

    # Sanity checks

    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")

    if args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    if args.validation_file is not None:
        extension = args.validation_file.split(".")[-1]
        assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

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


def reset_optimizer_buffers(optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            state = optimizer.state[param]
            if "momentum_buffer" in state:
                state["momentum_buffer"].zero_()
            if 'exp_avg' in state:
                state['exp_avg'].zero_()
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'].zero_()
            if 'max_exp_avg_sq' in state:
                state['max_exp_avg_sq'].zero_()


def main():
    # Parse the arguments
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir) if args.with_tracking else Accelerator()
    )

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
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

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

    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=args.trust_remote_code)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        assert (
            args.target_lang is not None and args.source_lang is not None
        ), "mBart requires --target_lang and --source_lang"
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if args.source_lang is not None:
            tokenizer.src_lang = args.source_lang
        if args.target_lang is not None:
            tokenizer.tgt_lang = args.target_lang

    # Get the language codes for input/target.
    source_lang = args.source_lang.split("_")[0]
    target_lang = args.target_lang.split("_")[0]

    padding = "max_length" if args.pad_to_max_length else False

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.mixed_precision == 'fp16' else None,
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
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
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2)
    )

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
        cycle_steps = args.cycle_steps or args.max_train_steps
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
            calibration_dataset = train_dataset.shuffle(args.seed).select(range(args.calibration_dataset_size))
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
        if args.teacher_model_name_or_path:
            teacher_model = AutoModelForSeq2SeqLM.from_pretrained(
                args.teacher_model_name_or_path,
                from_tf=bool(".ckpt" in args.teacher_model_name_or_path),
                config=config,
                trust_remote_code=args.trust_remote_code,
            )
        else:
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

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

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

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = args.lr_scheduler_type
            if args.report_to in ['wandb', 'all']:
                project_name = os.environ.get("WANDB_PROJECT", "translation_no_trainer")
            accelerator.init_trackers(project_name, experiment_config)

    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    # evaluation params setup
    if args.val_max_target_length is None:
         args.val_max_target_length = args.max_target_length
    generation_config = GenerationConfig(
        max_length=args.val_max_target_length if args is not None else config.max_length,
        num_beams=args.num_beams
    )

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
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    generation_config=generation_config
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                # If we are in a multiprocess environment, the last batch has duplicates
                if accelerator.num_processes > 1:
                    if step == len(eval_dataloader) - 1:
                        decoded_preds = decoded_preds[: len(eval_dataloader.dataset) - samples_seen]
                        decoded_labels = decoded_labels[: len(eval_dataloader.dataset) - samples_seen]
                    else:
                        samples_seen += len(decoded_labels)

                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        eval_metric = metric.compute()
        # return model back to train mode
        model.train()
        return eval_metric

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
            if args.sparsification_config and completed_steps % pruning_steps == 0:
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
                # reset optimizer buffers
                if args.reset_optimizer:
                    reset_optimizer_buffers(optimizer)
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
                encoder_mask = batch['attention_mask'].unsqueeze(-1)
                decoder_mask = batch['labels'].ne(-100).unsqueeze(-1)
                # mask logits
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
                        mask = encoder_mask if re.search('(encoder|EncDecAttention.(k|v))', feat_name) else decoder_mask
                        # option 1 - MSE
                        if args.feat_loss == 'l2':
                            feat_loss += F.mse_loss(x_student * mask, x_teacher * mask) / mask.float().mean()
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
                eval_metric = evaluation_loop()
                accelerator.wait_for_everyone()
                # re-assign hooks after evaluation
                if feat_distillation:
                    student_hooks = register_cache_output_hooks(
                        accelerator.unwrap_model(model),
                        args.feat_names,
                        student_features
                    )
                logger.info({"bleu": eval_metric["score"]})

                if args.with_tracking:
                    accelerator.log(
                        {
                            "bleu": eval_metric["score"],
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
        eval_metric = evaluation_loop()
        accelerator.wait_for_everyone()
        logger.info({"bleu": eval_metric["score"]})

        if args.with_tracking:
            accelerator.log(
                {
                    "bleu": eval_metric["score"],
                    # "epoch": epoch,
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
            json.dump({"eval_bleu": eval_metric["score"]}, f)


if __name__ == "__main__":
    main()
