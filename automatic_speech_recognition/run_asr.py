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
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import datasets
import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, broadcast
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    set_seed,
    SchedulerType,
    get_scheduler,
    GenerationConfig
)
from transformers.utils.versions import require_version
from transformers.trainer_pt_utils import LengthGroupedSampler

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


def str2bool(x: str) -> bool:
    if x == 'True':
        return True
    if x == 'False':
        return False
    raise ValueError("Expected either `True` of `False`.")


# Parsing input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a ASR task")
    # Model arguments
    parser.add_argument(
        '--model_name_or_path', 
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        '--config_name', 
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        '--tokenizer_name', 
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name"
    )
    parser.add_argument(
        '--feature_extractor_name', 
        type=str,
        default=None,
        help="feature extractor name or path if not the same as model_name"
    )
    parser.add_argument(
        '--cache_dir', 
        type=str,
        default=None,
        help="Where to store the pretrained models downloaded from huggingface.co"
    )
    parser.add_argument(
        '--use_fast_tokenizer', 
        type=bool,
        default=True,
        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
    )
    parser.add_argument(
        '--freeze_feature_encoder', 
        type=bool,
        default=True,
        help="Whether to freeze the feature encoder layers of the model."
    )
    parser.add_argument(
        '--freeze_encoder', 
        type=bool,
        default=False,
        help="Whether to freeze the entire encoder of the seq2seq model."
    )
    parser.add_argument(
        '--forced_decoder_ids', 
        type=int,
        nargs="+",
        default=None,
        help=(
            "A list of pairs of integers which indicates a mapping from generation indices to token indices "
            "that will be forced before sampling. For example, [[0, 123]] means the first generated token "
            "will always be a token of index 123."
        )
    )
    parser.add_argument(
        '--suppress_tokens', 
        type=int,
        nargs="+",
        default=None,
        help="A list of tokens that will be suppressed at generation."
    )
    parser.add_argument(
        '--apply_spec_augment', 
        type=bool,
        default=False,
        help="Whether to apply *SpecAugment* data augmentation to the input features. This is currently only relevant for Wav2Vec2, HuBERT, WavLM and Whisper models."
    )
    # DataTraining arguments
    parser.add_argument(
        '--dataset_name', 
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library)."
    )
    parser.add_argument(
        '--dataset_config_name', 
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument(
        '--overwrite_cache', 
        type=bool,
        default=False,
        help="Overwrite the cached training and evaluation sets."
    )
    parser.add_argument(
        '--preprocessing_num_workers', 
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing."
    )
    parser.add_argument(
        '--max_train_samples', 
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        )
    )
    parser.add_argument(
        '--max_eval_samples', 
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        )
    )
    parser.add_argument(
        '--audio_column_name', 
        type=str,
        default="audio",
        help="The name of the dataset column containing the audio data. Defaults to 'audio'."
    )
    parser.add_argument(
        '--text_column_name', 
        type=str,
        default="text",
        help="The name of the dataset column containing the text data. Defaults to 'text'."
    )
    parser.add_argument(
        '--max_duration_in_seconds', 
        type=float,
        default=20.0,
        help=(
            "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
            " 'max_duration_in_seconds`"
        )
    )
    parser.add_argument(
        '--min_duration_in_seconds', 
        type=float,
        default=0.0,
        help="Filter audio files that are shorter than `min_duration_in_seconds` seconds"
    )
    parser.add_argument(
        '--preprocessing_only', 
        type=bool,
        default=False,
        help=(
            "Whether to only do data preprocessing and skip training. This is especially useful when data"
            " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
            " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
            " can consequently be loaded in distributed training"
        )
    )
    parser.add_argument(
        '--train_split_name', 
        type=str,
        default="audio",
        help="The name of the training data set split to use (via the datasets library). Defaults to 'train'."
    )
    parser.add_argument(
        '--eval_split_name', 
        type=str,
        default="test",
        help="The name of the evaluation data set split to use (via the datasets library). Defaults to 'test'."
    )
    parser.add_argument(
        '--do_lower_case', 
        type=bool,
        default=True,
        help="Whether the target text should be lower cased."
    )
    parser.add_argument(
        '--language', 
        type=str,
        default=None,
        help=(
            "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
            "only. For English speech recognition, it should be set to `None`."
        )
    )
    parser.add_argument(
        '--task', 
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task, either `transcribe` for speech recognition or `translate` for speech translation."
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
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    # Training arguments
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
        "--group_by_length",
        action="store_true",
        help=(
            "Whether or not to group together samples of roughly the same length"
            " in the training dataset (to minimize padding applied and be more efficient)"
            ". Only useful if applying dynamic padding."
        )
    )
    parser.add_argument(
        "--length_column_name",
        type=str,
        default="length",
        help=(
            "Column name for precomputed lengths. "
            "If the column exists, grouping by length will use these values rather than computing them on train startup. "
            "Ignored unless group_by_length is True and the dataset is an instance of Dataset."
        )
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
        "--num_beams",
        type=int,
        default=1,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        '--gradient_checkpointing', 
        action="store_true",
        help="Whether to use gradient checkpointing."
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
        "--teacher_model_name_or_path",
        default=None,
        type=str,
        help="Path or name of teacher model.",
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


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def masked_norm_mse(x1, x2, mask, eps=1e-8):
    return (mask * (x1 - x2) ** 2).mean() / ((mask * x2 ** 2).mean() + eps)


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

    # 4. Load dataset
    raw_datasets = DatasetDict()
    # train dataset
    raw_datasets["train"] = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split=args.train_split_name,
        cache_dir=args.cache_dir,
    )
    # eval dataset
    raw_datasets["eval"] = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split=args.eval_split_name,
        cache_dir=args.cache_dir,
    )

    if args.audio_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--audio_column_name '{args.audio_column_name}' not found in dataset '{args.dataset_name}'. "
            "Make sure to set `--audio_column_name` to the correct audio column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    if args.text_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--text_column_name {args.text_column_name} not found in dataset '{args.dataset_name}'. "
            "Make sure to set `--text_column_name` to the correct text column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    # 5. Load pretrained model, tokenizer, and feature extractor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
    )

    config.update({"forced_decoder_ids": args.forced_decoder_ids, "suppress_tokens": args.suppress_tokens})

    # SpecAugment for whisper models
    if getattr(config, "model_type", None) == "whisper":
        config.update({"apply_spec_augment": args.apply_spec_augment})

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.feature_extractor_name if args.feature_extractor_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # if args.freeze_feature_encoder:
    #     model.freeze_feature_encoder()

    if args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.language is not None:
        # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
        tokenizer.set_prefix_tokens(language=args.language, task=args.task)

    # 6. Resample speech dataset if necessary
    dataset_sampling_rate = next(iter(raw_datasets.values())).features[args.audio_column_name].sampling_rate
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        raw_datasets = raw_datasets.cast_column(
            args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = args.min_duration_in_seconds * feature_extractor.sampling_rate
    audio_column_name = args.audio_column_name
    num_workers = args.preprocessing_num_workers
    text_column_name = args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    do_lower_case = args.do_lower_case
    # if SpecAugment is used for whisper models, return attention_mask to guide the mask along time axis
    forward_attention_mask = (
        getattr(config, "model_type", None) == "whisper"
        and getattr(config, "apply_spec_augment", False)
        and getattr(config, "mask_time_prob", 0) > 0
    )

    if args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(args.max_train_samples))

    if args.max_eval_samples is not None:
        raw_datasets["eval"] = raw_datasets["eval"].select(range(args.max_eval_samples))

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_attention_mask=forward_attention_mask
        )
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])
        if forward_attention_mask:
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        # process targets
        input_str = batch[text_column_name].lower() if do_lower_case else batch[text_column_name]
        batch["labels"] = tokenizer(input_str).input_ids
        return batch

    with accelerator.main_process_first():
        accelerator.print("dataset map pre-processing")
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=args.preprocessing_num_workers,
            desc="preprocess train dataset",
        )

    # filter data that is shorter than min_input_length or longer than
    # max_input_length
    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        num_proc=num_workers,
        input_columns=["input_length"],
    )

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if args.preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    # 8. Load Metric
    metric = evaluate.load("wer")

    # 9. Create a single speech processor
    # make sure all processes wait until data is saved
    with accelerator.main_process_first():
        # save feature extractor, tokenizer and config
        feature_extractor.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    # wait for other processes to save config and tokenizer
    accelerator.wait_for_everyone()

    accelerator.print('AutoProcessor.from_pretrained(args.output_dir)')
    processor = AutoProcessor.from_pretrained(args.output_dir)

    # 10. Define data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
    )

    train_dataset = vectorized_datasets["train"]
    eval_dataset = vectorized_datasets["eval"]

    # define sampler
    train_batch_size = args.per_device_train_batch_size * accelerator.num_processes
    train_sampler = None
    if args.group_by_length:
        if isinstance(train_dataset, datasets.Dataset):
            lengths = (
                train_dataset[args.length_column_name]
                if args.length_column_name in train_dataset.column_names
                else None
            )
        else:
            lengths = None
        model_input_name = tokenizer.model_input_names[0] if tokenizer is not None else None
        train_sampler =  LengthGroupedSampler(
            train_batch_size * args.gradient_accumulation_steps,
            dataset=train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
        )

    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True if train_sampler is None else None, 
        sampler=train_sampler,
        collate_fn=data_collator, 
        batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        collate_fn=data_collator, 
        batch_size=args.per_device_eval_batch_size
    )

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
            teacher_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                args.teacher_model_name_or_path,
                config=config,
                cache_dir=args.cache_dir,
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
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = args.lr_scheduler_type
            if args.report_to in ['wandb', 'all']:
                project_name = os.environ.get("WANDB_PROJECT", "automatic_speech_recogition_no_trainer")
            accelerator.init_trackers(project_name, experiment_config)

    metric = evaluate.load("wer")

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
    logger.info(f"  Num examples = {len(raw_datasets['train'])}")
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
                    batch["input_features"],
                    attention_mask=batch.get("attention_mask", None),
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

                # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

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
            completed_steps = resume_step // args.gradient_accumulation_stepp

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
                        mask = torch.ones_like(x_teacher) if 'encoder' in feat_name else decoder_mask
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
                logger.info({"wer": eval_metric})

                if args.with_tracking:
                    accelerator.log(
                        {
                            "wer": eval_metric,
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
        logger.info({"wer": eval_metric})

        if args.with_tracking:
            accelerator.log(
                {
                    "wer": eval_metric,
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
            json.dump({"eval_wer": eval_metric}, f)


if __name__ == "__main__":
    main()
