#!/bin/bash
export CUDA_VISIBLE_DEVICES=#TODO
export OMP_NUM_THREADS=#TODO

NUM_PROC=$(wc -w <<< $(tr ',' ' ' <<< $CUDA_VISIBLE_DEVICES))

MODEL=t5-small
BATCH_SIZE_PER_GPU=$(( 128 / ${NUM_PROC} ))

export WANDB_ENTITY=#TODO
export WANDB_PROJECT=#TODO

accelerate launch --multi_gpu --num_processes=${NUM_PROC} --main_process_port=29501 run_translation.py \
    --model_name_or_path ${MODEL} \
    --source_lang en \
    --target_lang de \
    --source_prefix "translate English to German: " \
    --dataset_name wmt14 \
    --dataset_config_name de-en \
    --max_source_length 256 \
    --max_target_length 256 \
    --output_dir output/${MODEL}/dense_training \
    --max_train_steps 100000 \
    --learning_rate 2e-3 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay 1e-4 \
    --num_beams 5 \
    --checkpointing_steps 100000 \
    --preprocessing_num_workers 8 \
    --lr_scheduler_type linear \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --num_warmup_steps 1000 \
    --report_to wandb \
    --with_tracking
