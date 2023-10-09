#!/bin/bash
export CUDA_VISIBLE_DEVICES=#TODO
export OMP_NUM_THREADS=#TODO

NUM_PROC=$(wc -w <<< $(tr ',' ' ' <<< $CUDA_VISIBLE_DEVICES))
BATCH_SIZE_PER_GPU=$(( 32 / ${NUM_PROC} ))

MODEL=openai/whisper-small

export WANDB_ENTITY=#TODO
export WANDB_PROJECT=#TODO

export WANDB_NAME=whisper-small,hindi,dense_training
accelerate launch --multi_gpu --num_processes=${NUM_PROC} --main_process_port=29501 run_asr.py \
	--model_name_or_path ${MODEL} \
	--dataset_name mozilla-foundation/common_voice_11_0 \
	--dataset_config_name 'hi' \
	--language hindi \
	--train_split_name 'train+validation' \
	--eval_split_name 'test' \
	--max_train_steps 5000 \
	--output_dir ./output/${MODEL}/hi/dense_training \
	--per_device_train_batch_size=$BATCH_SIZE_PER_GPU \
	--per_device_eval_batch_size=$BATCH_SIZE_PER_GPU \
	--logging_steps 25 \
	--learning_rate 1e-5 \
	--weight_decay 1e-4 \
	--val_max_target_length "225" \
	--preprocessing_num_workers "16" \
	--group_by_length \
	--length_column_name "input_length" \
	--max_duration_in_seconds "30" \
	--text_column_name "sentence" \
	--freeze_feature_encoder "False" \
	--gradient_checkpointing \
	--preprocessing_num_workers 8 \
	--lr_scheduler_type linear \
	--per_device_train_batch_size $BATCH_SIZE_PER_GPU \
	--per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
	--checkpointing_steps 5000 \
	--num_warmup_steps 100 \
	--report_to wandb \
	--with_tracking
