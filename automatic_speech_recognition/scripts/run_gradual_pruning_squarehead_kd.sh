#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=16

NUM_PROC=$(wc -w <<< $(tr ',' ' ' <<< $CUDA_VISIBLE_DEVICES))
BATCH_SIZE_PER_GPU=$(( 32 / ${NUM_PROC} ))

MODEL=openai/whisper-small
LEARNING_RATE=2e-4
SPARSITIES=(0.50 0.67 0.75 0.80 0.83 0.86 0.88 0.90)
EXP_NAME=squarehead_kd
DIST_LOSS_WEIGHT=0.0
FEAT_LOSS_WEIGHT=1.0
FEAT_LOSS=l2_norm

export WANDB_ENTITY=#TODO
export WANDB_PROJECT=#TODO

export WANDB_NAME=${MODEL},FastOBC,sparsity=0.50,${EXP_NAME}
accelerate launch --multi_gpu --num_processes=${NUM_PROC} --main_process_port=29501 run_asr.py \
	--model_name_or_path output/openai/whisper-small/hi/dense_training \
	--teacher_model_name_or_path output/openai/whisper-small/hi/dense_training \
	--dataset_name mozilla-foundation/common_voice_11_0 \
	--dataset_config_name 'hi' \
	--language hindi \
	--train_split_name 'train+validation' \
	--eval_split_name 'test' \
	--num_train_epochs 6 \
    --output_dir output/${MODEL}/hi/sparsity=0.50/${EXP_NAME} \
	--logging_steps 25 \
	--learning_rate ${LEARNING_RATE} \
	--weight_decay 1e-4 \
	--val_max_target_length "225" \
	--group_by_length \
	--length_column_name "input_length" \
	--max_duration_in_seconds "30" \
	--text_column_name "sentence" \
	--freeze_feature_encoder "False" \
	--gradient_checkpointing \
	--preprocessing_num_workers 8 \
    --checkpoint_before_pruning \
    --lr_scheduler_type linear \
	--per_device_train_batch_size $BATCH_SIZE_PER_GPU \
	--per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --num_warmup_steps 50 \
    --sparsification_config configs/fast_obc.yaml \
    --init_sparsity 0.50 \
    --final_sparsity 0.50 \
    --use_calibration_data \
    --calibration_dataset_size 512 \
    --calibration_batch_size 1 \
	--distillation \
    --orig_loss_weight 0.5 \
    --dist_loss_weight ${DIST_LOSS_WEIGHT} \
    --feat_loss_weight ${FEAT_LOSS_WEIGHT} \
    --feat_names '.*((en|de)coder.layers\.\d$)' \
    --feat_loss ${FEAT_LOSS} \
    --report_to wandb \
    --with_tracking

for (( i=0; i < $(( ${#SPARSITIES[@]}-1)); i++))
do
    export WANDB_NAME="${MODEL},FastOBC,sparsity=${SPARSITIES[$(( $i + 1 ))]},${EXP_NAME}"
	accelerate launch --multi_gpu --num_processes=${NUM_PROC} --main_process_port=29501 run_asr.py \
		--model_name_or_path "output/${MODEL}/hi/sparsity=${SPARSITIES[$i]}/${EXP_NAME}" \
		--teacher_model_name_or_path output/openai/whisper-small/hi/dense_training \
		--dataset_name mozilla-foundation/common_voice_11_0 \
		--dataset_config_name 'hi' \
		--language hindi \
		--train_split_name 'train+validation' \
		--eval_split_name 'test' \
		--num_train_epochs 6 \
		--output_dir "output/${MODEL}/hi/sparsity=${SPARSITIES[$(( $i + 1 ))]}/${EXP_NAME}" \
		--logging_steps 25 \
		--learning_rate ${LEARNING_RATE} \
		--weight_decay 1e-4 \
		--val_max_target_length "225" \
		--group_by_length \
		--length_column_name "input_length" \
		--max_duration_in_seconds "30" \
		--text_column_name "sentence" \
		--freeze_feature_encoder "False" \
		--gradient_checkpointing \
		--preprocessing_num_workers 8 \
		--checkpoint_before_pruning \
		--lr_scheduler_type linear \
		--per_device_train_batch_size $BATCH_SIZE_PER_GPU \
		--per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
		--num_warmup_steps 50 \
		--sparsification_config configs/fast_obc.yaml \
        --init_sparsity ${SPARSITIES[$(( $i + 1 ))]} \
        --final_sparsity ${SPARSITIES[$(( $i + 1 ))]} \
        --use_calibration_data \
        --calibration_dataset_size 512 \
        --calibration_batch_size 1 \
		--distillation \
		--orig_loss_weight 0.5 \
		--dist_loss_weight ${DIST_LOSS_WEIGHT} \
		--feat_loss_weight ${FEAT_LOSS_WEIGHT} \
		--feat_names '.*((en|de)coder.layers\.\d$)' \
		--feat_loss ${FEAT_LOSS} \
        --report_to wandb \
        --with_tracking
done
