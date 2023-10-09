#!/bin/bash
export CUDA_VISIBLE_DEVICES=#TODO
export OMP_NUM_THREADS=#TODO

NUM_PROC=$(wc -w <<< $(tr ',' ' ' <<< $CUDA_VISIBLE_DEVICES))

MODEL=t5-small
BATCH_SIZE_PER_GPU=$(( 128 / ${NUM_PROC} ))

export WANDB_ENTITY=#TODO
export WANDB_PROJECT=#TODO

LEARNING_RATE=2e-3
SPARSITIES=(0.50 0.67 0.75 0.80 0.83 0.86 0.88 0.90)
EXP_NAME=standard_kd
DIST_LOSS_WEIGHT=1.0
FEAT_LOSS_WEIGHT=0.0

# pruning to 0.50
export WANDB_NAME=${MODEL},FastOBC,sparsity=0.50,${EXP_NAME}
accelerate launch --multi_gpu --num_processes=${NUM_PROC} --main_process_port=29501 run_translation.py \
    --model_name_or_path output/${MODEL}/dense_training \
    --teacher_model_name_or_path output/${MODEL}/dense_training \
    --source_lang en \
    --target_lang de \
    --source_prefix "translate English to German: " \
    --dataset_name wmt14 \
    --dataset_config_name de-en \
    --output_dir output/${MODEL}/sparsity=0.50/${EXP_NAME} \
    --num_train_epochs 3 \
    --preprocessing_num_workers 8 \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 1e-4 \
    --num_beams 5 \
    --lr_scheduler_type linear \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --num_warmup_steps 300 \
    --eval_steps 3000 \
    --sparsification_config configs/fast_obc.yaml \
    --init_sparsity 0.50 \
    --final_sparsity 0.50 \
    --use_calibration_data \
    --calibration_dataset_size 512 \
    --calibration_batch_size 1 \
    --distillation \
    --orig_loss_weight 1.0 \
    --dist_loss_weight ${DIST_LOSS_WEIGHT} \
    --feat_loss_weight ${FEAT_LOSS_WEIGHT} \
    --report_to wandb \
    --with_tracking

for (( i=0; i < $(( ${#SPARSITIES[@]}-1)); i++))
do
    export WANDB_NAME="${MODEL},FastOBC,sparsity=${SPARSITIES[$(( $i + 1 ))]},${EXP_NAME}"
    accelerate launch --multi_gpu --num_processes=${NUM_PROC} --main_process_port=29502 run_translation.py \
        --model_name_or_path "output/${MODEL}/sparsity=${SPARSITIES[$i]}/${EXP_NAME}" \
        --teacher_model_name_or_path output/${MODEL}/dense_training \
        --source_lang en \
        --target_lang de \
        --source_prefix "translate English to German: " \
        --dataset_name wmt14 \
        --dataset_config_name de-en \
        --output_dir "output/${MODEL}/sparsity=${SPARSITIES[$(( $i + 1 ))]}/${EXP_NAME}" \
        --num_train_epochs 3 \
        --preprocessing_num_workers 8 \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay 1e-4 \
        --num_beams 5 \
        --lr_scheduler_type linear \
        --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
        --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
        --num_warmup_steps 300 \
        --eval_steps 3000 \
        --sparsification_config configs/fast_obc.yaml \
        --init_sparsity ${SPARSITIES[$(( $i + 1 ))]} \
        --final_sparsity ${SPARSITIES[$(( $i + 1 ))]} \
        --use_calibration_data \
        --calibration_dataset_size 512 \
        --calibration_batch_size 1 \
        --distillation \
        --orig_loss_weight 1.0 \
        --dist_loss_weight ${DIST_LOSS_WEIGHT} \
        --feat_loss_weight ${FEAT_LOSS_WEIGHT} \
        --report_to wandb \
        --with_tracking
done
