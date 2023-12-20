export NCCL_DEBUG=INFO
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PATH=/usr/local/cuda/bin:$PATH


whisper_path=pretrained_models/whisper-small
clip_path=pretrained_models/clip-vit-base-patch32
DATA_ROOT=data/now_how2
SAVE_ROOT=checkpoints/now_how2_3_self
mkdir -p $SAVE_ROOT

python -m torch.distributed.run --nproc_per_node=3 --master_port=52022 train.py \
    --deepspeed config/dp_config_zero1.json \
    --data $DATA_ROOT \
    --output_dir ${SAVE_ROOT} \
    --manifest_files "*.jsonl" \
    --remove_unused_columns False \
    --seed 1 \
    --do_train True \
    --bf16  True \
    \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps 1000 \
    \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 20 \
    \
    --whisper_model $whisper_path \
    \
    --disable_tqdm True \
    \
    --logging_steps 20 \
    --save_steps 600 \
    --save_total_limit 1 \
    --overwrite_output_dir 