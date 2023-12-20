export NCCL_DEBUG=INFO
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3

export PATH=/usr/local/cuda/bin:$PATH

# manifest_name=en-us_clean_matrix_test
# manifest_name=en-us_clean_nexus6_test
# manifest_name=en-us_clean_pseye_test
# manifest_name=en-us_clean_respeaker_test
# manifest_name=en-us_clean_shure_test
# manifest_name=en-us_clean_usb_test
# manifest_name=en-us_wind_matrix_test
# manifest_name=en-us_laughter_matrix_test
# manifest_name=en-us_rain_matrix_test
manifest_name=


whisper_path=pretrained_models/whisper-small
# DATA_ROOT=data/librispeech-evaluate/librispeech-test-clean/
DATA_ROOT=data/now_how2_val
# DATA_ROOT=data/librispeech-debug
SAVE_ROOT=checkpoints/now_how2_v2
SAVE_RESULT=results/
mkdir -p $SAVE_RESULT
mkdir -p $SAVE_ROOT

# python -m torch.distributed.run --nproc_per_node=1 --master_port=52022 test_model.py \
python test_model.py \
    --data $DATA_ROOT \
    --output_dir $SAVE_ROOT \
    --manifest_files "*.jsonl" \
    --remove_unused_columns False \
    --seed 1 \
    --do_train False \
    --evaluation_strategy 'steps' \
    --eval_accumulation_steps 1 \
    --label_names "labels" \
    --bf16  True \
    \
    --learning_rate 1.25e-6 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps 1000 \
    --per_device_eval_batch_size 8 \
    \
    --whisper_model $whisper_path \
    \
    --disable_tqdm False \
    \
    --logging_steps 20 \
    --save_steps 200 \
    --save_total_limit 1 \
    --overwrite_output_dir > ${SAVE_RESULT}/how2_val.txt
            # --predict_with_generate True \
