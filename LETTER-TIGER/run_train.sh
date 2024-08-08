# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1

DATASET=Instruments
OUTPUT_DIR=./ckpt/$DATASET/

torchrun --nproc_per_node=2 --master_port=2314 ./finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file .index.json \
    --temperature 1.0
