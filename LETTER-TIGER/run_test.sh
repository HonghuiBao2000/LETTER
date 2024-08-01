DATASET=Instruments
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET/
RESULTS_FILE=./results/$DATASET/xxx.json
CKPT_PATH=./ckpt/$DATASET/

python test.py \
    --gpu_id 2 \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 32 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.json
