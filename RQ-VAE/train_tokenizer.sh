python ./RQ-VAE/main.py \
  --device cuda:0 \
  --data_path ../data/Instruments/Instruments.emb-llama-td.npy\
  --alpha 0.01 \
  --beta 0.0001 \
  --cf_emb ./RQ-VAE/ckpt/Instruments-32d-sasrec.pt\
  --ckpt_dir ../checkpoint/