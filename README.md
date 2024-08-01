# LETTER

This is the pytorch implementation of our paper:

> [Learnable Item Tokenization for Generative Recommendation](https://arxiv.org/abs/2405.07314)

## Overview
We propose LETTER (a LEarnable Tokenizer for generaTivE Recommendation), which integrates hierarchical semantics, collaborative signals, and code assignment diversity to satisfy the essential requirements of identifiers. 
LETTER incorporates Residual Quantized VAE for semantic regularization, a contrastive alignment loss for collaborative regularization, and a diversity loss to mitigate code assignment bias. We instantiate LETTER on two generative recommender models and propose a ranking-guided generation loss to augment their ranking ability theoretically. 

![image.png](https://s2.loli.net/2024/05/12/PveBMV23SRa1lrJ.png)

## Requirements

```
torch==1.13.1+cu117
accelerate
bitsandbytes
deepspeed
evaluate
peft
sentencepiece
tqdm
transformers
```

## LETTER Tokenizer

### Train

```
bash LETTER-TIGER/train_tokenizer.sh 
```

### Tokenize

```
bash LETTER-TIGER/tokenize.sh 
```

## Instantiation

### LETTER-TIGER

```
cd LETTER-TIGER
bash run_train.sh
```

### LETTER-LC-Rec

```
cd LETTER-LC-Rec
bash run_train.sh
```
