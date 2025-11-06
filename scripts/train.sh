#!/bin/bash

# Single GPU training script (no distributed training)
# Usage: bash scripts/train_single_gpu.sh [other arguments]

# Get all arguments passed to the script
PY_ARGS="$@"

# Run the training script directly without torch.distributed.launch
python main_finetune.py \
    --model AIDE \
    --batch_size 32 \
    --blr 1e-5 \
    --epochs 20 \
    $PY_ARGS