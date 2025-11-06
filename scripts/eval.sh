#!/bin/bash

# Single GPU evaluation (no distributed training)
# Clear all distributed-related environment variables to ensure single GPU mode
unset RANK
unset WORLD_SIZE
unset MASTER_ADDR
unset MASTER_PORT
unset LOCAL_RANK
unset OMPI_COMM_WORLD_RANK
unset OMPI_COMM_WORLD_SIZE
unset OMPI_COMM_WORLD_LOCAL_RANK
unset SLURM_PROCID


PY_ARGS=${@:1}  # Any other arguments 
python main_finetune.py \
    --model AIDE \
    --batch_size 32 \
    --blr 5e-4 \
    --epochs 5 \
    --device cuda \
    --dist_eval False \
    ${PY_ARGS} 
