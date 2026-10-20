#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_CUMEM_ENABLE=0 # cluster specific setting
export NCCL_P2P_DISABLE=1 # cluster specific setting
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 # for e3nn loading

job_name=local_direct
out_dir=./out/$job_name

torchrun --nproc_per_node=2 ../scripts/train_local.py \
    --train_data ../data/local_direct/train.h5 \
    --val_data ../data/local_direct/val.h5 \
    --hidden_scalar 128 \
    --hidden_vector 64 \
    --hidden_tensor 32 \
    --q_irreps 256 \
    --p_irreps 256 \
    --num_layers 5 \
    --lmax 2 \
    --radial_hidden 128 \
    --head_layers 3 \
    --epochs 200 \
    --batch_size 16 \
    --lr 0.001 \
    --grad_clip 1.0 \
    --weight_decay 0.001 \
    --dropout 0.0 \
    --num_workers 8 \
    --outdir $out_dir
