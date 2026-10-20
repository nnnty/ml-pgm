#!/bin/bash

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # for e3nn loading

python ../../scripts/predict_local.py \
    --ckpt ../../train/out/local_delta/best.pt \
    --config ../../train/out/local_delta/config.json \
    --data ../../data/local_delta/test.h5 \
    --outdir . \
    --batch_size 8 \
    --device cuda:0 \
    --num_workers 2 "$@"
