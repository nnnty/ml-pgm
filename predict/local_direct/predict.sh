#!/bin/bash

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # for e3nn loading

python ../../scripts/predict_local.py \
    --ckpt ../../train/out/local_direct/best.pt \
    --config ../../train/out/local_direct/config.json \
    --data ../../data/local_direct/test.h5 \
    --outdir . \
    --batch_size 8 \
    --device cuda:0 \
    --num_workers 2 "$@"
