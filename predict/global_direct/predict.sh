#!/bin/bash

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # for e3nn loading

python ../../scripts/predict_global.py \
    --ckpt ../../train/out/global_direct/best.pt \
    --config ../../train/out/global_direct/config.json \
    --data ../../data/global_direct/test.h5 \
    --outdir . \
    --batch_size 8 \
    --device cuda:0 \
    --num_workers 2 "$@"
