#!/bin/bash

cd "$(dirname "$0")"

python ../predict_pgm.py --model global_direct --list mols.txt --batch_size 3 --outdir pred_global_direct

python ../predict_pgm.py --model l2g_equiv_delta --list mols.txt --batch_size 3 \
    --base_chg_dir base_chg --outdir pred_l2g_equiv_delta
