# ML-pGM

This is a repository of machine learning models for pGM parameterization.

| Model | Target | Model code | Trainer / predictor | Weights & Config | Data |
|---|---|---|---|---|---|
| Local frame model      | direct | `model/local_model.py`     | `scripts/train_local.py` / `scripts/predict_local.py`         | `train/out/local_direct/`    | `data/local_direct/`  |
| Global frame model     | direct | `model/global_model.py`    | `scripts/train_global.py` / `scripts/predict_global.py`       | `train/out/global_direct/`   | `data/global_direct/` |
| Local frame model      | Δ      | `model/local_model.py`     | `scripts/train_local.py` / `scripts/predict_local.py`         | `train/out/local_delta/`     | `data/local_delta/`   |
| Global frame model     | Δ      | `model/global_model.py`    | `scripts/train_global.py` / `scripts/predict_global.py`       | `train/out/global_delta/`    | `data/global_delta/`  |
| L2G    (local-to-global)        | Δ      | `model/l2g_model.py`       | `scripts/train_l2g.py` / `scripts/predict_l2g.py`             | `train/out/l2g_delta/`       | `data/l2g_delta/`     |
| L2G-Equiv (L2G + equivalence)   | Δ      | `model/l2g_equiv_model.py` | `scripts/train_l2g_equiv.py` / `scripts/predict_l2g_equiv.py` | `train/out/l2g_equiv_delta/` | `data/l2g_delta/`     |

Annotation: direct = absolute pGM parameters fitted using wB97XD/aug-cc-pVTZ ESP;
"Δ" = Δ-learning of wB97XD/aug-cc-pVTZ − wB97XD/3-21G corrections.

```
data/<dataset>/{train,val,test}.h5     featurized molecules; coordinates in Bohr
model/*.py                             the four architectures
scripts/train_*.py                     trainers, one per architecture
scripts/predict_*.py                   script for running predictions on h5 files
train/train_<model>.sh                 model training scripts
train/out/<model>/{best.pt,config.json,train_loss.txt} model weight and config
predict/<model>/predict.sh             front-end for running the predictions on the standardized test split
predict_pgm.py                         stand-alone prediction script for new molecules (pGM fit inputs + SDF)
example/                               three molecules and a list file for predict_pgm.py
```

## Environment

Python >= 3.10 with `torch`, `e3nn`, `h5py`, `numpy`, `scipy`, `rdkit`, `tqdm` (models were developed with torch 2.12 / e3nn 0.6).


## Predict pGM parameters for molecules: `predict_pgm.py`

```
# Option 1: run prediction by explicitly giving required input files
python predict_pgm.py --model l2g_equiv_delta \
    --in1 mol.in1 --in2 mol.in2 --sdf mol.sdf --outdir pred

# Option 2: run prediction by using a compact, formatted molecule list
python predict_pgm.py --model global_direct --list mols.txt --batch_size 16 --outdir pred
#   mols.txt:  name  stage1_input  stage2_input  sdf_file  [baseline_chg]
```

* `--in1` / `--in2` are the stage-1 and stage-2 PyRESP / PCMRESP inputs of
  the two-stage pGM fit; both are always required. Only the first molecule
  block of each is read. They supply the elements, the net charge, the
  number of permanent dipoles per atom and the equivalence (ivary) columns.
* `--sdf` gives the geometry (Angstrom) and the bond records (required). The molecule must have the same atom order as in PyRESP / PCMRESP inputs.
* The L2G-Equiv model always applies the equivalence constraints. The atom
  and dipole-pair classes are read from the ivary columns of the two inputs.
  The equivalence relations from both files will be merged using set union.
* `--batch_size` molecules are packed into one forward pass.
* Δ models output corrections. Pass the PyRESP / PCMRESP output parameter file(s)
  (`--base_chg ...`, `--base_chg_dir DIR [--base_chg_suffix _2nd.chg]`, or
  the 5th column of the list file) to also get the summed values
  in `total` columns.
* Output: `<outdir>/<name>.txt` per molecule (atomic units) with the atomic
  charges, the local-frame bond dipole scalars (Local, L2G, L2G-Equiv) and/or
  the global-frame atomic dipole vectors (Global, L2G, L2G-Equiv), plus a summary file
  `summary.csv`.
* `example/` contains a minimal example to run `predict_pgm.py` using its `--list` option.

## Reproduce predictions on the test split

```
cd predict/l2g_equiv_delta && bash predict.sh        # Output: per_mol/<mol>.txt
```

## Retrain

```
cd train && bash train_l2g_equiv_delta.sh
```
