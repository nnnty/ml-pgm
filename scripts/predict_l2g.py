#!/usr/bin/env python3

import os, sys, argparse, json, importlib.util
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(_PROJECT_ROOT)
from model.l2g_model import l2E3NN

# Reuse some modules from training script
TRAIN_PATH = os.path.join(_PROJECT_ROOT, "scripts", "train_l2g.py")
_spec = importlib.util.spec_from_file_location("train_l2g", TRAIN_PATH)
_train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_mod)
ChgDataset = _train_mod.ChgDataset
collate_batched = _train_mod.collate_batched
to_device = _train_mod.to_device


def write_per_mol_txt(path, name, n_atoms, n_dipoles, Q, ref_q, pred_q,
                      pairs, ref_uloc, pred_uloc, ref_u, pred_u):
    with open(path, "w") as fh:
        fh.write(f"# molecule: {name}\n")
        fh.write(f"# n_atoms: {n_atoms}   n_dipole_pairs: {n_dipoles}   total_charge_Q: {Q}\n")
        fh.write("# Atomic charges (e):  index  ref_q  pred_q\n")
        for a in range(n_atoms):
            fh.write(f"{a+1:4d}  {ref_q[a]: .8f}  {pred_q[a]: .8f}\n")
        fh.write("# Bond dipole magnitudes (e*Bohr, local frame):  index  i  j  ref_u  pred_u\n")
        for k in range(n_dipoles):
            i, j = int(pairs[0, k]), int(pairs[1, k])
            fh.write(f"{k+1:4d}  {i+1:4d}  {j+1:4d}  {ref_uloc[k]: .8f}  {pred_uloc[k]: .8f}\n")
        fh.write("# Atomic dipole vectors (e*Bohr, global frame):  "
                 "index  ref_ux ref_uy ref_uz  pred_ux pred_uy pred_uz\n")
        for a in range(n_atoms):
            fh.write(
                f"{a+1:4d}  "
                f"{ref_u[a,0]: .8f} {ref_u[a,1]: .8f} {ref_u[a,2]: .8f}  "
                f"{pred_u[a,0]: .8f} {pred_u[a,1]: .8f} {pred_u[a,2]: .8f}\n"
            )


def build_model(cfg, in_scalar, device):
    return l2E3NN(
        in_scalar=in_scalar,
        hidden_scalar=cfg["hidden_scalar"],
        hidden_vector=cfg["hidden_vector"],
        hidden_tensor=cfg["hidden_tensor"],
        q_irreps=cfg["q_irreps"],
        p_irreps=cfg["p_irreps"],
        num_layers=cfg["num_layers"],
        lmax=cfg["lmax"],
        radial_hidden=cfg["radial_hidden"],
        num_radial_layers=cfg["num_radial_layers"],
        num_bessel=cfg["num_bessel"],
        num_polynomial_cutoff=cfg["num_polynomial_cutoff"],
        dropout=cfg["dropout"],
        head_layers=cfg.get("head_layers", 3),
    ).to(device)


def main():
    ap = argparse.ArgumentParser(description="Script for L2G model prediction")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, 
                    help="Only HDF5 file with specific structures can be used for this script.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.config, "r") as f:
        cfg = json.load(f)

    per_mol_dir = os.path.join(args.outdir, "per_mol")
    os.makedirs(per_mol_dir, exist_ok=True)

    data_set = ChgDataset(args.data, dtype=torch.float32)

    # The trainer's ChgDataset does not hand out targets_u_loc; look it up by name.
    name_to_idx = {n: i for i, n in enumerate(data_set.data["idx"])}
    has_uloc = "targets_u_loc" in data_set.data

    sample_feat_dim = data_set.in_node_nf

    val_loader = DataLoader(
        data_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_batched, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Build model
    model = build_model(cfg, sample_feat_dim, device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Inference
    n_written = 0
    with torch.no_grad():
        for node_f, e_idx, crd, dip_idx, ref_chg, ref_u, Q, batch_idx, names in tqdm.tqdm(val_loader, desc="predicting"):
            node_f_d, e_idx_d, crd_d, dip_idx_d, Q_d, batch_idx_d = to_device(
                (node_f, e_idx, crd, dip_idx, Q, batch_idx), device
            )

            pred_q, pred_u, pred_uloc = model(node_f_d, crd_d, e_idx_d, dip_idx_d, Q_d, batch_idx_d,
                                              return_local=True)
            pred_q    = pred_q.squeeze(1).cpu().numpy()    # (N_total,)
            pred_u    = pred_u.cpu().numpy()               # (N_total, 3)
            pred_uloc = pred_uloc.squeeze(1).cpu().numpy() # (D_total,)

            ref_chg_np = ref_chg.numpy()
            ref_u_np   = ref_u.numpy()                     # (N_total, 3)
            batch_np   = batch_idx.numpy()
            dip_np     = dip_idx.numpy()                   # (2, D_total), offset node ids
            dip_batch  = batch_np[dip_np[0]]               # molecule id of each pair
            Q_np       = Q.numpy().astype(int)

            for m in range(len(names)):
                node_mask = batch_np == m
                n_atoms   = int(node_mask.sum())
                offset    = int(np.nonzero(node_mask)[0][0])

                pair_mask = dip_batch == m
                n_dip     = int(pair_mask.sum())
                pairs     = dip_np[:, pair_mask] - offset    # (2, D_i), 0-based within molecule

                if has_uloc:
                    ruloc = np.asarray(data_set.data["targets_u_loc"][name_to_idx[names[m]]],
                                       dtype=np.float64)
                    if ruloc.shape[0] != n_dip:
                        raise RuntimeError(f"{names[m]}: targets_u_loc has {ruloc.shape[0]} "
                                           f"entries but ppol_pairs has {n_dip}")
                else:
                    ruloc = np.full(n_dip, np.nan)

                write_per_mol_txt(
                    os.path.join(per_mol_dir, f"{names[m]}.txt"),
                    names[m], n_atoms, n_dip, int(Q_np[m]),
                    ref_chg_np[node_mask], pred_q[node_mask],
                    pairs, ruloc, pred_uloc[pair_mask],
                    ref_u_np[node_mask], pred_u[node_mask],
                )
                n_written += 1

if __name__ == "__main__":
    main()
