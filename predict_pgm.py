#!/usr/bin/env python3

import os, sys, argparse, json, csv

# e3nn loading workaround for PyTorch >= 2.6
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import torch
from rdkit import Chem
import tqdm as tq

_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _ROOT)

# --------------------------------------------------------------------------
# Constants shared with the data preparation
# --------------------------------------------------------------------------
BOHR2ANG = 0.529177249
EDGE_RADIUS_BOHR = 9.45   # radius graph in Bohr
ELE_NUM = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]   # element one-hot order


MODELS = {
    "local_direct":    dict(arch="local",     target="direct"),
    "global_direct":   dict(arch="global",    target="direct"),
    "local_delta":     dict(arch="local",     target="delta"),
    "global_delta":    dict(arch="global",    target="delta"),
    "l2g_delta":       dict(arch="l2g",       target="delta"),
    "l2g_equiv_delta": dict(arch="l2g_equiv", target="delta"),
}


def read_fit_input(path):
    """
    Returns dict(charge, elements (N,), ivary_q (N,), ivary_p (D,), npol_per_atom (N,), title).
    """
    with open(path) as fh:
        lines = fh.read().splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip().lower().startswith("&end")) + 1
    except StopIteration:
        raise ValueError(f"{path}: no '&end' line found")
    body = [l for l in lines[start:]]

    k = 0
    while k < len(body) and not body[k].strip():
        k += 1
    if k + 3 > len(body):
        raise ValueError(f"{path}: truncated molecule block")
    _weight = body[k].strip(); title = body[k + 1].strip()
    hdr = body[k + 2].split()
    if len(hdr) < 2:
        raise ValueError(f"{path}: bad 'charge natoms ndipoles' line: {body[k + 2]!r}")
    charge, natoms = int(hdr[0]), int(hdr[1])
    ndip = int(hdr[2]) if len(hdr) > 2 else None
    k += 3
    elements, ivary_q, ivary_p, npol = [], [], [], []
    for a in range(natoms):
        parts = body[k + a].split()
        if len(parts) < 2:
            raise ValueError(f"{path}: bad atom line {a + 1}: {body[k + a]!r}")
        elements.append(int(parts[0]))
        ivary_q.append(int(parts[1]))
        dips = [int(x) for x in parts[2:]]
        ivary_p.extend(dips)
        npol.append(len(dips))
    if ndip is not None and ndip != len(ivary_p):
        raise ValueError(f"{path}: header says {ndip} dipoles but the atom lines carry {len(ivary_p)}")
    return dict(charge=charge, elements=np.array(elements), ivary_q=np.array(ivary_q),
                ivary_p=np.array(ivary_p), npol=np.array(npol), title=title)


def read_sdf(path):
    """
    Returns (atomic numbers (N,), coords (N,3) Angstrom, bonds [(i,j), ...] 0-based,
    formal charge).
    """
    suppl = Chem.SDMolSupplier(path, sanitize=False, removeHs=False)
    mol = None
    for m in suppl:
        mol = m
        break
    if mol is None:
        raise ValueError(f"{path}: RDKit could not read a molecule from this file")
    if mol.GetNumConformers() == 0:
        raise ValueError(f"{path}: no coordinates in the first molecule")
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
    crd = np.array(mol.GetConformer(0).GetPositions(), dtype=float)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    if not bonds:
        raise ValueError(f"{path}: the first molecule has no bond block")
    return z, crd, bonds, Chem.GetFormalCharge(mol)


def read_base_chg(path):
    q, p, mu = [], {}, []
    section = None
    with open(path) as fh:
        seen_title = False
        for line in fh:
            if line.startswith("%FLAG"):
                if "TITLE" in line:
                    if seen_title:
                        break
                    seen_title = True
                    section = None
                elif "ATOM CHRG" in line:
                    section = "q"; next(fh)
                elif "PERM DIP LOCAL" in line:
                    section = "p"; next(fh)
                elif "PERM DIP GLOBAL" in line:
                    section = "mu"; next(fh)
                else:
                    section = None
                continue
            parts = line.split()
            if not parts:
                section = None
                continue
            if section == "q":
                q.append(float(parts[3]))
            elif section == "p":
                p[(int(parts[1]) - 1, int(parts[2]) - 1)] = float(parts[4])
            elif section == "mu":
                mu.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(q), p, (np.array(mu) if mu else None)


def build_ppol_pairs(bonds, n_atoms, npol_expected, name):
    nbr = [set() for _ in range(n_atoms)]
    for i, j in bonds:
        if i == j or not (0 <= i < n_atoms and 0 <= j < n_atoms):
            raise ValueError(f"{name}: invalid bond ({i + 1}, {j + 1}) in the SDF")
        nbr[i].add(j); nbr[j].add(i)
    pairs = []
    for i in range(n_atoms):
        js = sorted(nbr[i])
        if len(js) != int(npol_expected[i]):
            raise ValueError(
                f"{name}: atom {i + 1} has {len(js)} bonded neighbours "
                f"({[j + 1 for j in js]}) but the input file lists {int(npol_expected[i])} "
                f"permanent dipoles for it -- SDF bond list / input file mismatch")
        pairs.extend((i, j) for j in js)
    return np.array(pairs, dtype=int).T.reshape(2, -1)


def build_edge_index(crd_bohr, r=EDGE_RADIUS_BOHR):
    n = crd_bohr.shape[0]
    d = np.linalg.norm(crd_bohr[:, None, :] - crd_bohr[None, :, :], axis=-1)
    mask = (d <= r)
    np.fill_diagonal(mask, False)
    src, dst = np.nonzero(mask)
    return np.stack([src, dst], axis=0)


def merge_ivary(ivary2, ivary1, what, name):
    if len(ivary1) != len(ivary2):
        raise ValueError(f"{name}: {what} ivary length differs between the two inputs "
                         f"({len(ivary2)} vs {len(ivary1)})")
    merged = np.zeros(len(ivary2), dtype=int)
    for i, iv2 in enumerate(ivary2):
        if iv2 > 0:
            merged[i] = iv2
        elif iv2 == -1:
            iv1 = ivary1[i]
            if iv1 > 0 and ivary2[iv1 - 1] == -1:
                merged[i] = iv1
            elif iv1 > 0:
                print(f"Warning: {name}: {what} entry {i + 1} frozen in stage 2 with stage-1 tie to "
                      f"entry {iv1}, which was refit in stage 2 -- tie dropped")
    return merged


def resolve_equiv(ivary):
    n = len(ivary)
    root = np.arange(n, dtype=int)
    for i in range(n):
        j, steps = i, 0
        while ivary[j] > 0:
            j = ivary[j] - 1
            steps += 1
            if steps > n:
                raise ValueError(f"ivary cycle detected at entry {i + 1}")
        root[i] = j
    return root


def prepare_molecule(name, in1_path, in2_path, sdf_path, need_equiv):
    fit = read_fit_input(in2_path)
    fit1 = read_fit_input(in1_path)
    z_sdf, crd_ang, bonds, q_sdf = read_sdf(sdf_path)
    n = len(fit["elements"])
    if len(z_sdf) != n:
        raise ValueError(f"{name}: {in2_path} has {n} atoms but {sdf_path} has {len(z_sdf)}")
    if np.any(z_sdf != fit["elements"]):
        bad = int(np.nonzero(z_sdf != fit["elements"])[0][0])
        raise ValueError(f"{name}: element mismatch at atom {bad + 1}: input file {fit['elements'][bad]}, "
                         f"sdf {z_sdf[bad]} -- same atoms in the same order are required")
    if len(fit1["elements"]) != n or np.any(fit1["elements"] != fit["elements"]) \
            or np.any(fit1["npol"] != fit["npol"]):
        raise ValueError(f"{name}: {in1_path} does not describe the same atoms / dipoles as {in2_path}")
    for z in fit["elements"]:
        if int(z) not in ELE_NUM:
            raise ValueError(f"{name}: element {int(z)} is not covered by the models "
                             f"(supported: {ELE_NUM})")

    charge = fit["charge"]
    if charge == -99:
        charge = int(q_sdf)
        print(f"Warning: {name}: input has no total charge (-99); using Q = {charge} from the SDF")

    crd_bohr = crd_ang / BOHR2ANG
    ppol = build_ppol_pairs(bonds, n, fit["npol"], name)
    node_feat = np.array([[1.0 if int(z) == e else 0.0 for e in ELE_NUM] for z in fit["elements"]])
    mol = dict(name=name, in1=in1_path, in2=in2_path, sdf=sdf_path, elements=fit["elements"], charge=charge,
               crd=crd_bohr, node_features=node_feat, edge_index=build_edge_index(crd_bohr),
               ppol_pairs=ppol)
    if need_equiv:
        ivq = merge_ivary(fit["ivary_q"], fit1["ivary_q"], "charge", name)
        ivp = merge_ivary(fit["ivary_p"], fit1["ivary_p"], "dipole", name)
        mol["equiv"] = resolve_equiv(ivq)
        mol["dip_equiv"] = resolve_equiv(ivp)
    return mol


def load_model(model_name, weights_dir, device):
    arch = MODELS[model_name]["arch"]
    if arch == "local":
        from model.local_model import l2E3NN
    elif arch == "global":
        from model.global_model import l2E3NN
    elif arch == "l2g":
        from model.l2g_model import l2E3NN
    else:
        from model.l2g_equiv_model import l2E3NN
    cfg_path = os.path.join(weights_dir, "config.json")
    ckpt_path = os.path.join(weights_dir, "best.pt")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    model = l2E3NN(
        in_scalar=len(ELE_NUM),
        hidden_scalar=cfg["hidden_scalar"], hidden_vector=cfg["hidden_vector"],
        hidden_tensor=cfg["hidden_tensor"], q_irreps=cfg["q_irreps"], p_irreps=cfg["p_irreps"],
        num_layers=cfg["num_layers"], lmax=cfg["lmax"], radial_hidden=cfg["radial_hidden"],
        num_radial_layers=cfg["num_radial_layers"], num_bessel=cfg["num_bessel"],
        num_polynomial_cutoff=cfg["num_polynomial_cutoff"], dropout=cfg["dropout"],
        head_layers=cfg.get("head_layers", 3),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def collate(mols, model_Q, device):
    nf, ei, crd, dip, Q, batch, eq, deq = [], [], [], [], [], [], [], []
    node_off = pair_off = 0
    for b, m in enumerate(mols):
        n = m["crd"].shape[0]
        nf.append(m["node_features"]); crd.append(m["crd"])
        ei.append(m["edge_index"] + node_off)
        dip.append(m["ppol_pairs"] + node_off)
        Q.append(float(model_Q(m)))
        batch.append(np.full(n, b, dtype=int))
        if "equiv" in m:
            eq.append(m["equiv"] + node_off); deq.append(m["dip_equiv"] + pair_off)
        node_off += n; pair_off += m["ppol_pairs"].shape[1]
    out = dict(
        node_f=torch.tensor(np.concatenate(nf, 0), dtype=torch.float32, device=device),
        crd=torch.tensor(np.concatenate(crd, 0), dtype=torch.float32, device=device),
        e_idx=torch.tensor(np.concatenate(ei, 1), dtype=torch.long, device=device),
        dip_idx=torch.tensor(np.concatenate(dip, 1), dtype=torch.long, device=device),
        Q=torch.tensor(Q, dtype=torch.float32, device=device),
        batch=torch.tensor(np.concatenate(batch, 0), dtype=torch.long, device=device),
    )
    if eq:
        out["equiv"] = torch.tensor(np.concatenate(eq, 0), dtype=torch.long, device=device)
        out["dip_equiv"] = torch.tensor(np.concatenate(deq, 0), dtype=torch.long, device=device)
    return out


@torch.no_grad()
def run_batch(model, arch, b):
    if arch == "local":
        q, p = model(b["node_f"], b["crd"], b["e_idx"], b["dip_idx"], b["Q"], b["batch"])
        return q.squeeze(1).cpu().numpy(), p.squeeze(1).cpu().numpy(), None
    if arch == "global":
        q, mu = model(b["node_f"], b["crd"], b["e_idx"], b["dip_idx"], b["Q"], b["batch"])
        return q.squeeze(1).cpu().numpy(), None, mu.cpu().numpy()
    if arch == "l2g":
        q, mu, p = model(b["node_f"], b["crd"], b["e_idx"], b["dip_idx"], b["Q"], b["batch"],
                         return_local=True)
    else:
        q, mu, p = model(b["node_f"], b["crd"], b["e_idx"], b["dip_idx"], b["Q"], b["batch"],
                         b["equiv"], dip_equiv=b["dip_equiv"],
                         return_local=True)
    return q.squeeze(1).cpu().numpy(), p.squeeze(1).cpu().numpy(), mu.cpu().numpy()


def write_prediction(path, mol, model_name, q, p, mu, base):
    info = MODELS[model_name]
    delta = info["target"] == "delta"
    n = len(q)
    pairs = mol["ppol_pairs"]
    bq = bp = bmu = None
    if base is not None:
        bq, bp_dict, bmu = base
        if len(bq) != n:
            raise ValueError(f"{mol['name']}: baseline .chg has {len(bq)} atoms, molecule has {n}")
        if p is not None:
            try:
                bp = np.array([bp_dict[(int(i), int(j))] for i, j in pairs.T])
            except KeyError as e:
                raise ValueError(f"{mol['name']}: baseline .chg lacks dipole pair {e}") from None
        if mu is not None and bmu is not None and bmu.shape != (n, 3):
            bmu = None
    with open(path, "w") as fh:
        fh.write(f"# molecule: {mol['name']}\n")
        fh.write(f"# model: {model_name}  "
                 f"({'values are WATZ - W321 corrections' if delta else 'absolute WATZ-level values'})\n")
        fh.write(f"# in1: {mol['in1']}   in2: {mol['in2']}   sdf: {mol['sdf']}\n")
        fh.write(f"# n_atoms: {n}   n_dipole_pairs: {pairs.shape[1]}   total_charge_Q: {mol['charge']}\n")
        tot = "  total" if base is not None else ""
        fh.write(f"# Atomic charges (e):  index  element  q{tot}\n")
        for a in range(n):
            fh.write(f"{a + 1:4d}  {int(mol['elements'][a]):3d}  {q[a]: .8f}"
                     + (f"  {q[a] + bq[a]: .8f}" if bq is not None else "") + "\n")
        if p is not None:
            fh.write(f"# Bond dipole magnitudes (e*Bohr, local frame):  index  i  j  p{tot if bp is not None else ''}\n")
            for k in range(pairs.shape[1]):
                fh.write(f"{k + 1:4d}  {int(pairs[0, k]) + 1:4d}  {int(pairs[1, k]) + 1:4d}  {p[k]: .8f}"
                         + (f"  {p[k] + bp[k]: .8f}" if bp is not None else "") + "\n")
        if mu is not None:
            fh.write(f"# Atomic dipole vectors (e*Bohr, global frame):  index  ux uy uz"
                     f"{'  total_ux total_uy total_uz' if bmu is not None else ''}\n")
            for a in range(n):
                fh.write(f"{a + 1:4d}  {mu[a, 0]: .8f} {mu[a, 1]: .8f} {mu[a, 2]: .8f}"
                         + (f"  {mu[a, 0] + bmu[a, 0]: .8f} {mu[a, 1] + bmu[a, 1]: .8f} {mu[a, 2] + bmu[a, 2]: .8f}"
                            if bmu is not None else "") + "\n")



def parse_jobs(args):
    jobs = []
    if args.list:
        with open(args.list) as fh:
            for ln, line in enumerate(fh, 1):
                parts = line.split()
                if not parts or parts[0].startswith("#"):
                    continue
                if len(parts) < 4:
                    raise ValueError(f"{args.list}:{ln}: expected 'name in1 in2 sdf [base_chg]'")
                name, in1, in2, sdf = parts[:4]
                base = parts[4] if len(parts) > 4 else None
                jobs.append([name, in1, in2, sdf, base])
    if args.in1 or args.in2 or args.sdf:
        n = len(args.in1 or [])
        if not (args.in1 and args.in2 and args.sdf) or len(args.in2) != n or len(args.sdf) != n:
            raise ValueError("--in1, --in2 and --sdf must all be given simultaneously")
        if args.base_chg and len(args.base_chg) != n:
            raise ValueError("--base_chg must have one file per molecule")
        for k, (in1, in2, sdf) in enumerate(zip(args.in1, args.in2, args.sdf)):
            name = os.path.splitext(os.path.basename(sdf))[0]
            jobs.append([name, in1, in2, sdf, args.base_chg[k] if args.base_chg else None])
    if not jobs:
        raise ValueError("no molecules given")
    if args.base_chg_dir:
        for j in jobs:
            if j[4] is None:
                j[4] = os.path.join(args.base_chg_dir, j[0] + args.base_chg_suffix)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS), help="which model to use")
    ap.add_argument("--in1", nargs="+", help="stage 1 PyRESP / PCMRESP input file(s)")
    ap.add_argument("--in2", nargs="+", help="stage 2 PyRESP / PCMRESP input file(s), one per --in1")
    ap.add_argument("--sdf", nargs="+", help="SDF file(s) with bond records, one per --in1, same atom order "
                    "(only the first molecule of each SDF file is used)")
    ap.add_argument("--list", help="another option to use a more compact form of input, " \
                    "text file with one molecule per line: name in1 in2 sdf [base_chg]," \
                    "if [base_chg] is used, it must be provided for every molecule")
    ap.add_argument("--outdir", default="pgm_pred")
    ap.add_argument("--batch_size", type=int, default=8, help="molecules per forward pass for multi-molecule run")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--weights_dir", default=None,
                    help="directory with best.pt and config.json (default: train/out/<model>)")
    ap.add_argument("--base_chg", nargs="+", help="delta models only. " \
                    "Predicted values are added to the W321 baseline .chg file(s), " \
                    "one per molecule")
    ap.add_argument("--base_chg_dir", help="delta models only: directory of baseline .chg files named "
                    "<name><base_chg_suffix>")
    ap.add_argument("--base_chg_suffix", default="_2nd.chg")
    args = ap.parse_args()

    info = MODELS[args.model]
    need_equiv = info["arch"] == "l2g_equiv"
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    weights_dir = args.weights_dir or os.path.join(_ROOT, "train", "out", args.model)
    jobs = parse_jobs(args)
    if info["target"] == "direct" and any(j[4] for j in jobs):
        print("Warning: --base_chg is only meaningful for delta models; ignored")
        for j in jobs:
            j[4] = None

    model, cfg = load_model(args.model, weights_dir, device)
    print(f"Initializing model {args.model} from {weights_dir} on {device}", flush=True)

    # the charge constraint fed to the network: true Q for direct models, 0 for delta models
    model_Q = (lambda m: m["charge"]) if info["target"] == "direct" else (lambda m: 0)

    os.makedirs(args.outdir, exist_ok=True)
    mols, failed = [], []
    for name, in1, in2, sdf, base in jobs:
        try:
            m = prepare_molecule(name, in1, in2, sdf, need_equiv)
            m["base"] = read_base_chg(base) if base else None
            mols.append(m)
        except Exception as e:
            print(f"Warning: molecule {name} failed, {e}", flush=True)
            failed.append(name)

    summary = []
    for s in tq.tqdm(range(0, len(mols), args.batch_size), desc="Predicting batches"):
        chunk = mols[s:s + args.batch_size]
        b = collate(chunk, model_Q, device)
        q, p, mu = run_batch(model, info["arch"], b)
        batch_np = b["batch"].cpu().numpy()
        dip_mol = batch_np[b["dip_idx"][0].cpu().numpy()]
        for k, m in enumerate(chunk):
            nm = batch_np == k
            pm = dip_mol == k
            qk = q[nm]; pk = p[pm] if p is not None else None; muk = mu[nm] if mu is not None else None
            out = os.path.join(args.outdir, f"{m['name']}.txt")
            write_prediction(out, m, args.model, qk, pk, muk, m["base"])
            summary.append((m["name"], len(qk), m["ppol_pairs"].shape[1], m["charge"], float(qk.sum()), out))

    with open(os.path.join(args.outdir, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "n_atoms", "n_dipole_pairs", "Q", "sum_pred_q", "file"])
        w.writerows(summary)
    if failed:
        with open(os.path.join(args.outdir, "failed.txt"), "w") as fh:
            fh.write("\n".join(failed) + "\n")
    print(f"Job completed. {len(failed)} molecule(s) failed. See {os.path.join(args.outdir, 'failed.txt') if failed else 'no failures'}")


if __name__ == "__main__":
    main()
