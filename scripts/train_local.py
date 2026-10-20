#!/usr/bin/env python3

import os, sys, argparse, json, random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import tqdm
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.local_model import l2E3NN

import warnings, re
warnings.filterwarnings(
    "ignore",
    message=re.escape("The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`. Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`."),
    category=UserWarning,
    module="torch",
)

debug = sys.flags.debug


def set_seed(seed: int, rank: int = 0):
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def to_device(batch, device):
    return tuple(x.to(device, non_blocking=True) for x in batch)


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


class ChgDataset(Dataset):
    """
    Load HDF5 data.

    Each molecule group consists:
      - node_features : (N, F)   scalar node features
      - crds          : (N, 3)   coordinates
      - edge_index    : (2, E)   graph connectivity
      - ppol_pairs    : (2, D)   directed covalent-basis pairs (owner, reference)
      - targets_chg   : (N,)     per-node charge target
      - targets_u     : (D,)     per-pair local dipole scalar target
      - targets_Q     : ()       total formal charge
    """
    def __init__(self, data_file, dtype=torch.float32):
        self.data_file = data_file
        self.data = self.load_data_h5(data_file)
        self.dtype = dtype
        self.length = len(self.data['idx'])
        self.in_node_nf = self.data['node_features'][0].shape[-1]

    def load_data_h5(self, data_file):
        data = {}
        with h5py.File(data_file, 'r') as f:
            root = f['mols']
            for mol_name in root.keys():
                if 'idx' not in data:
                    data['idx'] = []
                data['idx'].append(mol_name)
                g = root[mol_name]
                for key in g.keys():
                    if key not in data:
                        data[key] = []
                    data[key].append(g[key][()])
        return data

    def __len__(self):
        return len(self.data['idx'])

    def __getitem__(self, idx):
        name = self.data['idx'][idx]
        h_f_i = torch.tensor(self.data['node_features'][idx], dtype=self.dtype)
        e_idx_i = torch.tensor(self.data['edge_index'][idx], dtype=torch.long)
        crd_i = torch.tensor(self.data['crds'][idx], dtype=self.dtype)
        ppol_pairs_i = torch.tensor(self.data['ppol_pairs'][idx], dtype=torch.long)
        targets_chg_i = torch.tensor(self.data['targets_chg'][idx], dtype=self.dtype)
        targets_u_i = torch.tensor(self.data['targets_u'][idx], dtype=self.dtype)
        targets_Q_i = torch.tensor(self.data['targets_Q'][idx], dtype=torch.int)
        return h_f_i, e_idx_i, crd_i, ppol_pairs_i, targets_chg_i, targets_u_i, targets_Q_i, name


def collate_batched(batch):

    node_feats_list, edge_idx_list, crd_list = [], [], []
    dip_idx_list, ref_chg_list, ref_u_list = [], [], []
    Q_list, batch_idx_list, names = [], [], []
    node_offset = 0

    for i, (h_f, e_idx, crd, ppol_pair, ref_chg, ref_u, ref_Q, name) in enumerate(batch):
        N = h_f.shape[0]
        node_feats_list.append(h_f)
        edge_idx_list.append(e_idx + node_offset)          # (2, E_i)
        crd_list.append(crd)

        # ppol_pairs: stored as (D, 2) in HDF5, model expects (2, D)
        if ppol_pair.dim() == 2 and ppol_pair.shape[1] == 2:
            dip_idx_list.append(ppol_pair.T + node_offset)  # (2, D_i)
        else:
            dip_idx_list.append(ppol_pair + node_offset)    # already (2, D_i)

        ref_chg_list.append(ref_chg)
        ref_u_list.append(ref_u)
        Q_list.append(ref_Q.float())
        batch_idx_list.append(torch.full((N,), i, dtype=torch.long))
        names.append(name)
        node_offset += N

    return (
        torch.cat(node_feats_list, dim=0),      # (N_total, F)
        torch.cat(edge_idx_list, dim=1),         # (2, E_total)
        torch.cat(crd_list, dim=0),              # (N_total, 3)
        torch.cat(dip_idx_list, dim=1),          # (2, D_total)
        torch.cat(ref_chg_list, dim=0),          # (N_total,)
        torch.cat(ref_u_list, dim=0),            # (D_total,)
        torch.stack(Q_list, dim=0),              # (B,)
        torch.cat(batch_idx_list, dim=0),        # (N_total,)
        names,                                    # list[str]
    )


def loss_fn(pred, target):
    return nn.MSELoss()(pred, target)

def comb_loss(loss_chg, loss_u):
    return loss_chg + loss_u


def train_one_epoch(model, loader, optimizer, device,
                    grad_clip=None, scaler=None, use_amp=False):
    model.train()
    total_loss = 0.0
    total_count = 0

    iterator = tqdm.tqdm(loader, desc="Training Batches") if is_main_process() else loader

    for node_f, e_idx, crd, dip_idx, ref_chg, ref_u, Q, batch_idx, names in iterator:
        node_f, e_idx, crd, dip_idx, ref_chg, ref_u, Q, batch_idx = to_device(
            (node_f, e_idx, crd, dip_idx, ref_chg, ref_u, Q, batch_idx), device
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred_q, pred_u = model(node_f, crd, e_idx, dip_idx, Q, batch_idx)
            pred_q = pred_q.squeeze(1)
            pred_u = pred_u.squeeze(1)
            loss_q = loss_fn(pred_q, ref_chg)
            loss_u = loss_fn(pred_u, ref_u)
            loss = comb_loss(loss_q, loss_u)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        B = Q.shape[0]
        total_loss += loss.item() * B
        total_count += B

    # Aggregate loss across all ranks
    if dist.is_initialized():
        stats = torch.tensor([total_loss, total_count], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_count = stats[0].item(), stats[1].item()

    return total_loss / max(total_count, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_count = 0

    for node_f, e_idx, crd, dip_idx, ref_chg, ref_u, Q, batch_idx, names in loader:
        node_f, e_idx, crd, dip_idx, ref_chg, ref_u, Q, batch_idx = to_device(
            (node_f, e_idx, crd, dip_idx, ref_chg, ref_u, Q, batch_idx), device
        )

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred_q, pred_u = model(node_f, crd, e_idx, dip_idx, Q, batch_idx)
            pred_q = pred_q.squeeze(1)
            pred_u = pred_u.squeeze(1)
            loss_q = loss_fn(pred_q, ref_chg)
            loss_u = loss_fn(pred_u, ref_u)
            loss = comb_loss(loss_q, loss_u)

        B = Q.shape[0]
        total_loss += loss.item() * B
        total_count += B

    # Aggregate loss across all ranks
    if dist.is_initialized():
        stats = torch.tensor([total_loss, total_count], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_count = stats[0].item(), stats[1].item()

    return total_loss / max(total_count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str,
                        default="data/local_delta/train.h5",
                        help="Path to training data HDF5 file")
    parser.add_argument("--val_data", type=str,
                        default="data/local_delta/val.h5",
                        help="Path to validation data HDF5 file")
    parser.add_argument("--hidden_scalar", type=int, default=128)
    parser.add_argument("--hidden_vector", type=int, default=32)
    parser.add_argument("--hidden_tensor", type=int, default=16)
    parser.add_argument("--q_irreps", type=int, default=256)
    parser.add_argument("--p_irreps", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--radial_hidden", type=int, default=256)
    parser.add_argument("--num_radial_layers", type=int, default=5)
    parser.add_argument("--num_bessel", type=int, default=8)
    parser.add_argument("--num_polynomial_cutoff", type=int, default=6)
    parser.add_argument("--head_layers", type=int, default=3, 
                        help="Number of stacked Linear layers in the readout heads")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2, 
                        help="Dropout rate")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="DataLoader workers")
    parser.add_argument("--amp", action="store_true", 
                        help="Enable mixed-precision training")
    args = parser.parse_args()

    # DDP
    local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    exec_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if is_main_process():
        os.makedirs(args.outdir, exist_ok=True)
        args.date_time = exec_time
        args.world_size = world_size
        with open(os.path.join(args.outdir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    if dist.is_initialized():
        dist.barrier()

    set_seed(args.seed, rank=local_rank)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True



    train_set = ChgDataset(args.train_data, dtype=torch.float32)
    val_set   = ChgDataset(args.val_data,   dtype=torch.float32)
    sample_feat_dim = train_set.in_node_nf

    train_sampler = DistributedSampler(train_set, shuffle=True) if dist.is_initialized() else None
    val_sampler   = DistributedSampler(val_set, shuffle=False)  if dist.is_initialized() else None

    use_pin = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_batched, num_workers=args.num_workers,
        pin_memory=use_pin, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_batched, num_workers=args.num_workers,
        pin_memory=use_pin, persistent_workers=args.num_workers > 0,
    )


    model = l2E3NN(
        in_scalar=sample_feat_dim,
        hidden_scalar=args.hidden_scalar,
        hidden_vector=args.hidden_vector,
        hidden_tensor=args.hidden_tensor,
        q_irreps=args.q_irreps,
        p_irreps=args.p_irreps,
        num_layers=args.num_layers,
        lmax=args.lmax,
        radial_hidden=args.radial_hidden,
        num_radial_layers=args.num_radial_layers,
        num_bessel=args.num_bessel,
        num_polynomial_cutoff=args.num_polynomial_cutoff,
        dropout=args.dropout,
        head_layers=args.head_layers,
    ).to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])




    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if args.amp and device.type == "cuda" else None

    best_val = float("inf")
    best_epoch = 0
    best_path = os.path.join(args.outdir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        # Set epoch for DistributedSampler to reshuffle each epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        tr_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            grad_clip=args.grad_clip, scaler=scaler, use_amp=args.amp,
        )
        val_loss = eval_epoch(model, val_loader, device, use_amp=args.amp)
        scheduler.step()

        # Only rank 0 logs and saves checkpoints
        if is_main_process():
            with open(os.path.join(args.outdir, "train_loss.txt"), "a") as f:
                f.write(f"[{epoch:03d}/{args.epochs}] train={tr_loss:.6e}  val={val_loss:.6e}  lr={optimizer.param_groups[0]['lr']:.2e}\n")

            if val_loss < best_val - 1e-8:
                best_epoch = epoch
                best_val = val_loss
                # Save the unwrapped model state_dict
                state_dict = model.module.state_dict() if dist.is_initialized() else model.state_dict()
                torch.save({"model": state_dict, "val_loss": best_val, "args": vars(args)},
                           best_path)

    if is_main_process():
        print(f"Best val loss: {best_val:.6e} at epoch {best_epoch}; saved to {best_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
