from __future__ import annotations

import math
import numpy as np
from scipy.fft import dst
import torch
from torch import nn
from e3nn import o3
from e3nn.o3 import Linear
from e3nn.nn import Gate


def _split_for_gate(irreps_out: o3.Irreps):
    """
    Split a target irreps into (scalars, gates, gated, tp_out) for use with Gate.
    """
    irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l == 0])
    irreps_gated = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l > 0])
    if irreps_gated.num_irreps == 0:
        irreps_gates = o3.Irreps("")
    else:
        irreps_gates = o3.Irreps(f"{irreps_gated.num_irreps}x0e")
    irreps_tp_out = (irreps_scalars + irreps_gates + irreps_gated).simplify()
    return irreps_scalars, irreps_gates, irreps_gated, irreps_tp_out


def _build_linear_stack(
        irreps_in: o3.Irreps,
        irreps_out_final: o3.Irreps,
        num_layers: int,
    ) -> list:
    """
    Stacked Linear blocks where each non-final layer halves all multiplicities
    and uses Gate as activation; the final layer maps to `irreps_out_final`.
    """
    if num_layers < 1:
        raise ValueError("head_layers must be >= 1")

    act = {
        1: torch.nn.functional.silu,
        -1: torch.tanh,
    }
    act_gates = {
        1: torch.sigmoid,
        -1: torch.tanh,
    }

    layers = []
    cur = irreps_in
    for _ in range(num_layers - 1):
        next_irreps = o3.Irreps(
            [(max(mul // 2, 1), ir) for mul, ir in cur]
        ).simplify()
        sc, ga, gd, _ = _split_for_gate(next_irreps)
        gate = Gate(
            sc, [act[ir.p] for _, ir in sc],
            ga, [act_gates[ir.p] for _, ir in ga],
            gd,
        )
        layers.append(Linear(cur, gate.irreps_in))
        layers.append(gate)
        cur = gate.irreps_out

    layers.append(Linear(cur, irreps_out_final))

    return layers


class BesselBasis(torch.nn.Module):
    """
    Code from J. Phys. Chem. Lett. 2025, 16, 25, 6334-6344
    """

    def __init__(self, r_max: float, num_basis=8, trainable=False):
        super().__init__()

        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


class PolynomialCutoff(torch.nn.Module):
    """
    Code from J. Phys. Chem. Lett. 2025, 16, 25, 6334-6344
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        envelope = (
                1.0
                - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.r_max, self.p)
                + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1)
                - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2)
        )

        return envelope * (x < self.r_max)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"


class NodeEmbedding(nn.Module):

    def __init__(self, irreps_in: o3.Irreps, out_irreps: o3.Irreps):
        super().__init__()
        self.linear = Linear(irreps_in, out_irreps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class RadialEmbeddingBlock(torch.nn.Module):

    """
    Code from J. Phys. Chem. Lett. 2025, 16, 25, 6334-6344
    """

    def __init__(
            self,
            r_max: float,
            num_bessel: int,
            num_polynomial_cutoff: int,
    ):
        super().__init__()
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
            self,
            edge_lengths: torch.Tensor,  # [n_edges, 1]
    ):
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]

        radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        return radial * cutoff  # [n_edges, n_basis]


class TensorProductMessagePass(nn.Module):
    """
    One equivariant message passing block
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        radial_in: int,
        radial_hidden: int = 128,
        num_radial_layers: int = 5, # of layers in the radial MLP
        lmax: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.lmax = lmax
        self.dropout = dropout

        self.layer_embedding = Linear(irreps_in, irreps_in)

        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)

        # Gate nonlinearity
        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }
        irreps_scalars, irreps_gates, irreps_gated, _ = _split_for_gate(irreps_out)
        self.act = Gate(
            irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],
            irreps_gates,   [act_gates[ir.p] for _, ir in irreps_gates],
            irreps_gated,
        )
        gate_in = self.act.irreps_in

        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in,
            self.irreps_sh,
            gate_in,
            shared_weights=False,
            internal_weights=False,
        )

        radial_layers = []
        for i in range(num_radial_layers):
            in_dim = radial_in if i == 0 else radial_hidden
            out_dim = self.tp.weight_numel if i == num_radial_layers - 1 else radial_hidden
            radial_layers.append(nn.Linear(in_dim, out_dim))
            if i < num_radial_layers - 1:
                radial_layers.append(nn.SiLU())
                if self.dropout > 0.0:
                    radial_layers.append(nn.Dropout(self.dropout))
        self.radial = nn.Sequential(*radial_layers)

        self.linear_out = Linear(gate_in, gate_in)

    def forward(
        self,
        x: torch.Tensor, # node features
        e: torch.Tensor, # edge features
        sh: torch.Tensor, # spherical harmonics
        edge_index: torch.Tensor # graph connectivity
    ) -> torch.Tensor:

        # Layer level embedding of node features
        x = self.layer_embedding(x)

        src, dst = edge_index

        w = self.radial(e)

        m = self.tp(x[src], sh, w)

        out = torch.zeros(
            x.shape[0], self.tp.irreps_out.dim, device=x.device, dtype=x.dtype
        )
        out.index_add_(0, dst, m)

        out = self.linear_out(out)

        return self.act(out)



class PolReadout(nn.Module):
    """
    Directed-edge readout.
    Output is directional
    """

    def __init__(self,
                 irreps_hidden: o3.Irreps,
                 radial_in: int,
                 radial_hidden: int = 128,
                 num_radial_layers: int = 5,
                 lmax: int = 2,
                 head_layers: int = 3,
                 dropout: float = 0.2,
                 ):
        super().__init__()
        self.irreps_hidden = irreps_hidden
        self.lmax = lmax
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.dropout = dropout

        self.tp_geom = o3.FullyConnectedTensorProduct(
            irreps_hidden,
            self.irreps_sh,
            irreps_hidden,
            shared_weights=False,
            internal_weights=False,
        )

        radial_layers = []
        for i in range(num_radial_layers):
            in_dim = radial_in if i == 0 else radial_hidden
            out_dim = self.tp_geom.weight_numel if i == num_radial_layers - 1 else radial_hidden
            radial_layers.append(nn.Linear(in_dim, out_dim))
            if i < num_radial_layers - 1:
                radial_layers.append(nn.SiLU())
                if self.dropout > 0.0:
                    radial_layers.append(nn.Dropout(self.dropout))
        self.radial = nn.Sequential(*radial_layers)

        self.tp_feat = o3.FullyConnectedTensorProduct(
            irreps_hidden,
            irreps_hidden,
            irreps_hidden,
            shared_weights=True,
        )

        irreps_cat = (
            self.irreps_hidden + self.irreps_hidden + o3.Irreps(f"{radial_in}x0e")
        )
        self.linear_out = nn.Sequential(
            *_build_linear_stack(irreps_cat, o3.Irreps("1x0e"), num_layers=head_layers)
        )

    def forward(
            self,
            x: torch.Tensor,
            e: torch.Tensor,
            sh: torch.Tensor,
            edge_index: torch.Tensor
            ) -> torch.Tensor:

        src, dst = edge_index

        w = self.radial(e)
        h_i, h_j = x[src], x[dst]

        e2 = self.tp_geom(h_i, sh, w)   # operator 1
        e3 = self.tp_feat(h_i, h_j)     # operator 2

        out = self.linear_out(torch.cat([e2, e3, e], dim=-1))

        return out


class l2E3NN(nn.Module):
    """
    Main l2E3NN model class.
    Batched input is supported.
    """

    def __init__(
        self,
        in_scalar: int,
        hidden_scalar: int = 128,
        hidden_vector: int = 16,
        hidden_tensor: int = 32,
        q_irreps: int = 256,
        p_irreps: int = 256,
        num_layers: int = 3,
        lmax: int = 2,
        radial_hidden: int = 128,
        num_radial_layers: int = 5,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 6,
        dropout: float = 0.2,
        head_layers: int = 3,
    ):
        super().__init__()

        self.lmax = lmax
        self.dropout = dropout


        self.irreps_in              = o3.Irreps(f"{in_scalar}x0e")
        irreps_node_embedding_out   = o3.Irreps(f"{hidden_scalar}x0e")
        self.irreps_hidden          = o3.Irreps(
                                        f"{hidden_scalar}x0e + {hidden_vector}x1o + {hidden_tensor}x2e"
                                    )

        # Embedding layer for node features
        self.node_embedding = NodeEmbedding(
            self.irreps_in, irreps_node_embedding_out
            )

        # Embedding layer for edge features
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=9.45, # !!! IMPORTANT: Units: Bohr， must be same as in the dataset
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        radial_in = self.radial_embedding.out_dim


        blocks = []
        irreps_prev = irreps_node_embedding_out
        for _ in range(num_layers):
            blocks.append(
                TensorProductMessagePass(
                    irreps_in=irreps_prev,
                    irreps_out=self.irreps_hidden,
                    radial_in=radial_in,
                    radial_hidden=radial_hidden,
                    num_radial_layers=num_radial_layers,
                    lmax=lmax,
                    dropout=self.dropout,
                )
            )
            irreps_prev = self.irreps_hidden
        self.blocks = nn.ModuleList(blocks)

        self.chg_head = nn.Sequential(
            *_build_linear_stack(self.irreps_hidden, o3.Irreps("1x0e"), num_layers=head_layers)
        )
        self.pol_head = PolReadout(
            self.irreps_hidden,
            radial_in=radial_in,
            radial_hidden=radial_hidden,
            num_radial_layers=num_radial_layers,
            lmax=lmax,
            head_layers=head_layers,
            dropout=self.dropout,
            )

    def chg_restraint(self, x, Q, batch):
        """
        Per-molecule charge restraint for batched graphs.
        """
        c = self.chg_head(x)  # (N_total, 1)

        B = Q.shape[0]

        c_sum = torch.zeros(B, 1, device=c.device, dtype=c.dtype)
        c_sum.scatter_add_(0, batch.unsqueeze(1), c)
        counts = torch.zeros(B, device=c.device, dtype=c.dtype)
        counts.scatter_add_(0, batch, torch.ones(batch.shape[0], device=c.device, dtype=c.dtype))

        # per-molecule mean
        c_mean = c_sum / counts.unsqueeze(1)  # (B, 1)

        # restraint
        q = c - c_mean[batch] + (Q[batch] / counts[batch]).unsqueeze(1)
        return q

    def forward(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        dipole_index: torch.Tensor,
        Q: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Batched forward pass for multiple molecules.

        Inputs
        ------
        - node_feats:   (N_total, F_in) float tensor of scalar node features
        - pos:          (N_total, 3)    float tensor of 3D coordinates
        - edge_index:   (2, E_total)    long tensor with offset source/target edges
        - dipole_index: (2, D_total)    long tensor with offset dipole pairs
        - Q:            (B,)            total formal charge per molecule
        - batch:        (N_total,)      long tensor mapping each node to molecule index [0..B-1]

        Outputs
        -------
        - q: (N_total, 1) node-level charge predictions with per-molecule sum constraint
        - u: (D_total, 1) dipole magnitude predictions
        """

        x = self.node_embedding(node_feats)

        src, dst = edge_index
        rel_vec = pos[dst] - pos[src]
        rel_len = rel_vec.norm(dim=1, keepdim=True) # (E_total, 1)

        e = self.radial_embedding(rel_len)

        sh = o3.spherical_harmonics(
                list(range(self.lmax + 1)),
                rel_vec,
                normalize=True,
                normalization="component",
            )

        for block in self.blocks:
            x = block(x, e, sh, edge_index)

        # Charge restraint
        q = self.chg_restraint(x, Q, batch)  # (N_total, 1)
        # q = self.chg_head(x)  # UNRESTRAINED

        # Compute radial embedding and spherical harmonics for dipole pairs
        dip_src, dip_dst = dipole_index
        dip_rel_vec = pos[dip_dst] - pos[dip_src]
        dip_rel_len = dip_rel_vec.norm(dim=1, keepdim=True)
        dip_e = self.radial_embedding(dip_rel_len)
        dip_sh = o3.spherical_harmonics(
            list(range(self.lmax + 1)),
            dip_rel_vec,
            normalize=True,
            normalization="component",
        )

        # local dipole scalar per directed pair
        u = self.pol_head(x, dip_e, dip_sh, dipole_index)  # (D_total, 1)

        return q, u


if __name__ == "__main__":
    torch.manual_seed(0)

    # Test with batched input (2 molecules)
    N1, E1, D1 = 5, 8, 4
    N2, E2, D2 = 7, 12, 6
    in_scalar = 3

    model = l2E3NN(in_scalar=in_scalar)

    # Molecule 1
    nf1 = torch.randn(N1, in_scalar)
    pos1 = torch.randn(N1, 3)
    ei1 = torch.randint(0, N1, (2, E1))
    di1 = torch.randint(0, N1, (2, D1))

    # Molecule 2 (offset indices)
    nf2 = torch.randn(N2, in_scalar)
    pos2 = torch.randn(N2, 3)
    ei2 = torch.randint(0, N2, (2, E2)) + N1
    di2 = torch.randint(0, N2, (2, D2)) + N1

    # Concatenate
    node_feats = torch.cat([nf1, nf2], dim=0)
    pos = torch.cat([pos1, pos2], dim=0)
    edge_index = torch.cat([ei1, ei2], dim=1)
    dipole_index = torch.cat([di1, di2], dim=1)
    Q = torch.tensor([0.0, 1.0])
    batch = torch.cat([torch.zeros(N1, dtype=torch.long), torch.ones(N2, dtype=torch.long)])

    q, u = model(node_feats, pos, edge_index, dipole_index, Q, batch)
    print("charges", q.shape, "dipoles", u.shape)
    print("charge sum mol0:", q[:N1].sum().item(), "expected ~0.0")
    print("charge sum mol1:", q[N1:].sum().item(), "expected ~1.0")
