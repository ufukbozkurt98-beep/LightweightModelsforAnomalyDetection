# methods/cflow_freia.py
import math
import torch
import torch.nn as nn
from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock


def positionalencoding2d(d_model: int, height: int, width: int) -> torch.Tensor:
    """
    2D sinusoidal positional encoding.
    Returns: (d_model, height, width)
    """
    if d_model % 4 != 0:
        raise ValueError("condition_vec (d_model) must be divisible by 4.")

    pe = torch.zeros(d_model, height, width)

    half = d_model // 2
    div_term = torch.exp(torch.arange(0, half, 2, dtype=torch.float) * (-math.log(10000.0) / half))

    pos_w = torch.arange(0, width, dtype=torch.float).unsqueeze(1)   # (W,1)
    pos_h = torch.arange(0, height, dtype=torch.float).unsqueeze(1)  # (H,1)

    # width encoding
    pe[0:half:2, :, :] = torch.sin(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)
    pe[1:half:2, :, :] = torch.cos(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)

    # height encoding
    pe[half::2, :, :] = torch.sin(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)
    pe[half + 1::2, :, :] = torch.cos(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)

    return pe


def _subnet_fc(c_in: int, c_out: int) -> nn.Module:
    """
    Small MLP used inside coupling blocks.
    """
    hidden = max(64, 2 * c_in)
    return nn.Sequential(
        nn.Linear(c_in, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, c_out),
    )


def freia_cflow_head(n_feat: int, n_coupling_blocks: int = 8, clamp: float = 1.9, cond_dim: int = 128) -> SequenceINN:
    """
    Build a conditional normalizing flow (FrEIA SequenceINN).

    Input: x (E, n_feat)
    Condition: c (E, cond_dim)
    """
    inn = SequenceINN(n_feat)
    for _ in range(n_coupling_blocks):
        inn.append(
            AllInOneBlock,
            subnet_constructor=_subnet_fc,
            affine_clamping=clamp,
            cond=0,
            cond_shape=(cond_dim,),
        )
    return inn


def get_logp(C: int, z: torch.Tensor, log_jac_det: torch.Tensor) -> torch.Tensor:
    """
    Standard normal log prob with jacobian correction.
    z: (E, C)
    log_jac_det: (E,)
    returns: (E,)
    """
    log_base = -0.5 * (z.pow(2).sum(dim=1) + C * math.log(2 * math.pi))
    return log_base + log_jac_det
