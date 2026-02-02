"""
CFLOW Decoder Architecture using FrEIA.

This module defines the normalizing flow decoders used in CFLOW for anomaly detection. It uses the FrEIA library to
build both conditional and unconditional normalizing flows.
"""
import math
import torch
from torch import nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm
from utils.constants import _GCONST_

# Global activation dict
# filled by TimmActivationEncoder wrapper instead of hooks
activation = {}


def positionalencoding2d(D, H, W):
    """
    Build 2D sin/cos positional encoding.
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D)
        )
    P = torch.zeros(D, H, W)
    # Each dimension (width and height); use half of D
    D = D // 2

    # Frequency terms
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)

    # Encode width with sin and cos
    P[0:D:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)

    # Encode height with sin and cos
    P[D::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


def subnet_fc(dims_in, dims_out):
    # Used inside flow coupling blocks
    return nn.Sequential(
        nn.Linear(dims_in, 2 * dims_in),
        nn.ReLU(),
        nn.Linear(2 * dims_in, dims_out),
    )


def freia_flow_head(c, n_feat):
    # Builds a normalizing flow head (FrEIA) for features
    coder = Ff.SequenceINN(n_feat)
    print("NF coder:", n_feat)
    for _k in range(c.coupling_blocks):
        coder.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_fc,
            affine_clamping=c.clamp_alpha,
            global_affine_type="SOFTPLUS",
            permute_soft=True,
        )
    return coder


def freia_cflow_head(c, n_feat):
    # Builds a conditional normalizing flow (FrEIA) head for features of size n_feat using a condition vector."
    n_cond = c.condition_vec
    coder = Ff.SequenceINN(n_feat)
    print("CNF coder:", n_feat)
    for _k in range(c.coupling_blocks):
        coder.append(
            Fm.AllInOneBlock,
            cond=0,
            cond_shape=(n_cond,),
            subnet_constructor=subnet_fc,
            affine_clamping=c.clamp_alpha,
            global_affine_type="SOFTPLUS",
            permute_soft=True,
        )
    return coder


def load_decoder_arch(c, dim_in):
    # Selects and builds the decoder (flow head) based on c.dec_arch.
    if c.dec_arch == "freia-flow":
        decoder = freia_flow_head(c, dim_in)
    elif c.dec_arch == "freia-cflow":
        decoder = freia_cflow_head(c, dim_in)
    else:
        raise NotImplementedError("{} is not supported NF!".format(c.dec_arch))
    return decoder


def get_logp(C, z, logdet_J):
    # Compute log-probability for a normalizing flow.
    logp = C * _GCONST_ - 0.5 * torch.sum(z**2, 1) + logdet_J
    return logp
