"""DNA blocks for Mobile-Former.

Ported from the official implementation:
  https://github.com/AAboys/MobileFormer

Paper: Mobile-Former: Bridging MobileNet and Transformer (CVPR 2022)
       https://arxiv.org/abs/2108.05895
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        channels_per_group = c // self.groups
        x = x.view(b, self.groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(b, -1, h, w)


class DyReLU(nn.Module):
    """Dynamic ReLU.
    num_func: -1=none, 0=relu, 1=SE, 2=dy-relu
    """

    def __init__(self, num_func=2, use_bias=False, scale=2.0, serelu=False):
        super().__init__()
        assert -1 <= num_func <= 2
        self.num_func = num_func
        self.scale = scale
        serelu = serelu and num_func == 1
        self.act = nn.ReLU6(inplace=True) if num_func == 0 or serelu else nn.Sequential()

    def forward(self, x):
        if isinstance(x, tuple):
            out, a = x
        else:
            out = x

        out = self.act(out)

        if self.num_func == 1:
            a = a * self.scale
            out = out * a
        elif self.num_func == 2:
            _, C, _, _ = a.shape
            a1, a2 = torch.split(a, [C // 2, C // 2], dim=1)
            a1 = (a1 - 0.5) * self.scale + 1.0
            a2 = (a2 - 0.5) * self.scale
            out = torch.max(out * a1, out * a2)

        return out


# ---------------------------------------------------------------------------
# Hyper-function (token → activation params)
# ---------------------------------------------------------------------------

class HyperFunc(nn.Module):
    def __init__(self, token_dim, oup, sel_token_id=0, reduction_ratio=4):
        super().__init__()
        self.sel_token_id = sel_token_id
        squeeze_dim = token_dim // reduction_ratio
        self.hyper = nn.Sequential(
            nn.Linear(token_dim, squeeze_dim),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze_dim, oup),
            h_sigmoid(),
        )

    def forward(self, x):
        if isinstance(x, tuple):
            x, attn = x

        if self.sel_token_id == -1:
            hp = self.hyper(x).permute(1, 2, 0)  # bs x hyper_dim x T
            bs, T, H, W = attn.shape
            attn = attn.view(bs, T, H * W)
            hp = torch.matmul(hp, attn)  # bs x hyper_dim x HW
            h = hp.view(bs, -1, H, W)
        else:
            t = x[self.sel_token_id]
            h = self.hyper(t)
            h = torch.unsqueeze(torch.unsqueeze(h, 2), 3)
        return h


# ---------------------------------------------------------------------------
# Max Depth Conv
# ---------------------------------------------------------------------------

class MaxDepthConv(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inp, oup, (3, 1), stride, (1, 0), bias=False, groups=inp),
            nn.BatchNorm2d(oup),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inp, oup, (1, 3), stride, (0, 1), bias=False, groups=inp),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return torch.max(self.conv1(x), self.conv2(x))


# ---------------------------------------------------------------------------
# Local ↔ Global bridges
# ---------------------------------------------------------------------------

class Local2Global(nn.Module):
    def __init__(
        self,
        inp,
        block_type="mlp",
        token_dim=128,
        token_num=6,
        inp_res=0,
        attn_num_heads=2,
        use_dynamic=False,
        norm_pos="post",
        drop_path_rate=0.0,
        remove_proj_local=True,
    ):
        super().__init__()
        self.num_heads = attn_num_heads
        self.token_num = token_num
        self.norm_pos = norm_pos
        self.block = block_type
        self.use_dynamic = use_dynamic

        if self.use_dynamic:
            self.alpha_scale = 2.0
            self.alpha = nn.Sequential(nn.Linear(token_dim, inp), h_sigmoid())

        if "mlp" in block_type:
            self.mlp = nn.Linear(inp_res, token_num)

        if "attn" in block_type:
            self.scale = (inp // attn_num_heads) ** -0.5
            self.q = nn.Linear(token_dim, inp)

        self.proj = nn.Linear(inp, token_dim)
        self.layer_norm = nn.LayerNorm(token_dim)
        self.drop_path = DropPath(drop_path_rate)

        self.remove_proj_local = remove_proj_local
        if not self.remove_proj_local:
            self.k = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)
            self.v = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)

    def forward(self, x):
        features, tokens = x
        bs, C, H, W = features.shape
        T, _, _ = tokens.shape
        attn = None

        if "mlp" in self.block:
            t_sum = self.mlp(features.view(bs, C, -1)).permute(2, 0, 1)

        if "attn" in self.block:
            t = self.q(tokens).view(T, bs, self.num_heads, -1).permute(1, 2, 0, 3)
            if self.remove_proj_local:
                k = features.view(bs, self.num_heads, -1, H * W)
                attn = (t @ k) * self.scale
                attn_out = attn.softmax(dim=-1)
                attn_out = attn_out @ k.transpose(-1, -2)
            else:
                k = self.k(features).view(bs, self.num_heads, -1, H * W)
                v = self.v(features).view(bs, self.num_heads, -1, H * W)
                attn = (t @ k) * self.scale
                attn_out = attn.softmax(dim=-1)
                attn_out = attn_out @ v.transpose(-1, -2)

            t_a = attn_out.permute(2, 0, 1, 3).reshape(T, bs, -1)
            t_sum = t_sum + t_a if "mlp" in self.block else t_a

        if self.use_dynamic:
            alp = self.alpha(tokens) * self.alpha_scale
            t_sum = t_sum * alp

        t_sum = self.proj(t_sum)
        tokens = tokens + self.drop_path(t_sum)
        tokens = self.layer_norm(tokens)

        if attn is not None:
            bs, Nh, Ca, HW = attn.shape
            attn = attn.view(bs, Nh, Ca, H, W)

        return tokens, attn


class GlobalBlock(nn.Module):
    def __init__(
        self,
        block_type="mlp",
        token_dim=128,
        token_num=6,
        mlp_token_exp=4,
        attn_num_heads=4,
        use_dynamic=False,
        use_ffn=False,
        norm_pos="post",
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.block = block_type
        self.num_heads = attn_num_heads
        self.token_num = token_num
        self.norm_pos = norm_pos
        self.use_dynamic = use_dynamic
        self.use_ffn = use_ffn
        self.ffn_exp = 2

        if self.use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(token_dim, token_dim * self.ffn_exp),
                nn.GELU(),
                nn.Linear(token_dim * self.ffn_exp, token_dim),
            )
            self.ffn_norm = nn.LayerNorm(token_dim)

        if self.use_dynamic:
            self.alpha_scale = 2.0
            self.alpha = nn.Sequential(nn.Linear(token_dim, token_dim), h_sigmoid())

        if "mlp" in self.block:
            self.token_mlp = nn.Sequential(
                nn.Linear(token_num, token_num * mlp_token_exp),
                nn.GELU(),
                nn.Linear(token_num * mlp_token_exp, token_num),
            )

        if "attn" in self.block:
            self.scale = (token_dim // attn_num_heads) ** -0.5
            self.q = nn.Linear(token_dim, token_dim)

        self.channel_mlp = nn.Linear(token_dim, token_dim)
        self.layer_norm = nn.LayerNorm(token_dim)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        tokens = x
        T, bs, C = tokens.shape

        if "mlp" in self.block:
            t = self.token_mlp(tokens.permute(1, 2, 0))
            t_sum = t.permute(2, 0, 1)

        if "attn" in self.block:
            t = self.q(tokens).view(T, bs, self.num_heads, -1).permute(1, 2, 0, 3)
            k = tokens.permute(1, 2, 0).view(bs, self.num_heads, -1, T)
            attn = (t @ k) * self.scale
            attn_out = attn.softmax(dim=-1)
            attn_out = attn_out @ k.transpose(-1, -2)
            t_a = attn_out.permute(2, 0, 1, 3).reshape(T, bs, -1)
            t_sum = t_sum + t_a if "mlp" in self.block else t_a

        if self.use_dynamic:
            alp = self.alpha(tokens) * self.alpha_scale
            t_sum = t_sum * alp

        t_sum = self.channel_mlp(t_sum)
        tokens = tokens + self.drop_path(t_sum)
        tokens = self.layer_norm(tokens)

        if self.use_ffn:
            t_ffn = self.ffn(tokens)
            tokens = tokens + t_ffn
            tokens = self.ffn_norm(tokens)

        return tokens


class Global2Local(nn.Module):
    def __init__(
        self,
        inp,
        inp_res=0,
        block_type="mlp",
        token_dim=128,
        token_num=6,
        attn_num_heads=2,
        use_dynamic=False,
        drop_path_rate=0.0,
        remove_proj_local=True,
    ):
        super().__init__()
        self.token_num = token_num
        self.num_heads = attn_num_heads
        self.block = block_type
        self.use_dynamic = use_dynamic

        if self.use_dynamic:
            self.alpha_scale = 2.0
            self.alpha = nn.Sequential(nn.Linear(token_dim, inp), h_sigmoid())

        if "mlp" in self.block:
            self.mlp = nn.Linear(token_num, inp_res)

        if "attn" in self.block:
            self.scale = (inp // attn_num_heads) ** -0.5
            self.k = nn.Linear(token_dim, inp)

        self.proj = nn.Linear(token_dim, inp)
        self.drop_path = DropPath(drop_path_rate)

        self.remove_proj_local = remove_proj_local
        if not self.remove_proj_local:
            self.q = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)
            self.fuse = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)

    def forward(self, x):
        out, tokens = x

        if self.use_dynamic:
            alp = self.alpha(tokens) * self.alpha_scale
            v = self.proj(tokens)
            v = (v * alp).permute(1, 2, 0)
        else:
            v = self.proj(tokens).permute(1, 2, 0)

        bs, C, H, W = out.shape
        if "mlp" in self.block:
            g_sum = self.mlp(v).view(bs, C, H, W)

        if "attn" in self.block:
            if self.remove_proj_local:
                q = out.view(bs, self.num_heads, -1, H * W).transpose(-1, -2)
            else:
                q = self.q(out).view(bs, self.num_heads, -1, H * W).transpose(-1, -2)

            k = self.k(tokens).permute(1, 2, 0).view(bs, self.num_heads, -1, self.token_num)
            attn = (q @ k) * self.scale
            attn_out = attn.softmax(dim=-1)
            vh = v.view(bs, self.num_heads, -1, self.token_num)
            attn_out = attn_out @ vh.transpose(-1, -2)

            g_a = attn_out.transpose(-1, -2).reshape(bs, C, H, W)

            if not self.remove_proj_local:
                g_a = self.fuse(g_a)

            g_sum = g_sum + g_a if "mlp" in self.block else g_a

        out = out + self.drop_path(g_sum)
        return out


# ---------------------------------------------------------------------------
# DNA Blocks
# ---------------------------------------------------------------------------

class DnaBlock3(nn.Module):
    """DNA block variant 3: dw→pw→dw→pw with L2G/G2G/G2L bridges."""

    def __init__(
        self,
        inp,
        oup,
        stride,
        exp_ratios,
        kernel_size=(3, 3),
        dw_conv="dw",
        group_num=1,
        se_flag=(2, 0, 2, 0),
        hyper_token_id=0,
        hyper_reduction_ratio=4,
        token_dim=128,
        token_num=6,
        inp_res=49,
        gbr_type="mlp",
        gbr_dynamic=(False, False, False),
        gbr_ffn=False,
        gbr_before_skip=False,
        mlp_token_exp=4,
        norm_pos="post",
        drop_path_rate=0.0,
        cnn_drop_path_rate=0.0,
        attn_num_heads=2,
        remove_proj_local=True,
    ):
        super().__init__()

        if isinstance(exp_ratios, tuple):
            e1, e2 = exp_ratios
        else:
            e1, e2 = exp_ratios, 4
        k1, k2 = kernel_size

        self.stride = stride
        self.hyper_token_id = hyper_token_id
        self.identity = stride == 1 and inp == oup
        self.use_conv_alone = False

        if e1 == 1 or e2 == 0:
            self.use_conv_alone = True
            if dw_conv == "dw":
                self.conv = nn.Sequential(
                    nn.Conv2d(inp, inp * e1, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp * e1),
                    nn.ReLU6(inplace=True),
                    ChannelShuffle(inp) if group_num > 1 else nn.Sequential(),
                    nn.Conv2d(inp * e1, oup, 1, 1, 0, groups=group_num, bias=False),
                    nn.BatchNorm2d(oup),
                )
            elif dw_conv == "sepdw":
                self.conv = nn.Sequential(
                    nn.Conv2d(inp, inp * e1 // 2, (3, 1), (stride, 1), (1, 0), groups=inp, bias=False),
                    nn.BatchNorm2d(inp * e1 // 2),
                    nn.Conv2d(inp * e1 // 2, inp * e1, (1, 3), (1, stride), (0, 1), groups=inp * e1 // 2, bias=False),
                    nn.BatchNorm2d(inp * e1),
                    nn.ReLU6(inplace=True),
                    ChannelShuffle(inp) if group_num > 1 else nn.Sequential(),
                    nn.Conv2d(inp * e1, oup, 1, 1, 0, groups=group_num, bias=False),
                    nn.BatchNorm2d(oup),
                )
        else:
            self.se_flag = se_flag
            hidden_dim1 = round(inp * e1)
            hidden_dim2 = round(oup * e2)

            if dw_conv == "dw":
                self.conv1 = nn.Sequential(
                    nn.Conv2d(inp, hidden_dim1, k1, stride, k1 // 2, groups=inp, bias=False),
                    nn.BatchNorm2d(hidden_dim1),
                    ChannelShuffle(inp) if group_num > 1 else nn.Sequential(),
                )
            elif dw_conv == "maxdw":
                self.conv1 = nn.Sequential(
                    MaxDepthConv(inp, hidden_dim1, stride),
                    ChannelShuffle(group_num) if group_num > 1 else nn.Sequential(),
                )
            elif dw_conv == "sepdw":
                self.conv1 = nn.Sequential(
                    nn.Conv2d(inp, hidden_dim1 // 2, (3, 1), (stride, 1), (1, 0), groups=inp, bias=False),
                    nn.BatchNorm2d(hidden_dim1 // 2),
                    nn.Conv2d(hidden_dim1 // 2, hidden_dim1, (1, 3), (1, stride), (0, 1), groups=hidden_dim1 // 2, bias=False),
                    nn.BatchNorm2d(hidden_dim1),
                    ChannelShuffle(inp) if group_num > 1 else nn.Sequential(),
                )

            num_func = se_flag[0]
            self.act1 = DyReLU(num_func=num_func, scale=2.0, serelu=True)
            self.hyper1 = (
                HyperFunc(token_dim, hidden_dim1 * num_func, sel_token_id=hyper_token_id, reduction_ratio=hyper_reduction_ratio)
                if se_flag[0] > 0
                else nn.Sequential()
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(hidden_dim1, oup, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(oup),
            )
            self.act2 = DyReLU(num_func=-1, scale=2.0)

            if dw_conv == "dw":
                self.conv3 = nn.Sequential(
                    nn.Conv2d(oup, hidden_dim2, k2, 1, k2 // 2, groups=oup, bias=False),
                    nn.BatchNorm2d(hidden_dim2),
                    ChannelShuffle(oup) if group_num > 1 else nn.Sequential(),
                )
            elif dw_conv == "maxdw":
                self.conv3 = nn.Sequential(MaxDepthConv(oup, hidden_dim2, 1))
            elif dw_conv == "sepdw":
                self.conv3 = nn.Sequential(
                    nn.Conv2d(oup, hidden_dim2 // 2, (3, 1), (1, 1), (1, 0), groups=oup, bias=False),
                    nn.BatchNorm2d(hidden_dim2 // 2),
                    nn.Conv2d(hidden_dim2 // 2, hidden_dim2, (1, 3), (1, 1), (0, 1), groups=hidden_dim2 // 2, bias=False),
                    nn.BatchNorm2d(hidden_dim2),
                    ChannelShuffle(oup) if group_num > 1 else nn.Sequential(),
                )

            num_func = se_flag[2]
            self.act3 = DyReLU(num_func=num_func, scale=2.0, serelu=True)
            self.hyper3 = (
                HyperFunc(token_dim, hidden_dim2 * num_func, sel_token_id=hyper_token_id, reduction_ratio=hyper_reduction_ratio)
                if se_flag[2] > 0
                else nn.Sequential()
            )

            self.conv4 = nn.Sequential(
                nn.Conv2d(hidden_dim2, oup, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(oup),
            )
            num_func = 1 if se_flag[3] == 1 else -1
            self.act4 = DyReLU(num_func=num_func, scale=2.0)
            self.hyper4 = (
                HyperFunc(token_dim, oup * num_func, sel_token_id=hyper_token_id, reduction_ratio=hyper_reduction_ratio)
                if se_flag[3] > 0
                else nn.Sequential()
            )

            self.drop_path = DropPath(cnn_drop_path_rate)

            # L2G, G2G, G2L
            self.local_global = Local2Global(
                inp,
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                inp_res=inp_res,
                use_dynamic=gbr_dynamic[0],
                norm_pos=norm_pos,
                drop_path_rate=drop_path_rate,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,
            )
            self.global_block = GlobalBlock(
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                mlp_token_exp=mlp_token_exp,
                use_dynamic=gbr_dynamic[1],
                use_ffn=gbr_ffn,
                norm_pos=norm_pos,
                drop_path_rate=drop_path_rate,
            )
            oup_res = inp_res // (stride * stride)
            self.global_local = Global2Local(
                oup,
                oup_res,
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                use_dynamic=gbr_dynamic[2],
                drop_path_rate=drop_path_rate,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,
            )

    def forward(self, x):
        features, tokens = x
        if self.use_conv_alone:
            out = self.conv(features)
        else:
            tokens, attn = self.local_global((features, tokens))
            tokens = self.global_block(tokens)

            out = self.conv1(features)

            if self.hyper_token_id == -1:
                attn = attn.mean(dim=1)
                if self.stride > 1:
                    _, _, H, W = out.shape
                    attn = F.adaptive_avg_pool2d(attn, (H, W))
                attn = torch.softmax(attn, dim=1)

            if self.se_flag[0] > 0:
                hp = self.hyper1((tokens, attn))
                out = self.act1((out, hp))
            else:
                out = self.act1(out)

            out = self.conv2(out)
            out = self.act2(out)

            out_cp = out
            out = self.conv3(out)
            if self.se_flag[2] > 0:
                hp = self.hyper3((tokens, attn))
                out = self.act3((out, hp))
            else:
                out = self.act3(out)

            out = self.conv4(out)
            if self.se_flag[3] > 0:
                hp = self.hyper4((tokens, attn))
                out = self.act4((out, hp))
            else:
                out = self.act4(out)

            out = self.drop_path(out) + out_cp
            out = self.global_local((out, tokens))

        if self.identity:
            out = out + features

        return (out, tokens)


class DnaBlock(nn.Module):
    """DNA block: pw→dw→pw with L2G/G2G/G2L bridges."""

    def __init__(
        self,
        inp,
        oup,
        stride,
        exp_ratios,
        kernel_size=(3, 3),
        dw_conv="dw",
        group_num=1,
        se_flag=(2, 0, 2, 0),
        hyper_token_id=0,
        hyper_reduction_ratio=4,
        token_dim=128,
        token_num=6,
        inp_res=49,
        gbr_type="mlp",
        gbr_dynamic=(False, False, False),
        gbr_ffn=False,
        gbr_before_skip=False,
        mlp_token_exp=4,
        norm_pos="post",
        drop_path_rate=0.0,
        cnn_drop_path_rate=0.0,
        attn_num_heads=2,
        remove_proj_local=True,
    ):
        super().__init__()

        if isinstance(exp_ratios, tuple):
            e1, e2 = exp_ratios
        else:
            e1, e2 = exp_ratios, 4
        k1, k2 = kernel_size

        self.stride = stride
        self.hyper_token_id = hyper_token_id
        self.gbr_before_skip = gbr_before_skip
        self.identity = stride == 1 and inp == oup
        self.use_conv_alone = False

        if e1 == 1 or e2 == 0:
            self.use_conv_alone = True
            self.conv = nn.Sequential(
                nn.Conv2d(inp, inp * e1, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp * e1),
                nn.ReLU6(inplace=True),
                ChannelShuffle(inp) if group_num > 1 else nn.Sequential(),
                nn.Conv2d(inp * e1, oup, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(oup),
            )
            self.drop_path = DropPath(cnn_drop_path_rate)
        else:
            self.se_flag = se_flag
            hidden_dim = round(inp * e1)

            self.conv1 = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(hidden_dim),
                ChannelShuffle(group_num) if group_num > 1 else nn.Sequential(),
            )

            num_func = se_flag[0]
            self.act1 = DyReLU(num_func=num_func, scale=2.0, serelu=True)
            self.hyper1 = (
                HyperFunc(token_dim, hidden_dim * num_func, sel_token_id=hyper_token_id, reduction_ratio=hyper_reduction_ratio)
                if se_flag[0] > 0
                else nn.Sequential()
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, k1, stride, k1 // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
            num_func = se_flag[2]
            self.act2 = DyReLU(num_func=num_func, scale=2.0, serelu=True)
            self.hyper2 = (
                HyperFunc(token_dim, hidden_dim * num_func, sel_token_id=hyper_token_id, reduction_ratio=hyper_reduction_ratio)
                if se_flag[2] > 0
                else nn.Sequential()
            )

            self.conv3 = nn.Sequential(
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(oup),
                ChannelShuffle(group_num) if group_num > 1 else nn.Sequential(),
            )
            num_func = 1 if se_flag[3] == 1 else -1
            self.act3 = DyReLU(num_func=num_func, scale=2.0)
            self.hyper3 = (
                HyperFunc(token_dim, oup * num_func, sel_token_id=hyper_token_id, reduction_ratio=hyper_reduction_ratio)
                if se_flag[3] > 0
                else nn.Sequential()
            )

            self.drop_path = DropPath(cnn_drop_path_rate)

            # L2G, G2G, G2L
            self.local_global = Local2Global(
                inp,
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                inp_res=inp_res,
                use_dynamic=gbr_dynamic[0],
                norm_pos=norm_pos,
                drop_path_rate=drop_path_rate,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,
            )
            self.global_block = GlobalBlock(
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                mlp_token_exp=mlp_token_exp,
                use_dynamic=gbr_dynamic[1],
                use_ffn=gbr_ffn,
                norm_pos=norm_pos,
                drop_path_rate=drop_path_rate,
            )
            oup_res = inp_res // (stride * stride)
            self.global_local = Global2Local(
                oup,
                oup_res,
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                use_dynamic=gbr_dynamic[2],
                drop_path_rate=drop_path_rate,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,
            )

    def forward(self, x):
        features, tokens = x
        if self.use_conv_alone:
            out = self.conv(features)
            if self.identity:
                out = self.drop_path(out) + features
        else:
            tokens, attn = self.local_global((features, tokens))
            tokens = self.global_block(tokens)

            out = self.conv1(features)

            if self.hyper_token_id == -1:
                attn = attn.mean(dim=1)
                if self.stride > 1:
                    _, _, H, W = out.shape
                    attn = F.adaptive_avg_pool2d(attn, (H, W))
                attn = torch.softmax(attn, dim=1)

            if self.se_flag[0] > 0:
                hp = self.hyper1((tokens, attn))
                out = self.act1((out, hp))
            else:
                out = self.act1(out)

            out = self.conv2(out)
            if self.se_flag[2] > 0:
                hp = self.hyper2((tokens, attn))
                out = self.act2((out, hp))
            else:
                out = self.act2(out)

            out = self.conv3(out)
            if self.se_flag[3] > 0:
                hp = self.hyper3((tokens, attn))
                out = self.act3((out, hp))
            else:
                out = self.act3(out)

            if self.gbr_before_skip:
                out = self.global_local((out, tokens))
                if self.identity:
                    out = self.drop_path(out) + features
            else:
                if self.identity:
                    out = self.drop_path(out) + features
                out = self.global_local((out, tokens))

        return (out, tokens)


# ---------------------------------------------------------------------------
# Classifier (kept for completeness, not used in feature extraction)
# ---------------------------------------------------------------------------

class MergeClassifier(nn.Module):
    def __init__(
        self,
        inp,
        oup=1280,
        ch_exp=6,
        num_classes=1000,
        drop_rate=0.0,
        drop_branch=(0.0, 0.0),
        group_num=1,
        token_dim=128,
        cls_token_num=1,
        last_act="relu",
        hyper_token_id=0,
        hyper_reduction_ratio=4,
    ):
        super().__init__()
        self.drop_branch = drop_branch
        self.cls_token_num = cls_token_num

        hidden_dim = inp * ch_exp
        self.conv = nn.Sequential(
            ChannelShuffle(group_num) if group_num > 1 else nn.Sequential(),
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, groups=group_num, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )

        self.last_act = last_act
        num_func = 2 if last_act == "dyrelu" else 0
        self.act = DyReLU(num_func=num_func, scale=2.0)

        self.hyper = (
            HyperFunc(token_dim, hidden_dim * num_func, sel_token_id=hyper_token_id, reduction_ratio=hyper_reduction_ratio)
            if last_act == "dyrelu"
            else nn.Sequential()
        )

        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), h_swish())

        if cls_token_num > 0:
            cat_token_dim = token_dim * cls_token_num
        elif cls_token_num == 0:
            cat_token_dim = token_dim
        else:
            cat_token_dim = 0

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + cat_token_dim, oup),
            nn.BatchNorm1d(oup),
            h_swish(),
        )
        self.classifier = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(oup, num_classes))

    def forward(self, x):
        features, tokens = x
        x = self.conv(features)

        if self.last_act == "dyrelu":
            hp = self.hyper(tokens)
            x = self.act((x, hp))
        else:
            x = self.act(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        ps = [x]
        if self.cls_token_num == 0:
            avg_token = torch.mean(F.relu6(tokens), dim=0)
            ps.append(avg_token)
        elif self.cls_token_num < 0:
            pass
        else:
            for i in range(self.cls_token_num):
                ps.append(tokens[i])

        if self.training and self.drop_branch[0] + self.drop_branch[1] > 1e-8:
            rd = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device)
            keep_local = 1 - self.drop_branch[0]
            keep_global = 1 - self.drop_branch[1]
            rd_local = (keep_local + rd).floor_()
            rd_global = -((rd - keep_global).floor_())
            ps[0] = ps[0].div(keep_local) * rd_local
            ps[1] = ps[1].div(keep_global) * rd_global

        x = torch.cat(ps, dim=1)
        x = self.fc(x)
        x = self.classifier(x)
        return x
