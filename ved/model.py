"""VideoMAEv2 ViT backbone (compatible with the official OpenGVLab checkpoints)
plus a small multi-label classification head.

The official distilled checkpoint `distill/vit_s_k710_dl_from_giant.pth`
(ViT-Small, 16 frames, 224px, tubelet 2, patch 16) loads into this module.
The architecture is inferred from the checkpoint itself, so ViT-Base
checkpoints also work.
"""
from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import neutral_metadata


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = self.v_bias = None
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        bias = None
        if self.q_bias is not None:
            bias = torch.cat([self.q_bias, torch.zeros_like(self.v_bias), self.v_bias])
        qkv = F.linear(x, self.qkv.weight, bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0, init_values=0.0):
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm(dim)
        self.attn = Attention(dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.norm2 = norm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1 = self.gamma_2 = None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


def sinusoid_table(n_position, d_hid):
    def angle_vec(pos):
        return [pos / np.power(10000, 2 * (j // 2) / d_hid) for j in range(d_hid)]

    table = np.array([angle_vec(p) for p in range(n_position)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.tensor(table, dtype=torch.float32).unsqueeze(0)


class VideoViT(nn.Module):
    """VideoMAEv2 encoder with mean pooling. Input: (B, 3, T, H, W)."""

    def __init__(self, img_size=224, patch_size=16, tubelet_size=2, num_frames=16,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 drop_path_rate=0.1, init_values=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv3d(
            3, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size))
        n_patches = (img_size // patch_size) ** 2 * (num_frames // tubelet_size)
        self.register_buffer("pos_embed", sinusoid_table(n_patches, embed_dim), persistent=True)
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, dpr[i], init_values) for i in range(depth)])
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed.to(x.dtype)
        for blk in self.blocks:
            x = blk(x)
        return self.fc_norm(x.mean(1))


class EventClassifier(nn.Module):
    """Backbone + multi-label head (one sigmoid output per event class)."""

    def __init__(self, backbone_cfg: dict, num_classes: int, head_dropout: float = 0.3):
        super().__init__()
        self.backbone_cfg = dict(backbone_cfg)
        self.backbone = VideoViT(**backbone_cfg)
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(self.backbone.embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.dropout(self.backbone(x)))


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
def _clean_state_dict(ckpt):
    for key in ("model", "module", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
    out = {}
    for k, v in ckpt.items():
        for prefix in ("module.", "backbone.", "encoder."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = v
    # checkpoints without mean pooling store the final norm as `norm.*`
    if "fc_norm.weight" not in out and "norm.weight" in out:
        out["fc_norm.weight"] = out.pop("norm.weight")
        out["fc_norm.bias"] = out.pop("norm.bias")
    return out


def infer_backbone_cfg(sd: dict, num_frames=16, img_size=224, drop_path_rate=0.1) -> dict:
    w = sd["patch_embed.proj.weight"]  # (D, 3, t, p, p)
    embed_dim, _, tubelet, patch, _ = w.shape
    depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    mlp_hidden = sd["blocks.0.mlp.fc1.weight"].shape[0]
    return dict(
        img_size=img_size, patch_size=int(patch), tubelet_size=int(tubelet),
        num_frames=num_frames, embed_dim=int(embed_dim), depth=int(depth),
        num_heads=int(embed_dim) // 64, mlp_ratio=mlp_hidden / embed_dim,
        drop_path_rate=drop_path_rate,
        init_values=0.1 if "blocks.0.gamma_1" in sd else 0.0,
    )


def build_from_pretrained(path: str, num_classes: int, num_frames=16, img_size=224,
                          drop_path_rate=0.1, head_dropout=0.3) -> EventClassifier:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = _clean_state_dict(ckpt)
    cfg = infer_backbone_cfg(sd, num_frames, img_size, drop_path_rate)
    model = EventClassifier(cfg, num_classes, head_dropout)
    bb_sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
    if "pos_embed" in bb_sd and bb_sd["pos_embed"].shape != model.backbone.pos_embed.shape:
        bb_sd.pop("pos_embed")
    missing, unexpected = model.backbone.load_state_dict(bb_sd, strict=False)
    missing = [k for k in missing if k != "pos_embed"]
    print(f"[pretrained] backbone cfg: {cfg}")
    print(f"[pretrained] missing keys: {missing}")
    print(f"[pretrained] unexpected keys: {unexpected}")
    if any(k.startswith("blocks.") or k.startswith("patch_embed") for k in missing):
        raise RuntimeError("Backbone weights did not load correctly - check the checkpoint.")
    return model


def save_export(model: EventClassifier, path: str, meta: dict):
    sd = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    cfg = dict(model.backbone_cfg)
    cfg["drop_path_rate"] = 0.0
    torch.save({"state_dict": sd, "backbone_cfg": cfg,
                "num_classes": model.head.out_features,
                "meta": neutral_metadata(meta, model.head.out_features)}, path)


def load_export(path: str, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = EventClassifier(ckpt["backbone_cfg"], ckpt["num_classes"], head_dropout=0.0)
    model.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
    return model.to(device).eval(), neutral_metadata(ckpt["meta"], ckpt["num_classes"])
