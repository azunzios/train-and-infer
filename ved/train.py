"""Dataset scanning, frame caching, MIL training and cross-validation."""
from __future__ import annotations

import copy
import hashlib
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .inference import MEAN, STD
from .model import build_from_pretrained
from .video_io import VIDEO_EXTS, read_frames


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
def find_dataset_root(base: str) -> str:
    """Return the first directory under `base` that has `positive` and `negative` subfolders."""
    for root, dirs, _ in os.walk(base):
        low = {d.lower() for d in dirs}
        if "positive" in low and "negative" in low:
            return root
    raise FileNotFoundError(f"No folder with 'positive' and 'negative' subfolders under {base}")


def _videos_in(folder: str) -> List[str]:
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                out.append(os.path.join(root, f))
    return sorted(out)


def scan_dataset(root: str, folder_to_class: Dict[str, str], class_names: List[str],
                 group_regex: Optional[str] = None) -> List[dict]:
    """positive/<sub>/ -> class from folder_to_class; negative/** -> all-zero label."""
    subdirs = {d.lower(): os.path.join(root, d) for d in os.listdir(root)}
    items = []
    pos_dir, neg_dir = subdirs["positive"], subdirs["negative"]
    for sub in sorted(os.listdir(pos_dir)):
        full = os.path.join(pos_dir, sub)
        if not os.path.isdir(full):
            continue
        cls = folder_to_class.get(sub.lower())
        if cls is None:
            print(f"[warn] unknown positive subfolder '{sub}' - skipped")
            continue
        y = np.zeros(len(class_names), np.float32)
        y[class_names.index(cls)] = 1
        for p in _videos_in(full):
            items.append(dict(path=p, y=y.copy(), source=f"positive/{sub}"))
    for p in _videos_in(neg_dir):
        rel = os.path.relpath(p, neg_dir).split(os.sep)[0]
        items.append(dict(path=p, y=np.zeros(len(class_names), np.float32), source=f"negative/{rel}"))
    for it in items:
        name = os.path.splitext(os.path.basename(it["path"]))[0]
        m = re.search(group_regex, name) if group_regex else None
        it["group"] = m.group(1) if m else it["path"]
    return items


def summarize(items, class_names):
    from collections import Counter
    c = Counter(it["source"] for it in items)
    for k in sorted(c):
        print(f"  {k:25s} {c[k]:5d}")
    Y = np.stack([it["y"] for it in items])
    for i, n in enumerate(class_names):
        print(f"  -> {n:20s} positives: {int(Y[:, i].sum())}")
    print(f"  -> negatives (all zero): {int((Y.sum(1) == 0).sum())}   total: {len(items)}")
    print(f"  -> unique groups: {len({it['group'] for it in items})}")


# --------------------------------------------------------------------------- #
# Frame cache (decode each clip once -> uint8 .npy)
# --------------------------------------------------------------------------- #
def build_cache(items, cache_dir, fps, size, workers=8):
    os.makedirs(cache_dir, exist_ok=True)

    def work(it):
        key = hashlib.md5(f"{it['path']}|{fps}|{size}".encode()).hexdigest()
        out = os.path.join(cache_dir, key + ".npy")
        if not os.path.exists(out):
            try:
                arr = read_frames(it["path"], fps, size)
            except Exception as e:  # noqa
                return it, None, str(e)
            np.save(out, arr)
        return it, out, None

    ok, bad = [], []
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, (it, out, err) in enumerate(ex.map(work, items)):
            if out is None:
                bad.append((it["path"], err))
            else:
                it["cache"] = out
                it["n_frames"] = int(np.load(out, mmap_mode="r").shape[0])
                ok.append(it)
            if (i + 1) % 25 == 0 or i + 1 == len(items):
                print(f"\r  cached {i + 1}/{len(items)} ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    for p, e in bad:
        print(f"[warn] failed to decode {p}: {e}")
    return ok


# --------------------------------------------------------------------------- #
# Dataset: K windows per clip (multiple-instance learning)
# --------------------------------------------------------------------------- #
class ClipDataset(Dataset):
    def __init__(self, items, num_frames, k_windows, train):
        self.items, self.T, self.K, self.train = items, num_frames, k_windows, train

    def __len__(self):
        return len(self.items)

    def _starts(self, n):
        max_s = max(n - self.T, 0)
        if self.train:
            edges = np.linspace(0, max_s, self.K + 1)
            return [int(np.random.randint(int(edges[i]), max(int(edges[i + 1]), int(edges[i]) + 1)))
                    for i in range(self.K)]
        return [int(round(v)) for v in np.linspace(0, max_s, self.K)]

    def __getitem__(self, i):
        it = self.items[i]
        arr = np.load(it["cache"], mmap_mode="r")
        n = arr.shape[0]
        wins = []
        for s in self._starts(n):
            idx = np.clip(np.arange(s, s + self.T), 0, n - 1)
            wins.append(np.asarray(arr[idx]))
        return torch.from_numpy(np.stack(wins)), torch.from_numpy(it["y"])


def gpu_prepare(x_uint8, size, train):
    """(B, K, T, H, W, 3) uint8 -> (B*K, 3, T, size, size) normalized, with augmentation."""
    B, K, T, H, W, _ = x_uint8.shape
    x = x_uint8.view(B * K, T, H, W, 3).permute(0, 1, 4, 2, 3).float().div_(255.0)  # N,T,3,H,W
    if train:
        out = []
        for v in x:
            scale = np.random.uniform(0.6, 1.0)
            ratio = math.exp(np.random.uniform(math.log(3 / 4), math.log(4 / 3)))
            h = min(H, int(round(math.sqrt(scale * H * W / ratio))))
            w = min(W, int(round(math.sqrt(scale * H * W * ratio))))
            y0, x0 = np.random.randint(0, H - h + 1), np.random.randint(0, W - w + 1)
            v = v[:, :, y0:y0 + h, x0:x0 + w]
            v = F.interpolate(v, size=(size, size), mode="bilinear", align_corners=False)
            if np.random.rand() < 0.5:
                v = v.flip(-1)
            c = np.random.uniform(0.75, 1.25)
            b = np.random.uniform(0.75, 1.25)
            m = v.mean()
            v = ((v - m) * c + m) * b
            out.append(v.clamp_(0, 1))
        x = torch.stack(out)
    elif H != size or W != size:
        x = F.interpolate(x.flatten(0, 1), size=(size, size), mode="bilinear",
                          align_corners=False).view(B * K, T, 3, size, size)
    x = x.permute(0, 2, 1, 3, 4)  # N,3,T,H,W
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def lse_pool(logits, r=5.0):
    """Smooth max over windows: (B, K, C) -> (B, C)."""
    K = logits.shape[1]
    return (torch.logsumexp(r * logits, dim=1) - math.log(K)) / r


# --------------------------------------------------------------------------- #
# Optimizer with layer-wise LR decay + partial freezing
# --------------------------------------------------------------------------- #
def make_optimizer(model, cfg):
    depth = model.backbone.depth
    groups = {}
    for n, p in model.named_parameters():
        if n.startswith("backbone.patch_embed"):
            lid = 0
        elif n.startswith("backbone.blocks."):
            lid = int(n.split(".")[2]) + 1
        elif n.startswith("backbone."):
            lid = depth + 1
        else:
            lid = None  # head
        if lid is not None and cfg["freeze_blocks"] > 0 and lid <= cfg["freeze_blocks"]:
            p.requires_grad_(False)
            continue
        no_wd = p.ndim == 1 or n.endswith(".bias") or "q_bias" in n or "v_bias" in n
        if lid is None:
            lr = cfg["head_lr"]
        else:
            lr = cfg["backbone_lr"] * cfg["layer_decay"] ** (depth + 1 - lid)
        key = (lr, 0.0 if no_wd else cfg["weight_decay"])
        groups.setdefault(key, []).append(p)
    param_groups = [dict(params=ps, lr=lr, weight_decay=wd, base_lr=lr) for (lr, wd), ps in groups.items()]
    n_train = sum(p.numel() for g in param_groups for p in g["params"])
    print(f"  trainable params: {n_train / 1e6:.2f}M")
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def lr_lambda(total, warmup):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog))
    return f


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def clip_metrics(y_true, y_prob, class_names, thresholds=None):
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
    res = {}
    for c, name in enumerate(class_names):
        yt, yp = y_true[:, c], y_prob[:, c]
        r = {"AP": float(average_precision_score(yt, yp)) if yt.sum() > 0 else float("nan"),
             "AUC": float(roc_auc_score(yt, yp)) if 0 < yt.sum() < len(yt) else float("nan")}
        if thresholds is None:
            prec, rec, thr = precision_recall_curve(yt, yp)
            f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
            best = int(np.nanargmax(f1[:-1])) if len(thr) else 0
            t = float(thr[best]) if len(thr) else 0.5
        else:
            t = float(thresholds[name])
        pred = yp >= t
        tp = int((pred & (yt == 1)).sum()); fp = int((pred & (yt == 0)).sum())
        fn = int((~pred & (yt == 1)).sum())
        p_ = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
        r.update(threshold=t, precision=p_, recall=r_, F1=2 * p_ * r_ / max(p_ + r_, 1e-9),
                 TP=tp, FP=fp, FN=fn)
        res[name] = r
    res["mAP"] = float(np.nanmean([res[n]["AP"] for n in class_names]))
    return res


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_clips(model, items, cfg, device):
    ds = ClipDataset(items, cfg["num_frames"], cfg["k_eval"], train=False)
    dl = DataLoader(ds, batch_size=max(1, cfg["batch_size"] // 2), shuffle=False,
                    num_workers=cfg["num_workers"], pin_memory=True)
    model.eval()
    out = []
    for x, _ in dl:
        B, K = x.shape[:2]
        x = gpu_prepare(x.to(device, non_blocking=True), cfg["img_size"], train=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            p = torch.sigmoid(model(x).float()).view(B, K, -1)
        out.append(p.max(1).values.cpu().numpy())  # clip score = best window
    return np.concatenate(out)


def train_model(cfg, pretrained_path, train_items, val_items=None, epochs=None, tag=""):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    C = len(cfg["class_names"])
    epochs = epochs or cfg["epochs"]
    model = build_from_pretrained(pretrained_path, C, cfg["num_frames"], cfg["img_size"],
                                  cfg["drop_path"], cfg["head_dropout"]).to(device)
    opt = make_optimizer(model, cfg)
    ds = ClipDataset(train_items, cfg["num_frames"], cfg["k_train"], train=True)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                    num_workers=cfg["num_workers"], pin_memory=True, persistent_workers=cfg["num_workers"] > 0)
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, cfg["warmup_epochs"] * len(dl)))
    Y = np.stack([it["y"] for it in train_items])
    pos = Y.sum(0); neg = len(Y) - pos
    pos_weight = torch.tensor(np.sqrt(neg / np.clip(pos, 1, None)), dtype=torch.float32, device=device)
    print(f"  {tag} train={len(train_items)} val={len(val_items) if val_items else 0} "
          f"epochs={epochs} pos_weight={pos_weight.cpu().numpy().round(2)}")

    history, best = [], dict(score=-1, state=None, epoch=0, probs=None)
    for ep in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y in dl:
            B, K = x.shape[:2]
            x = gpu_prepare(x.to(device, non_blocking=True), cfg["img_size"], train=True)
            y = y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x).float().view(B, K, -1)
            clip_logits = lse_pool(logits, cfg["lse_r"])
            loss = F.binary_cross_entropy_with_logits(clip_logits, y, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step(); sched.step()
            losses.append(loss.item())
        rec = dict(epoch=ep, loss=float(np.mean(losses)), time=time.time() - t0)
        if val_items:
            probs = predict_clips(model, val_items, cfg, device)
            yv = np.stack([it["y"] for it in val_items])
            m = clip_metrics(yv, probs, cfg["class_names"])
            rec["val_mAP"] = m["mAP"]
            for n in cfg["class_names"]:
                rec[f"AP_{n}"] = m[n]["AP"]
            if m["mAP"] > best["score"]:
                best.update(score=m["mAP"], epoch=ep, probs=probs,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        history.append(rec)
        print("  " + tag + " " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in rec.items()))
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, history, best


def make_folds(items, n_splits, seed=42):
    from sklearn.model_selection import StratifiedGroupKFold
    Y = np.stack([it["y"] for it in items])
    strat = (Y * (2 ** np.arange(Y.shape[1]))).sum(1).astype(int)
    groups = [it["group"] for it in items]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(items)), strat, groups))
