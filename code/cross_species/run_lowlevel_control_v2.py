#!/usr/bin/env python3
"""
run_lowlevel_control.py
=======================
Mechanism test for the resolution effect. Hypothesis: a model's V1 alignment is
driven by how much its early representation resembles low-level (local-contrast /
Gabor) image statistics. Untrained random filters stay Gabor-like (more so at high
resolution); training drifts representations away from Gabor-like -> lower V1
alignment, most at high resolution. Eval resolution modulates the gap.

For each resolution we build:
  - a Gabor filterbank RDM   (parameter-free low-level V1 model)
  - a raw-pixel (luminance) RDM   (cruder low-level baseline)
  - each model's early-layer RDM   (random CNN Conv1, backprop CNN Conv1, ResNet-50 layer1)
  - the human V1 fMRI RDM (fixed)
and report, per resolution:
  corr(model RDM, Gabor RDM)     -> how low-level-like is the model
  corr(model RDM, V1 RDM)        -> V1 alignment (= the sweep)
  corr(Gabor RDM, V1 RDM)        -> is V1 itself low-level?
  corr(pixel RDM, V1 RDM)        -> sanity baseline

If the hypothesis holds: random's Gabor-similarity is high & rises with resolution
and tracks its V1 alignment; trained models' Gabor-similarity is lower & falls and
tracks theirs; and a model's Gabor-similarity largely predicts its V1 alignment.

Run from the Cross_Species_RSA project root:
    python run_lowlevel_control.py
Output: results/rsa_lowlevel_control_bnfix.csv
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import fftconvolve
from scipy.stats import spearmanr
import torchvision.transforms as transforms
import torch
import torchvision.models as tv_models
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_loader import load_things_fmri
from models_v2 import load_model, extract_features   # v2 = BatchNorm-mode fix
from rsa_engine import build_rdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("lowlevel")

SWEEP_RES = [32, 64, 96, 128, 160, 224]
CNN_RULES = ["random", "backprop"]          # untrained vs trained (the divergence)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _resolve(c):
    for x in c:
        if Path(x).exists():
            return Path(x)
    return Path(c[0])

# v1 pointed at "Projekte/" (no _1), which does not exist here.
FMRI_DIR = _resolve([r"C:/Users/nilsl/Desktop/Projekte_1/learning-rules-rsa/outputs_720",
                     "/kaggle/input/datasets/nilsleutenegger/learning-rules-fmri/outputs_720"])
THINGS_DIR = _resolve([r"C:/Users/nilsl/Desktop/Projekte_1/RSA/Datensatz/images_THINGS/object_images",
                       "/kaggle/input/datasets/nilsleutenegger/things-object-images/images_THINGS/object_images"])
# BN-fixed re-run checkpoints, NOT the pre-fix outputs/.
WEIGHTS_DIR = _resolve([r"C:/Users/nilsl/Desktop/Projekte_1/learning-rules-rsa/learning_rules_outputs_bnfix/checkpoints",
                        "/kaggle/input/learning-rules-weights/outputs_bnfix/checkpoints"])
OUT = Path("results"); OUT.mkdir(parents=True, exist_ok=True)


def things_paths():
    with open(FMRI_DIR / "stim_order_sub-01.txt", encoding="utf-8") as f:
        names = [l.strip() for l in f if l.strip()]
    m = {p.name: str(p) for p in THINGS_DIR.rglob("*.jpg")}
    paths = [m[n] for n in names if n in m]
    log.info(f"Resolved {len(paths)}/{len(names)} stimuli")
    return paths


# ── Gabor filterbank (manual, no skimage dependency) ────────────────────────
def _gabor_kernel(theta, lam, sigma, gamma=0.5, ksize=21):
    half = ksize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    env = np.exp(-(xr ** 2 + (gamma ** 2) * yr ** 2) / (2 * sigma ** 2))
    even = env * np.cos(2 * np.pi * xr / lam)
    odd = env * np.sin(2 * np.pi * xr / lam)
    return even - even.mean(), odd - odd.mean()

_BANK = []
for th in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
    for lam in [4.0, 8.0, 16.0]:
        _BANK.append(_gabor_kernel(th, lam, sigma=0.5 * lam))


def gabor_features(paths, px):
    feats = []
    for p in paths:
        g = np.asarray(Image.open(p).convert("L").resize((px, px)), dtype=np.float64) / 255.0
        g = g - g.mean()
        v = []
        for even, odd in _BANK:
            re = fftconvolve(g, even, mode="same")
            im = fftconvolve(g, odd, mode="same")
            v.append(np.sqrt(re ** 2 + im ** 2).mean())   # energy per filter
        feats.append(v)
    return np.asarray(feats)


def pixel_features(paths, px):
    s = min(px, 32)  # cap luminance vector size
    return np.stack([np.asarray(Image.open(p).convert("L").resize((s, s)), dtype=np.float64).ravel() / 255.0
                     for p in paths])


# ── ResNet-50 early-layer extraction ────────────────────────────────────────
class _DS(Dataset):
    def __init__(s, p, t): s.p, s.t = list(p), t
    def __len__(s): return len(s.p)
    def __getitem__(s, i): return s.t(Image.open(s.p[i]).convert("RGB")), i

def resnet_layer1(paths, px):
    model = tv_models.resnet50(weights="IMAGENET1K_V2").to(DEVICE).eval()
    t = transforms.Compose([transforms.Resize(px), transforms.CenterCrop(px), transforms.ToTensor(),
                            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    out = []
    h = model.layer1.register_forward_hook(lambda m, i, o: out.append(o.mean(dim=[2, 3]).detach().cpu().numpy()))
    with torch.no_grad():
        for imgs, _ in DataLoader(_DS(paths, t), batch_size=64):
            model(imgs.to(DEVICE))
    h.remove()
    return np.concatenate(out)


def cnn_conv1(rule, paths, px):
    model = load_model(rule, weights_dir=WEIGHTS_DIR)
    t = transforms.Compose([transforms.Resize(px), transforms.CenterCrop(px), transforms.ToTensor(),
                            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    return extract_features(model, paths, t)["Conv1"]


def rdm_corr(a, b):
    n = min(a.shape[0], b.shape[0])
    iu = np.triu_indices(n, k=1)
    return float(spearmanr(a[:n, :n][iu], b[:n, :n][iu]).correlation)


def main():
    log.info(f"Device: {DEVICE}")
    paths = things_paths()
    fmri = load_things_fmri(str(FMRI_DIR))
    v1 = fmri["V1"]["rdm"]

    rows = []
    for px in SWEEP_RES:
        gab = build_rdm(gabor_features(paths, px))
        pix = build_rdm(pixel_features(paths, px))
        models = {f"cnn_{r}": build_rdm(cnn_conv1(r, paths, px)) for r in CNN_RULES}
        models["resnet50"] = build_rdm(resnet_layer1(paths, px))

        gab_v1, pix_v1 = rdm_corr(gab, v1), rdm_corr(pix, v1)
        for name, mrdm in models.items():
            rows.append({"model": name, "res": px,
                         "align_v1": round(rdm_corr(mrdm, v1), 6),
                         "sim_gabor": round(rdm_corr(mrdm, gab), 6),
                         "sim_pixel": round(rdm_corr(mrdm, pix), 6)})
        rows.append({"model": "GABOR", "res": px, "align_v1": round(gab_v1, 6),
                     "sim_gabor": 1.0, "sim_pixel": round(rdm_corr(gab, pix), 6)})
        rows.append({"model": "PIXEL", "res": px, "align_v1": round(pix_v1, 6),
                     "sim_gabor": round(rdm_corr(pix, gab), 6), "sim_pixel": 1.0})
        log.info(f"  @{px:>3}px  Gabor->V1={gab_v1:+.4f} | "
                 + " ".join(f"{n.split('_')[-1]}:align={r['align_v1']:+.3f},gab={r['sim_gabor']:+.3f}"
                            for n, r in [(k, [x for x in rows if x['model'] == k and x['res'] == px][0])
                                         for k in models]))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rsa_lowlevel_control_bnfix.csv", index=False)
    log.info(f"\nSaved {len(df)} rows -> {OUT / 'rsa_lowlevel_control_bnfix.csv'}")

    # ── Hypothesis test: does Gabor-similarity predict V1 alignment across models x res? ──
    net = df[df.model.isin([f"cnn_{r}" for r in CNN_RULES] + ["resnet50"])]
    r_all = spearmanr(net.sim_gabor, net.align_v1).correlation
    log.info(f"\nAcross all (model x resolution): corr(Gabor-similarity, V1-alignment) = {r_all:+.4f}")
    log.info("Per model — does Gabor-similarity rise/fall WITH V1 alignment over resolution?")
    for m in net.model.unique():
        s = net[net.model == m].sort_values("res")
        log.info(f"  {m:<14} align_v1 {s.align_v1.iloc[0]:+.3f}->{s.align_v1.iloc[-1]:+.3f} | "
                 f"sim_gabor {s.sim_gabor.iloc[0]:+.3f}->{s.sim_gabor.iloc[-1]:+.3f}")
    log.info("\nHypothesis supported if: (i) Gabor->V1 is high, (ii) corr(Gabor-sim, V1-align) strongly +,")
    log.info("(iii) random stays high-Gabor & rises, trained fall on BOTH Gabor-sim and V1-align.")


if __name__ == "__main__":
    main()
