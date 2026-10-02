"""Blur test, step 1: model RDMs on Gaussian low-passed 224 px stimuli (inference only).

Hypothesis: BP filters need image structure at the training pixel scale. Low-passing the
native 224 px stimulus removes fine structure while keeping the input size (224 px) and the
geometry identical to NATIVE@224; only the spectral content changes.

Low-pass. A stimulus at effective resolution r carries no frequencies above the Nyquist
frequency of an r-px grid, r/2 cycles per image width = f_c = r / (2 * 224) cycles/px at
224 px. The Gaussian filter is chosen so that its amplitude response
H(f) = exp(-2 pi^2 sigma^2 f^2) falls to 0.5 at f_c:
    sigma(r) = sqrt(ln 2 / (2 pi^2)) / f_c  ~=  84 / r  px    (r = 32 ... 160)
Applied per channel to the [0, 1] image after Resize(224) -> CenterCrop(224) -> ToTensor
and before Normalize (scipy.ndimage.gaussian_filter, mode='reflect', truncate=4).
Without blur this is exactly v12's NATIVE transform (asserted).

Diagnostic (blur_transform.json): mean per-image Pearson correlation of the blurred image
with the UPSAMPLED image of the same r (both at 224 px, before normalisation), and of the
unblurred NATIVE image with it, so the match to the upsampling arm can be judged.

Models: Random Weights and Backprop, all 5 v12 seeds; features/RDMs exactly as step1 of the
upsampling test (v12 extract_features + compute_rdm). Conv1 and Conv3 RDMs saved.
Output: data/blur/rdms/r{r}/seed_{i}/rdm_{rule}_{layer}.npy, results/blur/blur_transform.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "upsampling"))
from common_up import (REPO, SEEDS, TRI, stim_order, namespaces, transform, load_v12_model)  # noqa: E402

LEVELS = [32, 64, 96, 128, 160]
RULES = ["Random Weights", "Backprop"]
LAYERS = {"Conv1": 0, "Conv3": 2}
PX = 224
RDM_DIR = REPO / "data" / "blur" / "rdms"
OUT = REPO / "results" / "blur"


def sigma(r):
    fc = r / (2 * PX)
    return math.sqrt(math.log(2) / (2 * math.pi ** 2)) / fc


def rdm_path(r, si, rule, layer):
    return RDM_DIR / f"r{r}" / f"seed_{si}" / f"rdm_{rule.lower().replace(' ', '_')}_{layer}.npy"


def main():
    torch.set_num_threads(12)
    import torchvision.transforms as T
    from PIL import Image
    ns12, ns11 = namespaces()
    paths = [ns12["find_img"](s) for s in stim_order()]
    assert all(p is not None for p in paths)
    pre = T.Compose([T.Resize(PX), T.CenterCrop(PX), T.ToTensor()])
    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    nat = transform(ns12, ns11, "NATIVE", PX)
    imgs = [Image.open(p).convert("RGB") for p in paths]
    X0 = torch.stack([pre(im) for im in imgs])                               # [0,1], 224 px
    for i in range(0, 720, 97):                                              # no blur == v12 NATIVE
        assert torch.equal(norm(X0[i]), nat(imgs[i]))
    # content held at r px, shown at 224 px (the v11 UPSAMPLED construction with r instead of 32)
    up_pre = {r: T.Compose([T.Resize(r), T.CenterCrop(r), T.Resize(PX), T.ToTensor()]) for r in LEVELS}

    def corr(a, b):
        a = a.flatten() - a.mean(); b = b.flatten() - b.mean()
        return float((a @ b) / math.sqrt(float(a @ a) * float(b @ b)))

    diag, t0 = {}, time.time()
    for r in LEVELS:
        s = sigma(r)
        Xb = torch.from_numpy(np.stack([gaussian_filter(x.numpy(), sigma=(0, s, s), mode="reflect", truncate=4.0)
                                        for x in X0])).float()
        Xu = torch.stack([up_pre[r](im) for im in imgs])
        diag[r] = {"sigma_px": s, "cutoff_cycles_per_image": r / 2,
                   "corr_blur_vs_upsampled_r": float(np.mean([corr(a, b) for a, b in zip(Xb, Xu)])),
                   "corr_native_vs_upsampled_r": float(np.mean([corr(a, b) for a, b in zip(X0, Xu)]))}
        X = norm(Xb)
        print(f"[{time.time()-t0:5.0f}s] r={r} sigma={s:.3f}px  corr(blur,UP)={diag[r]['corr_blur_vs_upsampled_r']:.4f} "
              f"corr(native,UP)={diag[r]['corr_native_vs_upsampled_r']:.4f}", flush=True)
        for rule in RULES:
            for si in range(len(SEEDS)):
                if all(rdm_path(r, si, rule, l).exists() for l in LAYERS):
                    continue
                torch.manual_seed(SEEDS[si])
                m = load_v12_model(ns12, rule, si)
                feats = ns12["extract_features"](m, list(range(len(paths))), X)
                for l, fi in LAYERS.items():
                    rdm = ns12["compute_rdm"](feats[fi])
                    p = rdm_path(r, si, rule, l)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    np.save(p, np.asarray(rdm[TRI], np.float64))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "blur_transform.json").write_text(json.dumps(
        {"method": "Gaussian low-pass, amplitude response 0.5 at r/2 cycles per image (Nyquist of an r-px grid); "
                   "per channel on [0,1] 224px image before Normalize; mode=reflect, truncate=4",
         "levels": {str(k): v for k, v in diag.items()}}, indent=2))
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
