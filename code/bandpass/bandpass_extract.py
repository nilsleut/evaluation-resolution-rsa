"""Band-pass test, step 1: model RDMs on octave-band-filtered 224 px stimuli (inference only).

Octave bands by difference of Gaussians, with the sigma convention of the blur test: a Gaussian
low-pass L_r has amplitude response 0.5 at r/2 cycles per image (the Nyquist frequency of an
r-px grid), sigma(r) = sqrt(ln 2 / (2 pi^2)) * 2 * 224 / r  ~= 84 / r px. With cutoffs
r = 112, 56, 28, 14 (the native 224 px image is the upper end):

  band  "112-224"  = X - L112          finest octave
  band  "56-112"   = L112 - L56
  band  "28-56"    = L56 - L28
  band  "14-28"    = L28 - L14
  band  "0-14"     = L14               coarse residual, contains the image mean (DC)

Conditions per band:
  only   the band alone. Bands without DC (all but 0-14) are shifted by a constant mid-grey
         (the CIFAR-10 channel means), the same for every image, so no image-specific mean
         luminance re-enters.
  notch  native minus the band (X - band). For band 0-14 this removes the DC as well and is
         shifted by the same constant grey.
Everything else as in the blur test: filter on the [0,1] image after Resize(224) ->
CenterCrop(224) -> ToTensor, then Normalize; Random Weights and Backprop, 5 v12 seeds;
v12 extract_features + compute_rdm; Conv1 and Conv3 RDMs saved.

Output: data/bandpass/rdms/{cond}/seed_{i}/rdm_{rule}_{layer}.npy, results/bandpass/bandpass_transform.json
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
from common_up import REPO, SEEDS, TRI, stim_order, namespaces, transform, load_v12_model  # noqa: E402

PX = 224
CUTS = [112, 56, 28, 14]
BANDS = ["112-224", "56-112", "28-56", "14-28", "0-14"]
CONDS = [f"{kind}_{b}" for b in BANDS for kind in ("only", "notch")]
RULES = ["Random Weights", "Backprop"]
LAYERS = {"Conv1": 0, "Conv3": 2}
GREY = np.array([0.4914, 0.4822, 0.4465], np.float32)[:, None, None]
RDM_DIR = REPO / "data" / "bandpass" / "rdms"
OUT = REPO / "results" / "bandpass"


def sigma(r):
    return math.sqrt(math.log(2) / (2 * math.pi ** 2)) * 2 * PX / r


def rdm_path(cond, si, rule, layer):
    return RDM_DIR / cond / f"seed_{si}" / f"rdm_{rule.lower().replace(' ', '_')}_{layer}.npy"


def bands(x):
    """x: (3, H, W) float32 in [0,1] -> dict band -> array (sums back to x exactly)."""
    L = {r: gaussian_filter(x, sigma=(0, sigma(r), sigma(r)), mode="reflect", truncate=4.0) for r in CUTS}
    b = {"112-224": x - L[112], "56-112": L[112] - L[56], "28-56": L[56] - L[28],
         "14-28": L[28] - L[14], "0-14": L[14]}
    return b


def condition(x, b, cond):
    kind, band = cond.split("_", 1)
    if kind == "only":
        return b[band] + (0 if band == "0-14" else GREY)
    return x - b[band] + (GREY if band == "0-14" else 0)


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
    X0 = np.stack([pre(im).numpy() for im in imgs]).astype(np.float32)
    assert torch.equal(norm(torch.from_numpy(X0[0])), nat(imgs[0]))
    B20 = [bands(x) for x in X0[:20]]
    recon = max(float(np.abs(sum(b.values()) - x).max()) for b, x in zip(B20, X0[:20]))
    assert recon < 1e-5, recon                                           # bands sum to the image
    energy = {k: float(np.mean([np.var(b[k]) for b in B20])) for k in BANDS}   # first 20 images
    t0 = time.time()
    for cond in CONDS:
        X = norm(torch.from_numpy(np.stack([condition(x, bands(x), cond) for x in X0])))
        print(f"[{time.time()-t0:5.0f}s] {cond}", flush=True)
        for rule in RULES:
            for si in range(len(SEEDS)):
                if all(rdm_path(cond, si, rule, l).exists() for l in LAYERS):
                    continue
                torch.manual_seed(SEEDS[si])
                m = load_v12_model(ns12, rule, si)
                feats = ns12["extract_features"](m, list(range(len(paths))), X)
                for l, fi in LAYERS.items():
                    rdm = ns12["compute_rdm"](feats[fi])
                    p = rdm_path(cond, si, rule, l)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    np.save(p, np.asarray(rdm[TRI], np.float64))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bandpass_transform.json").write_text(json.dumps(
        {"sigma_px": {str(r): sigma(r) for r in CUTS}, "band_label_px_to_cycles_per_image": {"112-224": "56-112", "56-112": "28-56", "28-56": "14-28",
                                               "14-28": "7-14", "0-14": "0-7 (incl. DC)"},
         "mean_pixel_variance_per_band": energy, "reconstruction_max_abs_err": recon,
         "grey_shift": GREY.ravel().tolist()}, indent=2))
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
