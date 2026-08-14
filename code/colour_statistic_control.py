"""
colour_statistic_control.py
===========================
Does the Conv1 descriptor converge to a global COLOUR statistic as evaluation
resolution rises?

HYPOTHESIS UNDER TEST
Conv1 features are global-average-pooled over the post-block spatial map. At 32 px
that map is 16x16 = 256 positions; at 224 px it is 112x112 = 12544. As the number of
pooled positions grows, the pooled descriptor converges toward the expected filter
response over the whole image. For random colour-opponent filters that limit is close
to a global colour statistic, which would explain in one step why the untrained curve
RISES with resolution while every trained curve FALLS (selectivity is what averaging
destroys).

PREDICTION, FIXED BEFORE LOOKING
The untrained CNN's Conv1 RDM should become steeply more similar to MEANRGB /
COLORHIST as resolution rises, approaching a high value at 224 px, while the trained
conditions stay low or fall. And the colour references must themselves predict V1 --
a statistic that does not predict V1 explains nothing.

THIS IS NOT THE PIXEL CONTROL
rsa_lowlevel_control.csv's PIXEL RDM is correlation distance between flattened
224x224x3 images, which is dominated by spatial layout; the endpoint study's partial
RSA against it left the V1 ordering intact. MEANRGB / COLORHIST / MEANLUM discard
layout entirely, so they are a different account. PIXEL is recomputed here purely as
a same-table contrast.

REFERENCES (parameter-free, one fixed set over the same 720 THINGS stimuli)
  MEANRGB    per-image RGB channel means (3-dim)   -> correlation distance
  COLORHIST  per-image joint RGB histogram 8x8x8   -> correlation distance
  MEANLUM    per-image Rec.709 luminance (scalar)  -> absolute difference
  PIXEL      flattened 224x224x3                   -> correlation distance (contrast)

CPU only, no Modal, minutes.

Usage:
  python colour_statistic_control.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

ROOT       = Path(__file__).parent
FMRI_DIR   = ROOT / "outputs_720"
THINGS_DIR = Path(r"C:/Users/nilsl/Desktop/Projekte_1/RSA/Datensatz/images_THINGS/object_images")
RDM_DIR    = ROOT / "learning_rules_outputs_bnfix" / "rdms"
OUT_CSV    = ROOT / "colour_statistic_control.csv"

SUBJECTS = ["sub-01", "sub-02", "sub-03"]
RES      = [32, 64, 96, 128, 160, 224]
RULES    = ["Random Weights", "Random Weights (BN-calibrated)", "Backprop",
            "Feedback Alignment", "Predictive Coding", "STDP"]
SEEDS    = [0, 1, 2, 3, 4]
REFS     = ["MEANRGB", "COLORHIST", "MEANLUM", "PIXEL"]

REF_PX   = 224          # references are built once, at the largest eval geometry
HIST_BINS = 8


def rule_key(rule):
    return rule.lower().replace(" ", "_")


# ── Conv1 pooling geometry ───────────────────────────────────────────────────
def pooled_positions(px):
    """conv1 block = Conv2d(3,32,3,padding=1) -> BN -> ReLU -> MaxPool2d(2).
    The conv is size-preserving, so the pooled map is floor(px/2) squared, and the
    Conv1 feature is the mean over exactly that many positions."""
    return (px // 2) ** 2


# ── Stimuli ──────────────────────────────────────────────────────────────────
def load_stim_order(sub="sub-01"):
    with open(FMRI_DIR / f"stim_order_{sub}.txt", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def find_img(stimulus):
    """Identical resolution logic to the sweep, so the image ORDER matches the
    model and brain RDMs row for row."""
    name = stimulus.replace(".jpg", ""); parts = name.split("_"); last = parts[-1]
    concept = ("_".join(parts[:-1])
               if (len(parts) > 1 and len(last) <= 4 and any(c.isdigit() for c in last))
               else name)
    for pat in [f"{concept}/{name}.jpg", f"{concept}/*.jpg"]:
        hits = sorted(THINGS_DIR.glob(pat))
        if hits: return hits[0]
    for folder in THINGS_DIR.iterdir():
        if folder.name.lower() == concept.lower():
            imgs = sorted(folder.glob("*.jpg"))
            if imgs: return imgs[0]
    return None


def load_rgb(path, px):
    """Resize(px) + CenterCrop(px), RGB in [0,1] -- the eval geometry, without the
    CIFAR normalisation (an affine per-channel map; see the sensitivity check)."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = px / min(w, h)
    im = im.resize((max(px, int(round(w * s))), max(px, int(round(h * s)))),
                   Image.BILINEAR)
    w, h = im.size
    l, t = (w - px) // 2, (h - px) // 2
    return np.asarray(im.crop((l, t, l + px, t + px)), dtype=np.float32) / 255.0


# ── RDM construction ─────────────────────────────────────────────────────────
def corr_rdm(X):
    """Correlation-distance RDM. Row-wise z-scoring + a Gram matrix is identical to
    pdist(X,'correlation') but does not materialise 259k pairwise vectors, which
    matters for the 150k-dimensional PIXEL case."""
    X = np.asarray(X, dtype=np.float64)
    Z = X - X.mean(1, keepdims=True)
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    n[n == 0] = 1.0
    Z /= n
    R = np.clip(Z @ Z.T, -1.0, 1.0)
    D = 1.0 - R
    np.fill_diagonal(D, 0.0)
    return D


def abs_rdm(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return np.abs(v[:, None] - v[None, :])


def build_references(paths, px=REF_PX):
    imgs = [load_rgb(p, px) for p in paths]
    mean_rgb = np.stack([im.reshape(-1, 3).mean(0) for im in imgs])          # (720,3)
    lum = np.stack([(im @ np.array([0.2126, 0.7152, 0.0722],
                                   dtype=np.float32)).mean() for im in imgs])
    hist = np.stack([
        np.histogramdd(im.reshape(-1, 3), bins=(HIST_BINS,) * 3,
                       range=((0, 1), (0, 1), (0, 1)))[0].ravel() / im.shape[0] ** 2
        for im in imgs])                                                     # (720,512)
    pixel = np.stack([im.ravel() for im in imgs])                            # (720,150528)

    refs = {
        "MEANRGB":   corr_rdm(mean_rgb),
        "COLORHIST": corr_rdm(hist),
        "MEANLUM":   abs_rdm(lum),
        "PIXEL":     corr_rdm(pixel),
    }
    return refs, mean_rgb, lum


# ── Comparison ───────────────────────────────────────────────────────────────
def rsa(a, b):
    """Spearman over the upper triangle, min-n truncation -- the main pipeline's
    rsa_score()."""
    n = min(a.shape[0], b.shape[0]); idx = np.triu_indices(n, k=1)
    r, _ = spearmanr(a[:n, :n][idx], b[:n, :n][idx])
    return float(r)


def brain_v1():
    subs = [np.load(str(FMRI_DIR / f"fmri_rdm_V1_{s}.npy")) for s in SUBJECTS
            if (FMRI_DIR / f"fmri_rdm_V1_{s}.npy").exists()]
    return np.mean(subs, axis=0), subs


def main():
    stimuli = load_stim_order("sub-01")
    paths = [p for p in (find_img(s) for s in stimuli) if p is not None]
    print(f"THINGS: {len(paths)}/{len(stimuli)} images resolved\n")

    print("Conv1 pooling geometry")
    print(f"{'eval res (px)':<16}" + "".join(f"{r:>10}" for r in RES))
    print(f"{'pooled map':<16}" + "".join(f"{f'{px//2}x{px//2}':>10}" for px in RES))
    print(f"{'positions':<16}" + "".join(f"{pooled_positions(px):>10,}" for px in RES))

    print(f"\nBuilding reference RDMs at {REF_PX} px ...")
    refs, mean_rgb, lum = build_references(paths)

    # Sensitivity: are the colour references resolution-invariant, as claimed?
    refs32, _, _ = build_references(paths, px=32)
    print("\nReference RDM stability, built at 224 px vs at 32 px (Spearman):")
    for k in ["MEANRGB", "COLORHIST", "MEANLUM"]:
        print(f"  {k:<10} {rsa(refs[k], refs32[k]):.4f}")

    # Sensitivity: does the CIFAR normalisation the model applies change MEANRGB?
    m, s = np.array([0.4914, 0.4822, 0.4465]), np.array([0.247, 0.243, 0.261])
    print(f"  {'MEANRGB (CIFAR-normalised input)':<10} "
          f"{rsa(refs['MEANRGB'], corr_rdm((mean_rgb - m) / s)):.4f}")

    bmean, bsubs = brain_v1()

    # ── how well does each reference itself predict V1? ──────────────────────
    print("\n" + "=" * 78)
    print("Reference RDM vs the V1 brain RDM (the load-bearing question)")
    print("=" * 78)
    print(f"{'reference':<12}{'vs mean-V1':>12}{'sub-01':>10}{'sub-02':>10}{'sub-03':>10}")
    print("-" * 78)
    ref_v1 = {}
    for k in REFS:
        ref_v1[k] = rsa(refs[k], bmean)
        per = [rsa(refs[k], b) for b in bsubs]
        print(f"{k:<12}{ref_v1[k]:>12.4f}" + "".join(f"{p:>10.4f}" for p in per))

    # ── model Conv1 RDM vs each reference ────────────────────────────────────
    rows = []
    for rule in RULES:
        for px in RES:
            for si in SEEDS:
                f = RDM_DIR / f"res{px}" / f"seed_{si}" / f"rdm_{rule_key(rule)}_Conv1.npy"
                if not f.exists():
                    print(f"  MISSING {f}"); continue
                M = np.load(str(f))
                row = {"rule": rule, "res": px, "seed_idx": si,
                       "pooled_positions": pooled_positions(px),
                       "rho_v1": rsa(M, bmean)}
                for k in REFS:
                    row[f"rho_{k}"] = rsa(M, refs[k])
                rows.append(row)
        print(f"  done {rule}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n{len(df)} rows -> {OUT_CSV.name}")

    # Validation: rho_v1 recomputed here must match the sweep CSV exactly.
    sw = pd.read_csv(ROOT / "bnfix_sweep.csv")
    sw = sw[(sw.layer == "Conv1") & (sw.roi == "V1")]
    j = df.merge(sw[["rule", "res", "seed_idx", "rho"]], on=["rule", "res", "seed_idx"])
    print(f"VALIDATION  recomputed Conv1->V1 rho vs bnfix_sweep.csv: "
          f"max|diff| = {(j.rho_v1 - j.rho).abs().max():.2e}  (n={len(j)})")

    for k in REFS:
        print("\n" + "=" * 92)
        print(f"Conv1 RDM  vs  {k}   (Spearman, mean over 5 seeds)")
        print(f"   [{k} vs V1 = {ref_v1[k]:+.4f}]")
        print("=" * 92)
        t = df.groupby(["rule", "res"])[f"rho_{k}"].mean().unstack("res")
        hdr = f"{'rule':<32}" + "".join(f"{r:>11}" for r in RES) + f"{'224-32':>10}"
        print(hdr); print("-" * len(hdr))
        for rule in RULES:
            if rule not in t.index: continue
            v = [t.loc[rule, r] for r in RES]
            print(f"{rule:<32}" + "".join(f"{x:>11.4f}" for x in v)
                  + f"{v[-1]-v[0]:>+10.4f}")


if __name__ == "__main__":
    main()
