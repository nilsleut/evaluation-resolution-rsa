#!/usr/bin/env python3
"""
run_resolution_sweep_swin.py
============================
Architecture-generality control: does the early-visual resolution confound extend
to a transformer family? Swin-Tiny (ImageNet-pretrained, no training) evaluated
across the resolution sweep, same RSA pipeline as the CNN/ResNet sweeps.

Reports "extends to one transformer family" (Swin) — NOT a mechanism.

Run from the Cross_Species_RSA project root:
    python run_resolution_sweep_swin.py
Output (same schema as the other sweeps, rule="swin_t"):
    results/rsa_resolution_sweep_swin.csv   (rule, roi, layer, res, rho, ci_lo, ci_hi, n_stim)
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_loader import load_things_fmri
from rsa_engine import build_rdm, compare_rdms_bootstrap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("sweep-swin")

SWEEP_RES = [32, 64, 96, 128, 160, 224]
N_BOOT    = 500
# early stage -> V1/V2 ; late stage -> IT (mirrors the layer1->V1, layer4->IT ResNet choice)
ROI_LAYER = {"V1": "early", "V2": "early", "IT": "late"}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve(c):
    for x in c:
        if Path(x).exists():
            return Path(x)
    return Path(c[0])

FMRI_DIR = _resolve([r"C:/Users/nilsl/Desktop/Projekte/learning-rules-rsa/outputs_720",
                     "/kaggle/input/datasets/nilsleutenegger/learning-rules-fmri/outputs_720"])
THINGS_DIR = _resolve([r"C:/Users/nilsl/Desktop/Projekte/RSA/Datensatz/images_THINGS/object_images",
                       "/kaggle/input/datasets/nilsleutenegger/things-object-images/images_THINGS/object_images"])
OUT = Path("results"); OUT.mkdir(parents=True, exist_ok=True)


class _DS(Dataset):
    def __init__(s, p, t): s.p, s.t = list(p), t
    def __len__(s): return len(s.p)
    def __getitem__(s, i): return s.t(Image.open(s.p[i]).convert("RGB")), i


def things_paths():
    with open(FMRI_DIR / "stim_order_sub-01.txt", encoding="utf-8") as f:
        names = [l.strip() for l in f if l.strip()]
    m = {p.name: str(p) for p in THINGS_DIR.rglob("*.jpg")}
    paths = [m[n] for n in names if n in m]
    log.info(f"Resolved {len(paths)}/{len(names)} stimuli")
    return paths


def _tf(px):
    return transforms.Compose([transforms.Resize(px), transforms.CenterCrop(px), transforms.ToTensor(),
                               transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def _pick_stage_modules(model):
    """Return (early_module, late_module) robustly across torchvision/timm layouts."""
    if hasattr(model, "features"):           # torchvision: Sequential [embed, stage1, merge, stage2, ...]
        feats = model.features
        early = feats[1]
        late = feats[7] if len(feats) > 7 else feats[-1]
        log.info(f"Using torchvision .features: early=features[1], late=features[{7 if len(feats)>7 else len(feats)-1}]")
    elif hasattr(model, "layers"):           # timm-style: layers[0..3] are the 4 stages
        early, late = model.layers[0], model.layers[3]
        log.info("Using .layers: early=layers[0], late=layers[3]")
    else:
        raise RuntimeError("Cannot locate Swin stages (no .features or .layers)")
    return early, late


def _pool(out):
    """Swin emits channels-last [B,H,W,C] (4D) or [B,L,C] (3D); pool to [B,C]."""
    if out.ndim == 4:
        return out.mean(dim=[1, 2]).detach().cpu().numpy()
    if out.ndim == 3:
        return out.mean(dim=1).detach().cpu().numpy()
    return out.detach().cpu().numpy()


def extract_swin(model, early_m, late_m, paths, transform):
    store = {}
    h1 = early_m.register_forward_hook(lambda m, i, o: store.__setitem__("early", _pool(o)))
    h2 = late_m.register_forward_hook(lambda m, i, o: store.__setitem__("late", _pool(o)))
    out = {"early": [], "late": []}
    try:
        with torch.no_grad():
            for imgs, _ in DataLoader(_DS(paths, transform), batch_size=32):
                store.clear()
                model(imgs.to(DEVICE))
                out["early"].append(store["early"])
                out["late"].append(store["late"])
    finally:
        h1.remove(); h2.remove()
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    log.info(f"Device: {DEVICE}")
    paths = things_paths()
    fmri = load_things_fmri(str(FMRI_DIR))
    roi_rdm = {roi: fmri[roi]["rdm"] for roi in ROI_LAYER if roi in fmri}
    log.info(f"fMRI RDMs available: {list(roi_rdm)}")

    model = tv_models.swin_t(weights="IMAGENET1K_V1").to(DEVICE).eval()
    early_m, late_m = _pick_stage_modules(model)

    rows = []
    for res_px in SWEEP_RES:
        try:
            feats = extract_swin(model, early_m, late_m, paths, _tf(res_px))
        except Exception as e:
            log.warning(f"  @{res_px}px extraction failed ({type(e).__name__}: {e}) -> skip "
                        f"(Swin window may exceed feature-map size at low res)")
            continue
        if res_px == SWEEP_RES[0]:
            log.info(f"  feature dims: early={feats['early'].shape[1]}, late={feats['late'].shape[1]} "
                     f"(expect ~96 and ~768 for swin_t)")
        for roi, stage in ROI_LAYER.items():
            if roi not in roi_rdm:
                continue
            mdl = build_rdm(feats[stage])
            neu = roi_rdm[roi]
            n = min(mdl.shape[0], neu.shape[0])
            res = compare_rdms_bootstrap(mdl[:n, :n], neu[:n, :n], n_bootstrap=N_BOOT)
            lo, hi = res.get("ci_lower"), res.get("ci_upper")
            rows.append({"rule": "swin_t", "roi": roi, "layer": stage, "res": int(res_px),
                         "rho": round(float(res["rho"]), 6),
                         "ci_lo": round(float(lo), 6) if lo is not None else None,
                         "ci_hi": round(float(hi), 6) if hi is not None else None,
                         "n_stim": int(n)})
        v1 = next((r for r in rows if r["roi"] == "V1" and r["res"] == res_px), None)
        log.info(f"  @{res_px:>3}px  V1 rho={v1['rho']:.4f}" if v1 else f"  @{res_px:>3}px done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rsa_resolution_sweep_swin.csv", index=False)
    log.info(f"\nSaved {len(df)} rows -> {OUT / 'rsa_resolution_sweep_swin.csv'}")

    v1 = df[df.roi == "V1"].sort_values("res")
    if len(v1) > 1:
        log.info("\nSwin-Tiny V1 alignment by resolution:")
        for _, r in v1.iterrows():
            log.info(f"  @{int(r.res):>3}px:  rho = {r.rho:+.4f}")
        trend = "FALLS toward 224 (same as CNN/ResNet)" if v1.iloc[-1].rho < v1.iloc[0].rho else "rises toward 224"
        log.info(f"\n  V1 {trend}. (Reported as: 'extends to one transformer family'.)")


if __name__ == "__main__":
    main()
