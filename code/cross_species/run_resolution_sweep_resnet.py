#!/usr/bin/env python3
"""
run_resolution_sweep_resnet.py
==============================
Crossover test for the resolution confound: an ImageNet-pretrained ResNet-50
(trained at 224px) should show the OPPOSITE resolution dependence to the
CIFAR-trained custom CNNs (trained at 32px). If model-brain alignment peaks at
the model's training resolution, ResNet V1-alignment should RISE toward 224,
while the custom CNNs FALL toward 224.

Mirrors scripts/run_resnet50_baseline.py exactly (weights IMAGENET1K_V2, ImageNet
normalisation, layer1->V1/V2, layer2->V4, layer4->IT) but sweeps the eval
resolution instead of fixing it at 224.

Run from the Cross_Species_RSA project root:
    python run_resolution_sweep_resnet.py

Output (same schema as rsa_resolution_sweep_human.csv, with rule="resnet50"):
    results/rsa_resolution_sweep_resnet.csv  (rule, roi, layer, res, rho, ci_lo, ci_hi, n_stim)
    -> concat with the CNN sweep to plot both families on one curve.
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
log = logging.getLogger("sweep-resnet")

SWEEP_RES = [32, 64, 96, 128, 160, 224]
N_BOOT    = 500
# His Paper-2 mapping (layer1->V1/V2, layer2->V4, layer4->IT). Headline = V1.
ROI_LAYER = {"V1": "layer1", "V2": "layer1", "V4": "layer2", "IT": "layer4"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve(cands):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return Path(cands[0])

FMRI_DIR = _resolve([
    r"C:/Users/nilsl/Desktop/Projekte/learning-rules-rsa/outputs_720",
    "/kaggle/input/datasets/nilsleutenegger/learning-rules-fmri/outputs_720",
])
THINGS_DIR = _resolve([
    r"C:/Users/nilsl/Desktop/Projekte/RSA/Datensatz/images_THINGS/object_images",
    "/kaggle/input/datasets/nilsleutenegger/things-object-images/images_THINGS/object_images",
])
OUT = Path("results"); OUT.mkdir(parents=True, exist_ok=True)


class _ImgDS(Dataset):
    def __init__(self, paths, t): self.paths, self.t = list(paths), t
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return self.t(Image.open(str(self.paths[i])).convert("RGB")), i


def _tf(px):
    # match the CNN sweep geometry (Resize(px)+CenterCrop(px)) but ImageNet norm
    return transforms.Compose([
        transforms.Resize(px), transforms.CenterCrop(px), transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _resolve_things_paths():
    with open(FMRI_DIR / "stim_order_sub-01.txt", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    img_map = {p.name: str(p) for p in THINGS_DIR.rglob("*.jpg")}
    paths = [img_map[n] for n in names if n in img_map]
    log.info(f"Resolved {len(paths)}/{len(names)} THINGS stimuli")
    return paths


def extract_resnet50(model, paths, transform, batch_size=64):
    collected = {"layer1": [], "layer2": [], "layer4": []}
    current = {}
    handles = []
    for name in ("layer1", "layer2", "layer4"):
        def _hook(m, inp, out, _n=name):
            current[_n] = out.mean(dim=[2, 3]).detach().cpu().numpy()
        handles.append(getattr(model, name).register_forward_hook(_hook))
    loader = DataLoader(_ImgDS(paths, transform), batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for imgs, _ in loader:
            current.clear()
            model(imgs.to(DEVICE))
            for k in collected:
                collected[k].append(current[k])
    for h in handles:
        h.remove()
    return {k: np.concatenate(v) for k, v in collected.items()}


def main():
    log.info(f"Device: {DEVICE}")
    paths = _resolve_things_paths()
    fmri = load_things_fmri(str(FMRI_DIR))
    roi_rdm = {roi: fmri[roi]["rdm"] for roi in ROI_LAYER if roi in fmri}
    log.info(f"fMRI RDMs available: {list(roi_rdm)}")

    model = tv_models.resnet50(weights="IMAGENET1K_V2").to(DEVICE).eval()

    rows = []
    for res_px in SWEEP_RES:
        feats = extract_resnet50(model, paths, _tf(res_px))
        for roi, layer in ROI_LAYER.items():
            if roi not in roi_rdm:
                continue
            mdl = build_rdm(feats[layer])
            neu = roi_rdm[roi]
            n = min(mdl.shape[0], neu.shape[0])
            res = compare_rdms_bootstrap(mdl[:n, :n], neu[:n, :n], n_bootstrap=N_BOOT)
            lo, hi = res.get("ci_lower"), res.get("ci_upper")
            rows.append({"rule": "resnet50", "roi": roi, "layer": layer, "res": int(res_px),
                         "rho": round(float(res["rho"]), 6),
                         "ci_lo": round(float(lo), 6) if lo is not None else None,
                         "ci_hi": round(float(hi), 6) if hi is not None else None,
                         "n_stim": int(n)})
        v1 = next((r for r in rows if r["roi"] == "V1" and r["res"] == res_px), None)
        log.info(f"  @{res_px:>3}px  V1 rho={v1['rho']:.4f}" if v1 else f"  @{res_px:>3}px done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rsa_resolution_sweep_resnet.csv", index=False)
    log.info(f"\nSaved {len(df)} rows -> {OUT / 'rsa_resolution_sweep_resnet.csv'}")

    v1 = df[df.roi == "V1"].sort_values("res")
    log.info("\nResNet-50 V1 alignment by resolution (crossover check):")
    for _, r in v1.iterrows():
        log.info(f"  @{int(r.res):>3}px:  rho = {r.rho:+.4f}")
    if len(v1) > 1:
        peak = v1.loc[v1.rho.idxmax()]
        trend = "RISES toward 224 (crossover holds)" if v1.iloc[-1].rho > v1.iloc[0].rho else "falls toward 224 (no crossover)"
        log.info(f"\n  Peak @ {int(peak.res)}px (rho={peak.rho:.4f}); V1 {trend}")
        log.info("  Expectation: ImageNet-trained -> peak near 224, opposite of the CIFAR CNNs.")


if __name__ == "__main__":
    main()
