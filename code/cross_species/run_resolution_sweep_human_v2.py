#!/usr/bin/env python3
"""
run_resolution_sweep_human.py
=============================
Resolution SWEEP on the human THINGS-fMRI side, using the already-saved seed-0
weights — NO retraining. Loads each rule's seed-0 checkpoint, evaluates V1/V2/
LOC/IT alignment across a range of THINGS input resolutions, and shows that the
rule ranking is a tunable function of eval resolution (peak expected near the
32px training resolution).

Runs locally in minutes (tiny net, 720 stimuli, eval-only). No Brain-Score needed
— only the local human fMRI RDMs + THINGS images.

Run from the Cross_Species_RSA project root (same place as run_pipeline.py):
    python run_resolution_sweep_human.py

Output:
    results/rsa_resolution_sweep_human_bnfix.csv   (rule, roi, layer, res, rho, ci_lo, ci_hi, n_stim)
    + printed curve (V1 rho vs resolution per rule, and the Random-BP gap by resolution).
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_loader import load_things_fmri
from models_v2 import load_model, extract_features   # v2 = BatchNorm-mode fix
from rsa_engine import build_rdm, compare_rdms_bootstrap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("sweep-human")

# ── Config ────────────────────────────────────────────────────────────────
RULES      = ["random", "backprop", "feedback_alignment", "predictive_coding", "stdp"]
SWEEP_RES  = [32, 64, 96, 128, 160, 224]
N_BOOT     = 500
# Paper-1 fixed layer -> ROI mapping (V1/V2 from Conv1, LOC from Conv3, IT from FC1)
ROI_LAYER  = {"V1": "Conv1", "V2": "Conv1", "LOC": "Conv3", "IT": "FC1"}

def _resolve(cands):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return Path(cands[0])

# fMRI RDMs (outputs_720) + THINGS object images + Paper-1 weights — local defaults.
# The v1 paths pointed at "Projekte/" (no _1), which does not exist here, so _resolve
# fell through to a non-existent first candidate.
FMRI_DIR = _resolve([
    r"C:/Users/nilsl/Desktop/Projekte_1/learning-rules-rsa/outputs_720",
    "/kaggle/input/datasets/nilsleutenegger/learning-rules-fmri/outputs_720",
])
THINGS_DIR = _resolve([
    r"C:/Users/nilsl/Desktop/Projekte_1/RSA/Datensatz/images_THINGS/object_images",
    "/kaggle/input/datasets/nilsleutenegger/things-object-images/images_THINGS/object_images",
])
# BN-fixed re-run checkpoints. NOT outputs/, whose predictive_coding and stdp weights
# are pre-fix artifacts.
WEIGHTS_DIR = _resolve([
    r"C:/Users/nilsl/Desktop/Projekte_1/learning-rules-rsa/learning_rules_outputs_bnfix/checkpoints",
    "/kaggle/input/learning-rules-weights/outputs_bnfix/checkpoints",
])
OUT = Path("results"); OUT.mkdir(parents=True, exist_ok=True)

_WFILES = {
    "random": "model_weights_random_weights.pt", "backprop": "model_weights_backprop.pt",
    "feedback_alignment": "model_weights_feedback_alignment.pt",
    "predictive_coding": "model_weights_predictive_coding.pt", "stdp": "model_weights_stdp.pt",
}


def _check_weights():
    missing = []
    for r in RULES:
        f = WEIGHTS_DIR / _WFILES[r]
        ok = f.exists() or (r == "stdp" and (WEIGHTS_DIR / _WFILES[r].replace(".pt", "_seed0.pt")).exists())
        if not ok:
            missing.append(r)
    if missing:
        log.error(f"Missing weights for {missing} in {WEIGHTS_DIR} -> load_model would use random init. Abort.")
        sys.exit(1)
    log.info(f"Weights OK in {WEIGHTS_DIR}")


def _resolve_things_paths():
    order_file = FMRI_DIR / "stim_order_sub-01.txt"
    with open(order_file, encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    img_map = {p.name: str(p) for p in THINGS_DIR.rglob("*.jpg")}
    paths = [img_map[n] for n in names if n in img_map]
    log.info(f"Resolved {len(paths)}/{len(names)} THINGS stimuli")
    if len(paths) < 0.9 * len(names):
        log.warning("Many THINGS stimuli unresolved — check object_images layout")
    return paths


def _tf(px):
    return transforms.Compose([
        transforms.Resize(px), transforms.CenterCrop(px), transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])


def main():
    _check_weights()
    paths = _resolve_things_paths()

    fmri = load_things_fmri(str(FMRI_DIR))                 # {ROI: {"rdm": ...}}
    roi_rdm = {roi: fmri[roi]["rdm"] for roi in ROI_LAYER if roi in fmri}
    log.info(f"fMRI RDMs available: {list(roi_rdm)}")

    rows = []
    for rule in RULES:
        log.info(f"\n--- {rule} ---")
        model = load_model(rule, weights_dir=WEIGHTS_DIR)
        for res_px in SWEEP_RES:
            feats = extract_features(model, paths, _tf(res_px))
            for roi, layer in ROI_LAYER.items():
                if roi not in roi_rdm:
                    continue
                mdl_rdm = build_rdm(feats[layer])
                neu = roi_rdm[roi]
                n = min(mdl_rdm.shape[0], neu.shape[0])
                res = compare_rdms_bootstrap(mdl_rdm[:n, :n], neu[:n, :n], n_bootstrap=N_BOOT)
                lo, hi = res.get("ci_lower"), res.get("ci_upper")
                rows.append({"rule": rule, "roi": roi, "layer": layer, "res": int(res_px),
                             "rho": round(float(res["rho"]), 6),
                             "ci_lo": round(float(lo), 6) if lo is not None else None,
                             "ci_hi": round(float(hi), 6) if hi is not None else None,
                             "n_stim": int(n)})
            v1 = next((r for r in rows if r["rule"] == rule and r["roi"] == "V1" and r["res"] == res_px), None)
            log.info(f"    @{res_px:>3}px  V1 rho={v1['rho']:.4f}" if v1 else f"    @{res_px:>3}px done")
        del model

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rsa_resolution_sweep_human_bnfix.csv", index=False)
    log.info(f"\nSaved {len(df)} rows -> {OUT / 'rsa_resolution_sweep_human_bnfix.csv'}")

    # ── Curve summary ──
    v1 = df[df.roi == "V1"]
    tab = v1.pivot_table(index="res", columns="rule", values="rho")
    cols = [c for c in ["random", "stdp", "predictive_coding", "backprop", "feedback_alignment"] if c in tab.columns]
    log.info("\nV1 alignment by eval resolution (seed0):\n" + tab[cols].round(4).to_string())
    log.info("\nRandom - Backprop gap by resolution:")
    for res_px in SWEEP_RES:
        m = v1[v1.res == res_px].set_index("rule")["rho"]
        if {"random", "backprop"} <= set(m.index):
            log.info(f"  @{res_px:>3}px:  Rand-BP = {m['random'] - m['backprop']:+.4f}")
    # peak-alignment resolution per rule
    log.info("\nPeak V1-alignment resolution per rule:")
    for rule in cols:
        s = v1[v1.rule == rule]
        if not s.empty:
            peak = s.loc[s.rho.idxmax()]
            log.info(f"  {rule:<20} peak @ {int(peak.res)}px (rho={peak.rho:.4f})")


if __name__ == "__main__":
    main()
