#!/usr/bin/env python3
"""
run_resolution_control_macaque.py
=================================
32px RESOLUTION CONTROL for the macaque side of the cross-species study (Paper 2).

Question: do the macaque learning-rule effects (esp. STDP/PC > BP at V1/V2) survive
when the CIFAR-trained models are evaluated at their native 32x32 resolution instead
of 224x224? Neural RDMs are fixed; only the model INPUT resolution changes.

This mirrors run_pipeline.py (download -> extract -> rsa) for macaque only, looped over
[224, 32]. It imports the existing src/ modules and changes nothing in the pipeline.

Run from the project root (same place as run_pipeline.py):
    python run_resolution_control_macaque.py

Requires: brainscore-vision + internet/S3 (same as the normal pipeline) AND the
Paper-1 weight checkpoints (model_weights_*.pt) reachable via WEIGHTS_DIR below.

Output:
    results/rsa_resolution_control_macaque_bnfix.csv   (rule, layer, region, res, rho, ci_lo, ci_hi, n_stim)
    + a printed summary (rho@224 vs rho@32 per rule/region, and the STDP/PC-vs-BP gap).
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torchvision.transforms as transforms

# make src/ importable exactly like run_pipeline.py does
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_majajhong2015, load_freemanziemba2013, build_neural_rdm
from models_v2 import (                              # v2 = BatchNorm-mode fix
    load_model, extract_features, THINGS_TRANSFORM,
    LAYER_ROI_FREEMANZIEMBA, LAYER_ROI_MAJAJHONG,
)
from rsa_engine import build_rdm, compare_rdms_bootstrap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("res-control")

# ── Config ────────────────────────────────────────────────────────────────
RULES   = ["random", "backprop", "feedback_alignment", "predictive_coding", "stdp"]
ACCESS  = "public"
N_BOOT  = 1000

# Paper-1 weight checkpoints (model_weights_*.pt). Auto-resolve local vs Kaggle.
def _resolve_weights_dir():
    # BN-fixed re-run checkpoints. NOT outputs/, whose predictive_coding and stdp
    # weights are pre-fix artifacts. (v1's local candidate also said "Projekte/"
    # without the _1 and did not exist on this machine.)
    cands = [
        r"C:/Users/nilsl/Desktop/Projekte_1/learning-rules-rsa/learning_rules_outputs_bnfix/checkpoints",
        "/kaggle/input/learning-rules-weights/outputs_bnfix/checkpoints",
    ]
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return Path(cands[0])

WEIGHTS_DIR = _resolve_weights_dir()

# Weight filenames (mirror models._WEIGHT_FILES); STDP may also use the *_seed0.pt variant
_WFILES = {
    "random":             "model_weights_random_weights.pt",
    "backprop":           "model_weights_backprop.pt",
    "feedback_alignment": "model_weights_feedback_alignment.pt",
    "predictive_coding":  "model_weights_predictive_coding.pt",
    "stdp":               "model_weights_stdp.pt",
}

# 224 = the paper's transform; 32 = same CIFAR-norm, native training resolution
TRANSFORM_32 = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])
RES_VARIANTS = [(224, THINGS_TRANSFORM), (32, TRANSFORM_32)]

OUT = Path("results")
OUT.mkdir(parents=True, exist_ok=True)


def _check_weights():
    """Abort early if any trained checkpoint is missing (load_model would silently
    fall back to random init -> garbage 'trained' models)."""
    missing = []
    for rule in RULES:
        f = WEIGHTS_DIR / _WFILES[rule]
        ok = f.exists()
        if rule == "stdp":  # accept the full-retrain seed0 variant too
            ok = ok or (WEIGHTS_DIR / _WFILES[rule].replace(".pt", "_seed0.pt")).exists()
        if not ok:
            missing.append(rule)
    if missing:
        log.error(f"Missing weight files for {missing} in {WEIGHTS_DIR}")
        log.error("load_model would use RANDOM init for these -> results would be invalid.")
        log.error("Point WEIGHTS_DIR to the Paper-1 checkpoints (model_weights_*.pt) and retry.")
        sys.exit(1)
    log.info(f"Weights OK in {WEIGHTS_DIR}")


def _rsa_row(rule, layer, region, res_px, model_rdm, neural_rdm):
    n = min(model_rdm.shape[0], neural_rdm.shape[0])
    res = compare_rdms_bootstrap(model_rdm[:n, :n], neural_rdm[:n, :n], n_bootstrap=N_BOOT)
    lo, hi = res.get("ci_lower"), res.get("ci_upper")
    return {
        "rule": rule, "layer": layer, "region": region, "res": int(res_px),
        "rho": round(float(res["rho"]), 6),
        "ci_lo": round(float(lo), 6) if lo is not None else None,
        "ci_hi": round(float(hi), 6) if hi is not None else None,
        "n_stim": int(n),
    }


def _summary(df):
    log.info("\n" + "=" * 60)
    log.info("MACAQUE 32px RESOLUTION CONTROL — summary")
    log.info("=" * 60)
    for region in ["V1", "V2", "V4", "IT"]:
        sub = df[df["region"] == region]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="rule", columns="res", values="rho")
        log.info(f"\n{region}:   (rho@224  ->  rho@32   |   delta)")
        for rule in ["random", "stdp", "predictive_coding", "backprop", "feedback_alignment"]:
            if rule in piv.index:
                r224 = piv.loc[rule].get(224, float("nan"))
                r32 = piv.loc[rule].get(32, float("nan"))
                log.info(f"  {rule:<20} {r224:+.4f}  -> {r32:+.4f}   | {r32 - r224:+.4f}")
        for res in (224, 32):
            m = sub[sub["res"] == res].set_index("rule")["rho"]
            if {"stdp", "backprop"} <= set(m.index):
                extra = (f"  PC-BP={m['predictive_coding'] - m['backprop']:+.4f}"
                         if "predictive_coding" in m.index else "")
                log.info(f"    @{res}px gap:  STDP-BP={m['stdp'] - m['backprop']:+.4f}{extra}")
    log.info("\n  Erwartung: schrumpft STDP-BP / PC-BP bei 32px Richtung 0, ist der")
    log.info("  Makaken-Befund derselbe Resolution-Artefakt wie auf der Human-Seite.")


def main():
    _check_weights()

    # ── Macaque neural data (fixed across resolutions) ──
    log.info("Loading MajajHong2015 (V4/IT) via Brain-Score...")
    mh = load_majajhong2015(access=ACCESS)
    mh_stim = mh["stimulus_set"]
    mh_paths = [mh_stim.get_stimulus(i) for i in mh_stim["image_id"]]
    mh_rdm = {r: build_neural_rdm(mh[r]["responses"]) for r in ["V4", "IT"] if r in mh}
    log.info(f"  MajajHong: {len(mh_paths)} stimuli; regions {list(mh_rdm)}")

    log.info("Loading FreemanZiemba2013 (V1/V2) via Brain-Score...")
    import brainscore_vision
    fz = load_freemanziemba2013(access=ACCESS)
    fz_stim = brainscore_vision.load_stimulus_set(f"FreemanZiemba2013.aperture-{ACCESS}")
    fz_paths = [fz_stim.get_stimulus(i) for i in fz_stim["image_id"]]
    fz_rdm = {r: build_neural_rdm(fz[r]["responses"]) for r in ["V1", "V2"] if r in fz}
    log.info(f"  FreemanZiemba: {len(fz_paths)} stimuli; regions {list(fz_rdm)}")

    rows = []
    for rule in RULES:
        log.info(f"\n--- {rule} ---")
        model = load_model(rule, weights_dir=WEIGHTS_DIR)

        for res_px, tf in RES_VARIANTS:
            # MajajHong: Conv2 -> V4, FC1 -> IT
            mh_feat = extract_features(model, mh_paths, tf)
            for layer, regions in LAYER_ROI_MAJAJHONG.items():
                for region in regions:
                    if region in mh_rdm:
                        rows.append(_rsa_row(rule, layer, region, res_px,
                                             build_rdm(mh_feat[layer]), mh_rdm[region]))

            # FreemanZiemba: Conv1 -> V1, V2
            fz_feat = extract_features(model, fz_paths, tf)
            for layer, regions in LAYER_ROI_FREEMANZIEMBA.items():
                for region in regions:
                    if region in fz_rdm:
                        rows.append(_rsa_row(rule, layer, region, res_px,
                                             build_rdm(fz_feat[layer]), fz_rdm[region]))

            v1 = next((r for r in rows if r["rule"] == rule and r["region"] == "V1"
                       and r["res"] == res_px), None)
            msg = f"    @{res_px:>3}px done" + (f"  V1 rho={v1['rho']:.4f}" if v1 else "")
            log.info(msg)

        del model

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rsa_resolution_control_macaque_bnfix.csv", index=False)
    log.info(f"\nSaved {len(df)} rows -> {OUT / 'rsa_resolution_control_macaque_bnfix.csv'}")
    _summary(df)


if __name__ == "__main__":
    main()
