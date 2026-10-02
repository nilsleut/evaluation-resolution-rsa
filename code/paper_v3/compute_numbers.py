"""2608.12408 v3: every brain-dependent number of the paper, recomputed in the primary
convention of v3 (per subject, cross-run pairs), next to the v2 convention (mean RDM, all
pairs) used to validate each computation against the printed v2 value.

Model sets
  sweep / best layer / LOC / IT / content control : v12 checkpoints (the 30-checkpoint
      repaired retrain; data/upsampling/rdms/{NATIVE,UPSAMPLED}). Validation in the v2
      convention also runs on the published bnfix RDMs (learning-rules-rsa), which the v2
      text was computed from.
  BN calibration : outputs_bncal_rdms (Modal; data/bncal_rdms), Conv1.
  ResNet-50 / Swin-Tiny : data/arch_rdms (extract_arch.py, mirrors the v2 scripts).
  training dynamics : training-dynamics volume, outputs_bnfix (data/td_rdms), Conv1, epochs 0/40.
  low-level references : rebuilt with colour_statistic_control.py / run_lowlevel_control_v2.py.
Model-vs-reference similarities involve no brain data and keep all pairs (unchanged).

Statistic as in v2: mean over the 5 seeds, SEM across seeds, seeds positive; the V1
headline additionally carries the stimulus-bootstrap CI (results/upsampling/step4_gaps.csv).

Output: results/paper_v3/numbers_v3.csv  (key, section, quantity, v2_printed,
        v2_recomputed_bnfix, v2_recomputed_v12, v3_value, v3_seeds_pos, source)
"""
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE / "upsampling"))
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "cross_species"))
from common_up import (REPO, EXT, FMRI_DIR, THINGS_DIR, TRI, RES, SUBJECTS, ranks, run_labels,  # noqa: E402
                       pair_set, load_fmri_tri, rdm_path)

BNFIX = EXT / "learning_rules_outputs_bnfix" / "rdms"
BNCAL = REPO / "data" / "bncal_rdms" / "rdms"
TD = REPO / "data" / "td_rdms"
ARCH = REPO / "data" / "arch_rdms"
OUT = REPO / "results" / "paper_v3"
SEEDS = range(5)
RK = {"rnd": "Random Weights", "bp": "Backprop", "fa": "Feedback Alignment", "pc": "Predictive Coding",
      "stdp": "STDP", "cal": "Random Weights (BN-calibrated)"}

# ── brain side ──────────────────────────────────────────────────────────────
_, _, KEEP = pair_set(None, run_labels())
_BR = {}


def brain(roi, var):
    if (roi, var) not in _BR:
        F = load_fmri_tri(roi)
        k = KEEP if var == "cr" else slice(None)
        vecs = [f[k] for f in F]
        _BR[(roi, var)] = ([ranks(v) for v in vecs], ranks(np.mean(vecs, axis=0)), vecs)
    return _BR[(roi, var)]


def tri(x):
    x = np.asarray(x)
    return x[TRI].astype(np.float64) if x.ndim == 2 else x.astype(np.float64)


def score(m, roi, conv="persub", var="cr"):
    m = tri(m)
    mv = ranks(m[KEEP] if var == "cr" else m)
    sub, mean, _ = brain(roi, var)
    return float(np.mean([mv @ s for s in sub])) if conv == "persub" else float(mv @ mean)


OLD, NEW = ("meanrdm", "all"), ("persub", "cr")


# ── model RDM loaders ───────────────────────────────────────────────────────
def v12(rule, px, si, layer="Conv1", arm="NATIVE"):
    return np.load(rdm_path(arm, px, si, rule, layer))


def bnfix(rule, px, si, layer="Conv1"):
    return np.load(BNFIX / f"res{px}" / f"seed_{si}" / f"rdm_{rule.lower().replace(' ', '_')}_{layer}.npy")


def bncal(v, px, si):
    return np.load(BNCAL / f"res{px}" / f"seed_{si}" / f"rdm_{v}_Conv1.npy").astype(np.float64)


def td(rule, px, si, ep):
    return np.load(TD / f"res{px}" / f"{rule}_seed{si}_epoch{ep}" / f"rdm_{rule}_Conv1.npy")


# ── statistics ──────────────────────────────────────────────────────────────
def seedstat(vals):
    v = np.asarray(vals, float)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), int((v > 0).sum()), int((v < 0).sum())


ROWS = []


def add(key, section, quantity, printed, old_bnfix, old_v12, new, src, new_pos=None, note=""):
    ROWS.append(dict(key=key, section=section, quantity=quantity, v2_printed=printed,
                     v2_recomputed_bnfix=old_bnfix, v2_recomputed_v12=old_v12, v3_value=new,
                     v3_seeds_pos=new_pos, source=src, note=note))


def per_seed(fn, loader, conv_var, **kw):
    return [fn(loader, si, conv_var, **kw) for si in SEEDS]


# ── 3.1 sweep ───────────────────────────────────────────────────────────────
def sweep():
    for rk in ("bp", "fa", "pc", "stdp", "rnd"):
        for px in (32, 224):
            o = np.mean([score(bnfix(RK[rk], px, s), "V1", *OLD) for s in SEEDS])
            o12 = np.mean([score(v12(RK[rk], px, s), "V1", *OLD) for s in SEEDS])
            n = np.mean([score(v12(RK[rk], px, s), "V1", *NEW) for s in SEEDS])
            add(f"rho.{rk}.V1.{px}", "3.1", f"rho {RK[rk]} Conv1->V1 @{px}", None, o, o12, n, "v12 NATIVE")
    for px in RES:
        for roi, layer, a, b in (("V1", "Conv1", "rnd", "bp"), ("LOC", "Conv3", "bp", "rnd"),
                                 ("IT", "FC1", "bp", "rnd")):
            def g(load, conv, var):
                return [score(load(RK[a], px, s, layer), roi, conv, var) - score(load(RK[b], px, s, layer), roi, conv, var)
                        for s in SEEDS]
            mo, so, po, _ = seedstat(g(bnfix, *OLD))
            m12, _, _, _ = seedstat(g(v12, *OLD))
            mn, sn, pn, _ = seedstat(g(v12, *NEW))
            add(f"gap.{roi}.{px}", "3.1/3.7", f"{RK[a]}-{RK[b]} {roi} @{px}: mean", None, mo, m12, mn, "v12 NATIVE", pn)
            add(f"gap.{roi}.{px}.sem", "3.1/3.7", f"... SEM", None, so, None, sn, "v12 NATIVE")
            add(f"gap.{roi}.{px}.pos", "3.1/3.7", f"... seeds positive", None, po, None, pn, "v12 NATIVE")
    for roi, layer in (("LOC", "Conv3"),):
        for rk in ("bp", "rnd"):
            for px in (32, 224):
                o = np.mean([score(bnfix(RK[rk], px, s, layer), roi, *OLD) for s in SEEDS])
                n = np.mean([score(v12(RK[rk], px, s, layer), roi, *NEW) for s in SEEDS])
                add(f"rho.{rk}.LOC.{px}", "3.7", f"rho {RK[rk]} Conv3->LOC @{px}", None, o, None, n, "v12 NATIVE")
    # untrained LOC range over the sweep
    for conv_var, tag in ((OLD, "old"), (NEW, "new")):
        load = bnfix if tag == "old" else v12
        vals = [np.mean([score(load(RK["rnd"], px, s, "Conv3"), "LOC", *conv_var) for s in SEEDS]) for px in RES]
        add(f"rnd.LOC.range.{tag}", "3.7", f"untrained LOC min..max over sweep ({tag})", None,
            (min(vals), max(vals)) if tag == "old" else None, None, (min(vals), max(vals)) if tag == "new" else None,
            "v12 NATIVE")


def best_layer():
    layers = ["Conv1", "Conv2"]          # the layers mapped to V1 (six layer-ROI pairs), as in v2 App. C
    for px in (32, 96, 224):
        for tag, conv_var, load in (("old", OLD, bnfix), ("new", NEW, v12)):
            best = {}
            for rk in ("rnd", "bp", "fa", "pc", "stdp"):
                m = {l: np.mean([score(load(RK[rk], px, s, l), "V1", *conv_var) for s in SEEDS]) for l in layers}
                best[rk] = max(m, key=m.get)
                add(f"best.{rk}.{px}.{tag}", "App C", f"best V1 layer {RK[rk]} @{px} ({tag})", None,
                    best[rk] if tag == "old" else None, None, best[rk] if tag == "new" else None, "v12 NATIVE")
            d = [score(load(RK["rnd"], px, s, best["rnd"]), "V1", *conv_var)
                 - score(load(RK["bp"], px, s, best["bp"]), "V1", *conv_var) for s in SEEDS]
            m, se, pos, _ = seedstat(d)
            add(f"bestgap.{px}.{tag}", "App C", f"best-layer gap @{px} ({tag}) mean/SEM/pos", None,
                (m, se, pos) if tag == "old" else None, None, (m, se, pos) if tag == "new" else None, "v12 NATIVE")


# ── 3.2 BN calibration ──────────────────────────────────────────────────────
def bncal_section():
    V = ["random", "a", "b", "c", "d"]
    for tag, conv_var, bpload in (("old", OLD, bnfix), ("new", NEW, v12)):
        S = {(v, px, s): score(bncal(v, px, s), "V1", *conv_var) for v in V for px in (32, 224) for s in SEEDS}
        BP = {(px, s): score(bpload(RK["bp"], px, s), "V1", *conv_var) for px in (32, 224) for s in SEEDS}
        val = lambda q: q if tag == "old" else None  # noqa: E731
        nv = lambda q: q if tag == "new" else None   # noqa: E731

        def put(key, q, d):
            m, se, pos, _ = seedstat(d)
            add(f"{key}.{tag}", "3.2", q + f" ({tag})", None, val((m, se, pos)), None, nv((m, se, pos)), "bncal_rdms")
        for v in ("a", "b", "c", "d"):
            put(f"cal.{v}-random.224", f"{v} - identity @224", [S[(v, 224, s)] - S[("random", 224, s)] for s in SEEDS])
            put(f"cal.{v}-bp.224", f"{v} - BP @224", [S[(v, 224, s)] - BP[(224, s)] for s in SEEDS])
            put(f"cal.{v}-bp.32", f"{v} - BP @32", [S[(v, 32, s)] - BP[(32, s)] for s in SEEDS])
        put("cal.B-A.224", "B - A (match resolution, CIFAR)", [S[("b", 224, s)] - S[("a", 224, s)] for s in SEEDS])
        put("cal.D-C.224", "D - C (match resolution, THINGS)", [S[("d", 224, s)] - S[("c", 224, s)] for s in SEEDS])
        put("cal.D-B.224", "D - B (set at matched res)", [S[("d", 224, s)] - S[("b", 224, s)] for s in SEEDS])
        put("cal.random-bp.224", "identity - BP @224", [S[("random", 224, s)] - BP[(224, s)] for s in SEEDS])
        put("cal.random-bp.32", "identity - BP @32", [S[("random", 32, s)] - BP[(32, s)] for s in SEEDS])


# ── 3.3 ResNet-50 / Swin-Tiny ───────────────────────────────────────────────
def arch():
    for model, layer in (("resnet50", "layer1"), ("swin_t", "early")):
        f = ARCH / model / "res32" / f"{layer}.npy"
        if not f.exists():
            add(f"arch.{model}", "3.3", f"{model} missing", None, None, None, None, "arch_rdms", note="NOT COMPUTED")
            continue
        for tag, conv_var in (("old", OLD), ("new", NEW)):
            vals = {px: score(np.load(ARCH / model / f"res{px}" / f"{layer}.npy"), "V1", *conv_var) for px in RES}
            add(f"arch.{model}.V1.{tag}", "3.3", f"{model} V1 by res ({tag})", None,
                vals if tag == "old" else None, None, vals if tag == "new" else None, "arch_rdms")


# ── 3.5 content control (v12 UPSAMPLED vs NATIVE) ───────────────────────────
def content():
    for tag, conv_var in (("old", OLD), ("new", NEW)):
        R = {(arm, rk, px, s): score(v12(RK[rk], px, s, "Conv1", arm), "V1", *conv_var)
             for arm in ("NATIVE", "UPSAMPLED") for rk in ("rnd", "bp", "fa", "pc") for px in (32, 64, 224) for s in SEEDS}
        val = lambda q: q if tag == "old" else None  # noqa: E731
        nv = lambda q: q if tag == "new" else None   # noqa: E731

        def put(key, q, d, fmt=None):
            m, se, pos, neg = seedstat(d)
            add(f"{key}.{tag}", "3.5", q + f" ({tag})", None, val((m, se, pos)), None, nv((m, se, pos)), "v12 NATIVE/UPSAMPLED")
        for rk in ("bp", "rnd"):
            put(f"ups.step.{rk}", f"{rk} UP@64 - NATIVE@64 (first-step penalty)",
                [R[("UPSAMPLED", rk, 64, s)] - R[("NATIVE", rk, 64, s)] for s in SEEDS])
        for arm in ("NATIVE", "UPSAMPLED"):
            put(f"ups.gapopen.{arm}", f"gap(224)-gap(64) {arm}",
                [(R[(arm, "rnd", 224, s)] - R[(arm, "bp", 224, s)]) - (R[(arm, "rnd", 64, s)] - R[(arm, "bp", 64, s)]) for s in SEEDS])
            for rk in ("bp", "fa", "pc", "rnd"):
                put(f"ups.slope.{rk}.{arm}", f"{rk} @224 - @64 {arm}",
                    [R[(arm, rk, 224, s)] - R[(arm, rk, 64, s)] for s in SEEDS])
        d32 = max(abs(R[("NATIVE", rk, 32, s)] - R[("UPSAMPLED", rk, 32, s)]) for rk in ("rnd", "bp", "fa", "pc") for s in SEEDS)
        add(f"ups.arm32.maxdiff.{tag}", "3.5", "max|NATIVE-UPSAMPLED| @32", None, val(d32), None, nv(d32), "v12")


# ── 3.6 training dynamics (Conv1 -> V1, epoch 40 - epoch 0) ──────────────────
def dynamics():
    for rule in ("backprop", "feedback_alignment", "predictive_coding", "stdp"):
        for px in (224, 32):
            if not (TD / f"res{px}" / f"{rule}_seed4_epoch40" / f"rdm_{rule}_Conv1.npy").exists():
                continue
            for tag, conv_var in ((("old", ("persub", "all"))), ("new", NEW)):
                d = [score(td(rule, px, s, 40), "V1", *conv_var) - score(td(rule, px, s, 0), "V1", *conv_var) for s in SEEDS]
                m, se, pos, neg = seedstat(d)
                add(f"td.{rule}.{px}.{tag}", "3.6", f"{rule} epoch40-epoch0 @{px} ({tag}: persub, {'all' if tag=='old' else 'cr'})",
                    None, (m, se, neg) if tag == "old" else None, None, (m, se, neg) if tag == "new" else None,
                    "training-dynamics RDMs", note="v2 convention here is per subject, all pairs; tuple = mean, SEM, seeds negative")


def main():
    for f in (sweep, best_layer, bncal_section, arch, content, dynamics):
        f()
        print(f"{f.__name__} done", flush=True)
    df = pd.DataFrame(ROWS)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "numbers_v3_part1.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 500)
    print(df.drop(columns=["source", "note"]).to_string(index=False))


if __name__ == "__main__":
    main()
