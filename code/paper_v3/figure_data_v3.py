"""2608.12408 v3: data behind every figure, in the v3 convention.

Per cell (condition x resolution, or x epoch): mean over the 5 seeds of the per-subject
Spearman rho on cross-run stimulus pairs, with a 95% stimulus-bootstrap CI (1,000 resamples
of the noise_ceiling_v2 index matrix, seed 20261002; the cross-run pair set is recomputed in
every resample). Single checkpoints (ResNet-50, Swin-Tiny) have one RDM per cell.

Cells
  fig1/fig7  v12 NATIVE and UPSAMPLED, 5 conditions x 6 resolutions, Conv1 -> V1
  fig2       ResNet-50 layer1, Swin-Tiny early stage -> V1 (data/arch_rdms)
  fig3       training dynamics, 4 rules x 8 epochs x {32, 224} px, Conv1 -> V1 (data/td_rdms);
             the untrained baseline is Backprop at epoch 0 (identical initialization)
  fig4       v12 NATIVE, Random and Backprop, Conv3 -> LOC and FC1 -> IT
  fig6       BN-calibration RDMs, identity + A/B/C/D, Conv1 -> V1 (Backprop from fig1)
Fig. 5 (single seed, scatter) is computed in make_figures_v3.py without bootstrap.

Output: results/paper_v3/figure_data_v3.csv (figure, cell, condition, x, roi, mean, lo, hi, n_rdm);
        data/paper_v3/fig_boot.jsonl (resumable draws)
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "upsampling"))
from common_up import (REPO, RES, SEEDS, N_STIM, N_BOOT, N_PAIRS_ALL, TRI, run_labels, ranks,  # noqa: E402
                       tri_index, load_fmri_tri, rdm_path, boot_idx)

BNCAL = REPO / "data" / "bncal_rdms" / "rdms"
TD = REPO / "data" / "td_rdms"
ARCH = REPO / "data" / "arch_rdms"
CACHE = REPO / "data" / "paper_v3" / "fig_stack.npy"
PARTS = REPO / "data" / "paper_v3" / "fig_boot.jsonl"
OUT = REPO / "results" / "paper_v3" / "figure_data_v3.csv"
RULES5 = ["Random Weights", "Backprop", "Feedback Alignment", "Predictive Coding", "STDP"]
TDRULES = {"backprop": "Backprop", "feedback_alignment": "Feedback Alignment",
           "predictive_coding": "Predictive Coding", "stdp": "STDP"}
EPOCHS = [0, 1, 2, 5, 10, 20, 30, 40]
SI = range(len(SEEDS))


def cells():
    """(figure, cell id, condition, x, roi, [loader callables])"""
    C = []
    for arm, fig in (("NATIVE", "fig1"), ("UPSAMPLED", "fig7")):
        for r in RULES5:
            for px in RES:
                C.append((fig, f"{arm}|{r}|{px}", r, px, "V1",
                          [lambda s=s, r=r, px=px, arm=arm: np.load(rdm_path(arm, px, s, r, "Conv1")) for s in SI]))
    for m, layer in (("resnet50", "layer1"), ("swin_t", "early")):
        for px in RES:
            C.append(("fig2", f"{m}|{px}", m, px, "V1", [lambda m=m, px=px, layer=layer: np.load(ARCH / m / f"res{px}" / f"{layer}.npy")]))
    for key, r in TDRULES.items():
        for px in (224, 32):
            for ep in EPOCHS:
                C.append(("fig3", f"td|{r}|{px}|{ep}", r, ep, f"V1@{px}",
                          [lambda s=s, key=key, px=px, ep=ep: np.load(TD / f"res{px}" / f"{key}_seed{s}_epoch{ep}" / f"rdm_{key}_Conv1.npy")[TRI]
                           for s in SI]))
    for roi, layer in (("LOC", "Conv3"), ("IT", "FC1")):
        for r in ("Random Weights", "Backprop"):
            for px in RES:
                C.append(("fig4", f"{roi}|{r}|{px}", r, px, roi,
                          [lambda s=s, r=r, px=px, layer=layer: np.load(rdm_path("NATIVE", px, s, r, layer)) for s in SI]))
    for v in ("random", "a", "b", "c", "d"):
        for px in RES:
            C.append(("fig6", f"bncal|{v}|{px}", v, px, "V1",
                      [lambda s=s, v=v, px=px: np.load(BNCAL / f"res{px}" / f"seed_{s}" / f"rdm_{v}_Conv1.npy")[TRI].astype(np.float64)
                       for s in SI]))
    return C


CELLS = cells()
ITEMS = [(ci, j) for ci, c in enumerate(CELLS) for j in range(len(c[5]))]
_W = {}


def build_cache():
    if CACHE.exists() and np.load(CACHE, mmap_mode="r").shape == (len(ITEMS), N_PAIRS_ALL):
        return
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    M = np.lib.format.open_memmap(CACHE, mode="w+", dtype=np.float64, shape=(len(ITEMS), N_PAIRS_ALL))
    for k, (ci, j) in enumerate(ITEMS):
        M[k] = CELLS[ci][5][j]()
    M.flush()


def _init(boot):
    _W["boot"], _W["runs"] = boot, run_labels()
    _W["fmri"] = {roi: load_fmri_tri(roi) for roi in ("V1", "LOC", "IT")}
    _W["M"] = np.load(CACHE, mmap_mode="r")


def one(b):
    s = np.arange(N_STIM) if b < 0 else _W["boot"][b]
    a, c = s[TRI[0]], s[TRI[1]]
    keep = a != c
    for r in _W["runs"].values():
        keep &= r[a] != r[c]
    idx = tri_index(a[keep], c[keep])
    sub = {roi: [ranks(f[idx]) for f in F] for roi, F in _W["fmri"].items()}
    vals = {}
    for k, (ci, j) in enumerate(ITEMS):
        roi = CELLS[ci][4].split("@")[0]
        mv = ranks(np.asarray(_W["M"][k][idx], float))
        vals.setdefault(ci, []).append(float(np.mean([mv @ x for x in sub[roi]])))
    return {"boot": b, "cells": [float(np.mean(vals[ci])) for ci in range(len(CELLS))]}


def read_parts():
    done = {}
    if PARTS.exists():
        for line in PARTS.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[r["boot"]] = r
    return done


def main():
    t0 = time.time()
    boot = boot_idx()
    build_cache()
    print(f"{len(CELLS)} cells, {len(ITEMS)} RDMs; cache ready {time.time()-t0:.0f}s", flush=True)
    done = read_parts()
    todo = [b for b in range(-1, N_BOOT) if b not in done]
    if todo:
        with open(PARTS, "a") as fh, Pool(10, initializer=_init, initargs=(boot,)) as pool:
            for i, r in enumerate(pool.imap_unordered(one, todo, chunksize=2)):
                fh.write(json.dumps(r) + "\n"); fh.flush()
                if i % 100 == 0:
                    print(f"{i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    d = read_parts()
    D = np.array([d[b]["cells"] for b in range(-1, N_BOOT)])
    rows = []
    for ci, (fig, cid, cond, x, roi, ld) in enumerate(CELLS):
        rows.append(dict(figure=fig, cell=cid, condition=cond, x=x, roi=roi, mean=D[0, ci],
                         lo=np.quantile(D[1:, ci], .025), hi=np.quantile(D[1:, ci], .975), n_rdm=len(ld)))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"done {time.time()-t0:.0f}s -> {OUT.name}")


if __name__ == "__main__":
    main()
