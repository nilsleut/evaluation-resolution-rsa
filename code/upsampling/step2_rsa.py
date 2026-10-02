"""Step 2b: RSA point estimates for every (rule, seed, arm, res, layer->ROI), plus the
validation and the sanity gate.

Variants: cr  = pair set P (cross-run in every subject; PRIMARY pair set)
          all = all 258,840 pairs (only to validate against the v12 CSV)
Conventions: persub  = mean over subjects of rho(model, subject RDM)   (primary)
             meanrdm = rho(model, mean of the three subject RDMs)      (secondary)

Validation (reported):
  * NATIVE/all vs the v12 sweep CSV (same checkpoints, GPU vs CPU): rho (meanrdm) and
    rho_sub_mean (persub), all 1,080 rows.
  * UPSAMPLED/all vs results/rsa_resolution_sweep_upsample.csv (v11, retrained models;
    the four deterministic rules should agree, BP/FA only to ~1e-3).
Sanity gate (STOP on failure): NATIVE, cr, meanrdm, Random - Backprop at Conv1->V1,
mean over seeds: @224 within 0.005 of +0.043, @32 within 0.01 of 0.

Output: results/upsampling/rsa_seed{n}.csv (one per seed), results/upsampling/step2.json
"""
import json
import sys

import numpy as np
import pandas as pd

from common_up import (REPO, EXT, RESULTS, V12_CSV, ARMS, RES, RULES, SEEDS, LAYER_ROI, SUBJECTS,
                       ROIS, run_labels, pair_set, ranks, load_fmri_tri, rdm_path)

SANITY = {224: (0.043, 0.005), 32: (0.0, 0.01)}


def brain_ranks(keep):
    out = {}
    for roi in ROIS:
        F = load_fmri_tri(roi)
        for var, k in (("all", slice(None)), ("cr", keep)):
            vecs = [f[k] for f in F]
            out[(roi, var)] = ([ranks(v) for v in vecs], ranks(np.mean(vecs, axis=0)))
    return out


def main():
    _, _, keep = pair_set(None, run_labels())
    B = brain_ranks(keep)
    rows = []
    for arm in ARMS:
        for px in RES:
            for si, seed in enumerate(SEEDS):
                for rule in RULES:
                    cache = {}
                    for layer, roi in LAYER_ROI:
                        if layer not in cache:
                            x = np.load(rdm_path(arm, px, si, rule, layer))
                            cache[layer] = {"all": ranks(x), "cr": ranks(x[keep])}
                        for var in ("cr", "all"):
                            mv = cache[layer][var]
                            sub, mean = B[(roi, var)]
                            ps = [float(mv @ s) for s in sub]
                            base = dict(rule=rule, seed=seed, seed_idx=si, arm=arm, res=px,
                                        layer=layer, roi=roi, variant=var)
                            rows.append({**base, "convention": "persub", "rho": float(np.mean(ps)),
                                         **{f"rho_{s}": v for s, v in zip(SUBJECTS, ps)}})
                            rows.append({**base, "convention": "meanrdm", "rho": float(mv @ mean)})
        print(f"{arm} done", flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for si in range(len(SEEDS)):
        df[df.seed_idx == si].to_csv(RESULTS / f"rsa_seed{si}.csv", index=False)

    # ── validation vs v12 CSV (NATIVE, all pairs) ──
    v12 = pd.read_csv(V12_CSV)
    a = df[(df.arm == "NATIVE") & (df.variant == "all")]
    a = a.pivot_table(index=["rule", "seed_idx", "res", "layer", "roi"], columns="convention",
                      values="rho").reset_index()
    j = a.merge(v12, on=["rule", "seed_idx", "res", "layer", "roi"])
    j["d_mean"] = (j.meanrdm - j.rho).abs()
    j["d_sub"] = (j.persub - j.rho_sub_mean).abs()
    val12 = {"n_rows": int(len(j)),
             "max_abs_dev_meanrdm": float(j.d_mean.max()), "max_abs_dev_persub": float(j.d_sub.max()),
             "per_rule_max_meanrdm": j.groupby("rule").d_mean.max().to_dict()}

    # ── cross-check vs v11 upsample CSV (UPSAMPLED, all pairs) ──
    v11 = pd.read_csv(REPO / "results" / "rsa_resolution_sweep_upsample.csv")
    u = df[(df.variant == "all") & (df.convention == "meanrdm")]
    k = u.merge(v11, on=["rule", "seed_idx", "res", "layer", "roi", "arm"], suffixes=("", "_v11"))
    k["d"] = (k.rho - k.rho_v11).abs()
    val11 = {"n_rows": int(len(k)),
             "per_rule_arm_max": {f"{r}|{ar}": float(v) for (r, ar), v in
                                  k.groupby(["rule", "arm"]).d.max().items()}}

    # ── sanity gate ──
    v = df[(df.arm == "NATIVE") & (df.variant == "cr") & (df.convention == "meanrdm") &
           (df.layer == "Conv1") & (df.roi == "V1")]
    p = v.pivot_table(index=["res", "seed_idx"], columns="rule", values="rho")
    gap = (p["Random Weights"] - p["Backprop"]).groupby("res").mean()
    sanity = {int(r): {"gap": float(gap[r]), "target": t, "tol": tol,
                       "pass": bool(abs(gap[r] - t) <= tol)} for r, (t, tol) in SANITY.items()}
    ref = pd.read_csv(EXT / "results/crossrun/step2_effects.csv")
    ref = ref[(ref.effect == "E1 Random-BP V1") & (ref.variant == "cr") & (ref.convention == "meanrdm")]
    out = {"validation_v12_csv": val12, "crosscheck_v11_upsample_csv": val11, "sanity": sanity,
           "native_cr_meanrdm_gap_by_res": {int(r): float(g) for r, g in gap.items()},
           "crossrun_reference_bnfix": {int(r.res): float(r["mean"]) for _, r in ref.iterrows()}}
    (RESULTS / "step2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    if not all(s["pass"] for s in sanity.values()):
        print("SANITY GATE FAILED -> STOP"); sys.exit(2)
    if val12["max_abs_dev_meanrdm"] > 1e-3 or val12["max_abs_dev_persub"] > 1e-3:
        print("NATIVE does not reproduce the v12 CSV -> STOP"); sys.exit(3)
    print("SANITY GATE PASSED")


if __name__ == "__main__":
    main()
