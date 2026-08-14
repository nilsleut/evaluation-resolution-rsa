"""
colour_within_filter.py
=======================
COMPUTATION 1 -- the within-filter version of the pooling/global-statistic account.

The rank correlation currently supporting it (rho = 0.94, n = 6) is across the six
RULES of bnfix_sweep, whose convolutional filters all differ. It is therefore
observational: conditions that converge harder on luminance are also more V1-aligned,
but filters, training and normalisation all vary together.

The five conditions here share BIT-IDENTICAL convolutional weights (verified by
sha256 per seed) and identical pooling geometry. Only the BatchNorm statistics
differ. So if luminance convergence and V1 alignment still move together across
them, filters cannot be the confound.

B is the case that matters: the only calibration variant whose V1 alignment RISES
with resolution.

Note: bnfix_sweep's "Random Weights (BN-calibrated)" is the A_aug variant, retired in
3.2 as procedurally flawed, so its row in colour_statistic_control.csv is NOT used.

References, masking and rank procedure are reused unchanged from
colour_statistic_control.py.

CPU only, minutes.

Usage:
  python colour_within_filter.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import colour_statistic_control as C

ROOT     = Path(__file__).parent
RDM_DIR  = ROOT / "bncal_rdms" / "rdms"
ROWS_CSV = ROOT / "bncal_rdms" / "bn_calibration_control.csv"
REF_CSV  = ROOT / "bncal_rows.csv"           # the published factorial, for the equality check
OUT_CSV  = ROOT / "colour_within_filter.csv"

RES   = C.RES
SEEDS = C.SEEDS
REFS  = ["MEANRGB", "COLORHIST", "MEANLUM", "PIXEL"]

# variant -> (key used in the RDM filename, V1 slope already established)
VARIANTS = {
    "Random (identity BN)":  ("random",  +0.0110),
    "A CIFAR@32":            ("a",       -0.0122),
    "B CIFAR@eval-res":      ("b",       +0.0106),
    "C THINGS-disjoint@32":  ("c",       -0.0104),
    "D THINGS-disjoint@res": ("d",       -0.0023),
}


def main():
    stimuli = C.load_stim_order("sub-01")
    paths = [p for p in (C.find_img(s) for s in stimuli) if p is not None]
    print(f"THINGS: {len(paths)}/{len(stimuli)} images resolved")
    print("Building reference RDMs (identical procedure to colour_statistic_control) ...")
    refs, _, _ = C.build_references(paths)
    bmean, _ = C.brain_v1()

    rows = []
    for variant, (key, _) in VARIANTS.items():
        for px in RES:
            for si in SEEDS:
                f = RDM_DIR / f"res{px}" / f"seed_{si}" / f"rdm_{key}_Conv1.npy"
                if not f.exists():
                    print(f"  MISSING {f}"); continue
                M = np.load(str(f)).astype(np.float64)
                r = {"variant": variant, "res": px, "seed_idx": si,
                     "rho_v1": C.rsa(M, bmean)}
                for k in REFS:
                    r[f"rho_{k}"] = C.rsa(M, refs[k])
                rows.append(r)
        print(f"  done {variant}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n{len(df)} rows -> {OUT_CSV.name}")

    # ── the RDMs must be the ones behind the published factorial ─────────────
    pub = pd.read_csv(REF_CSV)
    pub = pub[(pub.layer == "Conv1") & (pub.roi == "V1")]
    j = df.merge(pub[["variant", "res", "seed_idx", "rho"]],
                 on=["variant", "res", "seed_idx"])
    print(f"VALIDATION  rho_v1 from saved RDMs vs bncal_rows.csv: "
          f"max|diff| = {(j.rho_v1 - j.rho).abs().max():.2e}  (n={len(j)})")

    g = df.groupby(["variant", "res"]).mean(numeric_only=True).reset_index()

    # ── per-variant 32 -> 224 change, per seed and mean ─────────────────────
    print("\n" + "=" * 100)
    print("CHANGE FROM 32 TO 224 px, per seed  (positive = converges further on the")
    print("reference as pooling grows from 256 to 12,544 positions)")
    print("=" * 100)
    for k in REFS:
        print(f"\n--- {k} ---")
        hdr = (f"{'variant':<24}" + "".join(f"{'s'+str(s):>9}" for s in SEEDS)
               + f"{'mean':>10}{'SEM':>9}{'n>0':>6}{'V1 slope':>11}")
        print(hdr); print("-" * len(hdr))
        for variant, (_, v1slope) in VARIANTS.items():
            d = df[df.variant == variant]
            per = []
            for s in SEEDS:
                a = d[(d.seed_idx == s) & (d.res == 32)][f"rho_{k}"]
                b = d[(d.seed_idx == s) & (d.res == 224)][f"rho_{k}"]
                per.append(float(b.iloc[0] - a.iloc[0]) if len(a) and len(b) else np.nan)
            per = np.array(per, dtype=float)
            print(f"{variant:<24}" + "".join(f"{v:>+9.4f}" for v in per)
                  + f"{per.mean():>+10.4f}{per.std(ddof=1)/np.sqrt(len(per)):>9.4f}"
                  + f"{int((per > 0).sum()):>6}{v1slope:>+11.4f}")

    # ── the five-point rank correlation ──────────────────────────────────────
    print("\n" + "=" * 100)
    print("FIVE-POINT RANK CORRELATION, within constant filters")
    print("  x = mean 32->224 change in similarity to the reference")
    print("  y = the established 32->224 change in V1 alignment")
    print("=" * 100)
    names = list(VARIANTS)
    y = np.array([VARIANTS[v][1] for v in names])
    for k in REFS:
        x = np.array([
            g[(g.variant == v) & (g.res == 224)][f"rho_{k}"].iloc[0]
            - g[(g.variant == v) & (g.res == 32)][f"rho_{k}"].iloc[0] for v in names])
        rho, p = spearmanr(x, y)
        pear = float(np.corrcoef(x, y)[0, 1])
        print(f"  {k:<12} Spearman rho = {rho:+.3f} (p={p:.3f}, n=5)   "
              f"Pearson r = {pear:+.3f}")
        for v, xv, yv in zip(names, x, y):
            print(f"      {v:<24} d_ref={xv:+.4f}   d_V1={yv:+.4f}")

    # ── absolute levels, for context ─────────────────────────────────────────
    print("\n" + "=" * 100)
    print("Absolute similarity to MEANLUM across the sweep (mean over 5 seeds)")
    print("=" * 100)
    hdr = f"{'variant':<24}" + "".join(f"{r:>10}" for r in RES) + f"{'224-32':>10}"
    print(hdr); print("-" * len(hdr))
    for variant in VARIANTS:
        d = g[g.variant == variant].sort_values("res")
        v = d.rho_MEANLUM.to_numpy()
        print(f"{variant:<24}" + "".join(f"{x:>10.4f}" for x in v) + f"{v[-1]-v[0]:>+10.4f}")


if __name__ == "__main__":
    main()
