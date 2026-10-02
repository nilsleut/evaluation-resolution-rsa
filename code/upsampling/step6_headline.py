"""Step 6: which pair set and convention gives the published V1 headline (+0.044 at 224 px)?

Random - BP at Conv1 -> V1, NATIVE arm, v12 checkpoints, mean over 5 seeds, for
  convention in {persub, meanrdm}  x  pair set in {all pairs, cross-run}
at every resolution. Cross-run rows are read from step4_gaps.csv (step3 draws). All-pairs
rows are bootstrapped here with the SAME stimulus resamples (noise_ceiling_v2 index matrix):
per resample, all pairs (s[i], s[j]), i < j, with s[i] != s[j].

Reference values: results/bnfix_sweep.csv (published, all pairs, mean-RDM) and the v12 CSV.

Output: results/upsampling/step6_headline.csv
"""
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

from common_up import (REPO, RESULTS, V12_CSV, RES, SEEDS, N_STIM, N_BOOT, TRI, ranks, tri_index,
                       load_fmri_tri, rdm_path, boot_idx)

RULES = ("Random Weights", "Backprop")
TQ = tdist.ppf(0.975, len(SEEDS) - 1)
_W = {}


def _init(boot):
    _W["boot"] = boot
    _W["F"] = load_fmri_tri("V1")
    _W["M"] = {(px, r, si): np.load(rdm_path("NATIVE", px, si, r, "Conv1"))
               for px in RES for r in RULES for si in range(len(SEEDS))}


def one(b):
    s = np.arange(N_STIM) if b < 0 else _W["boot"][b]
    a, c = s[TRI[0]], s[TRI[1]]
    k = a != c
    idx = tri_index(a[k], c[k])
    vecs = [f[idx] for f in _W["F"]]
    sub, mean = [ranks(v) for v in vecs], ranks(np.mean(vecs, axis=0))
    out = {"boot": b}
    for (px, r, si), x in _W["M"].items():
        mv = ranks(x[idx])
        out[f"persub|{px}|{r}|{si}"] = float(np.mean([mv @ y for y in sub]))
        out[f"meanrdm|{px}|{r}|{si}"] = float(mv @ mean)
    return out


def main():
    t0 = time.time()
    boot = boot_idx()
    with Pool(10, initializer=_init, initargs=(boot,)) as pool:
        D = pd.DataFrame(sorted(pool.map(one, range(-1, N_BOOT), chunksize=4), key=lambda r: r["boot"])).set_index("boot")
    rows = []
    for conv in ("persub", "meanrdm"):
        for px in RES:
            ds = np.stack([D[f"{conv}|{px}|Random Weights|{si}"] - D[f"{conv}|{px}|Backprop|{si}"]
                           for si in range(len(SEEDS))], axis=1)
            g = ds.mean(axis=1)
            s0 = ds[0]
            se = s0.std(ddof=1) / np.sqrt(len(s0))
            rows.append(dict(convention=conv, crossrun=False, res=px, gap=g[0],
                             boot_lo=np.quantile(g[1:], .025), boot_hi=np.quantile(g[1:], .975),
                             seed_lo=g[0] - TQ * se, seed_hi=g[0] + TQ * se, seeds_positive=int((s0 > 0).sum())))
    g4 = pd.read_csv(RESULTS / "step4_gaps.csv")
    g4 = g4[(g4.roi == "V1") & (g4.arm == "NATIVE")]
    for _, r in g4.iterrows():
        rows.append(dict(convention=r.convention, crossrun=True, res=int(r.res), gap=r.gap, boot_lo=r.boot_lo,
                         boot_hi=r.boot_hi, seed_lo=r.seed_t_lo, seed_hi=r.seed_t_hi, seeds_positive=r.seeds_positive))
    out = pd.DataFrame(rows).sort_values(["crossrun", "convention", "res"]).reset_index(drop=True)

    # references: identity all-pairs meanrdm must equal the v12 CSV (rounded to 6 decimals)
    v12 = pd.read_csv(V12_CSV)
    pub = pd.read_csv(REPO / "results" / "bnfix_sweep.csv")
    def ref(df, px, col="rho"):
        v = df[(df.layer == "Conv1") & (df.roi == "V1") & (df.res == px)].pivot_table(
            index="seed_idx", columns="rule", values=col)
        return float((v["Random Weights"] - v["Backprop"]).mean())
    for px in RES:
        x = out[(~out.crossrun) & (out.convention == "meanrdm") & (out.res == px)].iloc[0]
        assert abs(x.gap - ref(v12, px)) < 2e-6, (px, x.gap, ref(v12, px))
        y = out[(~out.crossrun) & (out.convention == "persub") & (out.res == px)].iloc[0]
        assert abs(y.gap - ref(v12, px, "rho_sub_mean")) < 2e-6
    out["v12_csv_meanrdm_allpairs"] = [ref(v12, p) for p in out.res]
    out["published_bnfix_sweep_meanrdm_allpairs"] = [ref(pub, p) for p in out.res]
    out.to_csv(RESULTS / "step6_headline.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
