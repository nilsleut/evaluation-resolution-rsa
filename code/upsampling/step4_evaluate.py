"""Step 3b: gaps, ratio R, bootstrap CIs, preregistered verdict, figure.

Gap (per convention, roi, arm, res) = mean over the 5 seeds of the paired per-seed
difference (V1: Random - BP at Conv1; LOC: BP - Random at Conv3), on pair set P.
Point = identity draw; 95% CI = 2.5/97.5 percentiles over the 1000 shared resamples.
R = Gap_UPSAMPLED(224) / Gap_NATIVE(224), formed per resample.

Verdict (fixed before the run; primary = persub, V1):
  A  R >= 0.5 and the Gap_UP(224) CI excludes 0  -> effect follows input size / RF geometry
  B  the R CI contains 0                         -> effect follows information content
  otherwise (or A and B both)                    -> not decided; propose (b1), do not start it
Also reported: seed-level t-CI (t_{0.975,4}), seeds positive, and the same for LOC.

Output: results/upsampling/step4_gaps.csv, step4_ratio.csv, step4.json,
        fig_gap_vs_resolution.{png,pdf}
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

from common_up import RESULTS, ARMS, RES, SEEDS, N_BOOT, N_PAIRS_CR
from step3_bootstrap import CASES, read_parts

CONVS = ["persub", "meanrdm"]
TQ = tdist.ppf(0.975, len(SEEDS) - 1)


def verdict(R, R_lo, R_hi, up_lo, up_hi):
    a = R >= 0.5 and (up_lo > 0 or up_hi < 0)
    b = R_lo <= 0 <= R_hi
    if a and not b:
        return "A", "Effekt hängt an Inputgrösse / Rezeptivfeld-Geometrie"
    if b and not a:
        return "B", "Effekt hängt am Informationsgehalt"
    return "C", "nicht entschieden -> (b1) Retrain-Sweep vorschlagen (nicht starten)"


def main():
    d = read_parts()
    miss = sorted(set(range(-1, N_BOOT)) - set(d))
    assert not miss, f"missing draws: {miss[:10]}"
    D = pd.DataFrame([d[b] for b in range(-1, N_BOOT)]).set_index("boot")
    assert D.loc[-1, "n_pairs"] == N_PAIRS_CR

    # identity draw must equal step2's cr rows
    s2 = pd.concat([pd.read_csv(RESULTS / f"rsa_seed{i}.csv") for i in range(len(SEEDS))])
    dev = 0.0
    for roi, (layer, a, b) in CASES.items():
        q = s2[(s2.variant == "cr") & (s2.layer == layer) & (s2.roi == roi) & s2.rule.isin([a, b])]
        for _, r in q.iterrows():
            dev = max(dev, abs(D.loc[-1, f"{r.convention}|{roi}|{r.arm}|{r.res}|{r.rule}|{r.seed_idx}"] - r.rho))
    assert dev < 1e-12, dev

    def seed_diffs(conv, roi, arm, px):
        _, a, b = CASES[roi]
        return np.stack([D[f"{conv}|{roi}|{arm}|{px}|{a}|{si}"] - D[f"{conv}|{roi}|{arm}|{px}|{b}|{si}"]
                         for si in range(len(SEEDS))], axis=1)       # (1001, 5)

    gaps, G = [], {}
    for conv in CONVS:
        for roi in CASES:
            for arm in ARMS:
                for px in RES:
                    sd = seed_diffs(conv, roi, arm, px)
                    g = sd.mean(axis=1)
                    G[(conv, roi, arm, px)] = g
                    pt, bt, s0 = g[0], g[1:], sd[0]
                    se = s0.std(ddof=1) / np.sqrt(len(s0))
                    gaps.append(dict(convention=conv, roi=roi, arm=arm, res=px, gap=pt,
                                     boot_lo=np.quantile(bt, .025), boot_hi=np.quantile(bt, .975),
                                     boot_sd=bt.std(ddof=1), seed_t_lo=pt - TQ * se, seed_t_hi=pt + TQ * se,
                                     seeds_positive=int((s0 > 0).sum()),
                                     **{f"gap_seed{i}": v for i, v in enumerate(s0)}))
    gaps = pd.DataFrame(gaps)
    gaps.to_csv(RESULTS / "step4_gaps.csv", index=False)

    ratios, out = [], {"n_boot": N_BOOT, "n_pairs_identity": int(D.loc[-1, "n_pairs"]),
                       "n_pairs_boot_median": int(D.n_pairs.iloc[1:].median()),
                       "identity_vs_step2_max_dev": dev, "verdicts": {}}
    for conv in CONVS:
        for roi in CASES:
            up, na = G[(conv, roi, "UPSAMPLED", 224)], G[(conv, roi, "NATIVE", 224)]
            R = up / na
            diff = up - na
            Rb, db = R[1:], diff[1:]
            rr = dict(convention=conv, roi=roi, R=R[0], R_lo=np.quantile(Rb, .025), R_hi=np.quantile(Rb, .975),
                      gap_up_224=up[0], gap_up_lo=np.quantile(up[1:], .025), gap_up_hi=np.quantile(up[1:], .975),
                      gap_nat_224=na[0], gap_nat_lo=np.quantile(na[1:], .025), gap_nat_hi=np.quantile(na[1:], .975),
                      up_minus_nat=diff[0], up_minus_nat_lo=np.quantile(db, .025),
                      up_minus_nat_hi=np.quantile(db, .975),
                      frac_boot_nat_le_0=float((na[1:] <= 0).mean()))
            code, text = verdict(rr["R"], rr["R_lo"], rr["R_hi"], rr["gap_up_lo"], rr["gap_up_hi"])
            rr["verdict"], rr["verdict_text"] = code, text
            ratios.append(rr)
            out["verdicts"][f"{conv}|{roi}"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                               for k, v in rr.items()}
    ratios = pd.DataFrame(ratios)
    ratios.to_csv(RESULTS / "step4_ratio.csv", index=False)
    out["primary"] = out["verdicts"]["persub|V1"]
    (RESULTS / "step4.json").write_text(json.dumps(out, indent=2))
    pd.set_option("display.width", 220)
    print(gaps[["convention", "roi", "arm", "res", "gap", "boot_lo", "boot_hi", "seeds_positive"]]
          .round(4).to_string(index=False))
    print(ratios.drop(columns=["verdict_text"]).round(4).to_string(index=False))
    print(json.dumps(out["primary"], indent=1))
    plot(gaps)


def plot(gaps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"NATIVE": "#2a78d6", "UPSAMPLED": "#eb6834"}
    mk = {"NATIVE": "o", "UPSAMPLED": "s"}
    lab = {"V1": "Random − Backprop, Conv1 → V1", "LOC": "Backprop − Random, Conv3 → LOC"}
    clab = {"persub": "pro Subject (primär)", "meanrdm": "Mittel-RDM (sekundär)"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    for i, roi in enumerate(CASES):
        for j, conv in enumerate(CONVS):
            ax = axes[i, j]
            ax.axhline(0, color="#52514e", lw=0.8, zorder=0)
            for k, arm in enumerate(ARMS):
                g = gaps[(gaps.roi == roi) & (gaps.convention == conv) & (gaps.arm == arm)].sort_values("res")
                x = g.res + (k - 0.5) * 3
                ax.errorbar(x, g.gap, yerr=[g.gap - g.boot_lo, g.boot_hi - g.gap], color=col[arm],
                            marker=mk[arm], ms=6, lw=2, capsize=0, elinewidth=1.2, label=arm)
            ax.set_title(f"{lab[roi]}\n{clab[conv]}", fontsize=10, color="#0b0b0b")
            ax.set_xticks(RES)
            ax.grid(axis="y", color="#e6e5e0", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if j == 0:
                ax.set_ylabel("Gap Δρ (Cross-Run, Mittel über 5 Seeds)")
            if i == 1:
                ax.set_xlabel("Evaluationsauflösung (px)")
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("(b2) Upsampling-Test: Gap vs. Auflösung, 95%-Stimulus-Bootstrap-CI (1000×)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_gap_vs_resolution.{ext}", dpi=160)


if __name__ == "__main__":
    main()
