"""Step 5: paired difference UPSAMPLED - NATIVE of the gap, per convention x ROI x res (res != 32).

Gap as in step4 (V1: Random - BP at Conv1; LOC: BP - Random at Conv3; cross-run pair set).
The difference is formed inside every draw of step3 (same stimulus resample for both arms,
same seeds), so it is paired over stimuli and over seeds:

  diff_b = mean_s [gap_UP(s, b)] - mean_s [gap_NAT(s, b)]

Point = identity draw; boot CI = 2.5/97.5 percentiles over the 1000 resamples;
seed t-CI = mean +- t_{0.975,4} SD/sqrt(5) over the 5 per-seed differences of the identity
draw; seeds_positive = number of seeds with a per-seed difference > 0.

Output: results/upsampling/step5_paired_diff.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import t as tdist

from common_up import RESULTS, RES, SEEDS, N_BOOT
from step3_bootstrap import CASES, read_parts

CONVS = ["persub", "meanrdm"]
TQ = tdist.ppf(0.975, len(SEEDS) - 1)


def main():
    d = read_parts()
    assert not (set(range(-1, N_BOOT)) - set(d)), "missing draws"
    D = pd.DataFrame([d[b] for b in range(-1, N_BOOT)]).set_index("boot")

    def per_seed_gap(conv, roi, arm, px):
        _, a, b = CASES[roi]
        return np.stack([D[f"{conv}|{roi}|{arm}|{px}|{a}|{si}"] - D[f"{conv}|{roi}|{arm}|{px}|{b}|{si}"]
                         for si in range(len(SEEDS))], axis=1)          # (1001, 5)

    rows = []
    for conv in CONVS:
        for roi in CASES:
            for px in [r for r in RES if r != 32]:
                ds = per_seed_gap(conv, roi, "UPSAMPLED", px) - per_seed_gap(conv, roi, "NATIVE", px)
                m = ds.mean(axis=1)
                pt, bt, s0 = m[0], m[1:], ds[0]
                se = s0.std(ddof=1) / np.sqrt(len(s0))
                rows.append(dict(convention=conv, roi=roi, res=px, diff=pt,
                                 boot_lo=np.quantile(bt, .025), boot_hi=np.quantile(bt, .975),
                                 seed_lo=pt - TQ * se, seed_hi=pt + TQ * se,
                                 seeds_positive=int((s0 > 0).sum()),
                                 **{f"diff_seed{i}": v for i, v in enumerate(s0)}))
    out = pd.DataFrame(rows)
    # consistency with step4 (224 px): UP - NAT there was formed the same way
    s4 = pd.read_csv(RESULTS / "step4_ratio.csv").set_index(["convention", "roi"])
    for (conv, roi), r in s4.iterrows():
        x = out[(out.convention == conv) & (out.roi == roi) & (out.res == 224)].iloc[0]
        assert abs(x["diff"] - r.up_minus_nat) < 1e-12 and abs(x.boot_lo - r.up_minus_nat_lo) < 1e-12
    out.to_csv(RESULTS / "step5_paired_diff.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.iloc[:, :9].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
