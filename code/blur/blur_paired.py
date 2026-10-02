"""Blur test, step 3: paired differences between conditions (per subject and mean-RDM, cross-run).

From the same draws as blur_gaps.csv (blur draws + step3 draws share the resample index), the
gap (V1: Random - BP, Conv1; LOC: BP - Random, Conv3) is differenced inside every draw and
inside every seed:
  blur_r - NATIVE_224, blur_r - UPSAMPLED_224 (all r), blur_96 - blur_32, blur_96 - blur_160.
Point = identity draw; boot CI = 2.5/97.5 percentiles; seeds_positive from the per-seed
differences of the identity draw.

Output: results/blur/blur_paired.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "upsampling"))
from common_up import REPO, SEEDS, N_BOOT  # noqa: E402
from step3_bootstrap import CASES, read_parts as read_step3  # noqa: E402
from blur_bootstrap import read_parts  # noqa: E402
from blur_extract import LEVELS  # noqa: E402


def main():
    d, d3 = read_parts(), read_step3()
    B = pd.DataFrame([d[b] for b in range(-1, N_BOOT)]).set_index("boot")
    B3 = pd.DataFrame([d3[b] for b in range(-1, N_BOOT)]).set_index("boot")

    def sg(conv, roi, cond):
        DF, key = (B3, cond) if cond in ("NATIVE|224", "UPSAMPLED|224") else (B, cond)
        _, a, b = CASES[roi]
        return np.stack([DF[f"{conv}|{roi}|{key}|{a}|{si}"] - DF[f"{conv}|{roi}|{key}|{b}|{si}"]
                         for si in range(len(SEEDS))], axis=1)

    pairs = [(f"blur{r}", "NATIVE|224", f"blur_r{r} - NATIVE_224") for r in LEVELS] + \
            [(f"blur{r}", "UPSAMPLED|224", f"blur_r{r} - UPSAMPLED_224") for r in LEVELS] + \
            [("blur96", "blur32", "blur_r96 - blur_r32"), ("blur96", "blur160", "blur_r96 - blur_r160")]
    rows = []
    for conv in ("persub", "meanrdm"):
        for roi in CASES:
            for x, y, name in pairs:
                ds = sg(conv, roi, x) - sg(conv, roi, y)
                m = ds.mean(axis=1)
                rows.append(dict(convention=conv, roi=roi, comparison=name, diff=m[0],
                                 boot_lo=np.quantile(m[1:], .025), boot_hi=np.quantile(m[1:], .975),
                                 seeds_positive=int((ds[0] > 0).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "results" / "blur" / "blur_paired.csv", index=False)
    pd.set_option("display.width", 200)
    print(out[out.convention == "persub"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
