"""Band-pass test, step 2: gaps per condition with stimulus-bootstrap CIs (as step4 / blur).

Gap: V1 = Random - BP (Conv1); LOC = BP - Random (Conv3); cross-run pair set; per subject
(primary) and mean-RDM; mean over 5 seeds of per-seed differences. Same 1000-row resample
matrix as step3, so every condition is paired with NATIVE@224 (from step3's draws); the
column minus_native gives condition - NATIVE@224 with its bootstrap CI.

Output: results/bandpass_exploratory/bandpass/bandpass_gaps.csv, data/bandpass/boot_parts.jsonl (resumable)
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "upsampling"))
from common_up import (REPO, SEEDS, N_STIM, N_BOOT, N_PAIRS_ALL, TRI, run_labels, ranks, tri_index,  # noqa: E402
                       load_fmri_tri, boot_idx)
from step3_bootstrap import CASES, read_parts as read_step3  # noqa: E402
from bandpass_extract import CONDS, rdm_path  # noqa: E402

ITEMS = [(roi, c, rule, si) for roi, (layer, a, b) in CASES.items() for c in CONDS
         for rule in (a, b) for si in range(len(SEEDS))]
CACHE = REPO / "data" / "bandpass" / "boot_stack.npy"
PARTS = REPO / "data" / "bandpass" / "boot_parts.jsonl"
OUT = REPO / "results" / "bandpass_exploratory" / "bandpass"
TQ = tdist.ppf(0.975, len(SEEDS) - 1)
_W = {}


def build_cache():
    if CACHE.exists() and np.load(CACHE, mmap_mode="r").shape == (len(ITEMS), N_PAIRS_ALL):
        return
    M = np.lib.format.open_memmap(CACHE, mode="w+", dtype=np.float64, shape=(len(ITEMS), N_PAIRS_ALL))
    for k, (roi, c, rule, si) in enumerate(ITEMS):
        M[k] = np.load(rdm_path(c, si, rule, CASES[roi][0]))
    M.flush()


def _init(boot):
    _W["boot"], _W["runs"] = boot, run_labels()
    _W["fmri"] = {roi: load_fmri_tri(roi) for roi in CASES}
    _W["M"] = np.load(CACHE, mmap_mode="r")


def one(b):
    s = np.arange(N_STIM) if b < 0 else _W["boot"][b]
    a, c = s[TRI[0]], s[TRI[1]]
    keep = a != c
    for r in _W["runs"].values():
        keep &= r[a] != r[c]
    idx = tri_index(a[keep], c[keep])
    out = {"boot": b}
    brain = {}
    for roi, F in _W["fmri"].items():
        vecs = [f[idx] for f in F]
        brain[roi] = ([ranks(v) for v in vecs], ranks(np.mean(vecs, axis=0)))
    for k, (roi, cond, rule, si) in enumerate(ITEMS):
        mv = ranks(np.asarray(_W["M"][k][idx], float))
        sub, mean = brain[roi]
        out[f"persub|{roi}|{cond}|{rule}|{si}"] = float(np.mean([mv @ x for x in sub]))
        out[f"meanrdm|{roi}|{cond}|{rule}|{si}"] = float(mv @ mean)
    return out


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
    done = read_parts()
    todo = [b for b in range(-1, N_BOOT) if b not in done]
    print(f"draws: {len(todo)} to do", flush=True)
    if todo:
        with open(PARTS, "a") as fh, Pool(10, initializer=_init, initargs=(boot,)) as pool:
            for i, r in enumerate(pool.imap_unordered(one, todo, chunksize=2)):
                fh.write(json.dumps(r) + "\n"); fh.flush()
                if i % 100 == 0:
                    print(f"{i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    d, d3 = read_parts(), read_step3()
    B = pd.DataFrame([d[b] for b in range(-1, N_BOOT)]).set_index("boot")
    B3 = pd.DataFrame([d3[b] for b in range(-1, N_BOOT)]).set_index("boot")

    def sg(DF, conv, roi, cond):
        _, a, b = CASES[roi]
        return np.stack([DF[f"{conv}|{roi}|{cond}|{a}|{si}"] - DF[f"{conv}|{roi}|{cond}|{b}|{si}"]
                         for si in range(len(SEEDS))], axis=1)

    rows = []
    for conv in ("persub", "meanrdm"):
        for roi in CASES:
            nat = sg(B3, conv, roi, "NATIVE|224")
            for name, ds in [("NATIVE_224", nat)] + [(c, sg(B, conv, roi, c)) for c in CONDS]:
                g, s0 = ds.mean(axis=1), ds[0]
                se = s0.std(ddof=1) / np.sqrt(len(s0))
                dn = g - nat.mean(axis=1)
                kind, band = (name.split("_", 1) if name != "NATIVE_224" else ("native", "all"))
                rows.append(dict(convention=conv, roi=roi, condition=kind, band=band, gap=g[0],
                                 boot_lo=np.quantile(g[1:], .025), boot_hi=np.quantile(g[1:], .975),
                                 seed_t_lo=g[0] - TQ * se, seed_t_hi=g[0] + TQ * se,
                                 seeds_positive=int((s0 > 0).sum()), minus_native=dn[0],
                                 minus_native_lo=np.quantile(dn[1:], .025), minus_native_hi=np.quantile(dn[1:], .975)))
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "bandpass_gaps.csv", index=False)
    pd.set_option("display.width", 220)
    print(out[out.convention == "persub"].round(4).to_string(index=False))
    print(f"finished {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
