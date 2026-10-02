"""Step 3a: stimulus bootstrap for the two preregistered quantities.

  V1  : Random Weights - Backprop, Conv1 -> V1
  LOC : Backprop - Random Weights, Conv3 -> LOC

Resamples: the noise_ceiling_v2 / crossrun index matrix (seed 20261002, 1000 x 720,
sha-checked), SHARED across arms, resolutions, seeds, rules and conventions, so every
difference and ratio is paired. Per resample b the pair set is recomputed as in
crossrun: pairs (s[i], s[j]), i < j, kept if s[i] != s[j] and the two stimuli lie in
different runs for every subject. Draw -1 is the identity sample (= the point estimate,
must equal step2's cr rows).

Each draw stores rho for every (convention, roi, arm, res, rule, seed_idx); gaps and
ratios are formed in step4. Resumable: one JSON line per finished draw in
data/upsampling/boot_parts.jsonl.

  py step3_bootstrap.py            # all remaining draws
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from common_up import (REPO, ARMS, RES, SEEDS, SUBJECTS, N_STIM, N_BOOT, N_PAIRS_ALL, TRI,
                       run_labels, ranks, tri_index, load_fmri_tri, rdm_path, boot_idx)

CASES = {"V1": ("Conv1", "Random Weights", "Backprop"),
         "LOC": ("Conv3", "Backprop", "Random Weights")}
ITEMS = [(roi, arm, px, rule, si) for roi, (layer, a, b) in CASES.items() for arm in ARMS
         for px in RES for rule in (a, b) for si in range(len(SEEDS))]
CACHE = REPO / "data" / "upsampling" / "boot_stack.npy"
PARTS = REPO / "data" / "upsampling" / "boot_parts.jsonl"
_W = {}


def build_cache():
    if CACHE.exists() and np.load(CACHE, mmap_mode="r").shape == (len(ITEMS), N_PAIRS_ALL):
        return
    M = np.lib.format.open_memmap(CACHE, mode="w+", dtype=np.float64, shape=(len(ITEMS), N_PAIRS_ALL))
    for k, (roi, arm, px, rule, si) in enumerate(ITEMS):
        M[k] = np.load(rdm_path(arm, px, si, rule, CASES[roi][0]))
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
    out = {"boot": b, "n_pairs": int(len(idx))}
    brain = {}
    for roi, F in _W["fmri"].items():
        vecs = [f[idx] for f in F]
        brain[roi] = ([ranks(v) for v in vecs], ranks(np.mean(vecs, axis=0)))
    M = _W["M"]
    for k, (roi, arm, px, rule, si) in enumerate(ITEMS):
        mv = ranks(np.asarray(M[k][idx], float))
        sub, mean = brain[roi]
        tag = f"{roi}|{arm}|{px}|{rule}|{si}"
        out[f"persub|{tag}"] = float(np.mean([mv @ x for x in sub]))
        out[f"meanrdm|{tag}"] = float(mv @ mean)
    return out


def read_parts():
    done = {}
    if PARTS.exists():
        for line in PARTS.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # truncated last line of a killed run
            done[r["boot"]] = r
    return done


def main():
    t0 = time.time()
    boot = boot_idx()
    build_cache()
    done = read_parts()
    todo = [b for b in range(-1, N_BOOT) if b not in done]
    print(f"draws: {len(todo)} to do, {len(done)} done", flush=True)
    if os.name == "nt":       # keep Windows awake while this runs
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    with open(PARTS, "a") as fh, Pool(10, initializer=_init, initargs=(boot,)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, todo, chunksize=2)):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if i % 50 == 0:
                print(f"{i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"finished {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
