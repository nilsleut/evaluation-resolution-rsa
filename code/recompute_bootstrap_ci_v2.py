"""
recompute_bootstrap_ci.py
=========================
Rebuild the ci_lo/ci_hi columns that learning_rules_v9_sweep_modal.py now skips on the
GPU path (COMPUTE_BOOTSTRAP_CI = False).

Nothing is lost by skipping them on-GPU: the sweep saves every model RDM to
  OUT_DIR/rdms/res{px}/seed_{i}/rdm_{rule_key}_{layer}.npy
and the brain RDMs live on the data volume at
  /data/outputs_720/fmri_rdm_{roi}_{sub}.npy
so the CIs are a pure function of files already on disk. This script recomputes them
with the IDENTICAL procedure (same n_boot, same ci, same rng seed, same triu masking
and same min-n truncation as the original inline bootstrap_ci), then merges them back
into the sweep CSV.

CPU only, no GPU, runs on Modal so it can reach both volumes:
  python -m modal run recompute_bootstrap_ci.py                    # all rows, merge in place
  python -m modal run recompute_bootstrap_ci.py --no-merge         # write a side CSV only

Output: OUT_DIR/rsa_resolution_sweep_ci.csv, and (unless --no-merge) ci_lo/ci_hi filled
in OUT_DIR/rsa_resolution_sweep.csv.
"""

import modal
from pathlib import Path

app = modal.App("learning-rules-rsa-ci")

data_vol    = modal.Volume.from_name("burstprop-data", create_if_missing=True)
results_vol = modal.Volume.from_name("learning-rules-rsa", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy", "pandas")
)

SUBJECTS = ["sub-01", "sub-02", "sub-03"]
LAYERS   = ["Conv1", "Conv2", "Conv3", "FC1"]
N_BOOT   = 500
CI       = 0.95
BOOT_SEED = 42


@app.function(
    image=image, timeout=60*60*6, cpu=8.0, memory=8192,
    volumes={"/data": data_vol, "/results": results_vol},
)
def recompute(out_subdir: str = "outputs_bnfix", merge: bool = True):
    import numpy as np, pandas as pd
    from scipy.stats import spearmanr

    FMRI_DIR = Path("/data/outputs_720")
    OUT_DIR  = Path("/results") / out_subdir
    RDM_DIR  = OUT_DIR / "rdms"
    csv_path = OUT_DIR / "rsa_resolution_sweep.csv"
    assert csv_path.exists(), f"no {csv_path}; run finalize_sweep first"

    df = pd.read_csv(str(csv_path))
    # Guarantee the access order the single-entry RDM cache assumes.
    df = df.sort_values(["rule", "seed_idx", "res", "layer", "roi"]).reset_index(drop=True)
    print(f"{len(df)} rows in {csv_path.name}")

    def rule_key(rule): return rule.lower().replace(" ", "_")

    # ── identical to the inline bootstrap_ci in the sweep script ──────────────
    def bootstrap_ci(rdm_a, rdm_b, n_boot=N_BOOT, ci=CI, seed=BOOT_SEED):
        n = min(rdm_a.shape[0], rdm_b.shape[0]); idx = np.triu_indices(n, k=1)
        va = rdm_a[:n, :n][idx]; vb = rdm_b[:n, :n][idx]
        rng = np.random.default_rng(seed); boot = []
        for _ in range(n_boot):
            s = rng.integers(0, len(va), len(va))
            boot.append(spearmanr(va[s], vb[s])[0])
        return (float(np.percentile(boot, (1 - ci) / 2 * 100)),
                float(np.percentile(boot, (1 + ci) / 2 * 100)))

    _brain = {}
    def brain_mean(roi):
        """Mean subject RDM for an ROI -- same construction as run_rsa()."""
        if roi not in _brain:
            subr = [np.load(str(FMRI_DIR / f"fmri_rdm_{roi}_{s}.npy"))
                    for s in SUBJECTS if (FMRI_DIR / f"fmri_rdm_{roi}_{s}.npy").exists()]
            _brain[roi] = np.mean(subr, axis=0) if subr else None
        return _brain[roi]

    # v1 cached every model RDM it touched. There are 6 res x 5 seeds x 6 rules x 4
    # layers = 720 of them at 720x720 float64 = ~4.1 MB each, so the cache grew to
    # ~3 GB and the container was killed part-way through. finalize_sweep sorts the
    # CSV by (rule, seed_idx, res, layer, roi), so the only reuse is across the ROIs
    # of one key -- a single-entry cache captures all of it at constant memory.
    _rdm_key, _rdm_val = None, None
    def model_rdm(res, seed_idx, rule, layer):
        nonlocal _rdm_key, _rdm_val
        k = (res, seed_idx, rule, layer)
        if k != _rdm_key:
            p = RDM_DIR / f"res{int(res)}" / f"seed_{int(seed_idx)}" / f"rdm_{rule_key(rule)}_{layer}.npy"
            _rdm_key, _rdm_val = k, (np.load(str(p)) if p.exists() else None)
        return _rdm_val

    out, missing, t0 = [], [], __import__("time").time()
    for i, r in enumerate(df.itertuples(index=False), 1):
        m = model_rdm(r.res, r.seed_idx, r.rule, r.layer)
        b = brain_mean(r.roi)
        if m is None or b is None:
            missing.append((r.rule, int(r.seed_idx), int(r.res), r.layer, r.roi))
            out.append((np.nan, np.nan)); continue
        n = min(m.shape[0], b.shape[0])
        out.append(bootstrap_ci(m[:n, :n], b[:n, :n]))
        if i % 100 == 0:
            el = (__import__("time").time() - t0) / 60
            print(f"  {i}/{len(df)} rows  ({el:.1f} min elapsed, "
                  f"~{el/i*(len(df)-i):.1f} min left)", flush=True)

    df["ci_lo"] = [round(lo, 6) for lo, _ in out]
    df["ci_hi"] = [round(hi, 6) for _, hi in out]

    side = OUT_DIR / "rsa_resolution_sweep_ci.csv"
    df[["rule", "layer", "roi", "res", "seed_idx", "rho", "ci_lo", "ci_hi"]].to_csv(str(side), index=False)
    if merge:
        df.to_csv(str(csv_path), index=False)
        print(f"merged ci_lo/ci_hi into {csv_path.name}")
    results_vol.commit()

    ok = int(df.ci_lo.notna().sum())
    print(f"\ncomputed CIs for {ok}/{len(df)} rows -> {side.name}")
    if missing:
        print(f"WARNING: {len(missing)} rows had no saved RDM; left as NaN")
        for m in missing[:10]: print(f"    {m}")
    # sanity: rho should sit inside its own CI for the overwhelming majority of rows
    inside = ((df.rho >= df.ci_lo) & (df.rho <= df.ci_hi)).sum()
    print(f"sanity: rho within [ci_lo, ci_hi] for {inside}/{ok} rows with CIs")
    return str(side)


@app.local_entrypoint()
def main(out_subdir: str = "outputs_bnfix", merge: bool = True):
    print(f"Recomputing bootstrap CIs offline (CPU) for /results/{out_subdir}")
    print(f"  n_boot={N_BOOT} ci={CI} seed={BOOT_SEED} (identical to the inline version)")
    print(f"\nDone: {recompute.remote(out_subdir, merge)}")
