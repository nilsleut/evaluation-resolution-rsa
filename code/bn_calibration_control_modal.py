"""
bn_calibration_control_modal.py
===============================
Eval-only control that decomposes the BN-calibration cost at V1.

WHY
---
The sweep's "Random Weights (BN-calibrated)" condition calibrates BatchNorm on
CIFAR-10 at 32 px and then evaluates on THINGS at up to 224 px. Those statistics are
mismatched to the evaluation distribution AND to the evaluation resolution, so the
-0.065 cost at 224 px is consistent with three different mechanisms. This script
separates them with a 2x2 factorial over {calibration distribution} x {calibration
resolution}, holding the convolutional weights fixed.

Plain "Random Weights" (identity BN: running_mean=0, running_var=1) is the reference.

A DEFECT IN THE ORIGINAL CONDITION
----------------------------------
nn.BatchNorm2d defaults to momentum=0.1, so 50 calibration batches give an
exponential moving average dominated by the last ~10 batches rather than the mean
over all 50. For a calibration procedure we want the unbiased estimate, so every
variant here sets momentum=None during calibration, which makes PyTorch accumulate
a cumulative moving average using num_batches_tracked. Variant "A_aug" reproduces
the original condition exactly except for that one change, so the re-run is a
one-variable comparison against the published -0.065.

(Caveat worth stating: PyTorch's cumulative running_var is the average of the
per-batch unbiased variances, not the pooled variance over all calibration images.
With equal batch sizes the running_mean IS the exact pooled mean; the variance
omits the between-batch variation of the means. That is what momentum=None means
in PyTorch and it is the estimator being requested.)

VARIANTS
--------
  Random (identity BN)         no calibration -- the reference
  A_aug  CIFAR-10 @ 32 px      the existing condition (augmented loader), momentum=None
  A      CIFAR-10 @ 32 px      deterministic loader
  B      CIFAR-10 @ eval res   CIFAR upsampled to whatever px is being evaluated
  C      THINGS   @ 32 px      disjoint THINGS images
  D      THINGS   @ eval res   disjoint THINGS images
  C_leak THINGS   @ 32 px      the SAME 720 evaluation images -- leakage probe
  D_leak THINGS   @ eval res   the SAME 720 evaluation images -- leakage probe

A_aug vs A isolates the augmented-vs-deterministic loader, so the factorial itself
(A/B/C/D) uses one matched, deterministic procedure throughout.

B and D recalibrate once per evaluation resolution; A and C calibrate once and are
evaluated at all six. B at 32 px is a no-op resize of A, so B@32 == A@32 exactly --
an internal consistency check.

LEAKAGE
-------
C and D calibrate on THINGS images drawn from concept folders DISJOINT from the 720
evaluation stimuli (1854 concepts exist, 720 are used), so there is no image-level or
concept-level overlap. C_leak/D_leak repeat the calibration on the 720 evaluation
images themselves to quantify what leakage would have bought.

READING IT (decided in advance)
-------------------------------
  only D recovers        -> matching to the evaluation distribution AND resolution
  B and D recover        -> resolution matching, not distribution
  C and D recover        -> distribution matching, not resolution
  nothing recovers       -> normalization per se costs alignment; 3.2 stands
  A recovers vs the published -0.065 -> the original condition was mis-estimated

RUN
---
  python -m modal deploy bn_calibration_control_modal.py
  python spawn_bncal.py --phase1      # Random + A_aug, 5 seeds each
  python spawn_bncal.py --phase2      # A B C D C_leak D_leak, 5 seeds each
  python -m modal run bn_calibration_control_modal.py::finalize
"""

import modal
from pathlib import Path

app = modal.App("bn-calibration-control")

data_vol    = modal.Volume.from_name("burstprop-data", create_if_missing=True)
results_vol = modal.Volume.from_name("learning-rules-rsa", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "torchvision", "numpy", "scipy", "pandas", "Pillow")
    .add_local_python_source("bn_guard")
)

# ── Constants: identical to learning_rules_v10_sweep_modal.py ────────────────
SUBJECTS   = ["sub-01", "sub-02", "sub-03"]
SEEDS      = [42, 123, 456, 789, 1337]
SWEEP_RES  = [32, 64, 96, 128, 160, 224]
N_BN_CALIB = 50            # calibration batches, as in the original condition
BATCH      = 128
N_CIFAR    = 8000
C1, C2, C3 = 32, 64, 128
FC1_DIM    = 512
N_CLS      = 10
FEAT_SIZE  = 4
FC1_IN     = C3 * FEAT_SIZE * FEAT_SIZE
NORM_MEAN  = (0.4914, 0.4822, 0.4465)
NORM_STD   = (0.247, 0.243, 0.261)

LAYER_ROI_FIXED = {
    "Conv1": (0, ["V1", "V2"]),
    "Conv2": (1, ["V1", "V2"]),
    "Conv3": (2, ["LOC"]),
    "FC1":   (3, ["IT"]),
}

# variant -> (calibration source, recalibrate per evaluation resolution?, augmented?)
#   source: None = no calibration; "cifar"; "things_disjoint"; "things_eval720"
VARIANTS = {
    "Random (identity BN)":   (None,              False, False),
    "A_aug CIFAR@32":         ("cifar",           False, True),
    "A CIFAR@32":             ("cifar",           False, False),
    "B CIFAR@eval-res":       ("cifar",           True,  False),
    "C THINGS-disjoint@32":   ("things_disjoint", False, False),
    "D THINGS-disjoint@res":  ("things_disjoint", True,  False),
    "C_leak THINGS-720@32":   ("things_eval720",  False, False),
    "D_leak THINGS-720@res":  ("things_eval720",  True,  False),
}
PHASE1 = ["Random (identity BN)", "A_aug CIFAR@32"]
PHASE2 = ["A CIFAR@32", "B CIFAR@eval-res", "C THINGS-disjoint@32",
          "D THINGS-disjoint@res", "C_leak THINGS-720@32", "D_leak THINGS-720@res"]


@app.function(
    image=image, gpu="T4", timeout=60 * 60,
    volumes={"/data": data_vol, "/results": results_vol},
)
def run_variant(variant: str, seed_idx: int, out_subdir: str = "outputs_bncal"):
    """One (variant, seed) shard: calibrate, evaluate at all six resolutions,
    and record the BN-statistics diagnostics. Forward passes only."""
    import hashlib, json, random, time
    import numpy as np, pandas as pd
    import torch, torch.nn as nn, torch.nn.functional as F
    import torchvision, torchvision.transforms as T
    from torch.utils.data import DataLoader, Subset, Dataset
    from scipy.stats import spearmanr
    from scipy.spatial.distance import pdist, squareform
    from PIL import Image
    from bn_guard import assert_bn_eval, assert_bn_train, iter_bn

    assert variant in VARIANTS, f"unknown variant {variant!r}"
    source, per_res, augmented = VARIANTS[variant]
    seed = SEEDS[seed_idx]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    FMRI_DIR   = Path("/data/outputs_720")
    THINGS_DIR = Path("/data/object_images")
    CIFAR_DIR  = Path("/results/cifar_data")
    OUT_DIR    = Path("/results") / out_subdir           # fresh: never outputs_bnfix
    SHARD_DIR  = OUT_DIR / "shards"
    DIAG_DIR   = OUT_DIR / "diagnostics"
    for d in [OUT_DIR, SHARD_DIR, DIAG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}   variant={variant!r}  seed_idx={seed_idx} (seed={seed})")

    # ── Architecture: byte-for-byte the v10 BP_CNN ───────────────────────────
    def make_conv_block(in_c, out_c):
        return nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                             nn.BatchNorm2d(out_c), nn.ReLU(True), nn.MaxPool2d(2))

    class BP_CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = make_conv_block(3, C1); self.conv2 = make_conv_block(C1, C2)
            self.conv3 = make_conv_block(C2, C3)
            self.fc1 = nn.Linear(FC1_IN, FC1_DIM); self.fc2 = nn.Linear(FC1_DIM, N_CLS)
            self.drop = nn.Dropout(0.3)
        def forward(self, x):
            x = self.conv3(self.conv2(self.conv1(x)))
            return self.fc2(self.drop(F.relu(self.fc1(x.view(x.size(0), -1)))))
        def calib_forward(self, x):
            """Conv trunk only: drives the three BatchNorms, skips the FC head
            (and its dropout, which would consume RNG for nothing)."""
            return self.conv3(self.conv2(self.conv1(x)))
        def get_features(self, x):
            with torch.no_grad():
                c1 = self.conv1(x); c2 = self.conv2(c1); c3 = self.conv3(c2)
                h1 = F.relu(self.fc1(F.adaptive_avg_pool2d(c3, FEAT_SIZE)
                                     .view(c3.size(0), -1)))
            return c1.mean([2, 3]), c2.mean([2, 3]), c3.mean([2, 3]), h1

    # Same seeding as the sweep, so the weights ARE the sweep's "Random Weights".
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = BP_CNN().to(DEVICE)

    conv_w = lambda: [model.conv1[0].weight, model.conv2[0].weight, model.conv3[0].weight]
    def conv_sha():
        h = hashlib.sha256()
        for w in conv_w():
            h.update(w.detach().cpu().numpy().tobytes())
        return h.hexdigest()[:16]
    sha_at_init = conv_sha()
    print(f"  conv-weight sha256[:16] at init: {sha_at_init}")

    # Seed 0 must equal the sweep's saved random-weights checkpoint, bit for bit.
    ckpt_match = None
    ref = Path("/results/outputs_bnfix/checkpoints/model_weights_random_weights.pt")
    if seed_idx == 0 and ref.exists():
        sd = torch.load(str(ref), map_location="cpu", weights_only=True)
        ckpt_match = all(
            torch.equal(w.detach().cpu(), sd[f"conv{i}.0.weight"])
            for i, w in enumerate(conv_w(), start=1))
        print(f"  conv weights identical to outputs_bnfix random checkpoint: {ckpt_match}")

    # ── Data ─────────────────────────────────────────────────────────────────
    def things_tf(px):
        return T.Compose([T.Resize(px), T.CenterCrop(px), T.ToTensor(),
                          T.Normalize(NORM_MEAN, NORM_STD)])

    def cifar_tf(px, aug):
        # CIFAR is 32x32 and square, so Resize(px) alone matches the eval geometry;
        # at px=32 it is a no-op, which is what makes B@32 identical to A@32.
        pre = ([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4)] if aug else [])
        return T.Compose(pre + [T.Resize(px), T.ToTensor(),
                                T.Normalize(NORM_MEAN, NORM_STD)])

    class ImgDS(Dataset):
        def __init__(self, paths, t): self.paths, self.t = paths, t
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            return self.t(Image.open(self.paths[i]).convert("RGB")), i

    def load_stim_order(sub="sub-01"):
        with open(FMRI_DIR / f"stim_order_{sub}.txt") as f:
            return [l.strip() for l in f if l.strip()]

    def find_img(stimulus):
        name = stimulus.replace(".jpg", ""); parts = name.split("_"); last = parts[-1]
        concept = ("_".join(parts[:-1])
                   if (len(parts) > 1 and len(last) <= 4 and any(c.isdigit() for c in last))
                   else name)
        for pat in [f"{concept}/{name}.jpg", f"{concept}/*.jpg"]:
            hits = sorted(THINGS_DIR.glob(pat))
            if hits: return hits[0]
        for folder in THINGS_DIR.iterdir():
            if folder.name.lower() == concept.lower():
                imgs = sorted(folder.glob("*.jpg"))
                if imgs: return imgs[0]
        return None

    stimuli = load_stim_order("sub-01")
    eval_paths = [p for p in [find_img(s) for s in stimuli] if p is not None]
    print(f"  THINGS eval set: {len(eval_paths)}/{len(stimuli)} images resolved")

    n_calib_imgs = N_BN_CALIB * BATCH

    def things_disjoint_paths():
        """Calibration images from concept folders that contribute NO evaluation
        stimulus. Excluding whole folders removes image- and concept-level overlap."""
        used = {p.parent.name for p in eval_paths}
        pool = []
        for folder in sorted(THINGS_DIR.iterdir()):
            if not folder.is_dir() or folder.name in used:
                continue
            pool.extend(sorted(folder.glob("*.jpg")))
        rng = np.random.default_rng(seed)
        if len(pool) < n_calib_imgs:
            raise RuntimeError(f"disjoint THINGS pool has only {len(pool)} images, "
                               f"need {n_calib_imgs}")
        sel = rng.choice(len(pool), size=n_calib_imgs, replace=False)
        print(f"  disjoint pool: {len(pool)} images over "
              f"{len(list(THINGS_DIR.iterdir())) - len(used)} unused concepts; "
              f"sampled {n_calib_imgs}")
        return [pool[i] for i in sel]

    def things_eval720_paths():
        """Leakage probe: calibrate on the very images the RDM is built from.
        Cycled to the same image budget so only the source differs."""
        reps = (n_calib_imgs + len(eval_paths) - 1) // len(eval_paths)
        return (list(eval_paths) * reps)[:n_calib_imgs]

    _things_calib_cache = {}
    def calib_loader(px):
        if source == "cifar":
            tf = cifar_tf(px, augmented)
            g = torch.Generator().manual_seed(seed)
            full = torchvision.datasets.CIFAR10(str(CIFAR_DIR), train=True,
                                                download=True, transform=tf)
            idx = torch.randperm(len(full), generator=torch.Generator().manual_seed(seed))
            idx = idx[:N_CIFAR].tolist()
            return DataLoader(Subset(full, idx), batch_size=BATCH, shuffle=True,
                              num_workers=4, drop_last=True, generator=g)
        if source not in _things_calib_cache:
            _things_calib_cache[source] = (things_disjoint_paths()
                                           if source == "things_disjoint"
                                           else things_eval720_paths())
        paths = _things_calib_cache[source]
        return DataLoader(ImgDS(paths, things_tf(px)), batch_size=BATCH,
                          shuffle=False, num_workers=4, drop_last=True)

    # ── Calibration ──────────────────────────────────────────────────────────
    def calibrate(px):
        """Estimate BN running stats from `source` at `px`. Weights must not move."""
        bns = iter_bn(model)
        for bn in bns:
            bn.reset_running_stats()   # mean=0, var=1, num_batches_tracked=0
            bn.momentum = None         # cumulative moving average, not EMA
        before = [p.detach().clone() for p in model.parameters()]
        model.train()
        assert_bn_train(model, context=f"calibrate/{variant}@{px}")
        loader = calib_loader(px)
        n = 0
        with torch.no_grad():
            for x, _ in loader:
                if n >= N_BN_CALIB: break
                model.calib_forward(x.to(DEVICE)); n += 1
        if n < N_BN_CALIB:
            raise RuntimeError(f"only {n} calibration batches available, need {N_BN_CALIB}")
        if not all(torch.equal(a, b) for a, b in zip(before, model.parameters())):
            raise RuntimeError("calibration changed weights; it must be stats-only")
        if conv_sha() != sha_at_init:
            raise RuntimeError("conv weights moved during calibration")
        nbt = [int(bn.num_batches_tracked) for bn in bns]
        print(f"    calibrated @{px}px on {source} ({n} batches, momentum=None, "
              f"num_batches_tracked={nbt})")
        return n

    # ── RSA ──────────────────────────────────────────────────────────────────
    def compute_rdm(f):
        return squareform(pdist(np.asarray(f), metric="correlation"))

    def rsa_score(a, b):
        n = min(a.shape[0], b.shape[0]); idx = np.triu_indices(n, k=1)
        r, _ = spearmanr(a[:n, :n][idx], b[:n, :n][idx]); return float(r)

    def load_fmri_rdm(roi, sub):
        p = FMRI_DIR / f"fmri_rdm_{roi}_{sub}.npy"
        return np.load(str(p)) if p.exists() else None

    def extract_features(paths, transform):
        loader = DataLoader(ImgDS(paths, transform), batch_size=128, shuffle=False,
                            num_workers=4)
        outs = [[], [], [], []]
        model.eval()
        assert_bn_eval(model, context=f"extract/{variant}")
        with torch.no_grad():
            for imgs, _ in loader:
                for acc, t in zip(outs, model.get_features(imgs.to(DEVICE))):
                    acc.append(t.cpu().numpy())
        return [np.concatenate(a) for a in outs]

    def run_rsa(px):
        rdms = [compute_rdm(f) for f in extract_features(eval_paths, things_tf(px))]
        rows = []
        for layer_name, (fi, rois) in LAYER_ROI_FIXED.items():
            for roi in rois:
                subr = [r for r in (load_fmri_rdm(roi, s) for s in SUBJECTS) if r is not None]
                if not subr: continue
                per_sub = [rsa_score(rdms[fi], r) for r in subr]
                bmean = np.mean(subr, axis=0)
                n = min(rdms[fi].shape[0], bmean.shape[0])
                rows.append({"variant": variant, "seed": seed, "seed_idx": seed_idx,
                             "res": int(px), "layer": layer_name, "roi": roi,
                             "rho": round(rsa_score(rdms[fi][:n, :n], bmean[:n, :n]), 6),
                             "n_subs": len(subr),
                             "rho_sub_mean": round(float(np.mean(per_sub)), 6)})
        return rows

    # ── Diagnostics: stored stats vs the statistics that actually arrive ──────
    def bn_diagnostics(px):
        """Per BN layer: the stored running stats, and the real per-channel mean and
        variance of that layer's INPUT over the evaluation set at `px`. The gap
        between the two is the mismatch this control is about, so it is measured
        rather than assumed."""
        bns = iter_bn(model)
        acc = [{"n": 0, "s": None, "ss": None} for _ in bns]

        def mk(i):
            def hook(mod, inp):
                x = inp[0].detach().float()
                s = x.sum(dim=[0, 2, 3]); ss = (x * x).sum(dim=[0, 2, 3])
                a = acc[i]
                a["s"] = s if a["s"] is None else a["s"] + s
                a["ss"] = ss if a["ss"] is None else a["ss"] + ss
                a["n"] += x.shape[0] * x.shape[2] * x.shape[3]
            return hook

        handles = [bn.register_forward_pre_hook(mk(i)) for i, bn in enumerate(bns)]
        try:
            model.eval()
            assert_bn_eval(model, context=f"diagnostics/{variant}")
            loader = DataLoader(ImgDS(eval_paths, things_tf(px)), batch_size=128,
                                shuffle=False, num_workers=4)
            with torch.no_grad():
                for imgs, _ in loader:
                    model.get_features(imgs.to(DEVICE))
        finally:
            for h in handles:
                h.remove()

        out = []
        for i, (bn, a) in enumerate(zip(bns, acc), start=1):
            amean = (a["s"] / a["n"]).cpu().numpy()
            avar = (a["ss"] / a["n"]).cpu().numpy() - amean ** 2
            rm = bn.running_mean.detach().cpu().numpy()
            rv = bn.running_var.detach().cpu().numpy()
            out.append({
                "variant": variant, "seed_idx": seed_idx, "diag_res": int(px),
                "layer": f"BN{i}",
                "num_batches_tracked": int(bn.num_batches_tracked),
                "stored_running_mean": round(float(rm.mean()), 6),
                "stored_running_var":  round(float(rv.mean()), 6),
                "actual_input_mean":   round(float(amean.mean()), 6),
                "actual_input_var":    round(float(avar.mean()), 6),
                # How wrong the stored stats are for what actually arrives.
                "mean_gap":            round(float(np.abs(rm - amean).mean()), 6),
                "var_ratio_stored_over_actual": round(float((rv / avar).mean()), 6),
                "conv_sha": sha_at_init,
                "conv_matches_bnfix_random_ckpt": ckpt_match,
            })
        return out

    # ── Main ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    rows, diags = [], []
    if source is not None and not per_res:
        calibrate(32)

    for px in SWEEP_RES:
        if source is not None and per_res:
            calibrate(px)
        rows.extend(run_rsa(px))
        v1 = next((r for r in rows[-6:] if r["layer"] == "Conv1" and r["roi"] == "V1"), None)
        if v1: print(f"    @{px:>3}px  V1(Conv1) rho={v1['rho']:.4f}")
        if px == 224:
            # Diagnostics under exactly the BN state used for the 224 px evaluation.
            diags.extend(bn_diagnostics(224))

    tag = variant.split()[0].replace("/", "_")
    pd.DataFrame(rows).to_csv(str(SHARD_DIR / f"rows_{tag}_seed{seed_idx}.csv"), index=False)
    pd.DataFrame(diags).to_csv(str(DIAG_DIR / f"diag_{tag}_seed{seed_idx}.csv"), index=False)
    results_vol.commit()

    mins = (time.time() - t0) / 60
    print(f"\nDONE {variant} seed {seed_idx}: {len(rows)} rows in {mins:.1f} min")
    return {"variant": variant, "seed_idx": seed_idx, "status": "ok",
            "rows": len(rows), "minutes": round(mins, 2),
            "conv_sha": sha_at_init, "ckpt_match": ckpt_match}


@app.function(image=image, timeout=60 * 20, volumes={"/results": results_vol})
def finalize(out_subdir: str = "outputs_bncal"):
    """Merge shards. Duplicated keys must agree exactly -- the run is deterministic,
    so a recomputed cell that changed value is a signal, not noise."""
    import numpy as np, pandas as pd
    OUT_DIR = Path("/results") / out_subdir
    for sub, key, name in [
        ("shards", ["variant", "seed_idx", "res", "layer", "roi"], "bn_calibration_control.csv"),
        ("diagnostics", ["variant", "seed_idx", "diag_res", "layer"], "bn_calibration_diagnostics.csv"),
    ]:
        files = sorted((OUT_DIR / sub).glob("*.csv"))
        if not files:
            print(f"{sub}: nothing to merge"); continue
        df = pd.concat([pd.read_csv(str(f)) for f in files], ignore_index=True)
        dup = df.duplicated(subset=key, keep=False)
        if dup.any():
            num = [c for c in df.columns
                   if c not in key and pd.api.types.is_numeric_dtype(df[c])]
            bad = []
            for k, g in df[dup].groupby(key, sort=True):
                for c in num:
                    v = g[c].to_numpy(dtype=float)
                    fin = ~np.isnan(v)
                    if not (fin.all() or (~fin).all()):
                        bad.append((k, c, list(g[c])))
                    elif fin.all() and not np.allclose(v, v[0], rtol=0, atol=1e-12):
                        bad.append((k, c, list(g[c])))
            if bad:
                raise RuntimeError(
                    f"{sub}: {len(bad)} numeric disagreement(s) across duplicated keys.\n"
                    + "\n".join(f"  {k} col={c} values={vv}" for k, c, vv in bad[:20]))
            print(f"{sub}: duplicates all numerically identical -- safe to collapse")
        df = df.drop_duplicates(subset=key, keep="last").sort_values(key).reset_index(drop=True)
        df.to_csv(str(OUT_DIR / name), index=False)
        print(f"{sub}: {len(files)} file(s) -> {len(df)} rows -> {name}")
        if sub == "shards":
            pairs = df.groupby(["variant", "seed_idx"]).res.nunique()
            print(f"  complete (variant,seed) pairs with 6 resolutions: "
                  f"{int((pairs == 6).sum())} / {len(pairs)}")
    results_vol.commit()
    return str(OUT_DIR / "bn_calibration_control.csv")
