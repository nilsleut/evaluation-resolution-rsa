"""Step 0 + Step 1 checks for (b2). No model evaluation beyond tiny probes.

  0.1  training config of the benchmark vs v12 (batches/epoch, epochs)
  0.2  checkpoint inventory (rule x seed), and why the benchmark counted 25
  0.3  PC/STDP (and all others) load into eval mode; BN guard passes; extraction does
       not mutate BN running stats
  0.4  arm definitions as executed; UPSAMPLED@32 == NATIVE@32 bit-exactly
  1    stimulus order + fMRI RDMs identical between learning-rules-rsa and the Modal
       copy v12 read; pair set P identical to crossrun's; |P| = 210,205

Output: results/upsampling/step0.json. Raises (STOP) on any failed check.
"""
import hashlib
import importlib.util
import json
import sys

import numpy as np

from common_up import (REPO, EXT, FMRI_DIR, CKPT_DIR, RESULTS, V12, V11, RULES, SEEDS, ARMS,
                       SUBJECTS, ROIS, N_STIM, N_PAIRS_CR, TRI, rule_key, stim_order, run_labels,
                       pair_set, namespaces, transform, load_v12_model, load_defs)

MODAL_720 = REPO / "data" / "modal_outputs_720"   # files fetched from burstprop-data:/outputs_720


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    import torch
    out = {}

    # ── 0.1 config ──
    c12 = load_defs(V12, "run_resolution_control", set(), {})
    c11 = load_defs(V11, "run_resolution_control", set(), {})
    bench_src = (REPO / "bench" / "bench_compute.py").read_text(encoding="utf-8")
    assert "N_EPOCHS_FULL = 40" in bench_src and 'ns["N_CIFAR"] // ns["BATCH"]' in bench_src
    cfg = {k: {"N_CIFAR": c["N_CIFAR"], "BATCH": c["BATCH"], "N_EPOCHS": c["N_EPOCHS"],
               "batches_per_epoch_drop_last": c["N_CIFAR"] // c["BATCH"]}
           for k, c in (("v12", c12), ("v11", c11))}
    cfg["bench"] = {"batches_per_epoch": c12["N_CIFAR"] // c12["BATCH"], "BATCH": c12["BATCH"],
                    "N_EPOCHS": 40, "source": "bench_compute.py reads N_CIFAR/BATCH from the v12 "
                                               "namespace; N_EPOCHS_FULL = 40"}
    b12 = cfg["v12"]["batches_per_epoch_drop_last"] * cfg["v12"]["N_EPOCHS"]
    bb = cfg["bench"]["batches_per_epoch"] * cfg["bench"]["N_EPOCHS"]
    cfg["projection_factor_v12_over_bench"] = b12 / bb
    out["0.1_config"] = cfg
    assert cfg["projection_factor_v12_over_bench"] == 1.0

    # ── 0.2 checkpoints ──
    inv = []
    for r in RULES:
        for si, sd in enumerate(SEEDS):
            f = CKPT_DIR / f"model_weights_{rule_key(r)}_seed{si}.pt"
            inv.append({"rule": r, "seed": sd, "seed_idx": si, "file": f.name,
                        "exists": f.exists(), "sha256": sha(f)[:16] if f.exists() else None})
    assert all(x["exists"] for x in inv), "missing checkpoint"
    assert len({x["sha256"] for x in inv}) == len(inv), "duplicate checkpoints"
    present = sorted(p.name for p in CKPT_DIR.glob("*.pt"))
    out["0.2_checkpoints"] = {
        "n_present": len(present), "n_expected": len(RULES) * len(SEEDS), "inventory": inv,
        "bench_b2_count": "25 = RULES5 (5 rules) x 5 seeds; bench_compute.py RULES5 omits "
                          "'Random Weights (BN-calibrated)', the 6th v12 condition. Resolution is "
                          "not a checkpoint dimension (training is resolution-independent).",
    }

    # ── 0.3 eval mode ──
    ns12, ns11 = namespaces()
    from bn_guard import assert_bn_eval, iter_bn
    x = torch.randn(4, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    bn_rep = []
    for r in RULES:
        for si in range(len(SEEDS)):
            m = load_v12_model(ns12, r, si)
            assert_bn_eval(m, context=f"step0/{r}/{si}")
            bns = iter_bn(m)
            before = [(b.running_mean.clone(), b.running_var.clone()) for b in bns]
            f1 = m.get_features(x); f2 = m.get_features(x)
            same = all(torch.equal(a, b) for a, b in zip(f1, f2))
            unchanged = all(torch.equal(b.running_mean, rm) and torch.equal(b.running_var, rv)
                            for b, (rm, rv) in zip(bns, before))
            identity = all(torch.all(b.running_mean == 0) and torch.all(b.running_var == 1) for b in bns)
            assert same and unchanged
            if si == 0 or r in ("Predictive Coding", "STDP"):
                bn_rep.append({"rule": r, "seed_idx": si, "n_bn": len(bns),
                               "all_bn_training_false": all(not b.training for b in bns),
                               "features_deterministic": same, "running_stats_unchanged": unchanged,
                               "bn_identity_stats": bool(identity),
                               "conv1_bn_running_var_mean": float(bns[0].running_var.mean())})
    out["0.3_bn_eval"] = bn_rep

    # ── 0.4 arms ──
    from PIL import Image
    order = stim_order()
    paths = [ns12["find_img"](s) for s in order]
    assert all(p is not None for p in paths), "unresolved THINGS image"
    arms = {a: {px: repr(transform(ns12, ns11, a, px)) for px in (32, 224)} for a in ARMS}
    t_n, t_u = transform(ns12, ns11, "NATIVE", 32), transform(ns12, ns11, "UPSAMPLED", 32)
    n_eq = 0
    for p in paths:
        im = Image.open(p).convert("RGB")
        n_eq += torch.equal(t_n(im), t_u(im))
    # v11 NATIVE (explicit bilinear+antialias) vs v12 NATIVE (defaults), on PIL input
    v11n = ns11["_tf"](224, "NATIVE")
    eq224 = sum(torch.equal(v11n(Image.open(p).convert("RGB")),
                            transform(ns12, ns11, "NATIVE", 224)(Image.open(p).convert("RGB")))
                for p in paths[:40])
    up224 = transform(ns12, ns11, "UPSAMPLED", 224)(Image.open(paths[0]).convert("RGB"))
    out["0.4_arms"] = {"transforms": arms, "n_images_resolved": len(paths),
                       "UP32_equals_NAT32_bitexact": f"{n_eq}/{len(paths)}",
                       "v11_NATIVE_equals_v12_NATIVE_224": f"{eq224}/40",
                       "UPSAMPLED_224_shape": list(up224.shape),
                       "matches_spec": True}
    assert n_eq == len(paths) and eq224 == 40

    # ── 1 stimulus order, fMRI RDMs, pair set ──
    files = [f"stim_order_{s}.txt" for s in SUBJECTS] + \
            [f"fmri_rdm_{r}_{s}.npy" for r in ROIS for s in SUBJECTS]
    same = {f: sha(FMRI_DIR / f) == sha(MODAL_720 / f) for f in files}
    assert all(same.values()), same
    sym = {f"{r}_{s}": bool(np.array_equal(a := np.load(FMRI_DIR / f"fmri_rdm_{r}_{s}.npy"), a.T))
           for r in ROIS for s in SUBJECTS}
    runs = run_labels()
    rp, cp, keep = pair_set(None, runs)
    spec = importlib.util.spec_from_file_location("common_cr", EXT / "scripts/crossrun/common_cr.py")
    sys.path.insert(0, str(EXT / "scripts/crossrun"))
    cr = importlib.util.module_from_spec(spec); spec.loader.exec_module(cr)
    rp2, cp2, keep2 = cr.pair_set(None, cr.run_labels())
    ident = bool(np.array_equal(rp, rp2) and np.array_equal(cp, cp2) and np.array_equal(keep, keep2))
    out["1_pairs"] = {"stim_order_sha256": sha(FMRI_DIR / "stim_order_sub-01.txt")[:16],
                      "stim_order_identical_to_modal_copy": all(v for k, v in same.items() if "stim" in k),
                      "fmri_rdms_identical_to_modal_copy": all(v for k, v in same.items() if "fmri" in k),
                      "fmri_rdms_symmetric": sym, "n_stim": len(order),
                      "n_pairs_all": len(TRI[0]), "n_pairs_cr": int(len(rp)),
                      "identical_to_crossrun_pair_set": ident}
    assert ident and len(rp) == N_PAIRS_CR and len(order) == N_STIM
    assert stim_order() == cr.stim_order()

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "step0.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "0.2_checkpoints"}, indent=1)[:6000])
    print("STEP 0 / STEP 1 CHECKS PASSED")


if __name__ == "__main__":
    main()
