"""Step 2a: model RDMs for all 30 v12 checkpoints x 6 resolutions x {NATIVE, UPSAMPLED}.

Feature extraction is v12's own `extract_features` (model.eval() + assert_bn_eval +
get_features in batches of 128) and v12's `compute_rdm` (correlation distance), lifted
unchanged. The only substitution: the image transform is applied once per (arm, px) and
the tensor stack is reused for all 30 models (CachedDS), instead of re-decoding the 720
JPEGs per model.

Output: data/upsampling/rdms/{arm}/res{px}/seed_{i}/rdm_{rule}_{layer}.npy, float64 upper
triangle in TRI order. Resumable: a (arm, px, rule, seed) whose four files exist is skipped.
"""
import time

import numpy as np
import torch

from common_up import (ARMS, RES, RULES, SEEDS, LAYERS, TRI, stim_order, namespaces, transform,
                       load_v12_model, rdm_path)


def main():
    torch.set_num_threads(12)
    ns12, ns11 = namespaces()
    from PIL import Image
    paths = [ns12["find_img"](s) for s in stim_order()]
    assert all(p is not None for p in paths)
    t0 = time.time()
    for arm in ARMS:
        for px in RES:
            todo = [(r, si) for r in RULES for si in range(len(SEEDS))
                    if not all(rdm_path(arm, px, si, r, l).exists() for l in LAYERS)]
            if not todo:
                continue
            tf = transform(ns12, ns11, arm, px)
            X = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
            print(f"[{time.time()-t0:6.0f}s] {arm} {px}px: stimuli {tuple(X.shape)}, "
                  f"{len(todo)} models", flush=True)
            for rule, si in todo:
                t1 = time.time()
                torch.manual_seed(SEEDS[si])
                m = load_v12_model(ns12, rule, si)
                feats = ns12["extract_features"](m, list(range(len(paths))), X)
                for layer, f in zip(LAYERS, feats):
                    rdm = ns12["compute_rdm"](f)
                    assert np.isfinite(rdm).all(), (arm, px, rule, si, layer)
                    p = rdm_path(arm, px, si, rule, layer)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    np.save(p, np.asarray(rdm[TRI], np.float64))
                print(f"    {rule:<32} s{si}  {time.time()-t1:5.1f}s", flush=True)
            del X
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
