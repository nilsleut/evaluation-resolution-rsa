"""2608.12408 v3, section 3.4: low-level references, partial RSA, luminance-convergence ordering.

References are rebuilt exactly as in code/colour_statistic_control.py (MEANRGB, COLORHIST,
MEANLUM, PIXEL at 224 px) and code/cross_species/run_lowlevel_control_v2.py (Gabor energy
bank per resolution). Reference-vs-V1 and model-vs-V1 values are computed in the v2
convention (mean RDM, all pairs; validation against the printed v2 values) and in the v3
convention (per subject, cross-run). Model-vs-reference similarities do not involve brain
data and are unchanged (all pairs).

Partial RSA ("rank-based linear regression", Methods): ranks of the model vector and of the
brain vector are each regressed (OLS, with intercept) on the ranks of the reference
vector(s); the Pearson correlation of the residuals is the partial rho. v3: per subject on
cross-run pairs, then averaged over subjects.

Model sets: the six-condition ordering (rho = 0.94) and the colour-convergence rates use
the bnfix RDMs that v2 used (they include the non-reproducible A_aug realization); the
within-filter ordering uses the BN-calibration RDMs; gaps with BP use v12 (as in 3.1).

Output: results/paper_v3/numbers_v3_lowlevel.csv
"""
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compute_numbers import (REPO, FMRI_DIR, THINGS_DIR, KEEP, TRI, RES, SEEDS, RK, OLD, NEW,  # noqa: E402
                             brain, score, tri, v12, bnfix, bncal, seedstat)
sys.path.insert(0, str(HERE.parent))
import colour_statistic_control as C  # noqa: E402

C.FMRI_DIR, C.THINGS_DIR = FMRI_DIR, THINGS_DIR
ROWS = []


def add(key, quantity, printed, old, new, src, note=""):
    ROWS.append(dict(key=key, section="3.4", quantity=quantity, v2_printed=printed, v2_recomputed=old,
                     v3_value=new, source=src, note=note))


def rank_resid(v, Z):
    r = rankdata(v)
    X = np.column_stack([np.ones(len(r))] + [rankdata(z) for z in Z])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    return r - X @ beta


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def partial(m, roi, refs, conv, var):
    m = tri(m)
    k = KEEP if var == "cr" else slice(None)
    Z = [tri(r)[k] for r in refs]
    rm = rank_resid(m[k], Z)
    _, _, vecs = brain(roi, var)
    if conv == "persub":
        return float(np.mean([pearson(rm, rank_resid(v, Z)) for v in vecs]))
    return pearson(rm, rank_resid(np.mean(vecs, axis=0), Z))


def gabor_rdm(paths, px):
    import run_lowlevel_control_v2 as L
    from scipy.spatial.distance import pdist
    return pdist(L.gabor_features(paths, px), metric="correlation")


def exact_perm_p(x, y):
    obs = abs(spearmanr(x, y).correlation)
    n = len(x)
    hits = sum(abs(spearmanr(x, np.asarray(y)[list(p)]).correlation) >= obs - 1e-12 for p in permutations(range(n)))
    import math
    return hits / math.factorial(n)


def main():
    stim = C.load_stim_order("sub-01")
    paths = [C.find_img(s) for s in stim]
    assert all(p is not None for p in paths)
    refs, _, _ = C.build_references(paths)
    R = {k: tri(v) for k, v in refs.items()}

    # reference vs V1
    for k, printed in (("MEANLUM", 0.074), ("MEANRGB", 0.045), ("COLORHIST", 0.028), ("PIXEL", 0.030)):
        add(f"ref.{k}.V1", f"{k} vs V1", printed, score(R[k], "V1", *OLD), score(R[k], "V1", *NEW), "colour refs")

    # Gabor bank vs V1 over resolutions (v2: 0.018-0.037)
    sys.path.insert(0, str(HERE.parent / "cross_species"))
    g_old, g_new = [], []
    for px in RES:
        G = gabor_rdm([str(p) for p in paths], px)
        g_old.append(score(G, "V1", *OLD)); g_new.append(score(G, "V1", *NEW))
    add("gabor.V1.range", "Gabor bank vs V1, min..max over res", "0.018-0.037",
        (min(g_old), max(g_old)), (min(g_new), max(g_new)), "gabor refs")

    # untrained V1 at 224 and partials
    for tag, cv in (("old", OLD), ("new", NEW)):
        unt = np.mean([score(v12(RK["rnd"], 224, s), "V1", *cv) for s in SEEDS])
        p_lum = np.mean([partial(v12(RK["rnd"], 224, s), "V1", [R["MEANLUM"]], *cv) for s in SEEDS])
        p_pix = np.mean([score(v12(RK["rnd"], 224, s), "V1", *cv) - partial(v12(RK["rnd"], 224, s), "V1", [R["PIXEL"]], *cv)
                         for s in SEEDS])
        val = (lambda q: q) if tag == "old" else (lambda q: None)
        nv = (lambda q: q) if tag == "new" else (lambda q: None)
        add(f"unt.V1.224.{tag}", f"untrained Conv1->V1 @224 ({tag})", 0.075, val(unt), nv(unt), "v12")
        add(f"unt.partial_lum.224.{tag}", f"untrained partial | MEANLUM @224 ({tag})", 0.038, val(p_lum), nv(p_lum), "v12")
        add(f"unt.pixel_decrease.224.{tag}", f"untrained decrease when PIXEL partialled @224 ({tag})", 0.004,
            val(p_pix), nv(p_pix), "v12")
        for px in (32, 224):
            def gap(refs_):
                return [partial(v12(RK["rnd"], px, s), "V1", refs_, *cv) - partial(v12(RK["bp"], px, s), "V1", refs_, *cv)
                        for s in SEEDS]
            j = seedstat(gap([R[k] for k in ("MEANRGB", "COLORHIST", "MEANLUM", "PIXEL")]))
            add(f"gap.joint4.{px}.{tag}", f"Random-BP gap, 4 refs partialled @{px} ({tag})",
                {32: -0.020, 224: 0.036}[px], val(j), nv(j), "v12")
            if px == 224:
                l = seedstat(gap([R["MEANLUM"]]))
                add(f"gap.lum.224.{tag}", f"Random-BP gap, MEANLUM partialled @224 ({tag})", 0.026, val(l), nv(l), "v12")

    # six-condition ordering (bnfix, incl. A_aug): change 32->224 in MEANLUM similarity vs in V1 alignment
    six = ["rnd", "cal", "bp", "fa", "pc", "stdp"]
    dl = {r: np.mean([spearmanr(tri(bnfix(RK[r], 224, s)), R["MEANLUM"]).correlation
                      - spearmanr(tri(bnfix(RK[r], 32, s)), R["MEANLUM"]).correlation for s in SEEDS]) for r in six}
    for tag, cv in (("old", OLD), ("new", NEW)):
        dv = {r: np.mean([score(bnfix(RK[r], 224, s), "V1", *cv) - score(bnfix(RK[r], 32, s), "V1", *cv) for s in SEEDS])
              for r in six}
        rho = spearmanr([dl[r] for r in six], [dv[r] for r in six]).correlation
        add(f"six.order.{tag}", f"Spearman(d MEANLUM sim, d V1), 6 conditions ({tag})", 0.94,
            rho if tag == "old" else None, rho if tag == "new" else None, "bnfix (incl. A_aug)")

    # within-filter: five BN variants (bncal RDMs)
    V = {"random": "Random (identity BN)", "a": "A", "b": "B", "c": "C", "d": "D"}
    dref = {k: {v: [spearmanr(tri(bncal(v, 224, s)), R[k]).correlation - spearmanr(tri(bncal(v, 32, s)), R[k]).correlation
                    for s in SEEDS] for v in V} for k in ("MEANLUM", "MEANRGB", "PIXEL")}
    for tag, cv in (("old", OLD), ("new", NEW)):
        dv = {v: [score(bncal(v, 224, s), "V1", *cv) - score(bncal(v, 32, s), "V1", *cv) for s in SEEDS] for v in V}
        val = (lambda q: q) if tag == "old" else (lambda q: None)
        nv = (lambda q: q) if tag == "new" else (lambda q: None)
        x = [np.mean(dref["MEANLUM"][v]) for v in V]; y = [np.mean(dv[v]) for v in V]
        r5 = spearmanr(x, y).correlation
        add(f"within.lum.rho5.{tag}", f"within-filter Spearman, MEANLUM, 5 variants ({tag})", 1.00, val(r5), nv(r5), "bncal")
        p5 = exact_perm_p(x, y)
        add(f"within.lum.p5.{tag}", f"exact permutation p ({tag})", 0.017, val(p5), nv(p5), "bncal")
        r25 = spearmanr(np.concatenate([dref["MEANLUM"][v] for v in V]), np.concatenate([dv[v] for v in V])).correlation
        add(f"within.lum.rho25.{tag}", f"within-filter Spearman, 25 variant-seed points ({tag})", 0.87, val(r25), nv(r25), "bncal")
        for k, pr in (("MEANRGB", -0.30), ("PIXEL", -0.70)):
            rk = spearmanr([np.mean(dref[k][v]) for v in V], y).correlation
            add(f"within.{k}.rho5.{tag}", f"within-filter Spearman, {k} ({tag})", pr, val(rk), nv(rk), "bncal")
        wv = [spearmanr(dref["MEANLUM"][v], dv[v]).correlation for v in V]
        wp = [float(np.corrcoef(dref["MEANLUM"][v], dv[v])[0, 1]) for v in V]
        add(f"within.seed.range.{tag}", f"within-variant across-seed corr range, Spearman | Pearson ({tag})", "-0.20..+0.90",
            val((min(wv), max(wv), min(wp), max(wp))), nv((min(wv), max(wv), min(wp), max(wp))), "bncal")
        b = seedstat(dv["b"])
        add(f"within.B.dV1.{tag}", f"B: V1 change 32->224 mean/SEM/pos ({tag})", "+0.011, 4/5", val(b), nv(b), "bncal")
        idv = seedstat(dv["random"])
        add(f"within.identity.dV1.{tag}", f"identity: V1 change 32->224 ({tag})", "+0.0110", val(idv), nv(idv), "bncal")

    df = pd.DataFrame(ROWS)
    df.to_csv(REPO / "results" / "paper_v3" / "numbers_v3_lowlevel.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print(df.drop(columns=["source", "note"]).to_string(index=False))


if __name__ == "__main__":
    main()
