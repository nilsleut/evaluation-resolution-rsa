"""Shared paths and helpers for experiment (b2): upsampling test, eval-only from the
30 v12 checkpoints, local CPU.

Arms (definitions lifted unchanged from the Modal scripts, see step1_extract.py):
  NATIVE     v12 `_tf(px)`:            Resize(px) -> CenterCrop(px)
  UPSAMPLED  v11 `_tf(px, "UPSAMPLED")`: Resize(32) -> CenterCrop(32) -> Resize(px)

Pair set P (cross-run): stimulus pairs that lie in different runs for EVERY subject,
exactly as Projekte_1/learning-rules-rsa/scripts/crossrun/common_cr.py::pair_set
(210,205 of 258,840 pairs for the identity sample). step0_checks.py asserts that the
two implementations give identical index arrays.

RSA = Spearman on the strict upper triangle, as centred unit-norm average ranks, so
rho(a, b) == ranks(a) @ ranks(b) (learning-rules-rsa noise_ceiling_v2 convention).
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CODE = REPO / "code"
V12 = CODE / "learning_rules_v12_allseeds_modal.py"
V11 = CODE / "learning_rules_v11_upsample_modal.py"

DESKTOP = REPO.parents[1]
EXT = DESKTOP / "Projekte_1" / "learning-rules-rsa"          # cross-run analysis repo
FMRI_DIR = EXT / "outputs_720"                               # == burstprop-data:/outputs_720 (sha-checked)
META_DIR = DESKTOP / "Projekte_1" / "RSA" / "Datensatz"      # THINGS-fMRI stimulus metadata
THINGS_DIR = META_DIR / "images_THINGS" / "object_images"    # == burstprop-data:/object_images
CKPT_DIR = REPO / "data" / "v12_checkpoints" / "checkpoints"  # learning-rules-rsa:/outputs_allseeds/checkpoints
V12_CSV = REPO / "data" / "v12_rsa_resolution_sweep.csv"      # learning-rules-rsa:/outputs_allseeds/rsa_resolution_sweep.csv
RDM_DIR = REPO / "data" / "upsampling" / "rdms"              # model RDM triangles (gitignored)
RESULTS = REPO / "results" / "upsampling"

SUBJECTS = ["sub-01", "sub-02", "sub-03"]
ROIS = ["V1", "V2", "LOC", "IT"]
LAYERS = ["Conv1", "Conv2", "Conv3", "FC1"]
# v12 LAYER_ROI_FIXED, flattened
LAYER_ROI = [("Conv1", "V1"), ("Conv1", "V2"), ("Conv2", "V1"), ("Conv2", "V2"),
             ("Conv3", "LOC"), ("FC1", "IT")]
RULES = ["Random Weights", "Backprop", "Feedback Alignment", "Predictive Coding", "STDP",
         "Random Weights (BN-calibrated)"]
SEEDS = [42, 123, 456, 789, 1337]
ARMS = ["NATIVE", "UPSAMPLED"]
RES = [32, 64, 96, 128, 160, 224]
N_STIM = 720
N_RUNS = 10
TRI = np.triu_indices(N_STIM, 1)
N_PAIRS_ALL = len(TRI[0])
N_PAIRS_CR = 210205

BOOT_SEED, N_BOOT = 20261002, 1000      # = noise_ceiling_v2 / crossrun resamples


def rule_key(rule):
    return rule.lower().replace(" ", "_")


def ranks(v):
    """Centred, unit-norm average ranks, so spearman(a, b) == ranks(a) @ ranks(b)."""
    r = rankdata(v)
    r -= r.mean()
    return r / np.linalg.norm(r)


def stim_order():
    with open(FMRI_DIR / "stim_order_sub-01.txt") as f:
        return [l.strip() for l in f if l.strip()]


def run_labels():
    """Run (within the session the 720 come from) of each of the 720 stimuli, per subject."""
    order = stim_order()
    out = {}
    for s in SUBJECTS:
        st = pd.read_csv(META_DIR / f"{s}_task-things_stimulus-metadata.csv")
        m = st.set_index("stimulus").loc[order]
        assert m.session.nunique() == 1
        out[s] = m.run.values
    return out


def pair_set(stims=None, runs=None):
    """(rows, cols, keep) of the pairs in P for a stimulus sample; keep is a mask on TRI.
    Same logic as learning-rules-rsa common_cr.pair_set."""
    runs = runs or run_labels()
    s = np.arange(N_STIM) if stims is None else np.asarray(stims)
    a, b = s[TRI[0]], s[TRI[1]]
    keep = a != b
    for r in runs.values():
        keep &= r[a] != r[b]
    return a[keep], b[keep], keep


def tri_index(a, b):
    """Index into the TRI-ordered upper triangle for pairs (a, b), a != b, any order."""
    i, j = np.minimum(a, b), np.maximum(a, b)
    return i * N_STIM - i * (i + 1) // 2 + (j - i - 1)


def load_fmri_tri(roi):
    """Per-subject fMRI RDM upper triangles (TRI order) for one ROI."""
    out = []
    for s in SUBJECTS:
        x = np.load(FMRI_DIR / f"fmri_rdm_{roi}_{s}.npy")
        assert x.shape == (N_STIM, N_STIM)
        out.append(np.asarray(x[TRI], float))
    return out


def rdm_path(arm, px, seed_idx, rule, layer):
    return RDM_DIR / arm / f"res{px}" / f"seed_{seed_idx}" / f"rdm_{rule_key(rule)}_{layer}.npy"


def boot_idx():
    """The noise_ceiling_v2 / crossrun stimulus resamples; sha-checked against that repo."""
    import hashlib, json
    boot = np.random.default_rng(BOOT_SEED).integers(0, N_STIM, size=(N_BOOT, N_STIM))
    v2 = json.loads((EXT / "results/noise_ceiling_v2/step3c_summary.json").read_text())
    assert hashlib.sha256(boot.tobytes()).hexdigest() == v2["boot_idx_sha256"], "resamples differ"
    return boot


# ── Lifting the original definitions out of the Modal scripts ───────────────────
# Same mechanism as bench/bench_compute.py: the models, transforms and RSA helpers live
# inside the Modal function body, so they are extracted by AST and exec'd with the
# source file as filename. Nothing under code/ is modified.

def _assigned_names(node):
    return {t.id for t in node.targets if isinstance(t, ast.Name)}


def load_defs(path, func, wanted, ns):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    modal_names = {"modal", "app", "image", "data_vol", "results_vol"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            used = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            if not used & modal_names and not _assigned_names(node) & modal_names:
                body.append(node)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func)
    found = set()
    for s in fn.body:
        if isinstance(s, (ast.FunctionDef, ast.ClassDef)) and s.name in wanted:
            body.append(s); found.add(s.name)
        elif isinstance(s, ast.Assign) and _assigned_names(s) & wanted:
            body.append(s); found |= _assigned_names(s) & wanted
    missing = set(wanted) - found
    if missing:
        raise RuntimeError(f"{Path(path).name}:{func}: definitions not found: {sorted(missing)}")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return ns


class CachedDS:
    """Stand-in for v12's ImgDS: `paths` are indices, `t` is the pre-transformed tensor
    stack (the same transform applied per image, done once per (arm, px) for all 30
    models instead of once per model)."""
    def __init__(self, paths, t): self.paths, self.X = paths, t
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return self.X[self.paths[i]], i


def namespaces():
    """(ns12, ns11): v12 models/feature extraction/NATIVE transform, v11 UPSAMPLED transform."""
    import random
    import torch, torch.nn as nn, torch.nn.functional as F
    import torchvision, torchvision.transforms as T
    from torch.utils.data import DataLoader as _DL, Subset, Dataset
    from scipy.stats import spearmanr
    from scipy.spatial.distance import pdist, squareform
    from PIL import Image
    sys.path.insert(0, str(CODE))
    from bn_guard import assert_bn_eval

    def DataLoader(*a, **kw):       # noqa: N802 -- cached tensors: no worker processes
        kw["num_workers"] = 0
        return _DL(*a, **kw)

    base = dict(np=np, torch=torch, nn=nn, F=F, T=T, torchvision=torchvision, random=random,
                DataLoader=DataLoader, Subset=Subset, Dataset=Dataset, spearmanr=spearmanr,
                pdist=pdist, squareform=squareform, Image=Image, pd=pd, Path=Path,
                assert_bn_eval=assert_bn_eval, DEVICE=torch.device("cpu"),
                THINGS_DIR=THINGS_DIR, FMRI_DIR=FMRI_DIR)
    ns12 = load_defs(V12, "run_resolution_control", {
        "make_conv_block", "_pool_for_fc", "BP_CNN", "FAConvFunction", "FAConv2d",
        "make_fa_conv_block", "FA_CNN", "PC_CNN", "STDP_Conv", "STDP_CNN",
        "find_img", "compute_rdm", "extract_features", "_tf"}, dict(base))
    ns12["ImgDS"] = CachedDS
    ns11 = load_defs(V11, "run_resolution_control", {"_BILIN", "_NORM", "_tf"}, dict(base))
    return ns12, ns11


def transform(ns12, ns11, arm, px):
    return ns12["_tf"](px) if arm == "NATIVE" else ns11["_tf"](px, "UPSAMPLED")


def load_v12_model(ns12, rule, seed_idx):
    """Rebuild the v12 class for `rule` and load its checkpoint. RAISES on any mismatch."""
    import torch
    f = CKPT_DIR / f"model_weights_{rule_key(rule)}_seed{seed_idx}.pt"
    if not f.exists():
        raise FileNotFoundError(f)
    sd = torch.load(f, map_location="cpu", weights_only=True)
    if rule == "STDP":
        m = ns12["STDP_CNN"]().to_device(torch.device("cpu"))
        m.L1.conv.weight.data = sd["conv1.weight"]
        m.L2.conv.weight.data = sd["conv2.weight"]
        m.L3.conv.weight.data = sd["conv3.weight"]
        for i, bn in enumerate([m.bn1, m.bn2, m.bn3], start=1):
            bn.weight.data = sd[f"bn{i}.weight"]; bn.bias.data = sd[f"bn{i}.bias"]
            bn.running_mean.copy_(sd[f"bn{i}.running_mean"])
            bn.running_var.copy_(sd[f"bn{i}.running_var"])
        for nm in ("fc1", "fc2"):
            getattr(m, nm).weight.data = sd[f"{nm}.weight"]
            getattr(m, nm).bias.data = sd[f"{nm}.bias"]
        expect = 3 + 3 * 4 + 4
        if len(sd) != expect:
            raise KeyError(f"{f.name}: {len(sd)} tensors, expected {expect}")
    else:
        cls = {"Feedback Alignment": "FA_CNN", "Predictive Coding": "PC_CNN"}.get(rule, "BP_CNN")
        m = ns12[cls]()
        m.load_state_dict(sd, strict=True)
    m.eval()
    return m
