"""
learning_rules_v12_allseeds_modal.py
====================================
v12 vs v10: ONE change, in the OUTPUT path only. v10 saved a checkpoint for
`seed_idx == 0` alone, so only 6 of the 30 (rule, seed) models were ever written to
disk. The ridge/readout study makes the seed the unit of inference, so it needs all
30. v12 writes a seed-suffixed checkpoint for every (rule, seed).

WHAT IS DELIBERATELY NOT CHANGED
--------------------------------
The training path is byte-for-byte identical to v10, because the whole point of this
run is a bit-identity gate: seed 42 must reproduce the six existing checkpoints
exactly (code/ridge/determinism_gate.py). Any change to model construction, seeding,
the CIFAR loader, the optimizers or the epoch loop would make a mismatch
uninterpretable — we could not tell nondeterminism from our own edit. So:

  * no change to torch.manual_seed / np.random.seed / random.seed placement;
  * no change to get_cifar_loader, the architectures, or the train_* functions;
  * no change to the evaluation path, so the CSV this run produces comes from the
    ORIGINAL evaluation function and can be compared directly against the published
    results/bnfix_sweep.csv rather than against a reimplementation of it.

Because all 30 pairs are recomputed, that comparison covers the whole 1,080-row
published table, not only seed 42. The checkpoint bit-identity gate is limited to
seed 42 because that is the only seed for which a reference checkpoint exists.

THE FOUR EDITS, exhaustively
---------------------------
  1. OUT_DIR default: outputs_bnfix -> outputs_allseeds (a fresh dir; nothing
     resumes from, or overwrites, the published run).
  2. The checkpoint block: `if seed_idx == 0:` removed; filenames gain _seed{n}.
  3. The checkpoint block's `except Exception: print(...)` becomes a raise, plus a
     reload-verify. Under v10 a checkpoint was a nice-to-have and swallowing the
     error was reasonable. Here the checkpoints ARE the deliverable, so a silently
     skipped one is exactly the failure shape that hid the original BatchNorm
     defect: the run looks successful and the artifact is missing.
  4. App name: learning-rules-rsa -> learning-rules-rsa-allseeds, so deploying this
     does not replace the published v10 deployment that spawn_runs.py resolves by
     name. The results volume is unchanged.

Everything below this docstring, apart from those four edits, is v10. Verify with:
    diff code/learning_rules_v10_sweep_modal.py code/learning_rules_v12_allseeds_modal.py

---- v10 docstring follows ----

v10 vs v9: finalize_sweep no longer collapses a duplicated
(rule, seed_idx, res, layer, roi) key by keeping one row. The run is seeded and
deterministic, so two computations of the same cell must agree exactly; if they do
not, a respawned shard changed a number and that is a signal, not noise. The merge
now verifies every numeric column across duplicates and raises on disagreement.
Everything else is identical to v9.

Paper-1 endpoint RSA (epoch 40) for all 5 conditions x 5 seeds, evaluated across a
THINGS resolution SWEEP [32,64,96,128,160,224] -> 5-seed resolution sweep (error bars).

Why: the V1 rule ranking depends on eval resolution. This run gives the full
5-seed curve (alignment vs eval resolution) with error bars -- the publication
version of the single-seed run_resolution_sweep_human.py. Trains once per
(seed, rule); evaluates at every sweep resolution (training is res-independent).

Uses the EXACT v8 implementations of BP/FA/PC/STDP (mirrored from
training_dynamics_rsa_modal.py).

SETUP:
  python -m modal volume create burstprop-data        (already exists; has the data)
  python -m modal volume create learning-rules-rsa

DATA (on burstprop-data volume):
  outputs_720/       fMRI RDMs + stim_order files
  object_images/     THINGS images

RUN:
  python -m modal run learning_rules_v10_sweep_modal.py

DOWNLOAD:
  python -m modal volume get learning-rules-rsa outputs_bnfix ./learning_rules_outputs_bnfix --force
  # key file: outputs_bnfix/rsa_resolution_sweep.csv

BN-MODE FIX (supersedes the /results/outputs run):
  PC_CNN and STDP_CNN used to define `def eval(self): pass`, so extract_features()'s
  model.eval() silently did nothing for them: their BatchNorms stayed in training mode
  and normalised THINGS eval batches by *their own* per-batch statistics at whatever
  resolution was being swept -- while Random/Backprop/FA used frozen CIFAR-32px running
  stats. That is a rule- AND resolution-correlated confound in a resolution claim.
  Fixed by giving both classes working train()/eval(), setting the mode explicitly at
  each phase of the main loop, and asserting BN.training is False inside
  extract_features(). Results go to a FRESH dir so nothing resumes from the old run.
"""

import modal
import time
from pathlib import Path

# EDIT 4/4 vs v10: a distinct app name. Deploying under v10's name would replace the
# published deployment, and spawn_runs.py resolves
# modal.Function.from_name("learning-rules-rsa", "run_resolution_control") -- which
# would then silently point at v12. The results VOLUME is unchanged; only the app
# differs, so outputs still land beside the published ones.
app = modal.App("learning-rules-rsa-allseeds")

data_vol    = modal.Volume.from_name("burstprop-data", create_if_missing=True)
results_vol = modal.Volume.from_name("learning-rules-rsa", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "torchvision", "numpy", "scipy", "pandas", "Pillow", "matplotlib")
    .add_local_python_source("bn_guard")        # shared BatchNorm mode guards
)

# ── Constants matching v8 ─────────────────────────────────────────────────
SUBJECTS   = ["sub-01", "sub-02", "sub-03"]
ROIS       = ["V1", "V2", "LOC", "IT"]          # Paper 1 ROIs
LAYERS     = ["Conv1", "Conv2", "Conv3", "FC1"]
RULES      = ["Random Weights", "Backprop", "Feedback Alignment", "Predictive Coding", "STDP",
              "Random Weights (BN-calibrated)"]   # appended last: order of the 5 originals is untouched
RANDOM_RULES = ("Random Weights", "Random Weights (BN-calibrated)")   # no weight updates
N_BN_CALIB = 50                                 # forward-only batches for BN-stat calibration
COMPUTE_BOOTSTRAP_CI = False                    # OFF: CIs rebuilt on CPU by recompute_bootstrap_ci.py
N_BOOT     = 500                                # bootstrap iterations (used by the offline script too)
SEEDS      = [42, 123, 456, 789, 1337]
N_EPOCHS   = 40
IMG_SIZE   = 224                                # primary eval resolution
CTRL_SIZE  = 32                                 # control = training resolution
BATCH      = 128
LR         = 1e-3
N_CIFAR    = 8000
C1, C2, C3 = 32, 64, 128
FC1_DIM    = 512
N_CLS      = 10
FEAT_SIZE  = 4
FC1_IN     = C3 * FEAT_SIZE * FEAT_SIZE
T_INF      = 10
LR_R       = 0.02
LR_W       = 1e-4
A_P        = 0.003
A_M        = 0.003
TAU_P      = 20.0
TAU_M      = 20.0
T_SIM      = 10

# fixed layer->ROI mapping (Paper 1, Table 2)
LAYER_ROI_FIXED = {
    "Conv1": (0, ["V1", "V2"]),
    "Conv2": (1, ["V1", "V2"]),
    "Conv3": (2, ["LOC"]),
    "FC1":   (3, ["IT"]),
}


@app.function(
    image=image, gpu="T4", timeout=60*60*24,
    volumes={"/data": data_vol, "/results": results_vol},
)
def run_resolution_control(only_rule: str = None, only_seed: int = None,
                           out_subdir: str = None, force_recompute: bool = False,
                           shard_suffix: str = ""):
    """Full sweep, or a single (rule, seed) shard.

    Sharded mode (only_rule and only_seed given) writes this pair's rows to their own
    file under OUT_DIR/shards/ instead of the cumulative partial CSV, so 30 concurrent
    calls cannot clobber each other's writes. finalize_sweep() merges them afterwards.
    The pre-existing rsa_resolution_sweep_partial.csv is READ for resume, never written.

    force_recompute ignores the resume set, so an already-complete pair is recomputed.
    shard_suffix writes the rows to rows_<rule>_seed<n><suffix>.csv instead of the
    canonical name -- together they give a determinism re-check that lands beside the
    original shard rather than overwriting it.
    """
    import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    import torchvision, torchvision.transforms as T, random
    from torch.utils.data import DataLoader, Subset, Dataset
    from scipy.stats import spearmanr
    from scipy.spatial.distance import pdist, squareform
    from PIL import Image
    import pandas as pd
    from bn_guard import assert_bn_eval

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    FMRI_DIR   = Path("/data/outputs_720")
    THINGS_DIR = Path("/data/object_images")
    CIFAR_DIR  = Path("/results/cifar_data")
    # EDIT 1/3 vs v10: fresh dir, so this run neither resumes from nor overwrites the
    # published outputs_bnfix.
    OUT_DIR    = Path("/results/" + (out_subdir or "outputs_allseeds"))
    RDM_DIR    = OUT_DIR / "rdms"
    CKPT_DIR   = OUT_DIR / "checkpoints"
    for d in [OUT_DIR, RDM_DIR, CKPT_DIR, CIFAR_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    assert FMRI_DIR.exists(),   f"fMRI data not at {FMRI_DIR}"
    assert THINGS_DIR.exists(), f"THINGS not at {THINGS_DIR}"

    # ── Architectures (exact v8) ──────────────────────────────────────────
    def make_conv_block(in_c, out_c):
        return nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                             nn.BatchNorm2d(out_c), nn.ReLU(True), nn.MaxPool2d(2))

    def _pool_for_fc(c3):
        return F.adaptive_avg_pool2d(c3, FEAT_SIZE)

    class BP_CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1=make_conv_block(3,C1); self.conv2=make_conv_block(C1,C2)
            self.conv3=make_conv_block(C2,C3)
            self.fc1=nn.Linear(FC1_IN,FC1_DIM); self.fc2=nn.Linear(FC1_DIM,N_CLS)
            self.drop=nn.Dropout(0.3)
        def forward(self,x):
            x=self.conv3(self.conv2(self.conv1(x)))
            return self.fc2(self.drop(F.relu(self.fc1(x.view(x.size(0),-1)))))
        def get_features(self,x):
            with torch.no_grad():
                c1=self.conv1(x);c2=self.conv2(c1);c3=self.conv3(c2)
                h1=F.relu(self.fc1(_pool_for_fc(c3).view(c3.size(0),-1)))
            return c1.mean([2,3]),c2.mean([2,3]),c3.mean([2,3]),h1

    class FAConvFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,W,b,B,stride,padding):
            ctx.save_for_backward(x,W,b,B); ctx.stride=stride; ctx.padding=padding
            return F.conv2d(x,W,b,stride=stride,padding=padding)
        @staticmethod
        def backward(ctx,grad_output):
            x,W,b,B=ctx.saved_tensors; s,p=ctx.stride,ctx.padding
            gW=torch.nn.grad.conv2d_weight(x,W.shape,grad_output,stride=s,padding=p)
            gb=grad_output.sum([0,2,3]) if b is not None else None
            gx=F.conv_transpose2d(grad_output,B,stride=s,padding=p)
            return gx,gW,gb,None,None,None

    class FAConv2d(nn.Module):
        def __init__(self,in_c,out_c,ks=3,stride=1,padding=1):
            super().__init__()
            self.W=nn.Parameter(nn.init.kaiming_normal_(torch.empty(out_c,in_c,ks,ks)))
            self.b=nn.Parameter(torch.zeros(out_c))
            self.register_buffer("B_feedback",nn.init.xavier_normal_(torch.randn(out_c,in_c,ks,ks)))
            self.stride=stride; self.padding=padding
        def forward(self,x):
            return FAConvFunction.apply(x,self.W,self.b,self.B_feedback,self.stride,self.padding)

    def make_fa_conv_block(in_c,out_c):
        return nn.Sequential(FAConv2d(in_c,out_c,3,padding=1),nn.BatchNorm2d(out_c),nn.ReLU(True),nn.MaxPool2d(2))

    class FA_CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1=make_fa_conv_block(3,C1); self.conv2=make_fa_conv_block(C1,C2)
            self.conv3=make_fa_conv_block(C2,C3)
            self.fc1=nn.Linear(FC1_IN,FC1_DIM); self.fc2=nn.Linear(FC1_DIM,N_CLS)
            self.drop=nn.Dropout(0.3)
        def forward(self,x):
            x=self.conv3(self.conv2(self.conv1(x)))
            return self.fc2(self.drop(F.relu(self.fc1(x.view(x.size(0),-1)))))
        def get_features(self,x):
            with torch.no_grad():
                c1=self.conv1(x);c2=self.conv2(c1);c3=self.conv3(c2)
                h1=F.relu(self.fc1(_pool_for_fc(c3).view(c3.size(0),-1)))
            return c1.mean([2,3]),c2.mean([2,3]),c3.mean([2,3]),h1

    class PC_CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.W1=nn.Conv2d(3,C1,3,padding=1,bias=False)
            self.W2=nn.Conv2d(C1,C2,3,padding=1,bias=False)
            self.W3=nn.Conv2d(C2,C3,3,padding=1,bias=False)
            self.pool=nn.MaxPool2d(2)
            self.bn1=nn.BatchNorm2d(C1);self.bn2=nn.BatchNorm2d(C2);self.bn3=nn.BatchNorm2d(C3)
            self.P2=nn.ConvTranspose2d(C2,C1,3,padding=1,bias=False)
            self.P3=nn.ConvTranspose2d(C3,C2,3,padding=1,bias=False)
            self.fc1=nn.Linear(FC1_IN,FC1_DIM);self.fc2=nn.Linear(FC1_DIM,N_CLS)
            self.clf_opt=None
        def _make_opt(self):
            self.clf_opt=torch.optim.Adam(list(self.fc1.parameters())+list(self.fc2.parameters()),lr=LR)
        def infer(self,x):
            with torch.no_grad():
                r1=self.pool(F.relu(self.bn1(self.W1(x))))
                r2=self.pool(F.relu(self.bn2(self.W2(r1))))
                r3=self.pool(F.relu(self.bn3(self.W3(r2))))
                for _ in range(T_INF):
                    pred1=torch.tanh(F.interpolate(self.P2(r2),size=r1.shape[2:],mode="nearest"))
                    pred2=torch.tanh(F.interpolate(self.P3(r3),size=r2.shape[2:],mode="nearest"))
                    e1=r1-pred1;e2=r2-pred2
                    r1=F.relu(r1+LR_R*(-e1))
                    r2=F.relu(r2+LR_R*(-e2+F.avg_pool2d(F.conv2d(e1,self.W2.weight,padding=1),2)))
                    r3=F.relu(r3+LR_R*F.avg_pool2d(F.conv2d(e2,self.W3.weight,padding=1),2))
            return r1,r2,r3
        def weight_update(self,x,r1,r2,r3):
            with torch.no_grad():
                r1i=self.pool(F.relu(self.bn1(self.W1(x))))
                r2i=self.pool(F.relu(self.bn2(self.W2(r1i))))
                e1=(r1-r1i).clamp(-0.5,0.5);e2=(r2-r2i).clamp(-0.5,0.5)
                dW1=(e1.mean([0,2,3]).unsqueeze(1)*x.mean([0,2,3]).unsqueeze(0)).unsqueeze(-1).unsqueeze(-1).expand_as(self.W1.weight).clamp(-0.01,0.01)
                self.W1.weight.data+=LR_W*dW1
                dW2=(e2.mean([0,2,3]).unsqueeze(1)*r1.mean([0,2,3]).unsqueeze(0)).unsqueeze(-1).unsqueeze(-1).expand_as(self.W2.weight).clamp(-0.01,0.01)
                self.W2.weight.data+=LR_W*dW2
        def get_features(self,x):
            with torch.no_grad():
                r1,r2,r3=self.infer(x)
                h1=F.relu(self.fc1(_pool_for_fc(r3).view(r3.size(0),-1)))
            return r1.mean([2,3]),r2.mean([2,3]),r3.mean([2,3]),h1
        def step(self,x,y):
            r1,r2,r3=self.infer(x);self.weight_update(x,r1,r2,r3)
            self.clf_opt.zero_grad()
            logit=self.fc2(F.relu(self.fc1(r3.detach().view(r3.size(0),-1))))
            loss=F.cross_entropy(logit,y);loss.backward();self.clf_opt.step()
            return loss.item(),(logit.argmax(1)==y).float().mean().item()

    class STDP_Conv:
        def __init__(self,in_c,out_c,ks=3,padding=1):
            self.conv=nn.Conv2d(in_c,out_c,ks,padding=padding,bias=False)
            nn.init.kaiming_normal_(self.conv.weight)
        def poisson_spikes(self,rates):
            r=rates.clamp(0,1).unsqueeze(-1).expand(*rates.shape,T_SIM)
            return (torch.rand_like(r)<r/T_SIM).float()
        def first_spike(self,spikes):
            has=spikes.any(-1);t=torch.argmax(spikes,dim=-1).float();t[~has]=float(T_SIM+1);return t
        def stdp_update(self,pre_act,post_act,lr=5e-4):
            with torch.no_grad():
                pre_s=self.poisson_spikes(torch.sigmoid(pre_act))
                post_s=self.poisson_spikes(torch.sigmoid(post_act))
                t_pre=self.first_spike(pre_s);t_post=self.first_spike(post_s)
                dt=t_post.unsqueeze(1)-t_pre.unsqueeze(2)
                dW=(A_P*torch.exp(-dt.clamp(min=0)/TAU_P)-A_M*torch.exp(dt.clamp(max=0)/TAU_M)).mean(0).clamp(-0.002,0.002)
                dW_conv=dW.T.view(self.conv.weight.size(0),self.conv.weight.size(1),1,1).expand_as(self.conv.weight)
                self.conv.weight.data+=lr*dW_conv;self.conv.weight.data.clamp_(-1.0,1.0)
        def forward(self,x,do_stdp=False,pre_act=None):
            out=F.relu(self.conv(x))
            if do_stdp and pre_act is not None: self.stdp_update(pre_act,out.mean([2,3]))
            return out

    class STDP_CNN:
        def __init__(self):
            self.L1=STDP_Conv(3,C1);self.L2=STDP_Conv(C1,C2);self.L3=STDP_Conv(C2,C3)
            self.pool=nn.MaxPool2d(2)
            self.bn1=nn.BatchNorm2d(C1);self.bn2=nn.BatchNorm2d(C2);self.bn3=nn.BatchNorm2d(C3)
            self.fc1=nn.Linear(FC1_IN,FC1_DIM);self.fc2=nn.Linear(FC1_DIM,N_CLS);self.clf_opt=None
        def to_device(self,device):
            self.L1.conv=self.L1.conv.to(device);self.L2.conv=self.L2.conv.to(device);self.L3.conv=self.L3.conv.to(device)
            self.pool=self.pool.to(device)
            self.bn1=self.bn1.to(device);self.bn2=self.bn2.to(device);self.bn3=self.bn3.to(device)
            self.fc1=self.fc1.to(device);self.fc2=self.fc2.to(device)
            self.clf_opt=torch.optim.Adam(list(self.fc1.parameters())+list(self.fc2.parameters())+
                [self.bn1.weight,self.bn1.bias,self.bn2.weight,self.bn2.bias,self.bn3.weight,self.bn3.bias],lr=LR)
            return self
        def train(self,mode=True):
            # plain class -> nn.Module.train() is not inherited; drive the BNs by hand
            self.bn1.train(mode);self.bn2.train(mode);self.bn3.train(mode)
            return self
        def eval(self):
            return self.train(False)
        def _forward(self,x,do_stdp=False):
            c1=self.pool(F.relu(self.bn1(self.L1.forward(x,do_stdp,x.mean([2,3])))))
            c2=self.pool(F.relu(self.bn2(self.L2.forward(c1,do_stdp,c1.mean([2,3])))))
            c3=self.pool(F.relu(self.bn3(self.L3.forward(c2,do_stdp,c2.mean([2,3])))))
            return c1,c2,c3
        def get_features(self,x):
            with torch.no_grad():
                c1,c2,c3=self._forward(x,False)
                h1=F.relu(self.fc1(_pool_for_fc(c3).view(c3.size(0),-1)))
            return c1.mean([2,3]),c2.mean([2,3]),c3.mean([2,3]),h1
        def step(self,x,y):
            self._forward(x,True)
            self.clf_opt.zero_grad()
            _,_,c3b=self._forward(x,False)
            logit=self.fc2(F.relu(self.fc1(c3b.view(c3b.size(0),-1))))
            F.cross_entropy(logit,y).backward();self.clf_opt.step()
            return F.cross_entropy(logit.detach(),y).item(),(logit.argmax(1)==y).float().mean().item()

    # ── Training functions (exact v8) ─────────────────────────────────────
    def train_bp(model,loader,opt,sched):
        model.train()
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE);opt.zero_grad()
            loss=F.cross_entropy(model(x),y);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        sched.step();model.eval()

    def train_fa(model,loader,opt,sched):
        model.train()
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE);opt.zero_grad()
            loss=F.cross_entropy(model(x),y);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        sched.step();model.eval()

    def train_pc(model,loader):
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE);model.step(x,y)

    def train_stdp(model,loader):
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE);model.step(x,y)

    # ── Data loading ──────────────────────────────────────────────────────
    def get_cifar_loader(seed):
        tf=T.Compose([T.RandomHorizontalFlip(),T.RandomCrop(32,padding=4),
            T.ToTensor(),T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        full=torchvision.datasets.CIFAR10(str(CIFAR_DIR),train=True,download=True,transform=tf)
        idx=torch.randperm(len(full))[:N_CIFAR].tolist()
        return DataLoader(Subset(full,idx),batch_size=BATCH,shuffle=True,num_workers=4,drop_last=True)

    def load_stim_order(sub="sub-01"):
        with open(FMRI_DIR/f"stim_order_{sub}.txt") as f:
            return [l.strip() for l in f if l.strip()]

    def find_img(stimulus):
        name=stimulus.replace(".jpg","");parts=name.split("_");last=parts[-1]
        concept="_".join(parts[:-1]) if (len(parts)>1 and len(last)<=4 and any(c.isdigit() for c in last)) else name
        for pat in [f"{concept}/{name}.jpg",f"{concept}/*.jpg"]:
            hits=sorted(THINGS_DIR.glob(pat))
            if hits: return hits[0]
        for folder in THINGS_DIR.iterdir():
            if folder.name.lower()==concept.lower():
                imgs=sorted(folder.glob("*.jpg"))
                if imgs: return imgs[0]
        return None

    class ImgDS(Dataset):
        def __init__(self,paths,t): self.paths,self.t=paths,t
        def __len__(self): return len(self.paths)
        def __getitem__(self,i): return self.t(Image.open(self.paths[i]).convert("RGB")),i

    # ── RSA (Paper-1 style: mean-brain + bootstrap, fixed layer->ROI) ──────
    def compute_rdm(f):
        f=f.detach().cpu().numpy() if torch.is_tensor(f) else np.array(f)
        return squareform(pdist(f,metric="correlation"))

    def rsa_score(a,b):
        n=min(a.shape[0],b.shape[0]);idx=np.triu_indices(n,k=1)
        r,p=spearmanr(a[:n,:n][idx],b[:n,:n][idx]);return float(r),float(p)

    def bootstrap_ci(rdm_a,rdm_b,n_boot=500,ci=0.95,seed=42):
        n=min(rdm_a.shape[0],rdm_b.shape[0]);idx=np.triu_indices(n,k=1)
        va=rdm_a[:n,:n][idx];vb=rdm_b[:n,:n][idx]
        rng=np.random.default_rng(seed);boot=[]
        for _ in range(n_boot):
            s=rng.integers(0,len(va),len(va));boot.append(spearmanr(va[s],vb[s])[0])
        return float(np.percentile(boot,(1-ci)/2*100)),float(np.percentile(boot,(1+ci)/2*100))

    def load_fmri_rdm(roi,sub):
        p=FMRI_DIR/f"fmri_rdm_{roi}_{sub}.npy"
        return np.load(str(p)) if p.exists() else None

    def extract_features(model,paths,transform):
        loader=DataLoader(ImgDS(paths,transform),batch_size=128,shuffle=False,num_workers=4)
        c1s,c2s,c3s,h1s=[],[],[],[]
        model.eval()
        assert_bn_eval(model)          # shared guard -- see bn_guard.py
        with torch.no_grad():
            for imgs,_ in loader:
                c1,c2,c3,h1=model.get_features(imgs.to(DEVICE))
                np_=lambda t:t.cpu().numpy() if torch.is_tensor(t) else np.array(t)
                c1s.append(np_(c1));c2s.append(np_(c2));c3s.append(np_(c3));h1s.append(np_(h1))
        return np.concatenate(c1s),np.concatenate(c2s),np.concatenate(c3s),np.concatenate(h1s)

    def run_rsa(model,paths,transform,rule,res_px):
        feats=extract_features(model,paths,transform)
        rdms=[compute_rdm(f) for f in feats]
        rows=[]
        for layer_name,(fi,rois) in LAYER_ROI_FIXED.items():
            for roi in rois:
                subr=[load_fmri_rdm(roi,s) for s in SUBJECTS]
                subr=[r for r in subr if r is not None]
                if not subr: continue
                per_sub=[rsa_score(rdms[fi],r)[0] for r in subr]
                brain_mean=np.mean(subr,axis=0)
                n=min(rdms[fi].shape[0],brain_mean.shape[0])
                rho_mb,_=rsa_score(rdms[fi][:n,:n],brain_mean[:n,:n])
                # ci_lo/ci_hi appear in no figure and dominate GPU wall time
                # (500 Spearman calls over ~258k pairs, per layer-ROI, per resolution).
                # Off by default; recompute_bootstrap_ci.py rebuilds them on CPU from the
                # saved RDMs. NaN keeps the CSV schema identical to the previous run.
                lo,hi=(bootstrap_ci(rdms[fi][:n,:n],brain_mean[:n,:n])
                       if COMPUTE_BOOTSTRAP_CI else (float("nan"),float("nan")))
                rows.append({"rule":rule,"layer":layer_name,"roi":roi,"res":int(res_px),
                             "rho":round(float(rho_mb),6),"ci_lo":round(lo,6),"ci_hi":round(hi,6),
                             "n_subs":len(subr),"rho_sub_mean":round(float(np.mean(per_sub)),6)})
        return rows,rdms

    def rule_key(rule): return rule.lower().replace(" ","_")

    # ══════════════════════════════════════════════════════════════════════
    # MAIN LOOP — train once per (seed, rule), eval at 224 AND 32
    # ══════════════════════════════════════════════════════════════════════
    SWEEP_RES = [32, 64, 96, 128, 160, 224]
    print(f"\nLearning-Rules Resolution SWEEP")
    print(f"  Rules: {RULES}\n  Seeds: {SEEDS}\n  Resolutions: {SWEEP_RES}\n  Epochs: {N_EPOCHS}\n")

    shard_mode  = (only_rule is not None and only_seed is not None)
    partial_csv = OUT_DIR / "rsa_resolution_sweep_partial.csv"
    SHARD_DIR   = OUT_DIR / "shards"
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    # Resume set = the pre-existing partial CSV (read-only) + every shard already written.
    prior = pd.read_csv(str(partial_csv)).to_dict("records") if partial_csv.exists() else []
    n_from_partial = len(prior)
    for f in sorted(SHARD_DIR.glob("rows_*.csv")):
        prior += pd.read_csv(str(f)).to_dict("records")
    done = set((r["rule"], int(r["seed_idx"]), int(r["res"])) for r in prior)
    def combo_done(rule, si):
        if force_recompute: return False
        return all((rule, si, px) in done for px in SWEEP_RES)

    if shard_mode:
        print(f"SHARD: rule={only_rule!r} seed_idx={only_seed}")
        print(f"  resume set: {n_from_partial} rows from partial CSV + "
              f"{len(prior)-n_from_partial} from {len(list(SHARD_DIR.glob('rows_*.csv')))} shard file(s)")
        if combo_done(only_rule, only_seed):
            print(f"  ALREADY COMPLETE (all {len(SWEEP_RES)} resolutions) -> nothing to do")
            return {"rule": only_rule, "seed_idx": only_seed, "status": "already_done", "rows": 0}
        all_rows = []                       # this shard contributes only its own rows
    else:
        all_rows = prior
        if all_rows: print(f"Resuming: {len(all_rows)} rows already present\n")

    stimuli = load_stim_order("sub-01")
    paths = [p for p in [find_img(s) for s in stimuli] if p is not None]
    print(f"  THINGS: {len(paths)}/{len(stimuli)} images resolved")

    def _tf(px):
        return T.Compose([T.Resize(px), T.CenterCrop(px), T.ToTensor(),
                          T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
    RES_VARIANTS = [(px, _tf(px)) for px in SWEEP_RES]

    t_start = time.time()
    for seed_idx, seed in enumerate(SEEDS):
        if only_seed is not None and seed_idx != only_seed: continue
        print(f"\n{'='*60}\nSEED {seed_idx}/{len(SEEDS)-1}  (seed={seed})\n{'='*60}")
        cifar_loader = None  # built lazily (skip cost if whole seed already done)

        for rule in RULES:
            if only_rule is not None and rule != only_rule: continue
            if combo_done(rule, seed_idx):
                print(f"  {rule}: already done (all resolutions) -> skip")
                continue

            print(f"\n--- {rule} ---")
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

            # Build model. BN mode is set explicitly at each phase below -- never
            # inherited from construction (that is what hid the BN bug).
            if rule in RANDOM_RULES + ("Backprop",):    model = BP_CNN().to(DEVICE)
            elif rule == "Feedback Alignment":          model = FA_CNN().to(DEVICE)
            elif rule == "Predictive Coding":           model = PC_CNN().to(DEVICE); model._make_opt()
            elif rule == "STDP":                        model = STDP_CNN().to_device(DEVICE)

            # Optimizer (BP / FA only)
            opt = sched = None
            if rule == "Backprop":
                opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)
            elif rule == "Feedback Alignment":
                opt = torch.optim.SGD(model.parameters(), lr=LR*0.5, momentum=0.9, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)

            # Train to endpoint (Random Weights = untrained)
            if rule not in RANDOM_RULES:
                if cifar_loader is None:
                    cifar_loader = get_cifar_loader(seed)
                model.train()                       # BN: batch stats + running-stat updates
                for epoch in range(N_EPOCHS):
                    if rule == "Backprop":               train_bp(model, cifar_loader, opt, sched)
                    elif rule == "Feedback Alignment":   train_fa(model, cifar_loader, opt, sched)
                    elif rule == "Predictive Coding":    train_pc(model, cifar_loader)
                    elif rule == "STDP":                 train_stdp(model, cifar_loader)
                print(f"  trained {N_EPOCHS} epochs")

            elif rule == "Random Weights (BN-calibrated)":
                # Untrained filters, but BN running stats estimated from CIFAR-10 at the
                # 32px training resolution -- the same statistics the trained rules carry.
                # Isolates "learned filters" from "input normalisation" in the Rand-BP gap.
                if cifar_loader is None:
                    cifar_loader = get_cifar_loader(seed)
                before = [p.detach().clone() for p in model.parameters()]
                model.train()
                with torch.no_grad():
                    for n_batch, (x, _) in enumerate(cifar_loader):
                        if n_batch >= N_BN_CALIB: break
                        model(x.to(DEVICE))
                if not all(torch.equal(a, b) for a, b in zip(before, model.parameters())):
                    raise RuntimeError("BN calibration changed weights; it must be stats-only")
                print(f"  BN running stats calibrated on {n_batch+1} CIFAR batches "
                      f"@{CTRL_SIZE}px (no weight updates)")

            # Evaluate at every sweep resolution (training is resolution-independent)
            model.eval()                            # BN: frozen running stats
            for res_px, tf in RES_VARIANTS:
                rows, rdms = run_rsa(model, paths, tf, rule, res_px)
                for r in rows:
                    r["seed"] = seed; r["seed_idx"] = seed_idx
                all_rows.extend(rows)
                rdir = RDM_DIR / f"res{res_px}" / f"seed_{seed_idx}"
                rdir.mkdir(parents=True, exist_ok=True)
                for ln, rdm in zip(LAYERS, rdms):
                    np.save(str(rdir / f"rdm_{rule_key(rule)}_{ln}.npy"), rdm)
                v1 = next((r for r in rows if r["layer"] == "Conv1" and r["roi"] == "V1"), None)
                if v1: print(f"    @{res_px:>3}px  V1(Conv1) rho={v1['rho']:.4f}")

            # EDIT 2/3 and 3/3 vs v10. v10 read:
            #     if seed_idx == 0:
            #         try:
            #             ...
            #         except Exception as e:
            #             print(f"    [checkpoint skipped for {rule}: {e}]")
            # Both the seed guard and the swallowed exception are gone: every
            # (rule, seed) is written, seed-suffixed, and a failure to write raises.
            # The checkpoints are this run's deliverable -- a silently skipped one is
            # the same failure shape as the original BatchNorm defect, where the run
            # looked successful and the artifact was wrong.
            ckpt_path = CKPT_DIR / f"model_weights_{rule_key(rule)}_seed{seed_idx}.pt"
            if hasattr(model, "state_dict"):
                torch.save(model.state_dict(), str(ckpt_path))
            elif hasattr(model, "L1"):
                # STDP_CNN is a plain class: no state_dict(). Saving only the three
                # conv weights loses the BN running stats and the FC head, so a
                # reloaded model would evaluate with identity BN statistics and a
                # random FC1 -- i.e. NOT the model that produced these rows. Save
                # every learned tensor so eval-only scripts can reproduce this run.
                sd = {
                    "conv1.weight": model.L1.conv.weight.detach().cpu(),
                    "conv2.weight": model.L2.conv.weight.detach().cpu(),
                    "conv3.weight": model.L3.conv.weight.detach().cpu(),
                }
                for i, bn in enumerate([model.bn1, model.bn2, model.bn3], start=1):
                    sd[f"bn{i}.weight"]       = bn.weight.detach().cpu()
                    sd[f"bn{i}.bias"]         = bn.bias.detach().cpu()
                    sd[f"bn{i}.running_mean"] = bn.running_mean.detach().cpu()
                    sd[f"bn{i}.running_var"]  = bn.running_var.detach().cpu()
                for nm, fc in [("fc1", model.fc1), ("fc2", model.fc2)]:
                    sd[f"{nm}.weight"] = fc.weight.detach().cpu()
                    sd[f"{nm}.bias"]   = fc.bias.detach().cpu()
                torch.save(sd, str(ckpt_path))
            else:
                raise RuntimeError(
                    f"{rule}: model has neither state_dict() nor .L1, so it cannot be "
                    "checkpointed. The checkpoints are this run's deliverable."
                )

            # Verify the artifact exists and reloads before moving on. A torch.save
            # that appears to succeed but leaves an unreadable file is exactly the
            # silent-artifact failure this run cannot tolerate.
            if not ckpt_path.exists():
                raise RuntimeError(f"{rule} seed{seed_idx}: {ckpt_path} was not written")
            _reloaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            if not _reloaded:
                raise RuntimeError(f"{rule} seed{seed_idx}: {ckpt_path} reloaded empty")
            print(f"    checkpoint -> {ckpt_path.name} ({len(_reloaded)} tensors)")
            del _reloaded

            # Persist + commit after each (seed, rule).
            # Sharded: own file, so concurrent shards never clobber each other and the
            # pre-existing partial CSV is left byte-for-byte untouched.
            if shard_mode:
                out_f = SHARD_DIR / f"rows_{rule_key(rule)}_seed{seed_idx}{shard_suffix}.csv"
                pd.DataFrame(all_rows).to_csv(str(out_f), index=False)
                results_vol.commit()
                print(f"    committed {len(all_rows)} rows -> shards/{out_f.name}")
            else:
                pd.DataFrame(all_rows).to_csv(str(partial_csv), index=False)
                results_vol.commit()
                print(f"    committed ({len(all_rows)} rows)")

            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    if shard_mode:
        mins = (time.time() - t_start) / 60
        print(f"\nSHARD DONE: {only_rule} seed {only_seed} -- {len(all_rows)} rows in {mins:.1f} min")
        return {"rule": only_rule, "seed_idx": only_seed, "status": "ok",
                "rows": len(all_rows), "minutes": round(mins, 2)}

    # ── Finalize ──────────────────────────────────────────────────────────
    final = pd.DataFrame(all_rows)
    final.to_csv(str(OUT_DIR / "rsa_resolution_sweep.csv"), index=False)
    results_vol.commit()
    print(f"\nFinal: {len(final)} rows -> rsa_resolution_sweep.csv")

    # ── Summary: V1 Random vs BP at each resolution ───────────────────────
    v1 = final[(final["layer"] == "Conv1") & (final["roi"] == "V1")]
    print("\nV1 alignment (Conv1->V1), mean +/- std across seeds, per resolution\n")
    summ = (v1.groupby(["res", "rule"])["rho"].agg(["mean", "std"])
              .reset_index().sort_values(["res", "mean"], ascending=[True, False]))
    print(summ.to_string(index=False))

    def paired_gap(a, b, res_px):
        """Per-seed paired difference a-b at one resolution: mean, SEM, #seeds > 0."""
        p = v1[v1["res"] == res_px].pivot_table(index="seed_idx", columns="rule", values="rho")
        if not {a, b} <= set(p.columns): return None
        d = (p[a] - p[b]).dropna()
        return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()), len(d)

    for a, b in [("Random Weights", "Backprop"),
                 ("Random Weights (BN-calibrated)", "Backprop"),
                 ("Random Weights (BN-calibrated)", "Random Weights")]:
        print(f"\nHeadline -- {a} minus {b} at V1 (Conv1), paired across seeds:")
        for res_px in sorted(v1["res"].unique()):
            g = paired_gap(a, b, res_px)
            if g is None: continue
            mean, sem, npos, n = g
            print(f"  @{res_px:>3}px:  delta_rho={mean:+.4f} +/- {sem:.4f} (SEM)   "
                  f"seeds_positive={npos}/{n}")

    print("\nPeak V1-alignment resolution per rule (mean across seeds):")
    for rule in v1["rule"].unique():
        sr = v1[v1["rule"] == rule].groupby("res")["rho"].mean()
        print(f"  {rule:<20} peak @ {int(sr.idxmax())}px (rho={sr.max():.4f})")

    print("\n* Done -- committed to 'learning-rules-rsa' *")
    return str(OUT_DIR / "rsa_resolution_sweep.csv")


@app.function(image=image, timeout=60*30, volumes={"/results": results_vol})
def finalize_sweep(out_subdir="outputs_bnfix"):
    """Merge the pre-existing partial CSV + every shard file into the final CSV.
    CPU only. Idempotent -- safe to call repeatedly while shards are still landing."""
    import pandas as pd, numpy as np
    OUT_DIR = Path("/results") / out_subdir
    parts = []
    p = OUT_DIR / "rsa_resolution_sweep_partial.csv"
    if p.exists(): parts.append(pd.read_csv(str(p)))
    shards = sorted((OUT_DIR / "shards").glob("rows_*.csv"))
    parts += [pd.read_csv(str(f)) for f in shards]
    if not parts:
        print("nothing to merge"); return None

    final = pd.concat(parts, ignore_index=True)
    key = ["rule", "seed_idx", "res", "layer", "roi"]
    before = len(final)

    # A duplicated key means the same cell was computed twice -- by the pre-existing
    # partial CSV and a shard, or by a respawned shard. The run is seeded and
    # deterministic, so the two values must agree exactly; a retry that produces a
    # different rho is a bug signal, not noise. Verify before collapsing.
    dup_mask = final.duplicated(subset=key, keep=False)
    if dup_mask.any():
        num_cols = [c for c in final.columns
                    if c not in key and pd.api.types.is_numeric_dtype(final[c])]
        conflicts = []
        for k, g in final[dup_mask].groupby(key, sort=True):
            for c in num_cols:
                v = g[c].to_numpy(dtype=float)
                finite = ~np.isnan(v)
                # NaN in the same places counts as agreement (e.g. unfilled ci_lo/ci_hi);
                # NaN in only some rows does not.
                if not (finite.all() or (~finite).all()):
                    conflicts.append((k, c, list(g[c])))
                elif finite.all() and not np.allclose(v, v[0], rtol=0, atol=1e-12):
                    conflicts.append((k, c, list(g[c])))
        n_dup_keys = int(final[dup_mask].groupby(key, sort=False).ngroups)
        if conflicts:
            shown = "\n".join(f"  {k}  col={c}  values={vals}" for k, c, vals in conflicts[:20])
            raise RuntimeError(
                f"{len(conflicts)} numeric disagreement(s) across {n_dup_keys} duplicated "
                f"key(s): a recomputation of a deterministic cell changed its value. "
                f"Refusing to merge.\n{shown}"
                + (f"\n  ... and {len(conflicts)-20} more" if len(conflicts) > 20 else ""))
        print(f"duplicate check: {n_dup_keys} duplicated key(s), all numerically identical "
              f"(atol=1e-12) across {len(num_cols)} numeric column(s) -- safe to collapse")

    final = final.drop_duplicates(subset=key, keep="last").sort_values(key).reset_index(drop=True)
    final.to_csv(str(OUT_DIR / "rsa_resolution_sweep.csv"), index=False)
    results_vol.commit()

    pairs = final.groupby(["rule", "seed_idx"]).res.nunique()
    complete = int((pairs == 6).sum())
    print(f"merged {len(shards)} shard file(s) + partial -> {len(final)} rows "
          f"({before-len(final)} dupes dropped)")
    print(f"complete (rule,seed) pairs: {complete} / {len(RULES)*len(SEEDS)}")
    if complete < len(RULES) * len(SEEDS):
        missing = [f"{r} s{s}" for (r, s), v in pairs.items() if v != 6]
        absent = [f"{r} s{s}" for r in RULES for s in range(len(SEEDS))
                  if (r, s) not in pairs.index]
        print(f"  incomplete: {missing}")
        print(f"  not started: {absent}")
    return str(OUT_DIR / "rsa_resolution_sweep.csv")


@app.local_entrypoint()
def main():
    print("Learning-Rules Resolution SWEEP (5 seeds x [32..224], exact v8)")
    print(f"  Rules: {RULES}\n  Seeds: {SEEDS}\n  Epochs: {N_EPOCHS}\n")
    result = run_resolution_control.remote()
    print(f"\nDone: {result}")
    print("Download: python -m modal volume get learning-rules-rsa outputs_bnfix ./learning_rules_outputs_bnfix --force")
    print("  key file: outputs_bnfix/rsa_resolution_sweep.csv")
