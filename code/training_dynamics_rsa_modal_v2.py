"""
training_dynamics_rsa_modal.py
==============================
Full Phase 2: All 5 learning rules + Random, 5 seeds, 8 milestones.
Uses EXACT v8 implementations of FA, PC, STDP.

SETUP:
  python -m modal volume create burstprop-data      (if not done)
  python -m modal volume create training-dynamics

DATA (on burstprop-data volume):
  outputs_720/       — fMRI RDMs + stim_order files
  object_images/     — THINGS images

RUN:
  python -m modal run training_dynamics_rsa_modal.py

DOWNLOAD:
  python -m modal volume get training-dynamics outputs_bnfix ./training_dynamics_outputs_bnfix --force

BN-MODE FIX (supersedes the /results/outputs run that produced Figure 4):
  PC_CNN and STDP_CNN used to define `def eval(self): pass`, so extract_features()'s
  model.eval() did nothing for them and their BatchNorms normalised the THINGS eval
  batches by their own per-batch statistics. Fixed via the shared guards in bn_guard.py.

  This script is the harder case: it evaluates at MILESTONES, so each rule cycles
  train -> eval -> train eight times. Both directions are now asserted every cycle:
    assert_bn_eval(model)  before each milestone extraction  (eval-set stat leakage)
    assert_bn_train(model) before each training epoch        (running stats frozen for
                                                              the rest of the run)
  Results go to a FRESH dir so nothing resumes from the old run.
"""

import modal
from pathlib import Path

app = modal.App("training-dynamics-rsa")

data_vol    = modal.Volume.from_name("burstprop-data", create_if_missing=True)
results_vol = modal.Volume.from_name("training-dynamics", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "torchvision", "numpy", "scipy", "pandas", "Pillow", "matplotlib")
    .add_local_python_source("bn_guard")        # shared BatchNorm mode guards
)

# Constants matching v8
SUBJECTS   = ["sub-01", "sub-02", "sub-03"]
ROIS       = ["V1", "V2", "V3", "V4", "LOC", "IT"]
LAYERS     = ["Conv1", "Conv2", "Conv3", "FC1"]
N_EPOCHS   = 40
MILESTONES = [0, 1, 2, 5, 10, 20, 30, 40]
RULES      = ["Random Weights", "Backprop", "Feedback Alignment", "Predictive Coding", "STDP"]
SEEDS      = [42, 123, 456, 789, 1337]
IMG_SIZE   = 224
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

COLORS = {
    "Random Weights": "#999999", "Backprop": "#2E86AB",
    "Feedback Alignment": "#E84855", "Predictive Coding": "#3BB273", "STDP": "#F4A261",
}


@app.function(
    image=image, gpu="T4", timeout=60*60*24,
    volumes={"/data": data_vol, "/results": results_vol},
)
def run_training_dynamics():
    import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    import torchvision, torchvision.transforms as T, random
    from torch.utils.data import DataLoader, Subset, Dataset
    from scipy.stats import spearmanr
    from scipy.spatial.distance import pdist, squareform
    from PIL import Image
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import pandas as pd
    from bn_guard import assert_bn_eval, assert_bn_train

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    FMRI_DIR   = Path("/data/outputs_720")
    THINGS_DIR = Path("/data/object_images")
    CIFAR_DIR  = Path("/results/cifar_data")
    OUT_DIR    = Path("/results/outputs_bnfix")   # fresh dir: do NOT resume the pre-BN-fix run
    CKPT_DIR   = OUT_DIR / "checkpoints"
    RDM_DIR    = OUT_DIR / "rdms"
    PLOT_DIR   = OUT_DIR / "plots"
    for d in [OUT_DIR, CKPT_DIR, RDM_DIR, PLOT_DIR, CIFAR_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    assert FMRI_DIR.exists(), f"fMRI data not at {FMRI_DIR}"
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

    # ── Training functions ────────────────────────────────────────────────

    def train_bp(model,loader,opt,sched):
        model.train()
        tl,tc,tn=0.0,0,0
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE);opt.zero_grad()
            loss=F.cross_entropy(model(x),y);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            with torch.no_grad(): tl+=loss.item();tc+=(model(x).argmax(1)==y).sum().item();tn+=y.size(0)
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

    # ── RSA ────────────────────────────────────────────────────────────────

    def compute_rdm(f):
        f=f.detach().cpu().numpy() if torch.is_tensor(f) else np.array(f)
        return squareform(pdist(f,metric="correlation"))

    def rsa_score(a,b):
        n=min(a.shape[0],b.shape[0]);idx=np.triu_indices(n,k=1)
        r,p=spearmanr(a[:n,:n][idx],b[:n,:n][idx]);return float(r),float(p)

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

    def compute_rsa_rows(feats,rule,seed_idx,epoch,res_px):
        rdms=[compute_rdm(f) for f in feats];rows=[]
        for li,ln in enumerate(LAYERS):
            for roi in ROIS:
                for sub in SUBJECTS:
                    brain=load_fmri_rdm(roi,sub)
                    if brain is None: continue
                    rho,pval=rsa_score(rdms[li],brain)
                    rows.append({"rule":rule,"layer":ln,"roi":roi,"subject":sub,
                                 "seed_idx":seed_idx,"epoch":epoch,"res":res_px,
                                 "rho":round(rho,6),"pval":round(pval,6)})
        return rows,rdms

    def rule_key(rule): return rule.lower().replace(" ","_")

    def already_done(all_rows,rule,seed_idx,epoch,res_px):
        return any(r["rule"]==rule and r["seed_idx"]==seed_idx and r["epoch"]==epoch
                   and r.get("res",IMG_SIZE)==res_px for r in all_rows)

    # ── Plots ─────────────────────────────────────────────────────────────

    def plot_v1(df):
        rules=[r for r in df["rule"].unique() if r in COLORS];epochs=sorted(df["epoch"].unique())
        fig,ax=plt.subplots(figsize=(9,5))
        for rule in rules:
            means,sds=[],[]
            for ep in epochs:
                sub=df[(df["rule"]==rule)&(df["roi"]=="V1")&(df["epoch"]==ep)]
                if sub.empty: means.append(np.nan);sds.append(0);continue
                bps=sub.groupby(["seed_idx","layer"])["rho"].mean().reset_index().groupby("seed_idx")["rho"].max()
                means.append(float(bps.mean()));sds.append(float(bps.std()) if len(bps)>1 else 0)
            x=[max(e,0.5) for e in epochs]
            ax.plot(x,means,"o-",color=COLORS[rule],label=rule,lw=2,ms=5)
            ax.fill_between(x,[m-s for m,s in zip(means,sds)],[m+s for m,s in zip(means,sds)],color=COLORS[rule],alpha=0.15)
        ax.set_xscale("symlog",linthresh=1);ax.set_xlabel("Epoch");ax.set_ylabel("Spearman ρ")
        ax.set_title("V1 alignment across training");ax.axhline(0,color="black",lw=0.5)
        ax.legend(fontsize=9);ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
        plt.tight_layout();plt.savefig(str(PLOT_DIR/"epoch_vs_rho_v1.png"),dpi=150);plt.close()
        print("  Saved: epoch_vs_rho_v1.png")

    def plot_all_rois(df):
        rules=[r for r in df["rule"].unique() if r in COLORS];epochs=sorted(df["epoch"].unique())
        fig,axes=plt.subplots(2,3,figsize=(14,8));axes=axes.flatten()
        for ax,roi in zip(axes,ROIS):
            for rule in rules:
                means=[float(df[(df["rule"]==rule)&(df["roi"]==roi)&(df["epoch"]==ep)].groupby("layer")["rho"].mean().max()) if not df[(df["rule"]==rule)&(df["roi"]==roi)&(df["epoch"]==ep)].empty else np.nan for ep in epochs]
                ax.plot([max(e,0.5) for e in epochs],means,"o-",color=COLORS[rule],label=rule,lw=1.5,ms=4)
            ax.set_xscale("symlog",linthresh=1);ax.set_title(roi,fontweight="bold")
            ax.axhline(0,color="black",lw=0.5);ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
        h,l=axes[0].get_legend_handles_labels()
        fig.legend(h,l,loc="lower center",ncol=3,fontsize=9,bbox_to_anchor=(0.5,-0.02))
        fig.suptitle("fMRI alignment across training — all ROIs",fontsize=13,y=1.01)
        plt.tight_layout();plt.savefig(str(PLOT_DIR/"epoch_vs_rho_all_rois.png"),dpi=150);plt.close()
        print("  Saved: epoch_vs_rho_all_rois.png")

    def plot_delta(df):
        rules=[r for r in df["rule"].unique() if r in COLORS];epochs=sorted(df["epoch"].unique())
        fig,axes=plt.subplots(2,3,figsize=(14,8),sharey=True);axes=axes.flatten()
        for ax,roi in zip(axes,ROIS):
            for rule in rules:
                means={}
                for ep in epochs:
                    sub=df[(df["rule"]==rule)&(df["roi"]==roi)&(df["epoch"]==ep)]
                    if not sub.empty: means[ep]=float(sub.groupby("layer")["rho"].mean().max())
                if 0 not in means or means[0]==0: continue
                bl=means[0];eps_p=[e for e in epochs if e in means]
                ax.plot([max(e,0.5) for e in eps_p],[means[e]-bl for e in eps_p],"o-",color=COLORS[rule],label=rule,lw=1.5,ms=4)
            ax.set_xscale("symlog",linthresh=1);ax.set_title(roi,fontweight="bold")
            ax.axhline(0,color="black",lw=1,ls="--");ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
        h,l=axes[0].get_legend_handles_labels()
        fig.legend(h,l,loc="lower center",ncol=3,fontsize=9,bbox_to_anchor=(0.5,-0.02))
        fig.suptitle("Change in alignment relative to untrained baseline",fontsize=13,y=1.01)
        plt.tight_layout();plt.savefig(str(PLOT_DIR/"delta_rho_normalized.png"),dpi=150);plt.close()
        print("  Saved: delta_rho_normalized.png")

    # ══════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ══════════════════════════════════════════════════════════════════════

    print(f"\nTraining Dynamics RSA — Phase 2 (exact v8)")
    print(f"  Rules: {RULES}\n  Seeds: {SEEDS}\n  Milestones: {MILESTONES}\n  Epochs: {N_EPOCHS}\n")

    partial_csv=OUT_DIR/"training_dynamics_results_partial.csv"
    all_rows=pd.read_csv(str(partial_csv)).to_dict("records") if partial_csv.exists() else []
    for r in all_rows: r.setdefault("res", IMG_SIZE)   # alte 224-only-Zeilen als 224 taggen
    if all_rows: print(f"Resuming: {len(all_rows)} rows\n")

    stimuli=load_stim_order("sub-01")
    paths=[p for p in [find_img(s) for s in stimuli] if p is not None]
    print(f"  THINGS: {len(paths)}/{len(stimuli)} images")
    tf_things=T.Compose([T.Resize(IMG_SIZE),T.CenterCrop(IMG_SIZE),T.ToTensor(),
                         T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
    tf_things_32=T.Compose([T.Resize(32),T.CenterCrop(32),T.ToTensor(),
                            T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])

    for seed_idx,seed in enumerate(SEEDS):
        print(f"\n{'='*60}\nSEED {seed_idx}/{len(SEEDS)-1}  (seed={seed})\n{'='*60}")
        cifar_loader=get_cifar_loader(seed)

        for rule in RULES:
            print(f"\n--- {rule} ---")
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)

            # Build model. This script flips train -> eval -> train once per milestone,
            # so BN mode is set explicitly at BOTH transitions below and never inherited
            # from construction or from the previous milestone.
            if rule in ("Random Weights","Backprop"): model=BP_CNN().to(DEVICE)
            elif rule=="Feedback Alignment": model=FA_CNN().to(DEVICE)
            elif rule=="Predictive Coding": model=PC_CNN().to(DEVICE);model._make_opt()
            elif rule=="STDP": model=STDP_CNN().to_device(DEVICE)

            # Optimizer
            opt,sched=None,None
            if rule=="Backprop":
                opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-4)
                sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,N_EPOCHS)
            elif rule=="Feedback Alignment":
                opt=torch.optim.SGD(model.parameters(),lr=LR*0.5,momentum=0.9,weight_decay=1e-4)
                sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,N_EPOCHS)

            for epoch in range(N_EPOCHS+1):
                where=f"{rule} seed{seed_idx} ep{epoch}"

                # Milestone: extract RSA at BOTH resolutions (224 + 32px control)
                if epoch in MILESTONES:
                    model.eval()                       # frozen BN running stats
                    assert_bn_eval(model, where)
                    did_any=False
                    for res_px,res_tf in [(IMG_SIZE,tf_things),(32,tf_things_32)]:
                        if already_done(all_rows,rule,seed_idx,epoch,res_px):
                            continue
                        did_any=True
                        feats=extract_features(model,paths,res_tf)
                        rows,rdms=compute_rsa_rows(feats,rule,seed_idx,epoch,res_px)
                        all_rows.extend(rows)

                        v1=[r for r in rows if r["roi"]=="V1"]
                        if v1:
                            by_l={}
                            for r in v1: by_l.setdefault(r["layer"],[]).append(r["rho"])
                            bl=max(by_l,key=lambda l:np.mean(by_l[l]))
                            print(f"  ep {epoch:3d} @{res_px:>3}px: V1 best={bl} rho={np.mean(by_l[bl]):.4f}")

                        rdir=RDM_DIR/f"res{res_px}"/f"{rule_key(rule)}_seed{seed_idx}_epoch{epoch}"
                        rdir.mkdir(parents=True,exist_ok=True)
                        for ln,rdm in zip(LAYERS,rdms):
                            np.save(str(rdir/f"rdm_{rule_key(rule)}_{ln}.npy"),rdm)

                    if did_any:
                        pd.DataFrame(all_rows).to_csv(str(partial_csv),index=False)
                        results_vol.commit()
                        print(f"    committed ({len(all_rows)} rows)")

                    if rule=="Random Weights": break

                # Train one epoch. Must flip BACK to train mode after a milestone eval --
                # otherwise BN running stats freeze for the rest of the run (the same bug
                # in the opposite direction), silently and with no error.
                if epoch<N_EPOCHS and rule!="Random Weights":
                    model.train()
                    assert_bn_train(model, where)
                    if rule=="Backprop": train_bp(model,cifar_loader,opt,sched)
                    elif rule=="Feedback Alignment": train_fa(model,cifar_loader,opt,sched)
                    elif rule=="Predictive Coding": train_pc(model,cifar_loader)
                    elif rule=="STDP": train_stdp(model,cifar_loader)

            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Final
    final_df=pd.DataFrame(all_rows)
    final_df.to_csv(str(OUT_DIR/"training_dynamics_results.csv"),index=False)
    print(f"\nFinal: {len(final_df)} rows")

    # Standard figures use the 224px subset (unchanged Paper-3 plots)
    df224=final_df[final_df["res"]==IMG_SIZE]
    print("\nPlots (224px)...");plot_v1(df224);plot_all_rois(df224);plot_delta(df224)

    # ── 32x32 RESOLUTION CONTROL: does the V1 degradation survive at native res? ──
    print("\n32x32 RESOLUTION CONTROL -- V1 best-layer change, epoch0 -> last epoch:")
    print(f"  {'rule':<20} {'d_rho@224':>12} {'d_rho@32':>12}")
    for rule in RULES:
        line=f"  {rule:<20}"
        for res_px in (IMG_SIZE,32):
            sub=final_df[(final_df["rule"]==rule)&(final_df["roi"]=="V1")&(final_df["res"]==res_px)]
            if sub.empty:
                line+=f" {'n/a':>12}"; continue
            g=sub.groupby(["epoch","layer"])["rho"].mean().reset_index()
            r0=g[g["epoch"]==g["epoch"].min()]["rho"].max()
            rN=g[g["epoch"]==g["epoch"].max()]["rho"].max()
            line+=f" {rN-r0:>+12.4f}"
        print(line)
    print("  (negativ = Training degradiert V1. Wenn @32 ~0 oder viel kleiner als @224,")
    print("   ist die 'Degradation' groesstenteils Resolution-Drift -> Paper-3-These faellt.)")

    results_vol.commit()
    print("\nDone -- committed to 'training-dynamics'")
    return str(OUT_DIR/"training_dynamics_results.csv")


@app.local_entrypoint()
def main():
    print("Training Dynamics RSA — Phase 2 (exact v8)")
    print(f"  Rules: {RULES}\n  Seeds: {SEEDS}\n  Milestones: {MILESTONES}\n  Epochs: {N_EPOCHS}\n")
    result=run_training_dynamics.remote()
    print(f"\nDone: {result}")
    print("Download: python -m modal volume get training-dynamics outputs_bnfix ./training_dynamics_outputs_bnfix --force")