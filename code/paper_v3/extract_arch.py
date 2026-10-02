"""2608.12408 v3: model RDMs of the ImageNet ResNet-50 and Swin-Tiny at the six sweep resolutions.

Mirrors code/cross_species/run_resolution_sweep_resnet.py and ..._swin.py exactly (torchvision
IMAGENET1K_V2 / IMAGENET1K_V1 weights, Resize(px) -> CenterCrop(px) -> ImageNet normalisation,
ResNet layer1/layer2/layer4, Swin features[1] (early) / features[7] (late), global average
pooling, correlation-distance RDM). Those scripts stored only the RSA values, not the RDMs; the
RDMs are needed for the per-subject, cross-run convention of v3.

Output: data/arch_rdms/{model}/res{px}/{layer}.npy  (upper triangle, TRI order, float64)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.models as tv
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import pdist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "upsampling"))
from common_up import REPO, RES, stim_order, namespaces  # noqa: E402

OUT = REPO / "data" / "arch_rdms"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def tf(px):
    return T.Compose([T.Resize(px), T.CenterCrop(px), T.ToTensor(), T.Normalize(MEAN, STD)])


def run(model, mods, pool, name, imgs, bs=32):
    for px in RES:
        if all((OUT / name / f"res{px}" / f"{k}.npy").exists() for k in mods):
            continue
        store, outs = {}, {k: [] for k in mods}
        hs = [m.register_forward_hook(lambda _m, _i, o, _k=k: store.__setitem__(_k, pool(o)))
              for k, m in mods.items()]
        t = tf(px)
        with torch.no_grad():
            for i in range(0, len(imgs), bs):
                store.clear()
                model(torch.stack([t(im) for im in imgs[i:i + bs]]))
                for k in mods:
                    outs[k].append(store[k])
        for h in hs:
            h.remove()
        for k, v in outs.items():
            f = np.concatenate(v)
            p = OUT / name / f"res{px}" / f"{k}.npy"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, pdist(f, metric="correlation"))          # pdist order == TRI order
        print(f"{name} {px}px done", flush=True)


def main():
    torch.set_num_threads(12)
    ns12, _ = namespaces()
    imgs = [Image.open(ns12["find_img"](s)).convert("RGB") for s in stim_order()]
    r50 = tv.resnet50(weights="IMAGENET1K_V2").eval()
    run(r50, {"layer1": r50.layer1, "layer2": r50.layer2, "layer4": r50.layer4},
        lambda o: o.mean(dim=[2, 3]).numpy(), "resnet50", imgs)
    sw = tv.swin_t(weights="IMAGENET1K_V1").eval()

    def spool(o):
        return (o.mean(dim=[1, 2]) if o.ndim == 4 else o.mean(dim=1)).numpy()
    run(sw, {"early": sw.features[1], "late": sw.features[7]}, spool, "swin_t", imgs)


if __name__ == "__main__":
    main()
