"""Which definition reproduces v2's joint partial gap (-0.020 at 32 px, +0.036 at 224 px)?
Probe only (no paper number comes from this file directly)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compute_numbers import RK, SEEDS, OLD, v12, bnfix, tri  # noqa: E402
from compute_lowlevel import partial, C, gabor_rdm  # noqa: E402
sys.path.insert(0, str(HERE.parent / "cross_species"))
import run_lowlevel_control_v2 as L  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402


def main():
    paths = [C.find_img(s) for s in C.load_stim_order("sub-01")]
    refs, _, _ = C.build_references(paths)
    R = {k: tri(v) for k, v in refs.items()}
    for px in (32, 224):
        R[f"GABOR{px}"] = gabor_rdm([str(p) for p in paths], px)
        R[f"PIX32g_{px}"] = pdist(L.pixel_features([str(p) for p in paths], px), metric="correlation")
    sets = {
        "colour4 (RGB,HIST,LUM,PIXEL)": lambda px: ["MEANRGB", "COLORHIST", "MEANLUM", "PIXEL"],
        "Gabor+pixel32g+RGB+LUM": lambda px: [f"GABOR{px}", f"PIX32g_{px}", "MEANRGB", "MEANLUM"],
        "Gabor+PIXEL+RGB+LUM": lambda px: [f"GABOR{px}", "PIXEL", "MEANRGB", "MEANLUM"],
        "Gabor+PIXEL+HIST+LUM": lambda px: [f"GABOR{px}", "PIXEL", "COLORHIST", "MEANLUM"],
        "Gabor+pixel32g+HIST+LUM": lambda px: [f"GABOR{px}", f"PIX32g_{px}", "COLORHIST", "MEANLUM"],
    }
    for name, fs in sets.items():
        for bpname, bpl in (("bnfix", bnfix), ("v12", v12)):
            out = []
            for px in (32, 224):
                Z = [R[k] for k in fs(px)]
                g = [partial(bnfix(RK["rnd"], px, s), "V1", Z, *OLD) - partial(bpl(RK["bp"], px, s), "V1", Z, *OLD) for s in SEEDS]
                out.append(np.mean(g))
            print(f"{name:<32} BP={bpname:<6} gap32={out[0]:+.4f} gap224={out[1]:+.4f}", flush=True)


if __name__ == "__main__":
    main()
