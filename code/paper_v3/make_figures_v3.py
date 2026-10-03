"""2608.12408 v3: the seven figures in the v3 convention.

Same layout, colours, markers and line styles as code/make_figures.py (imported, unchanged);
the data are per subject on cross-run pairs (results/paper_v3/figure_data_v3.csv), and the
error bars are 95% stimulus-bootstrap CIs of the 5-seed mean (single CI for the single
ResNet-50 / Swin-Tiny checkpoints) instead of across-seed SEM.

Fig. 5 (Gabor similarity vs V1 alignment, single seed as in v2): x = Gabor similarity from
results/rsa_lowlevel_control.csv (model vs reference, no brain data, unchanged); y = V1
alignment per subject on cross-run pairs, recomputed here (seed 0 of the v12 checkpoints,
ResNet-50 layer1, and the Gabor filterbank RDM itself).

Output: figures/fig{1..7}_*_v3.pdf (and fig1 PNG for the README)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import make_figures as MF  # noqa: E402  (style, labels, helpers; applies the rcParams)
from compute_numbers import REPO, RES, score, v12  # noqa: E402

plt = MF.plt
OUT = REPO / "figures"
FD = pd.read_csv(REPO / "results" / "paper_v3" / "figure_data_v3.csv")
YLAB_V1 = r"Spearman $\rho$ (Conv1$\to$V1), per subject"


def cell(fig, prefix):
    d = FD[(FD.figure == fig) & FD.cell.str.startswith(prefix)].sort_values("x")
    return d.x.to_numpy(), d["mean"].to_numpy(), np.vstack([d["mean"] - d.lo, d.hi - d["mean"]])


def eb(ax, x, m, e, rule=None, colour=None, marker=None, ls="-", label=None, **kw):
    if rule is not None:
        colour, marker, ls = MF.STYLE[rule]
        label = label or MF.LABEL[rule]
    return ax.errorbar(x, m, yerr=e, color=colour, marker=marker, linestyle=ls, capsize=2,
                       elinewidth=0.9, label=label, **kw)


def fig1():
    """(a) absolute rho per condition (native); (b) paired Random - Backprop gap, native and
    content limited to 32 px, from results/upsampling/step4_gaps.csv (per subject, cross-run,
    paired stimulus bootstrap shared by both arms)."""
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.4, 5.2), gridspec_kw={"height_ratios": [1.35, 1]})
    for r in MF.PAPER_RULES:
        eb(ax, *cell("fig1", f"NATIVE|{r}|"), rule=r)
    ax.set_ylim(0, None)
    ax.axvline(32, color="0.6", lw=0.7, ls=":", zorder=0)
    ax.annotate("training resolution", xy=(34, ax.get_ylim()[0]), va="bottom", ha="left", fontsize=6, color="0.45")
    MF.res_axis(ax)
    ax.set_xlabel("")
    ax.set_ylabel(YLAB_V1)
    ax.set_title("(a)", loc="left", fontsize=8)
    ax.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              frameon=False, handlelength=2.2, columnspacing=1.4)
    g = pd.read_csv(REPO / "results" / "upsampling" / "step4_gaps.csv")
    g = g[(g.convention == "persub") & (g.roi == "V1")]
    bx.axhline(0, color="0.5", lw=0.7, zorder=0)
    bx.axvline(32, color="0.6", lw=0.7, ls=":", zorder=0)
    for arm, colour, marker, ls, lab in (("NATIVE", "#333333", "o", "-", "native"),
                                         ("UPSAMPLED", "#D55E00", "s", (0, (4, 2)), "content limited to 32 px")):
        d = g[g.arm == arm].sort_values("res")
        bx.errorbar(d.res, d.gap, yerr=np.vstack([d.gap - d.boot_lo, d.boot_hi - d.gap]), color=colour, marker=marker,
                    linestyle=ls, capsize=2, elinewidth=0.9, label=lab)
    MF.res_axis(bx)
    bx.set_ylabel(r"Random $-$ Backprop, V1")
    bx.set_title("(b)", loc="left", fontsize=8)
    bx.legend(frameon=False, loc="lower right", handlelength=2.4)
    fig.subplots_adjust(hspace=0.78)
    fig.savefig(OUT / "fig1_v1_sweep_v3.pdf")
    fig.savefig(OUT / "fig1_v1_sweep_v3.png")
    plt.close(fig)


def fig2():
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    eb(ax, *cell("fig1", "NATIVE|Random Weights|"), rule="Random Weights", label="CNN, untrained")
    eb(ax, *cell("fig1", "NATIVE|Backprop|"), rule="Backprop", label="CNN, backprop (32 px)")
    for key, colour, marker, lab in (("resnet50", "#009E73", "^", "ResNet-50 (224 px)"),
                                     ("swin_t", "#CC79A7", "v", "Swin-Tiny (224 px)")):
        eb(ax, *cell("fig2", f"{key}|"), colour=colour, marker=marker, label=lab)
    MF.res_axis(ax)
    ax.set_ylabel(r"Spearman $\rho$ (early layer$\to$V1)")
    MF.legend_below(fig, ax, ncol=2)
    fig.savefig(OUT / "fig2_architectures_v3.pdf")
    plt.close(fig)


def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, px in zip(axes, [224, 32]):
        base = FD[FD.cell == f"td|Backprop|{px}|0"]["mean"].iloc[0]
        ax.axhline(base, color=MF.BASELINE_C, linestyle=(0, (5, 2)), linewidth=1.4, zorder=1,
                   label="Untrained (random), epoch 0")
        for r in ("Backprop", "Feedback Alignment", "Predictive Coding", "STDP"):
            eb(ax, *cell("fig3", f"td|{r}|{px}|"), rule=r, zorder=3)
        ax.set_title(f"Evaluated at {px} px" + ("  (training resolution)" if px == 32 else ""))
        ax.set_xlabel("Training epoch")
    axes[0].set_ylabel(YLAB_V1)
    MF.legend_below(fig, axes[0], ncol=5, y=-0.08)
    fig.savefig(OUT / "fig3_dynamics_v3.pdf")
    plt.close(fig)


def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, roi in zip(axes, ["LOC", "IT"]):
        for r in ("Random Weights", "Backprop"):
            eb(ax, *cell("fig4", f"{roi}|{r}|"), rule=r)
        MF.res_axis(ax)
        ax.set_title(f"{roi}  ({MF.ROI_LAYER[roi]})")
        ax.legend(frameon=False, handlelength=2.2)
    axes[0].set_ylabel(r"Spearman $\rho$, per subject")
    fig.savefig(OUT / "fig4_higher_areas_v3.pdf")
    plt.close(fig)


def fig5():
    from compute_lowlevel import gabor_rdm, C
    ll = pd.read_csv(REPO / "results" / "rsa_lowlevel_control.csv")
    paths = [str(C.find_img(s)) for s in C.load_stim_order("sub-01")]
    rows = []
    for px in RES:
        g = gabor_rdm(paths, px)
        y = {"cnn_random": score(v12("Random Weights", px, 0), "V1"),
             "cnn_backprop": score(v12("Backprop", px, 0), "V1"),
             "resnet50": score(np.load(REPO / "data" / "arch_rdms" / "resnet50" / f"res{px}" / "layer1.npy"), "V1"),
             "GABOR": score(g, "V1")}
        for k, v in y.items():
            sim = ll[(ll.model == k) & (ll.res == px)].sim_gabor.iloc[0]
            rows.append(dict(model=k, res=px, sim_gabor=sim, align_v1=v))
    d = pd.DataFrame(rows)
    d.to_csv(REPO / "results" / "paper_v3" / "fig5_data_v3.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    groups = [("cnn_random", MF.BASELINE_C, "o", "CNN, untrained"), ("cnn_backprop", "#0072B2", "s", "CNN, backprop"),
              ("resnet50", "#009E73", "^", "ResNet-50"), ("GABOR", "#E69F00", "D", "Gabor filterbank")]
    for key, colour, marker, lab in groups:
        g = d[d.model == key].sort_values("res")
        ax.plot(g.sim_gabor, g.align_v1, color=colour, marker=marker, linestyle=":", linewidth=0.8, alpha=0.9, label=lab)
        for _, r in g.iloc[[0, -1]].iterrows():
            ax.annotate(f"{int(r.res)}", (r.sim_gabor, r.align_v1), textcoords="offset points", xytext=(4, 3),
                        fontsize=5.5, color=colour)
    ax.set_xlabel(r"Gabor-model similarity (Spearman $\rho$)")
    ax.set_ylabel(r"V1 alignment (Spearman $\rho$), per subject")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(OUT / "fig5_gabor_v3.pdf")
    plt.close(fig)


def fig6():
    CIFAR_C, THINGS_C = "#A6761D", "#B2182B"
    SOLID, DOTTED = "-", (0, (1.5, 1.5))
    series = [("bncal|random|", MF.BASELINE_C, "o", (0, (5, 2)), 2.0, 5, "Untrained (identity BN)"),
              ("bncal|b|", CIFAR_C, "^", SOLID, 1.6, 4, "CIFAR @ eval-res  (B)"),
              ("bncal|d|", THINGS_C, "D", SOLID, 1.6, 4, "THINGS @ eval-res  (D)"),
              ("NATIVE|Backprop|", "#0072B2", "s", SOLID, 1.6, 3, "Backprop"),
              ("bncal|a|", CIFAR_C, "v", DOTTED, 1.6, 3, "CIFAR @ 32 px  (A)"),
              ("bncal|c|", THINGS_C, "P", DOTTED, 1.6, 3, "THINGS @ 32 px  (C)")]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    hs = []
    for pre, c, mk, ls, lw, z, lab in series:
        fig_key = "fig1" if pre.startswith("NATIVE") else "fig6"
        hs.append(eb(ax, *cell(fig_key, pre), colour=c, marker=mk, ls=ls, label=lab, linewidth=lw, zorder=z))
    MF.res_axis(ax)
    ax.set_ylabel(YLAB_V1)
    fig.legend(hs, [s[-1] for s in series], loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
               frameon=False, handlelength=2.4, columnspacing=1.2)
    fig.savefig(OUT / "fig6_bn_calibrated_v3.pdf")
    plt.close(fig)


def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, arm, fig_key, title in zip(axes, ["NATIVE", "UPSAMPLED"], ["fig1", "fig7"],
                                       ["native", "content capped at 32 px"]):
        ax.axvspan(64, 224, color="0.5", alpha=0.07, lw=0, zorder=0)
        for r in MF.PAPER_RULES:
            eb(ax, *cell(fig_key, f"{arm}|{r}|"), rule=r, zorder=3)
        MF.res_axis(ax)
        ax.set_title(title)
    axes[0].set_ylabel(YLAB_V1)
    lo = 0.0
    hi = FD[FD.figure.isin(["fig1", "fig7"]) & FD.roi.eq("V1")].hi.max() * 1.15
    axes[0].set_ylim(lo, hi)
    axes[0].annotate("64$\\to$224 px: pooled positions $\\times$12", xy=(0.50, 0.965), xycoords="axes fraction",
                     ha="center", va="top", fontsize=6, color="0.35")
    axes[1].annotate("content fixed across this span", xy=(0.50, 0.965), xycoords="axes fraction",
                     ha="center", va="top", fontsize=6, color="0.35")
    for ax in axes:
        ax.axvline(32, color="0.55", lw=0.7, ls=":", zorder=1)
    axes[1].annotate("32 px: identical to native by construction", xy=(36, hi * 0.04), fontsize=6, color="0.35",
                     ha="left", va="bottom")
    MF.legend_below(fig, axes[0], ncol=5, y=-0.06)
    fig.savefig(OUT / "fig7_content_control_v3.pdf")
    plt.close(fig)


def main():
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
        print(f"{f.__name__} done", flush=True)


if __name__ == "__main__":
    main()
