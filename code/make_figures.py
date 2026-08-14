"""
make_figures.py
===============
Regenerates the five figures of Evaluation_Resolution_Confounds from the BN-fixed
(outputs_bnfix) result set.

  fig1_v1_sweep.pdf       V1 alignment (Conv1->V1) vs evaluation resolution, per rule
  fig2_architectures.pdf  same curve for CNN / ResNet-50 / Swin-Tiny
  fig3_dynamics.pdf       V1 alignment over training, at 224 px and at 32 px
  fig4_higher_areas.pdf   Backprop vs untrained at LOC and IT, across resolution
  fig5_gabor.pdf          Gabor-model similarity vs V1 alignment

Plus one figure outside the paper set, for the BN-calibration control:

  fig6_bn_calibrated.pdf  the BN-calibration 2x2 (set x resolution) at V1
  fig7_content_control.pdf  NATIVE vs content-capped-at-32px, V1 alignment

COLOUR
------
The untrained network is a BASELINE, not a fifth rule, so it is drawn as a near-black
dashed line everywhere -- including fig3, where it exists only at epoch 0 and coincides
exactly with Backprop's epoch-0 value. The previous red-vs-orange pairing for
Random vs Predictive Coding is unreadable at print size; near-black vs orange is the
largest separation available and also reads correctly as "reference line".

The four rule hues are Okabe-Ito steps, validated with the dataviz palette checker
(all-pairs): lightness band PASS, chroma floor PASS, normal-vision worst pair
dE 18.7 PASS. STDP<->FA sits at dE 7.6 under deuteranopia, in the 6-8 band that is
permitted only with secondary encoding -- hence a distinct marker shape per series
and a legend in every panel.

Usage:
  python make_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
# Repointed for this repository: the original script read from two separate working
# trees (Projekte_1/learning-rules-rsa and Projekte_2/Cross_Species_RSA). Everything
# now comes from ../results/, which holds the same files under the names documented
# in results/README.md. This path block is the only edit; the plotting code below is
# unchanged from the version that produced the figures in the paper.
RESULTS = Path(__file__).resolve().parent.parent / "results"

SWEEP   = RESULTS / "bnfix_sweep.csv"                       # 6 conditions x 5 seeds
BNCAL   = RESULTS / "bncal_rows.csv"                        # BN-calibration factorial
UPSAMP  = RESULTS / "rsa_resolution_sweep_upsample.csv"     # NATIVE vs UPSAMPLED arms
DYN     = RESULTS / "training_dynamics_results.csv"
RESNET  = RESULTS / "rsa_resolution_sweep_resnet.csv"
SWIN    = RESULTS / "rsa_resolution_sweep_swin.csv"
# The BN-fixed low-level control is the one the paper uses. The pre-repair file ships
# alongside it as `rsa_lowlevel_control_superseded.csv` for comparison only; it carries
# a `#` comment header, hence comment="#" on the read below.
LOWLVL_NEW = RESULTS / "rsa_lowlevel_control.csv"
LOWLVL_OLD = RESULTS / "rsa_lowlevel_control_superseded.csv"

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

RES = [32, 64, 96, 128, 160, 224]

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "font.size":         8,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "legend.fontsize":   7,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.5,
    "lines.linewidth":   1.6,
    "lines.markersize":  4.0,
})

BASELINE_C = "#333333"      # untrained network: reference line, not a categorical hue
STYLE = {   # rule -> (colour, marker, linestyle)
    "Random Weights":                 (BASELINE_C, "o", (0, (5, 2))),
    "Backprop":                       ("#0072B2",  "s", "-"),
    "Feedback Alignment":             ("#009E73",  "^", "-"),
    "Predictive Coding":              ("#E69F00",  "D", "-"),
    "STDP":                           ("#CC79A7",  "v", "-"),
    # Appears only in fig6. Deliberately NOT a second blue: against Backprop's
    # #0072B2 this clears every check with room to spare (normal dE 25.4, CVD 22.1).
    "Random Weights (BN-calibrated)": ("#A6761D",  "P", (0, (1, 1.2))),
}
LABEL = {
    "Random Weights":                 "Untrained (random)",
    "Backprop":                       "Backprop",
    "Feedback Alignment":             "Feedback alignment",
    "Predictive Coding":              "Predictive coding",
    "STDP":                           "STDP",
    "Random Weights (BN-calibrated)": "Random, BN-calibrated",
}
PAPER_RULES = ["Random Weights", "Backprop", "Feedback Alignment",
               "Predictive Coding", "STDP"]

ROI_LAYER = {"V1": "Conv1", "V2": "Conv1", "LOC": "Conv3", "IT": "FC1"}


def sweep_cell(df, roi):
    return df[(df.layer == ROI_LAYER[roi]) & (df.roi == roi)]


def mean_sem(df, rule, roi):
    """Mean and across-seed SEM of rho at each resolution."""
    d = sweep_cell(df, roi)
    d = d[d.rule == rule]
    g = d.groupby("res").rho
    m = g.mean().reindex(RES)
    s = (g.std(ddof=1) / np.sqrt(g.count())).reindex(RES)
    return m.to_numpy(), s.to_numpy()


def plot_rule(ax, df, rule, roi, label=None):
    m, s = mean_sem(df, rule, roi)
    c, mk, ls = STYLE[rule]
    ax.errorbar(RES, m, yerr=s, color=c, marker=mk, linestyle=ls,
                capsize=2, elinewidth=0.9, label=label or LABEL[rule])
    return m


def mean_sem_by(df, col, value, roi):
    """mean_sem() for frames keyed by something other than `rule` (the BN-calibration
    factorial is keyed by `variant`)."""
    d = df[(df.layer == ROI_LAYER[roi]) & (df.roi == roi) & (df[col] == value)]
    g = d.groupby("res").rho
    m = g.mean().reindex(RES)
    s = (g.std(ddof=1) / np.sqrt(g.count())).reindex(RES)
    return m.to_numpy(), s.to_numpy()


def res_axis(ax):
    ax.set_xticks(RES)
    ax.set_xticklabels([str(r) for r in RES])
    ax.set_xlabel("Evaluation resolution (px)")


def legend_below(fig, ax, ncol, y=-0.02):
    """Legend under the axes. These panels are dense enough that any in-axes
    placement lands on a curve."""
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
               frameon=False, handlelength=2.2, columnspacing=1.4)


# ═══════════════════════════════════════════════════════════════════════════
def fig1(sweep):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for rule in PAPER_RULES:
        plot_rule(ax, sweep, rule, "V1")
    ax.axvline(32, color="0.6", lw=0.7, ls=":", zorder=0)
    ax.annotate("training resolution", xy=(34, ax.get_ylim()[0]),
                va="bottom", ha="left", fontsize=6, color="0.45")
    res_axis(ax)
    ax.set_ylabel(r"Spearman $\rho$ (Conv1$\to$V1)")
    legend_below(fig, ax, ncol=2)
    fig.savefig(OUT / "fig1_v1_sweep.pdf")
    # Also as PNG: GitHub does not render PDFs inline, and fig1 is embedded in the
    # repository README. The PDF remains the artifact of record for the paper.
    fig.savefig(OUT / "fig1_v1_sweep.png")
    plt.close(fig)
    print("  fig1_v1_sweep.pdf  (+ .png for the README)")


def fig2(sweep):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    plot_rule(ax, sweep, "Random Weights", "V1", label="CNN, untrained")
    plot_rule(ax, sweep, "Backprop", "V1", label="CNN, backprop (32 px)")

    for path, key, colour, marker, lab in [
        (RESNET, "resnet50", "#009E73", "^", "ResNet-50 (224 px)"),
        (SWIN,   "swin_t",   "#CC79A7", "v", "Swin-Tiny (224 px)"),
    ]:
        d = pd.read_csv(path)
        d = d[d.roi == "V1"].sort_values("res")
        ax.plot(d.res, d.rho, color=colour, marker=marker, linestyle="-", label=lab)

    res_axis(ax)
    ax.set_ylabel(r"Spearman $\rho$ (early layer$\to$V1)")
    legend_below(fig, ax, ncol=2)
    fig.savefig(OUT / "fig2_architectures.pdf")
    plt.close(fig)
    print("  fig2_architectures.pdf")


def fig3(dyn):
    """V1 alignment over training at 224 px and 32 px.

    Random Weights exists only at epoch 0 and equals Backprop's epoch-0 value
    exactly, so plotting it as a series would draw a single point on top of another
    series' first point. It is drawn as a horizontal dashed reference line instead.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    trained = ["Backprop", "Feedback Alignment", "Predictive Coding", "STDP"]

    for ax, res_px in zip(axes, [224, 32]):
        d = dyn[(dyn.res == res_px) & (dyn.layer == "Conv1") & (dyn.roi == "V1")]

        base = d[(d.rule == "Random Weights") & (d.epoch == 0)].rho.mean()
        ax.axhline(base, color=BASELINE_C, linestyle=(0, (5, 2)), linewidth=1.4,
                   zorder=1, label="Untrained (random), epoch 0")

        for rule in trained:
            g = d[d.rule == rule].groupby("epoch").rho
            m, e = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
            c, mk, ls = STYLE[rule]
            ax.errorbar(m.index, m.to_numpy(), yerr=e.to_numpy(), color=c, marker=mk,
                        linestyle=ls, capsize=2, elinewidth=0.9, zorder=3,
                        label=LABEL[rule])

        ax.set_title(f"Evaluated at {res_px} px" +
                     ("  (training resolution)" if res_px == 32 else ""))
        ax.set_xlabel("Training epoch")
    axes[0].set_ylabel(r"Spearman $\rho$ (Conv1$\to$V1)")
    legend_below(fig, axes[0], ncol=5, y=-0.08)
    fig.savefig(OUT / "fig3_dynamics.pdf")
    plt.close(fig)
    print("  fig3_dynamics.pdf")


def fig4(sweep):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, roi in zip(axes, ["LOC", "IT"]):
        for rule in ["Random Weights", "Backprop"]:
            plot_rule(ax, sweep, rule, roi)
        res_axis(ax)
        ax.set_title(f"{roi}  ({ROI_LAYER[roi]})")
        ax.legend(frameon=False, handlelength=2.2)
    axes[0].set_ylabel(r"Spearman $\rho$")
    fig.savefig(OUT / "fig4_higher_areas.pdf")
    plt.close(fig)
    print("  fig4_higher_areas.pdf")


def fig5(lowlevel_path):
    """Gabor-similarity vs V1 alignment.

    No correlation statistic in the title: the paper text no longer claims one, and
    with three model families the analysis is not powered to support or exclude a
    weak relationship.
    """
    d = pd.read_csv(lowlevel_path, comment="#")   # the superseded fallback has a header
    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    groups = [
        ("cnn_random",   BASELINE_C, "o", "CNN, untrained"),
        ("cnn_backprop", "#0072B2",  "s", "CNN, backprop"),
        ("resnet50",     "#009E73",  "^", "ResNet-50"),
        ("GABOR",        "#E69F00",  "D", "Gabor filterbank"),
    ]
    for key, colour, marker, lab in groups:
        g = d[d.model == key].sort_values("res")
        if g.empty:
            continue
        ax.plot(g.sim_gabor, g.align_v1, color=colour, marker=marker,
                linestyle=":", linewidth=0.8, alpha=0.9, label=lab)

    # Label the resolution extremes of each track rather than every point.
    for key, colour, _, _ in groups:
        g = d[d.model == key].sort_values("res")
        if g.empty:
            continue
        for _, r in g.iloc[[0, -1]].iterrows():
            ax.annotate(f"{int(r.res)}", (r.sim_gabor, r.align_v1),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=5.5, color=colour)

    ax.set_xlabel(r"Gabor-model similarity (Spearman $\rho$)")
    ax.set_ylabel(r"V1 alignment (Spearman $\rho$)")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(OUT / "fig5_gabor.pdf")
    plt.close(fig)
    print(f"  fig5_gabor.pdf   (source: {lowlevel_path.name})")


def fig6(sweep, bncal):
    """The BN-calibration 2x2 control -- not one of the paper's five.

    The retired A_aug condition is NOT plotted: it calibrated through an augmented
    loader whose zero-padded RandomCrop put black borders into the calibration
    images, and it overstated the calibration cost by roughly 2.4x. The two leakage
    probes (C_leak / D_leak) are diagnostics discussed in the text and would only
    crowd the panel.

    Encoding: hue = calibration SET (ochre CIFAR, dark red THINGS), linestyle =
    calibration RESOLUTION (solid = evaluation resolution, dotted = 32 px), so the
    2x2 reads straight off the legend. The two reference conditions keep the colours
    they carry in the other figures (near-black untrained baseline, blue backprop).
    Hues validated all-pairs: worst CVD dE 10.6 (deuteranopia) / 14.1 (tritanopia),
    worst normal-vision dE 18.7, contrast all >= 3:1. Every series also keeps its own
    marker shape.
    """
    CIFAR_C, THINGS_C = "#A6761D", "#B2182B"
    SOLID, DOTTED = "-", (0, (1.5, 1.5))

    # (frame, key column, value, colour, marker, linestyle, linewidth, zorder, label)
    series = [
        (bncal, "variant", "Random (identity BN)", BASELINE_C, "o", (0, (5, 2)),
         2.0, 5, "Untrained (identity BN)"),
        (bncal, "variant", "B CIFAR@eval-res", CIFAR_C, "^", SOLID,
         1.6, 4, "CIFAR @ eval-res  (B)"),
        (bncal, "variant", "D THINGS-disjoint@res", THINGS_C, "D", SOLID,
         1.6, 4, "THINGS @ eval-res  (D)"),
        (sweep, "rule", "Backprop", "#0072B2", "s", SOLID,
         1.6, 3, "Backprop"),
        (bncal, "variant", "A CIFAR@32", CIFAR_C, "v", DOTTED,
         1.6, 3, "CIFAR @ 32 px  (A)"),
        (bncal, "variant", "C THINGS-disjoint@32", THINGS_C, "P", DOTTED,
         1.6, 3, "THINGS @ 32 px  (C)"),
    ]

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    handles = []
    for df, col, val, c, mk, ls, lw, z, lab in series:
        m, s = mean_sem_by(df, col, val, "V1")
        h = ax.errorbar(RES, m, yerr=s, color=c, marker=mk, linestyle=ls, linewidth=lw,
                        capsize=2, elinewidth=0.9, zorder=z, label=lab)
        handles.append(h)

    res_axis(ax)
    ax.set_ylabel(r"Spearman $\rho$ (Conv1$\to$V1)")
    # Legend order is column-major at ncol=2, so this gives a left column of
    # "tracks the untrained baseline" and a right column of "sits below it".
    fig.legend(handles, [s[-1] for s in series], loc="upper center",
               bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False,
               handlelength=2.4, columnspacing=1.2)
    fig.savefig(OUT / "fig6_bn_calibrated.pdf")
    plt.close(fig)
    print("  fig6_bn_calibrated.pdf")


def fig7(ups):
    """NATIVE vs content capped at 32 px.

    The endpoints of the two arms look alike; the SHAPE does not, and the shape is
    the result. In UPSAMPLED the trained rules complete their whole 32->224 change in
    the single first step -- which is where the resize chain is introduced, not a
    pooling manipulation -- and are flat or reversed afterwards. In NATIVE they
    decline steadily across the sweep.

    The shaded 64->224 span is the window the text rests on: there the UPSAMPLED
    arm's content is identical at both ends, so only the pooled positions change
    (1,024 -> 12,544). Rule colours, markers and line styles are fig1's, untouched,
    so the panels read as the same experiment under one changed condition.
    """
    c = ups[(ups.roi == "V1") & (ups.layer == "Conv1")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)

    for ax, arm, title in zip(axes, ["NATIVE", "UPSAMPLED"],
                              ["native", "content capped at 32 px"]):
        d = c[c.arm == arm]
        # The comparison window: content fixed at both ends in the right panel.
        ax.axvspan(64, 224, color="0.5", alpha=0.07, lw=0, zorder=0)
        for rule in PAPER_RULES:
            g = d[d.rule == rule].groupby("res").rho
            m = g.mean().reindex(RES)
            e = (g.std(ddof=1) / np.sqrt(g.count())).reindex(RES)
            col, mk, ls = STYLE[rule]
            ax.errorbar(RES, m.to_numpy(), yerr=e.to_numpy(), color=col, marker=mk,
                        linestyle=ls, capsize=2, elinewidth=0.9, zorder=3,
                        label=LABEL[rule])
        res_axis(ax)
        ax.set_title(title)

    axes[0].set_ylabel(r"Spearman $\rho$ (Conv1$\to$V1)")
    axes[0].set_ylim(0.000, 0.093)          # headroom: data+SEM spans 0.0069-0.0792

    # Name the shaded span once, and flag that the arms start from the same point.
    axes[0].annotate("64$\\to$224 px: pooled positions $\\times$12",
                     xy=(0.50, 0.965), xycoords="axes fraction", ha="center",
                     va="top", fontsize=6, color="0.35")
    axes[1].annotate("content fixed across this span",
                     xy=(0.50, 0.965), xycoords="axes fraction", ha="center",
                     va="top", fontsize=6, color="0.35")
    # The 32 px column is shared by both arms by construction; say so next to the
    # guide line rather than with a leader, which would have to cross five curves.
    for ax in axes:
        ax.axvline(32, color="0.55", lw=0.7, ls=":", zorder=1)
    axes[1].annotate("32 px: identical to native by construction", xy=(36, 0.004),
                     fontsize=6, color="0.35", ha="left", va="bottom")

    legend_below(fig, axes[0], ncol=5, y=-0.06)
    fig.savefig(OUT / "fig7_content_control.pdf")
    plt.close(fig)
    print("  fig7_content_control.pdf")


def main():
    sweep = pd.read_csv(SWEEP)
    dyn = pd.read_csv(DYN)
    lowlevel = LOWLVL_NEW if LOWLVL_NEW.exists() else LOWLVL_OLD
    if not LOWLVL_NEW.exists():
        print(f"! {LOWLVL_NEW.name} not present -- fig5 falls back to the pre-fix "
              f"{LOWLVL_OLD.name}")

    print(f"writing to {OUT}")
    fig1(sweep)
    fig2(sweep)
    fig3(dyn)
    fig4(sweep)
    fig5(lowlevel)
    fig6(sweep, pd.read_csv(BNCAL))
    fig7(pd.read_csv(UPSAMP))


if __name__ == "__main__":
    main()
