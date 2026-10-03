r"""2608.12408 v3: build paper/Evaluation_Resolution_Confounds_paper_v3.tex from the v2 source.

Only text corrections: every changed number becomes \NV{key} (paper/numbers_v3.tex, from
make_manifest.py); the convention, the noise-bound paragraph, the dataset description and a
version note are updated. No new results (no upsampling follow-up, no blur or band-pass test).
Each replacement must match exactly once in the v2 source, otherwise the script stops.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "paper" / "Evaluation_Resolution_Confounds_paper_v2.tex"
DST = REPO / "paper" / "arxiv_upload_v3" / "Evaluation_Resolution_Confounds_paper_v3.tex"

R = []


def r(old, new):
    R.append((old, new))


# ── preamble ────────────────────────────────────────────────────────────────
r(r"\usepackage{caption}", "\\usepackage{caption}\n\\input{numbers_v3.tex}")

# ── abstract ────────────────────────────────────────────────────────────────
r(r"from $-0.001 \pm 0.007$ at the $32$\,px training resolution to $+0.044 \pm 0.006$ at $224$\,px, growing monotonically across six resolutions ($n=5$ seeds); the gap at the training resolution is $\approx 0$ under the fixed Conv1$\to$V1 mapping and $+0.014$ under best-layer selection",
  r"from $\NV{gap.V1.32}$ (95\% CI $[\NV{boot.V1.32.lo}, \NV{boot.V1.32.hi}]$) at the $32$\,px training resolution to $\NV{boot.V1.224}$ $[\NV{boot.V1.224.lo}, \NV{boot.V1.224.hi}]$ at $224$\,px, growing monotonically across six resolutions ($n=5$ seeds; per-subject RSA on stimulus pairs from different fMRI runs); the gap at the training resolution is $\approx 0$ under the fixed Conv1$\to$V1 mapping and $\NV{bestgap.32}$ under best-layer selection")
r(r"the gap opens by $+0.003 \pm 0.001$ against $+0.030 \pm 0.002$ when content is free to vary, and backprop's decline is abolished ($-0.023 \to -0.000$, $0/5$ and $2/5$ seeds).",
  r"the gap opens by $\NV{ups.gapopen.UPSAMPLED} \pm \NV{ups.gapopen.UPSAMPLED.sem}$ against $\NV{ups.gapopen.NATIVE} \pm \NV{ups.gapopen.NATIVE.sem}$ when content is free to vary, and backprop's decline is abolished ($\NV{ups.bp.NATIVE} \to \NV{ups.bp.UPSAMPLED}$, $0/5$ and $\NV{ups.bp.UPSAMPLED.pos}/5$ seeds).")
r(r"a single scalar luminance value per image reaches $\rho = 0.074$ against V1, essentially matching the untrained network's $0.075$.",
  r"a single scalar luminance value per image reaches $\rho = \NV{ref.lum.V1}$ against V1, essentially matching the untrained network's $\NV{rho.rnd.V1.224}$.")

# ── version note (inserted before the v2 note) ──────────────────────────────
r(r"\textbf{Note on version 2.}",
  r"""\textbf{Note on version 3.} This version changes the convention behind the numbers, not the analyses. (i)~Every RSA value is now computed per subject and averaged over the three subjects (v2: against the RDM averaged over subjects), on the $210{,}205$ of $258{,}840$ stimulus pairs whose two stimuli lie in different fMRI runs for all three subjects. Two subjects saw the stimuli in identical run order, and same-run pairs carry a run component they share (see the revised endpoint study, \citealt{leutenegger2026}). Values are lower in this convention; no comparison between conditions changes sign, and the seed counts are unchanged. The headline V1 gap at $224$\,px is $\NV{boot.V1.224}$, 95\% stimulus-bootstrap CI $[\NV{boot.V1.224.lo}, \NV{boot.V1.224.hi}]$. (ii)~Sweep, best-layer, higher-area and content-control values are from the $30$-checkpoint retrain of the repaired five-seed sweep, which reproduces the published sweep to within $6\times10^{-4}$ at the headline cell; backprop differs between the two runs only within its run-to-run reproducibility (\S4.1). (iii)~The scale section reports the leave-one-subject-out lower bound \citep{nili2014} on cross-run pairs. (iv)~The figures are those of v2 and show the v2 convention (RDM averaged over subjects, all pairs; Fig.~\ref{fig:dynamics}: per subject, all pairs). (v)~One value is not recomputed, because its RDMs were not stored: the calibration on the evaluation stimuli in \S3.2, given in the v2 convention. The joint four-reference partial correlation in \S3.4 could not be reproduced exactly at $32$\,px from its stated definition ($-0.017$ against the printed $-0.020$, v2 convention); the v3 value follows the stated definition. (vi)~The dataset description is corrected to 3T. A manifest of every changed number, with its source, is in the code repository (\texttt{results/paper\_v3/NUMBERS\_MANIFEST\_v3.md}).

\textbf{Note on version 2.}""")

# ── introduction ────────────────────────────────────────────────────────────
r(r"\citep[at $224$\,px, untrained $\rho = 0.075$ vs.\ backprop $\rho = 0.033$; $\Delta\rho = +0.042$, $p<0.001$;][]{leutenegger2026}",
  r"\citep[at $224$\,px, per subject on cross-run pairs, untrained $\rho = \NV{p1.rnd}$ vs.\ backprop $\rho = \NV{p1.bp}$; $\Delta\rho = \NV{p1.gap}$, 95\% CI {$[\NV{p1.gap.lo}, \NV{p1.gap.hi}]$};][]{leutenegger2026}")
r(r"and it vanishes at the $32$\,px training resolution ($-0.001 \pm 0.007$; Fig.~\ref{fig:sweep}, $n=5$ seeds)",
  r"and it vanishes at the $32$\,px training resolution ($\NV{gap.V1.32} \pm \NV{gap.V1.32.sem}$; Fig.~\ref{fig:sweep}, $n=5$ seeds)")
r(r"a single scalar luminance value per image reaches $\rho = 0.074$ against the V1 RDM, essentially matching the untrained network's own $0.075$, and partialling",
  r"a single scalar luminance value per image reaches $\rho = \NV{ref.lum.V1}$ against the V1 RDM, essentially matching the untrained network's own $\NV{rho.rnd.V1.224}$, and partialling")
r(r"(LOC backprop$-$untrained $\approx +0.019$, $5/5$ seeds at both $32$\,px and $224$\,px)",
  r"(LOC backprop$-$untrained $\NV{gap.LOC.32}$ at $32$\,px and $\NV{gap.LOC.224}$ at $224$\,px, $5/5$ seeds at both)")

# ── methods ─────────────────────────────────────────────────────────────────
r(r"over $720$ object images \citep{hebart2023}, averaged across $3$ subjects.",
  r"over $720$ object images from THINGS-fMRI \citep[3T Siemens Magnetom Prisma, $3$ subjects;][]{hebart2023}; each model RDM is compared with each subject's RDM separately.")
r(r"""The main figures report the mean $\pm$ SEM across the $5$ seeds. Two averaging schemes appear in this paper. Figures~\ref{fig:sweep}, \ref{fig:arch} and \ref{fig:higher} correlate each model RDM against the RDM averaged over the three subjects; Figure~\ref{fig:dynamics} correlates against each subject separately and averages the resulting correlations. Averaging RDMs before correlating reduces noise in the brain estimate and yields uniformly higher $\rho$, so the two schemes give different absolute values for the same models. The main effect is unchanged under either: the V1 Random$-$Backprop gap runs from $-0.001 \pm 0.007$ ($3/5$ seeds) at $32$\,px to $+0.044 \pm 0.006$ ($5/5$) at $224$\,px under RDM averaging, and from $+0.000 \pm 0.005$ ($3/5$) to $+0.032 \pm 0.004$ ($5/5$) under per-subject averaging.""",
  r"""Values in the text are the mean $\pm$ SEM across the $5$ seeds, with the number of seeds in which a difference is positive, in the per-subject convention: each model RDM is correlated with each subject's RDM and the three correlations are averaged, using only the $210{,}205$ of $258{,}840$ stimulus pairs whose two stimuli lie in different fMRI runs for all three subjects (two subjects saw the stimuli in identical run order, so same-run pairs share a run component). The headline V1 gap additionally carries a 95\% stimulus-bootstrap CI ($1{,}000$ resamples of the $720$ stimuli, with the cross-run pair set recomputed in each). The figures are unchanged from v2: Figures~\ref{fig:sweep}, \ref{fig:bncal}, \ref{fig:arch}, \ref{fig:content} and \ref{fig:higher} correlate each model RDM with the RDM averaged over the three subjects on all stimulus pairs, and Figure~\ref{fig:dynamics} averages per-subject correlations on all pairs. The conventions give different absolute values for the same models and the same pattern: the V1 Random$-$Backprop gap runs from $\NV{gap.V1.32}$ ($\NV{gap.V1.32.pos}/5$ seeds) at $32$\,px to $\NV{gap.V1.224}$ ($\NV{gap.V1.224.pos}/5$) at $224$\,px per subject on cross-run pairs, from $\NV{mr.cr.V1.32}$ to $\NV{mr.cr.V1.224}$ with RDM averaging on cross-run pairs, and from $\NV{mr.all.V1.32}$ to $\NV{mr.all.V1.224}$ with RDM averaging on all pairs.""")
r(r"""\textbf{Interpreting the scale.} Absolute Spearman $\rho$ values in this setting are low, and we report no noise ceiling for them. The $720$ evaluation stimuli are single-presentation, so no within-subject reliability can be estimated on them, and with three subjects a between-subject estimate is not usable either: at $N=3$ each subject contributes a third of its own target, and the upper bound we obtain sits barely above the value it takes when all shared structure is removed by permutation. A ceiling figure reported in our earlier work \citep{leutenegger2026} rests on such an estimate. We withdraw our use of it here: the reported bounds are not reproducible from the code that produced them, and the quantity they estimate is between-subject consistency rather than measurement reliability.""",
  r"""\textbf{Interpreting the scale.} Absolute Spearman $\rho$ values in this setting are low. The $720$ evaluation stimuli are single-presentation, so no within-subject reliability can be estimated on them. With three subjects the informative reference is the leave-one-subject-out lower bound of \citet{nili2014}, each subject's RDM against the mean of the other two: on cross-run pairs it is $\NV{lb.V1}$ (95\% CI $[\NV{lb.V1.lo}, \NV{lb.V1.hi}]$) at V1, $\NV{lb.V2}$ at V2, $\NV{lb.LOC}$ at LOC and $\NV{lb.IT}$ at IT \citep{leutenegger2026}. The corresponding upper bound is not informative at $N=3$, where each subject contributes a third of its own target. The ceiling figure of earlier versions of our endpoint study, described there as split-half reliability, was not reproducible and is not used.""")
r(r"a single scalar luminance value per image reaches $\rho = 0.074$ against the V1 RDM, essentially matching the best model we tested at $0.075$.",
  r"a single scalar luminance value per image reaches $\rho = \NV{ref.lum.V1}$ against the V1 RDM, essentially matching the untrained network at $224$\,px ($\NV{rho.rnd.V1.224}$) and the lower bound at V1.")

# ── 3.1 ─────────────────────────────────────────────────────────────────────
r(r"backprop from $\rho=0.065$ at $32$\,px to $\rho=0.031$ at $224$\,px, feedback alignment from $0.020$ to $0.012$, predictive coding from $0.026$ to $0.016$, STDP from $0.059$ to $0.037$ ($5$ seeds). The untrained network goes the other way, climbing from $\rho=0.064$ to $\rho=0.075$.",
  r"backprop from $\rho=\NV{rho.bp.V1.32}$ at $32$\,px to $\rho=\NV{rho.bp.V1.224}$ at $224$\,px, feedback alignment from $\NV{rho.fa.V1.32}$ to $\NV{rho.fa.V1.224}$, predictive coding from $\NV{rho.pc.V1.32}$ to $\NV{rho.pc.V1.224}$, STDP from $\NV{rho.stdp.V1.32}$ to $\NV{rho.stdp.V1.224}$ ($5$ seeds). The untrained network goes the other way, climbing from $\rho=\NV{rho.rnd.V1.32}$ to $\rho=\NV{rho.rnd.V1.224}$.")
r(r"It runs from $-0.001 \pm 0.007$ at the $32$\,px training resolution ($3/5$ seeds positive, not significant) to $+0.044 \pm 0.006$ at $224$\,px ($5/5$ seeds positive), growing monotonically across the sweep.",
  r"It runs from $\NV{gap.V1.32} \pm \NV{gap.V1.32.sem}$ at the $32$\,px training resolution ($\NV{gap.V1.32.pos}/5$ seeds positive, not significant; 95\% CI $[\NV{boot.V1.32.lo}, \NV{boot.V1.32.hi}]$) to $\NV{gap.V1.224} \pm \NV{gap.V1.224.sem}$ at $224$\,px ($\NV{gap.V1.224.pos}/5$ seeds positive; 95\% CI $[\NV{boot.V1.224.lo}, \NV{boot.V1.224.hi}]$), growing monotonically across the sweep.")
r(r"and the gap grows in the same way and further: $+0.014 \pm 0.006$ ($4/5$ seeds) at $32$\,px to $+0.060 \pm 0.004$ ($5/5$) at $224$\,px (Appendix~C). One thing does change: under best-layer selection the gap at the training resolution is no longer $\approx 0$ but $+0.014$.",
  r"and the gap grows in the same way and further: $\NV{bestgap.32} \pm \NV{bestgap.32.sem}$ ($\NV{bestgap.32.pos}/5$ seeds) at $32$\,px to $\NV{bestgap.224} \pm \NV{bestgap.224.sem}$ ($\NV{bestgap.224.pos}/5$) at $224$\,px (Appendix~C). One thing does change: under best-layer selection the gap at the training resolution is no longer $\approx 0$ but $\NV{bestgap.32}$.")
r(r"The Random$-$Backprop gap is $\approx 0$ at $32$\,px and grows to $+0.044$ at $224$\,px;",
  r"The Random$-$Backprop gap is $\approx 0$ at $32$\,px and grows to $+0.044$ at $224$\,px in the plotted convention (RDM averaged over subjects, all stimulus pairs; per subject on cross-run pairs, $\NV{gap.V1.224}$);")

# ── 3.2 ─────────────────────────────────────────────────────────────────────
r(r"$-0.003 \pm 0.009$ at $224$\,px with CIFAR-10 statistics ($2/5$ seeds positive) and $-0.011 \pm 0.006$ with THINGS statistics from a disjoint image pool ($2/5$). Calibrating at $32$\,px and evaluating higher costs more ($-0.026 \pm 0.006$ and $-0.020 \pm 0.006$ at $224$\,px). The $2\times2$ design separates the two factors: matching the calibration resolution is worth $+0.023 \pm 0.006$ for CIFAR and $+0.008 \pm 0.002$ for THINGS ($5/5$ seeds each), whereas changing the calibration set at matched resolution is worth $-0.008 \pm 0.006$ ($2/5$).",
  r"$\NV{cal.b-random.224} \pm \NV{cal.b-random.224.sem}$ at $224$\,px with CIFAR-10 statistics ($\NV{cal.b-random.224.pos}/5$ seeds positive) and $\NV{cal.d-random.224} \pm \NV{cal.d-random.224.sem}$ with THINGS statistics from a disjoint image pool ($\NV{cal.d-random.224.pos}/5$). Calibrating at $32$\,px and evaluating higher costs more ($\NV{cal.a-random.224} \pm \NV{cal.a-random.224.sem}$ and $\NV{cal.c-random.224} \pm \NV{cal.c-random.224.sem}$ at $224$\,px). The $2\times2$ design separates the two factors: matching the calibration resolution is worth $\NV{cal.B-A.224} \pm \NV{cal.B-A.224.sem}$ for CIFAR and $\NV{cal.D-C.224} \pm \NV{cal.D-C.224.sem}$ for THINGS ($5/5$ seeds each), whereas changing the calibration set at matched resolution is worth $\NV{cal.D-B.224} \pm \NV{cal.D-B.224.sem}$ ($\NV{cal.D-B.224.pos}/5$).")
r(r"by $+0.041$ (CIFAR calibration) or $+0.033$ (THINGS calibration), $5/5$ seeds in both cases, against $+0.044$ for the uncalibrated baseline;",
  r"by $\NV{cal.b-bp.224}$ (CIFAR calibration) or $\NV{cal.d-bp.224}$ (THINGS calibration), $5/5$ seeds in both cases, against $\NV{gap.V1.224}$ for the uncalibrated baseline;")
r(r"it ranges from $0.001$ to $0.007$ across variants",
  r"it ranges from $\NV{cal.sem.min}$ to $\NV{cal.sem.max}$ across variants")
r(r"costing $-0.030 \pm 0.006$ relative to the identity baseline ($0/5$ seeds positive).",
  r"costing $-0.030 \pm 0.006$ relative to the identity baseline ($0/5$ seeds positive; v2 convention, RDM averaged over subjects on all pairs: the RDMs of this probe were not stored, so it is the one value not recomputed per subject on cross-run pairs).")

# ── 3.3 ─────────────────────────────────────────────────────────────────────
r(r"ResNet-50 from $\rho=0.045$ to $\rho=0.032$, Swin-Tiny from $\rho=0.079$ to $\rho=0.052$.",
  r"ResNet-50 from $\rho=\NV{arch.resnet50.32}$ to $\rho=\NV{arch.resnet50.224}$, Swin-Tiny from $\rho=\NV{arch.swin_t.32}$ to $\rho=\NV{arch.swin_t.224}$.")

# ── 3.4 ─────────────────────────────────────────────────────────────────────
r(r"($\rho\approx0.018$--$0.037$)", r"($\rho\approx\NV{gabor.V1.min}$--$\NV{gabor.V1.max}$)")
r(r"It is itself weakly related to V1 ($\rho=0.030$)", r"It is itself weakly related to V1 ($\rho=\NV{ref.pixel.V1}$)")
r(r"with decreases of $0.004$--$0.008$ \citep{leutenegger2026}; our own partial-RSA implementation reproduces that band ($0.004$ at $224$\,px)",
  r"with decreases of $\NV{p1.pr.min}$--$\NV{p1.pr.max}$ \citep{leutenegger2026}; our own partial-RSA implementation reproduces that band ($\NV{unt.pixel_decrease}$ at $224$\,px)")
r(r"a single scalar luminance value per image reaches $\rho = 0.074$ against the V1 RDM, essentially matching the untrained network's own $0.075$ at $224$\,px, and partialling luminance out of both model and brain RDMs halves the untrained network's alignment ($0.075 \to 0.038$).",
  r"a single scalar luminance value per image reaches $\rho = \NV{ref.lum.V1}$ against the V1 RDM, essentially matching the untrained network's own $\NV{rho.rnd.V1.224}$ at $224$\,px, and partialling luminance out of both model and brain RDMs halves the untrained network's alignment ($\NV{rho.rnd.V1.224} \to \NV{unt.partial_lum}$).")
r(r"leaves the untrained$-$backprop gap climbing from $-0.020$ at $32$\,px to $+0.036$ at $224$\,px, against $+0.044$ uncorrected, so the resolution dependence is not simply a colour effect being read twice; partialling mean luminance alone cuts the gap at $224$\,px from $+0.044$ to $+0.026$ without flattening it.",
  r"leaves the untrained$-$backprop gap climbing from $\NV{gap.joint4.32}$ at $32$\,px to $\NV{gap.joint4.224}$ at $224$\,px, against $\NV{gap.V1.224}$ uncorrected, so the resolution dependence is not simply a colour effect being read twice; partialling mean luminance alone cuts the gap at $224$\,px from $\NV{gap.V1.224}$ to $\NV{gap.lum.224}$ without flattening it.")
r(r"rank together at Spearman $\rho = 0.94$", r"rank together at Spearman $\rho = \NV{six.order}$")
r(r"$\rho = 0.87$ over all $25$ variant-seed points", r"$\rho = \NV{within.rho25}$ over all $25$ variant-seed points")
r(r"($\rho = 0.074$ against $0.045$, $0.028$ and $0.030$).",
  r"($\rho = \NV{ref.lum.V1}$ against $\NV{ref.hist.V1}$, $\NV{ref.rgb.V1}$ and $\NV{ref.pixel.V1}$ for colour histogram, mean RGB and pixels).")
r(r"while raising V1 alignment ($+0.011$, $4/5$)", r"while raising V1 alignment ($\NV{within.B.dV1}$, $\NV{within.B.dV1.pos}/5$)")

# ── 3.5 ─────────────────────────────────────────────────────────────────────
r(r"(backprop $-0.033$ at $64$\,px) while costing the untrained network nothing ($-0.000$).",
  r"(backprop $\NV{ups.step.bp}$ at $64$\,px) while costing the untrained network nothing ($\NV{ups.step.rnd}$).")
r(r"The Random$-$Backprop gap opens by $+0.030 \pm 0.002$ ($5/5$ seeds) with content free to vary and by $+0.003 \pm 0.001$ with it fixed, about a tenth as much. Backprop's decline is abolished outright ($-0.023 \pm 0.002$, $0/5$ seeds positive, to $-0.000 \pm 0.001$, $2/5$), and feedback alignment and predictive coding reverse sign ($-0.004 \to +0.003$ and $-0.005 \to +0.004$, $5/5$ seeds each). One residual survives on the pooling axis, and for one condition only: the untrained network still rises with content fixed ($+0.0029 \pm 0.0003$, $5/5$ seeds), about $44\%$ of its rise in the native arm.",
  r"The Random$-$Backprop gap opens by $\NV{ups.gapopen.NATIVE} \pm \NV{ups.gapopen.NATIVE.sem}$ ($\NV{ups.gapopen.NATIVE.pos}/5$ seeds) with content free to vary and by $\NV{ups.gapopen.UPSAMPLED} \pm \NV{ups.gapopen.UPSAMPLED.sem}$ with it fixed, about a tenth as much. Backprop's decline is abolished outright ($\NV{ups.bp.NATIVE} \pm \NV{ups.bp.NATIVE.sem}$, $0/5$ seeds positive, to $\NV{ups.bp.UPSAMPLED} \pm \NV{ups.bp.UPSAMPLED.sem}$, $\NV{ups.bp.UPSAMPLED.pos}/5$), and feedback alignment and predictive coding reverse sign ($\NV{ups.fa.NATIVE} \to \NV{ups.fa.UPSAMPLED}$ and $\NV{ups.pc.NATIVE} \to \NV{ups.pc.UPSAMPLED}$, $5/5$ seeds each). One residual survives on the pooling axis, and for one condition only: the untrained network still rises with content fixed ($\NV{ups.rnd.UPSAMPLED} \pm \NV{ups.rnd.UPSAMPLED.sem}$, $5/5$ seeds), about $\NV{ups.rnd.frac}\%$ of its rise in the native arm.")
r(r"and the Random$-$Backprop gap opens by $+0.003$ there against $+0.030$ on the left.",
  r"and the Random$-$Backprop gap opens by $+0.003$ there against $+0.030$ on the left (plotted convention, v2 run; per subject on cross-run pairs $\NV{ups.gapopen.UPSAMPLED}$ against $\NV{ups.gapopen.NATIVE}$).")

# ── 3.6 ─────────────────────────────────────────────────────────────────────
r(r"($-0.031 \pm 0.005$ from epoch $0$ to $40$, $5/5$ seeds negative)",
  r"($\NV{td.bp.224} \pm \NV{td.bp.224.sem}$ from epoch $0$ to $40$, $\NV{td.bp.224.neg}/5$ seeds negative)")
r(r"returns to its starting value ($-0.000 \pm 0.005$, $3/5$ seeds negative)",
  r"returns to its starting value ($\NV{td.bp.32} \pm \NV{td.bp.32.sem}$, $\NV{td.bp.32.neg}/5$ seeds negative)")
r(r"($-0.026 \pm 0.004$ and $-0.021 \pm 0.007$; $5/5$ and $4/5$ seeds negative)",
  r"($\NV{td.pc.32} \pm \NV{td.pc.32.sem}$ and $\NV{td.fa.32} \pm \NV{td.fa.32.sem}$; $5/5$ and $4/5$ seeds negative)")

# ── 3.7 ─────────────────────────────────────────────────────────────────────
r(r"(backprop$-$untrained $= +0.019 \pm 0.001$ at $32$\,px and $+0.018 \pm 0.001$ at $224$\,px; $5/5$ seeds throughout). IT shows a weaker but consistently positive version of the same thing ($+0.015 \pm 0.002$ at $32$\,px, $+0.005 \pm 0.001$ at $224$\,px; $5/5$ seeds).",
  r"(backprop$-$untrained $= \NV{gap.LOC.32} \pm \NV{gap.LOC.32.sem}$ at $32$\,px and $\NV{gap.LOC.224} \pm \NV{gap.LOC.224.sem}$ at $224$\,px; $5/5$ seeds throughout). IT shows a weaker but consistently positive version of the same thing ($\NV{gap.IT.32} \pm \NV{gap.IT.32.sem}$ at $32$\,px, $\NV{gap.IT.224} \pm \NV{gap.IT.224.sem}$ at $224$\,px; $5/5$ seeds).")
r(r"backprop's own LOC alignment falls from $0.017$ to $0.013$ over the sweep, but the untrained baseline sits near zero and slightly negative throughout ($-0.002$ to $-0.005$)",
  r"backprop's own LOC alignment falls from $\NV{rho.bp.LOC.32}$ to $\NV{rho.bp.LOC.224}$ over the sweep, but the untrained baseline sits near zero and slightly negative throughout ($\NV{rnd.LOC.max}$ to $\NV{rnd.LOC.min}$)")

# ── discussion / conclusion ─────────────────────────────────────────────────
r(r"letting the pooled positions grow $12$-fold removes about $90\%$ of the effect (\S3.5)",
  r"letting the pooled positions grow $12$-fold removes about $\NV{ups.removed}\%$ of the effect (\S3.5)")
r(r"Our own repaired sweep gives $\Delta\rho = +0.044$ at the same cell.",
  r"Our own repaired sweep gives $\Delta\rho = \NV{gap.V1.224}$ at the same cell in the same convention (the endpoint study's revised value is $\NV{p1.gap}$).")
r(r"letting the pooled positions grow $12$-fold removes about $90\%$ of the effect, so what varies",
  r"letting the pooled positions grow $12$-fold removes about $\NV{ups.removed}\%$ of the effect, so what varies")
r(r"Code and results: \url{https://github.com/nilsleut}.",
  r"Code and results: \url{https://github.com/nilsleut/evaluation-resolution-rsa/tree/arxiv-v3}.")

# ── appendix C table ────────────────────────────────────────────────────────
r(r"Gap, fixed Conv1 & $-0.001$ & $+0.025$ & $+0.044$ \\",
  r"Gap, fixed Conv1 & $\NV{gap.V1.32}$ & $\NV{gap.V1.96}$ & $\NV{gap.V1.224}$ \\")
r(r"Gap, best layer & $+0.014$ & $+0.041$ & $+0.060$ \\",
  r"Gap, best layer & $\NV{bestgap.32}$ & $\NV{bestgap.96}$ & $\NV{bestgap.224}$ \\")
r(r"Seeds positive & $4/5$ & $5/5$ & $5/5$ \\",
  r"Seeds positive (best layer) & $\NV{bestgap.32.pos}/5$ & $\NV{bestgap.96.pos}/5$ & $\NV{bestgap.224.pos}/5$ \\")

# ── bibliography ────────────────────────────────────────────────────────────
r(r"Hebart, M.~N., Contier, O., Teichmann, L., et~al. (2023). THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior. \textit{eLife}, 12:e82580.",
  r"Hebart, M.~N., Contier, O., Teichmann, L., Rockter, A.~H., Zheng, C.~Y., Kidder, A., Corriveau, A., Vaziri-Pashkam, M., and Baker, C.~I. (2023). THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior. \textit{eLife}, 12:e82580.")
r(r"Leutenegger, N. (2026). Untrained CNNs match backpropagation at V1: A systematic RSA comparison of four learning rules against human fMRI. \textit{arXiv:2604.16875}.",
  r"""Leutenegger, N. (2026). Untrained CNNs exceed backpropagation in V1 alignment at high evaluation resolution: A systematic RSA comparison of four learning rules against human fMRI. \textit{arXiv:2604.16875}, v4.

\bibitem[Nili et~al.(2014)]{nili2014}
Nili, H., Wingfield, C., Walther, A., Su, L., Marslen-Wilson, W., and Kriegeskorte, N. (2014). A toolbox for representational similarity analysis. \textit{PLoS Computational Biology}, 10:e1003553.""")


# ── second pass: figures recomputed in the v3 convention (applied after R) ──
R2 = []
for f in ("fig1_v1_sweep", "fig6_bn_calibrated", "fig2_architectures", "fig5_gabor", "fig7_content_control",
          "fig3_dynamics", "fig4_higher_areas"):
    R2.append((f"{{figures/{f}.pdf}}", f"{{figures/{f}_v3.pdf}}"))
R2 += [
    (r"Mean Spearman $\rho$ (Conv1$\to$V1) $\pm$ SEM across $5$ seeds.",
     r"Mean Spearman $\rho$ (Conv1$\to$V1, per subject, cross-run pairs) across $5$ seeds, with 95\% stimulus-bootstrap CIs."),
    (r"The Random$-$Backprop gap is $\approx 0$ at $32$\,px and grows to $+0.044$ at $224$\,px in the plotted convention (RDM averaged over subjects, all stimulus pairs; per subject on cross-run pairs, $\NV{gap.V1.224}$);",
     r"The Random$-$Backprop gap is $\approx 0$ at $32$\,px and grows to $\NV{gap.V1.224}$ at $224$\,px;"),
    (r"mean $\pm$ SEM over $5$ seeds, with stimuli",
     r"mean over $5$ seeds with 95\% stimulus-bootstrap CIs (per subject, cross-run pairs), with stimuli"),
    (r"and the Random$-$Backprop gap opens by $+0.003$ there against $+0.030$ on the left (plotted convention, v2 run; per subject on cross-run pairs $\NV{ups.gapopen.UPSAMPLED}$ against $\NV{ups.gapopen.NATIVE}$).",
     r"and the Random$-$Backprop gap opens by $\NV{ups.gapopen.UPSAMPLED}$ there against $\NV{ups.gapopen.NATIVE}$ on the left."),
    (r"mean across seeds and subjects.",
     r"mean over seeds of per-subject $\rho$ on cross-run pairs, with 95\% stimulus-bootstrap CIs."),
    (r"($\pm$ SEM, $5$ seeds)", r"(per subject, cross-run pairs; 95\% stimulus-bootstrap CIs, $5$ seeds)"),
    (r"The figures are unchanged from v2: Figures~\ref{fig:sweep}, \ref{fig:bncal}, \ref{fig:arch}, \ref{fig:content} and \ref{fig:higher} correlate each model RDM with the RDM averaged over the three subjects on all stimulus pairs, and Figure~\ref{fig:dynamics} averages per-subject correlations on all pairs.",
     r"The figures use the same convention; their error bars are 95\% stimulus-bootstrap CIs of the seed mean (Fig.~\ref{fig:arch}: of the single ResNet-50 and Swin-Tiny checkpoints), and Fig.~\ref{fig:gabor} shows single-seed points as in v2. The model set is the $30$-checkpoint retrain of the repaired sweep; the endpoint study \citep{leutenegger2026} uses a nominally identical retrain of the same five-seed configuration, and its gap at the headline cell ($\NV{p1.gap}$) differs from ours ($\NV{gap.V1.224}$) by less than the seed spread."),
    (r"(iv)~The figures are those of v2 and show the v2 convention (RDM averaged over subjects, all pairs; Fig.~\ref{fig:dynamics}: per subject, all pairs).",
     r"(iv)~All figures are recomputed in this convention; their error bars are now 95\% stimulus-bootstrap CIs rather than across-seed SEM."),
]


# ── third pass: §3.5 corrected (upsampling test, criterion fixed before the run), training-
#    dynamics wording, version-note items, limitation, acknowledgement. No blur / band-pass. ──
SPANS = [   # (start marker, end marker inclusive, replacement)
    (r"A fifth experiment does locate the effect.", r"remains unexplained.",
     r"A fifth experiment, analysed against a criterion fixed before the analysis, separates the two things resolution changes at once. Repeating the sweep on stimuli first reduced to $32$\,px and then upsampled limits the image content to the training resolution while the input still grows. The V1 gap does not shrink: at $224$\,px it is $\NV{s4.V1.up224}$ (95\% CI $[\NV{s4.V1.up224.lo}, \NV{s4.V1.up224.hi}]$) with content limited, against $\NV{boot.V1.224}$ with native content (ratio $\NV{s4.V1.R}$ $[\NV{s4.V1.R.lo}, \NV{s4.V1.R.hi}]$). The gap therefore arises once the input exceeds the training resolution, driven by a decline of backprop; image content above $32$\,px is not needed for it and narrows it. The backprop advantage at LOC, by contrast, disappears under band-limited upsampling."),
    (r"Read at the endpoints the two arms look alike: every sign is preserved", r"could have lived.",
     r"""We evaluated the comparison against a criterion fixed before the analysis (October 2026): the ratio $R = \mathrm{Gap}_{\mathrm{UP}}(224)/\mathrm{Gap}_{\mathrm{NAT}}(224)$ of the V1 Random$-$Backprop gap with content limited to the gap with native content, with a stimulus-bootstrap CI on resamples shared by both arms. $R \geq 0.5$ with a CI of $\mathrm{Gap}_{\mathrm{UP}}$ excluding $0$ was to mean that the effect depends on input size; a CI of $R$ including $0$, that it depends on image content. The result is the first case. At $224$\,px the gap is $\NV{s4.V1.up224}$ $[\NV{s4.V1.up224.lo}, \NV{s4.V1.up224.hi}]$ with content limited and $\NV{boot.V1.224}$ $[\NV{boot.V1.224.lo}, \NV{boot.V1.224.hi}]$ with native content, $R = \NV{s4.V1.R}$ $[\NV{s4.V1.R.lo}, \NV{s4.V1.R.hi}]$; limiting the content widens the gap by $\NV{s4.V1.diff}$ $[\NV{s4.V1.diff.lo}, \NV{s4.V1.diff.hi}]$.

The shape shows where the gap comes from. With content limited it appears in full at the first step ($\NV{s4.V1.up64}$ at $64$\,px) and changes little thereafter: it opens by $\NV{ups.gapopen.UPSAMPLED}$ from $64$ to $224$\,px, against $\NV{ups.gapopen.NATIVE}$ with native content, so growing the number of pooled positions from $1{,}024$ to $12{,}544$ does not carry it either. It is carried by backprop. With content limited, backprop's V1 alignment falls from $\NV{rho.bp.V1.32}$ at $32$\,px to $\NV{s4.bp.up.min}$--$\NV{s4.bp.up.max}$ at every larger input size; with native content it declines gradually to $\NV{rho.bp.V1.224}$ at $224$\,px. The untrained network stays between $\NV{rho.rnd.V1.32}$ and $\NV{rho.rnd.V1.224}$ in both arms. Image content above the training resolution is therefore not needed for the V1 gap; where it is present, it keeps backprop's alignment higher and narrows the gap. The gap arises once the input exceeds the training resolution.

LOC behaves differently. There the backprop advantage is invariant to resolution with native content ($\NV{gap.LOC.32}$ at $32$\,px, $\NV{gap.LOC.224}$ at $224$\,px) but disappears with content limited ($\NV{s4.LOC.up224}$ $[\NV{s4.LOC.up224.lo}, \NV{s4.LOC.up224.hi}]$ at $224$\,px; difference from native $\NV{s4.LOC.diff}$ $[\NV{s4.LOC.diff.lo}, \NV{s4.LOC.diff.hi}]$): it needs natural broadband image content. The ratio $R$ is not informative at LOC ($\NV{s4.LOC.R}$ $[\NV{s4.LOC.R.lo}, \NV{s4.LOC.R.hi}]$) because its denominator is close to zero.

One caution applies to both areas. Upsampled images are smoother than natural images of the same size; the comparison separates information content from input size, but it does not isolate which property of the image is responsible for backprop's decline at V1 or for the loss of its advantage at LOC."""),
]
R3 = [
    (r"\subsection{Content, not pooled positions}", r"\subsection{Input size, not image content}"),
    (r"\textbf{The dependence is on image content, not on pooled positions.}",
     r"\textbf{With image content limited to the training resolution, the V1 gap is as large as with native content.}"),
    (r"with content fixed they drop once at the first step, where the resize chain is introduced, and then run flat or turn back up.",
     r"with content limited they drop at the first step, as soon as the input exceeds the training resolution, and then run flat."),
    (r"(3)~We separate the two things evaluation resolution changes at once and locate the dependence on the image-content axis rather than the pooling axis.",
     r"(3)~We separate the two things evaluation resolution changes at once and show that the V1 gap does not require image content above the training resolution: it arises once the input exceeds that resolution, driven by a decline of backprop."),
    (r"Section~3.5 then separates the two things resolution changes at once and finds the dependence on the content axis rather than the pooling axis.",
     r"Section~3.5 then separates the two things resolution changes at once and finds that the V1 gap follows the input size, not the image content."),
    (r"Section~3.5 constrains the pure form of it: in the upsampled arm the filters cover exactly the same fraction of the image as in the native arm at every resolution, so a receptive-field account alone predicts no difference between the arms, and backprop's decline nevertheless disappears. What survives is a mixed statement, that the filter needs detail on its own spatial scale, and we have not tested it.",
     r"Section~3.5 fits it: in the upsampled arm the filters cover the same fraction of the image as in the native arm, and the gap appears in full as soon as the input exceeds the training resolution, whether or not image content above that resolution is present. What the account does not explain is why that content, where present, narrows the gap."),
    (r"holding image content fixed at the training resolution while letting the pooled positions grow $12$-fold removes about $\NV{ups.removed}\%$ of the effect (\S3.5), so the dependence lives on the content axis. Why detail above $32$\,px should help random filters and hurt trained ones is the question that remains.",
     r"limiting the image content to the training resolution leaves the V1 gap at $224$\,px intact ($R = \NV{s4.V1.R}$, \S3.5), so the gap follows the input size relative to the training resolution and is carried by a decline of backprop; image content above the training resolution narrows it. Why trained filters lose alignment once the input exceeds their training scale is the question that remains."),
    (r"A fifth experiment locates it: capping image detail at the training resolution while letting the pooled positions grow $12$-fold removes about $\NV{ups.removed}\%$ of the effect, so what varies with evaluation resolution is the image detail and not the number of averaged positions.",
     r"A fifth experiment, analysed against a criterion fixed before the analysis, locates it: with image content limited to the training resolution the V1 gap at $224$\,px is as large as with native content ($R = \NV{s4.V1.R}$ $[\NV{s4.V1.R.lo}, \NV{s4.V1.R.hi}]$), so it arises once the input exceeds the training resolution, driven by a decline of backprop, and content above that resolution narrows it rather than creating it."),
    (r"Section~3.5 places the dependence on the content axis, which narrows the search without ending it: the next question is which property of the detail above the training resolution is responsible, and a bandpass decomposition of the stimuli would be the natural way to ask.",
     r"Section~3.5 ties the V1 gap to the input size relative to the training resolution, which narrows the search without ending it: the next question is which property of the enlarged input lowers backprop's alignment, and why image content above the training resolution partly offsets it."),
    (r"tracks the same dependence, accumulating epoch by epoch.", r"tracks the same dependence."),
    (r"At $224$\,px backprop V1 alignment falls epoch by epoch ($\NV{td.bp.224} \pm \NV{td.bp.224.sem}$ from epoch $0$ to $40$, $\NV{td.bp.224.neg}/5$ seeds negative),",
     r"At $224$\,px backprop V1 alignment drops sharply in the first epoch, from $\NV{td.bp224.e0}$ to $\NV{td.bp224.e1}$, and then recovers partly, to $\NV{td.bp224.e40}$ at epoch $40$ (net change $\NV{td.bp.224} \pm \NV{td.bp.224.sem}$, $\NV{td.bp.224.neg}/5$ seeds negative),"),
    (r"At $224$\,px backprop appears to degrade; at $32$\,px it returns to baseline, showing no net degradation.",
     r"At $224$\,px backprop drops in the first epoch and recovers only partly; at $32$\,px it returns to baseline, showing no net degradation."),
    (r"but the absolute scale should not be read as a property of V1.",
     r"but the absolute scale should not be read as a property of V1. (13)~The upsampled images of \S3.5 are smoother than natural images of the same size. The comparison separates information content from input size, but it does not isolate which image property is responsible for backprop's decline."),
    (r"(vi)~The dataset description is corrected to 3T.",
     r"(vi)~The dataset description is corrected to 3T. (vii)~v2 stated that image detail above the $32$\,px training resolution carries the V1 effect. A preregistered evaluation-only upsampling analysis shows the opposite: with image content limited to $32$\,px the gap is larger ($R = \NV{s4.V1.R}$ $[\NV{s4.V1.R.lo}, \NV{s4.V1.R.hi}]$); it arises when the input size exceeds the training resolution, driven by a decline of backpropagation. v2 had measured only the growth of the gap between $64$ and $224$\,px within each arm and set aside the jump at the first upsampled step as a resizing artifact; that jump is the effect. (viii)~The training-dynamics description is corrected: at $224$\,px backprop does not decline epoch by epoch but drops in the first epoch and then recovers partly (\S3.6)."),
    (r"(a)~Mean Spearman", r"(a)~Mean Spearman"),  # placeholder kept idempotent (see fig. 1 caption below)
    (r"Mean Spearman $\rho$ (Conv1$\to$V1, per subject, cross-run pairs) across $5$ seeds, with 95\% stimulus-bootstrap CIs.",
     r"(a)~Mean Spearman $\rho$ (Conv1$\to$V1, per subject, cross-run pairs) across $5$ seeds, with 95\% stimulus-bootstrap CIs."),
    (r"after low-level statistics are partialled out (\S3.4).}",
     r"after low-level statistics are partialled out (\S3.4). (b)~Random$-$Backprop gap, paired per seed and averaged over the $5$ seeds, with paired 95\% stimulus-bootstrap CIs, for native stimuli and for stimuli with content limited to $32$\,px (\S3.5).}"),
    (r"The author thanks Martin Schrimpf for the arXiv endorsement and helpful feedback, and the creators of",
     r"The author thanks the creators of"),
    (r"datasets and the Brain-Score team for their infrastructure.",
     r"datasets and the Brain-Score team for their infrastructure. Analysis code and text drafts were developed with the assistance of an AI coding assistant; all results were verified by the author."),
]
R3 = [x for x in R3 if x[0] != x[1]]


def main():
    t = SRC.read_text(encoding="utf-8")
    for old, new in R:
        n = t.count(old)
        assert n == 1, (n, old[:90])
        t = t.replace(old, new)
    for old, new in R2:
        n = t.count(old)
        assert n == 1, (n, old[:90])
        t = t.replace(old, new)
    for start, end, new in SPANS:
        assert t.count(start) == 1, start[:80]
        i = t.index(start)
        j = t.index(end, i) + len(end)
        t = t[:i] + new + t[j:]
    for old, new in R3:
        n = t.count(old)
        assert n == 1, (n, old[:90])
        t = t.replace(old, new)
    DST.write_text(t, encoding="utf-8")
    print(f"{len(R)} replacements -> {DST.name}")


if __name__ == "__main__":
    main()
