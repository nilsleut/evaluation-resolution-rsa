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
    DST.write_text(t, encoding="utf-8")
    print(f"{len(R)} replacements -> {DST.name}")


if __name__ == "__main__":
    main()
