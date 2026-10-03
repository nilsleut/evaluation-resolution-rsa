# 2608.12408 v3 — Section 3.5: conflict with the data, and a proposed correction

Status: **proposal, not applied.** The v3 source (`paper/arxiv_upload_v3/`) still carries the
v2 interpretation of §3.5 with v3 numbers. This note documents why that interpretation does
not hold and offers replacement text. Every number below is per subject on cross-run pairs,
v12 checkpoints, mean over 5 seeds (sources in brackets).

## What §3.5 claims (v2, unchanged in the v3 draft)

- Abstract: "The dependence is carried by image detail above the training resolution rather
  than by the number of pooled positions."
- §3.5: the first upsampled step (32 → 64 px) is "a large one-off penalty" caused by
  introducing the resize chain, "a property of the content and not of the pooling"; over
  64 → 224 px with content fixed the gap barely opens (+0.002 vs +0.021 native), so the
  dependence lives on the content axis; "the decline of the trained conditions requires that
  detail entirely"; "detail above 32 px helps random filters slightly and hurts trained ones
  considerably".
- Discussion / Conclusion: capping detail "removes about 89 % of the effect".

## What the data show

| | 32 | 64 | 96 | 128 | 160 | 224 |
|---|---|---|---|---|---|---|
| gap NATIVE (Random − BP, V1) | −0.001 | +0.010 | +0.018 | +0.023 | +0.026 | +0.030 |
| gap UPSAMPLED (content ≤ 32 px) | −0.001 | +0.032 | +0.036 | +0.036 | +0.035 | +0.035 |
| BP NATIVE | 0.046 | 0.039 | 0.033 | 0.029 | 0.026 | 0.023 |
| BP UPSAMPLED | 0.046 | 0.016 | 0.014 | 0.015 | 0.015 | 0.016 |
| Random NATIVE | 0.045 | 0.048 | 0.050 | 0.051 | 0.052 | 0.053 |
| Random UPSAMPLED | 0.045 | 0.048 | 0.050 | 0.050 | 0.050 | 0.051 |

[`results/upsampling/step4_gaps.csv`, `rsa_seed*.csv`]

1. **Removing detail above 32 px widens the gap at every resolution ≥ 64 px**:
   UPSAMPLED − NATIVE = +0.023 [0.017, 0.028] at 64 px … +0.004 [0.001, 0.008] at 224 px,
   5/5 seeds at 64–160 px [`step5_paired_diff.csv`]. Detail above the training resolution
   therefore does not hurt backprop; its absence does. BP aligns *worse* without it at every
   resolution.
2. **The gap at 224 px is fully present without that detail**: 0.035 vs 0.030, R = 1.14
   [1.03, 1.34]. Under the preregistered criterion of the upsampling test this is verdict A,
   "the effect depends on input size / receptive-field geometry", not on information content
   [`results/upsampling/UPSAMPLING_REPORT.md`].
3. **The "first-step penalty" is not a resize-chain artifact.** Gaussian low-pass of the
   native 224 px image, with no resize chain and no change of input size, also widens the gap:
   +0.006 to +0.010 above NATIVE at every blur level, 4–5/5 seeds
   [`results/bandpass_exploratory/blur/blur_paired.csv`; exploratory].
4. What *is* true in §3.5: with content capped, the gap barely changes between 64 and 224 px
   (+0.002 vs +0.021 native). That is a statement about the *growth* of the gap across the
   sweep, not about whether the gap needs detail.

So the native sweep's resolution dependence is not "carried by detail above the training
resolution". At a given input size, that detail *reduces* the gap (it keeps backprop's
alignment up); once the image is presented larger than the training scale, backprop loses
alignment whether or not the detail is there, and loses more without it.

## Proposed replacement text (for decision)

**Abstract**, replacing "A fifth experiment does locate the effect … remains unexplained.":

> A fifth experiment separates the two things resolution changes at once. Repeating the sweep
> on stimuli first reduced to 32 px and then upsampled caps the image detail at the training
> resolution while the network still sees the larger input. With detail capped, the gap
> appears in full at the first step (64 px) and changes little thereafter (+0.002 from 64 to
> 224 px, against +0.021 with native content), and at 224 px it is as large as with native
> content (+0.035 against +0.030). Detail above the training resolution is therefore not
> what creates the gap; at a given input size it narrows it.

**§3.5, paragraphs 2–4**, replacing the "resize chain … property of the content" reading:

> Read at the endpoints the two arms look alike, and they are: at 224 px the gap is
> $+0.035$ with detail capped and $+0.030$ without, a ratio of 1.14 [1.03, 1.34]. The arms
> differ in shape. With detail capped, backprop loses its alignment at the first step
> ($0.046 \to 0.016$ at 64 px) and stays there, while the untrained network is unaffected;
> with native detail, backprop declines gradually. At every resolution from 64 px on,
> removing the detail widens the gap (by $+0.023$ at 64 px and $+0.004$ at 224 px). The
> first step is not an artifact of the resize chain: low-passing the native 224 px image,
> which leaves input size and resizing unchanged, widens the gap as well (exploratory).
>
> The dependence is therefore not carried by detail above the training resolution. What the
> experiment shows is that backprop's V1 alignment drops once the image is presented larger
> than the scale it was trained at, and that the detail available at the new scale partly
> offsets that drop, the less so the larger the image. Why the trained filters behave this
> way remains open.

**Discussion "Mechanism"** and **Conclusion**: replace "removes about 89 % of the effect …
so what varies with evaluation resolution is the image detail and not the number of averaged
positions" with a sentence consistent with the above (e.g. "capping image detail at the
training resolution leaves the gap at 224 px intact (ratio 1.14) and removes only its gradual
growth across 64–224 px; detail above the training resolution narrows the gap rather than
creating it").

**Contribution (3)**: "We separate the two things evaluation resolution changes at once and
locate the dependence on the image-content axis rather than the pooling axis" → "…and show
that the gap does not require image detail above the training resolution".

**Discussion "Which resolution is correct?"** already anticipates this ("the filter needs
detail on its own spatial scale"); it can stay with one sentence adjusted to the above.

## What this would not change

The headline (+0.030 at 224 px, ≈ 0 at 32 px), §3.1–3.4, §3.6–3.7, and the recommendation to
evaluate at the training resolution and report the resolution are unaffected. The blur and
band-pass results are exploratory and would be cited only as supporting the "not a
resize-chain artifact" point, or left out.
