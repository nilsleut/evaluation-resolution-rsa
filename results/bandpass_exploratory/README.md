# Blur and band-pass tests — exploratory

**Exploratory. Not part of any paper version. No correction for multiple comparisons.**
These analyses were run after the upsampling test (`results/upsampling/`) to ask which
spatial-frequency content carries the V1 gap (Random − Backprop) and the LOC advantage
(Backprop − Random). They are inference-only (no retraining), on the 30 v12 checkpoints
(Random Weights and Backprop, 5 seeds each), and use the pipeline and conventions of
`results/upsampling/step4`: Conv1→V1 and Conv3→LOC, per subject (primary) and
mean-RDM, cross-run stimulus pairs, mean over 5 seeds of per-seed differences, 95%
stimulus-bootstrap CIs from the same 1,000 resamples (noise_ceiling_v2 index matrix,
seed 20261002), so every condition is paired with NATIVE@224 and UPSAMPLED@224.

Many conditions are reported side by side (5 blur levels, 10 band-pass conditions, two
ROIs, two conventions, several paired contrasts). The CIs are per comparison; none is
adjusted for the number of comparisons, and no result here was specified in advance.

## Files

| File | Content | Code |
|---|---|---|
| `blur/blur_gaps.csv` | gap per blur level, plus NATIVE@224 and UPSAMPLED@224 | `code/blur/blur_extract.py`, `blur_bootstrap.py` |
| `blur/blur_paired.csv` | blur_r − NATIVE, blur_r − UPSAMPLED, blur96 − blur32, blur96 − blur160 | `code/blur/blur_paired.py` |
| `blur/blur_transform.json` | sigma per level; per-image correlation with the UPSAMPLED image of the same r | `code/blur/blur_extract.py` |
| `bandpass/bandpass_gaps.csv` | gap per band condition (band only, notch) and minus NATIVE | `code/bandpass/bandpass_extract.py`, `bandpass_bootstrap.py` |
| `bandpass/bandpass_transform.json` | sigmas, band labels, per-band pixel variance, reconstruction error | `code/bandpass/bandpass_extract.py` |

Model RDMs and bootstrap draws are in `data/blur/` and `data/bandpass/` (not committed).

## Filters

**Blur.** Gaussian low-pass of the native 224 px image whose amplitude response is 0.5 at
the Nyquist frequency of an r-px grid (r/2 cycles per image):
σ(r) = √(ln 2 / (2π²)) · 2·224 / r ≈ 84 / r px, r ∈ {32, 64, 96, 128, 160}; applied per
channel to the [0, 1] image after Resize(224) → CenterCrop(224), before normalisation.

**Band-pass.** Octave bands by difference of Gaussians with the same σ convention,
cutoffs at effective resolutions 112, 56, 28, 14 px: 112–224, 56–112, 28–56, 14–28 and
the residual 0–14 (which contains the image mean). The bands sum to the image
(max reconstruction error < 1e−5). Two conditions per band: the band alone, and the
native image minus the band ("notch").

### Design decision: grey shift

A band-pass image other than the residual has zero mean, and the notch of the residual
removes the mean. Fed to the network unchanged, such an image would sit at the CIFAR-10
normalisation offset rather than at a plausible image mean. These conditions are therefore
shifted by a **constant mid-grey equal to the CIFAR-10 channel means (0.4914, 0.4822,
0.4465), identical for every image**. The shift restores a plausible input range without
reintroducing any image-specific mean luminance, which matters here because a single
luminance value per image predicts V1 about as well as the untrained network. The band 0–14
alone and the notches of the four finer bands keep their own image means and are not
shifted. Other choices (no shift; shift by each image's own mean) would test different
questions and were not run.

## What the numbers suggest (exploratory reading)

- V1: removing fine detail (blur, notch of the finest octave) widens the gap; the coarse
  band (≤ 14 px effective resolution, including the image mean) alone reproduces the
  native gap, and removing it reverses the gap.
- LOC: no single band carries the Backprop advantage; removing the finest octave or the
  coarse band reduces it, removing the middle octaves does not; blur reduces it
  monotonically.

These readings are not tested claims.
