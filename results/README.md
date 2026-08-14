# `results/` — data dictionary

Every number in the paper comes from a file in this directory. All of them are
plain CSV, comma-separated, UTF-8, with a header row. Together they regenerate all
seven figures:

```bash
cd code && python make_figures.py      # reads only ../results/, writes ../figures/
```

> **All files here come from the repaired implementation** described in §4.1 of the
> paper, in which the predictive-coding and STDP model classes no longer override
> `eval()` with a no-op. Two pre-repair files ship alongside their repaired
> counterparts for comparison and are suffixed `_superseded`; they are not used by
> any figure or by any number in the paper, and each carries a `#` comment header
> saying so. See [Superseded files](#superseded-files).

**Contents**

- [Two things that are easy to get wrong](#two-things-that-are-easy-to-get-wrong)
  - [1. `rho` vs `rho_sub_mean`](#1-rho-vs-rho_sub_mean)
  - [2. Which BN-calibration variant is which](#2-which-bn-calibration-variant-is-which)
- [Shared column conventions](#shared-column-conventions)
- [File-by-file](#file-by-file)
- [Superseded files](#superseded-files)

---

## Two things that are easy to get wrong

These two distinctions are load-bearing and are documented nowhere else, including
in the paper itself beyond a sentence in Methods.

### 1. `rho` vs `rho_sub_mean`

Several files carry both columns. They are two different averaging schemes over the
three THINGS-fMRI subjects, computed from the same model RDM and the same three
subject RDMs. They are **not** interchangeable, and they do not differ by a constant.

| Column | Procedure | In words |
|---|---|---|
| `rho` | `brain_mean = mean(RDM_sub01, RDM_sub02, RDM_sub03)` element-wise, then `spearmanr(model_utri, brain_mean_utri)` — **one** correlation | *Average the RDMs, then correlate* |
| `rho_sub_mean` | `spearmanr(model_utri, RDM_sub_i_utri)` for each subject *i*, then take the arithmetic mean of the three values | *Correlate per subject, then average* |

`utri` is the strict upper triangle (`np.triu_indices(n, k=1)`) of the 720×720
stimulus RDM; when model and brain RDMs disagree in size, both are truncated to the
smaller `n` first.

**Why it matters.** Averaging RDMs before correlating cancels independent
measurement noise in the brain estimate, so `rho` is **uniformly higher** than
`rho_sub_mean` for the same model. Absolute values from the two schemes are not
comparable, and a value quoted from one will not reproduce against the other.

**Why the paper reports both.** The main effect survives either choice, which is the
point of showing them. For the headline Random − Backprop gap at Conv1→V1:

| Scheme | 32 px | 224 px |
|---|---|---|
| `rho` (RDM-averaged) | −0.001 ± 0.007, 3/5 seeds | +0.044 ± 0.006, 5/5 seeds |
| `rho_sub_mean` (per-subject) | +0.000 ± 0.005, 3/5 seeds | +0.032 ± 0.004, 5/5 seeds |

Which scheme each figure uses:

| Figures | Scheme | Source |
|---|---|---|
| fig1, fig2, fig4, fig6, fig7 | `rho` | `bnfix_sweep.csv`, `bncal_rows.csv`, `rsa_resolution_sweep_upsample.csv`, ResNet/Swin files |
| fig3 (training dynamics) | per-subject, averaged | `training_dynamics_results.csv`, which stores one row **per subject** and is averaged downstream |

`training_dynamics_results.csv` is therefore the per-subject scheme by construction —
it has a `subject` column and no `rho_sub_mean` column. Averaging its `rho` over
`subject` gives the `rho_sub_mean` quantity; there is no RDM-averaged column in that
file at all.

### 2. Which BN-calibration variant is which

`bncal_rows.csv` and `colour_within_filter.csv` use short variant labels for the
2×2 calibration factorial of §3.2. The label alone does not say what was done.

Calibration means: starting from the untrained network, estimate BatchNorm
statistics from 50 forward passes in training mode with **no weight updates**, using
an unbiased cumulative average (`momentum=None`) and a deterministic loader with no
augmentation. Convolutional weights are verified bit-identical to the plain untrained
condition afterwards (sha256 per seed).

The factorial crosses **calibration set** × **calibration resolution**:

| Label in CSV | Calibration set | Calibration resolution | Role |
|---|---|---|---|
| `Random (identity BN)` | *none* | — | **Reference.** BN at initialization: `running_mean=0`, `running_var=1`, `num_batches_tracked=0`, so in eval mode BN is the identity. This is the "untrained" / "Random Weights" condition of the main sweep. |
| `A CIFAR@32` | CIFAR-10 | 32 px (fixed) | Calibrated once at the training resolution, then evaluated at all six resolutions. |
| `B CIFAR@eval-res` | CIFAR-10 | = evaluation resolution | CIFAR-10 upsampled to whatever px is being evaluated; recalibrated once per resolution. |
| `C THINGS-disjoint@32` | THINGS | 32 px (fixed) | Images drawn from the 1,134 THINGS concepts that contribute **no** evaluation stimulus. |
| `D THINGS-disjoint@res` | THINGS | = evaluation resolution | As C, recalibrated once per resolution. |

Reading the design: **A/B** vs **C/D** is the distribution factor; **A/C** vs **B/D**
is the resolution factor. `B@32` is a no-op resize of `A@32`, so `B@32 == A@32`
exactly — an internal consistency check that holds in the data.

Two further labels appear in `bncal_rows.csv` only:

| Label | What it is |
|---|---|
| `C_leak THINGS-720@32` | Leakage probe. Calibrated on the **720 evaluation stimuli themselves**, at 32 px. |
| `D_leak THINGS-720@res` | Leakage probe. Same, at the evaluation resolution. |

The leak probes are deliberate positive controls for what calibrating on the test
images would have bought; they are **not** part of the factorial. `D_leak` gives, by
construction, an almost exact estimate of the evaluation activations and is the worst
of the eight variants — the result discussed at the end of §3.2.

#### `A_aug` is retired and appears in no paper number

| Label | Status |
|---|---|
| `A_aug CIFAR@32` | **Retired as procedurally flawed. Does not appear in the paper.** |

`A_aug` reproduces the original, pre-factorial BN-calibrated condition: it uses the
**augmented** CIFAR-10 loader (random crop/flip), so its calibration statistics are
estimated from a stochastic input distribution that the model is never evaluated on,
and the run is not reproducible across invocations. The factorial arms A/B/C/D all
use one matched, deterministic loader instead. `A_aug` was run only as a
one-variable bridge to the previously published −0.065 figure and is retained in
`bncal_rows.csv` for that audit trail alone.

> **This is the trap.** The condition labelled **`Random Weights (BN-calibrated)` in
> `bnfix_sweep.csv` is the `A_aug` variant**, not any arm of the factorial. Its rows
> in `bnfix_sweep.csv` and in `colour_statistic_control.csv` are therefore **not**
> used for any calibration claim in the paper, and should not be read as "the
> calibrated baseline". The calibrated baselines are A/B/C/D in `bncal_rows.csv`.
> `colour_within_filter.csv` exists precisely because of this: it re-runs the
> within-filter colour test over the five sound variants only.

---

## Shared column conventions

| Column | Meaning |
|---|---|
| `rule` / `variant` / `model` | Condition label. See per-file notes; capitalisation and spelling differ between files because they were written by different scripts. |
| `layer` | Model layer the RDM was built from: `Conv1`, `Conv2`, `Conv3`, `FC1` for the custom CNN; `layer1`/`layer2`/`layer4` for ResNet-50; `early`/`late` for Swin-Tiny. Convolutional and stage feature maps are global-average-pooled before the RDM is built. |
| `roi` / `region` | Brain region. Human (THINGS-fMRI): `V1`, `V2`, `V3`, `V4`, `LOC`, `IT`. Macaque: `V1`, `V2` from FreemanZiemba2013, `V4`, `IT` from MajajHong2015. |
| `res` | **Evaluation** resolution in pixels, one of `32, 64, 96, 128, 160, 224`. Training is always at 32 px; only evaluation varies. This is the manipulation the paper is about. |
| `rho` | Spearman rank correlation between the strict upper triangles of the model RDM and the brain RDM. Model RDMs use correlation distance on globally pooled features. |
| `seed` | Training seed: one of `42, 123, 456, 789, 1337`. |
| `seed_idx` | Index of the same seed, `0`–`4`, in that order. Files that carry only one of the two use whichever their script wrote. |
| `n_subs` | Number of THINGS-fMRI subjects contributing (always `3`). |
| `n_stim` | Number of stimuli behind the brain RDM: `720` (THINGS), `135` (FreemanZiemba2013 textures), `3200` (MajajHong2015 objects). |
| `ci_lo`, `ci_hi` | 95% bootstrap CI over **stimulus pairs** (not seeds): 500 resamples of the RDM upper-triangle vector, `numpy` default RNG seeded at 42, 2.5/97.5 percentiles. **Mostly empty — see the note under `bnfix_sweep.csv`.** |

Error bars in the figures are **SEM across the 5 seeds**, computed at plot time from
the per-seed rows. They are not the `ci_lo`/`ci_hi` columns, which quantify a
different thing (stimulus-sampling uncertainty within a single run).

---

## File-by-file

### `bnfix_sweep.csv` — main 5-seed resolution sweep

The central result. Endpoint (epoch 40) RSA for every condition × seed × layer × ROI ×
resolution. **1,080 data rows.** Produced by `code/learning_rules_v10_sweep_modal.py`
(CIs backfilled by `code/recompute_bootstrap_ci_v2.py`).

| Column | Type | Notes |
|---|---|---|
| `rule` | str | `Random Weights`, `Backprop`, `Feedback Alignment`, `Predictive Coding`, `STDP`, `Random Weights (BN-calibrated)` |
| `layer` | str | `Conv1`, `Conv2`, `Conv3`, `FC1` |
| `roi` | str | `V1`, `V2`, `LOC`, `IT` |
| `res` | int | 32 / 64 / 96 / 128 / 160 / 224 |
| `rho` | float | RDM-averaged scheme — see [§1](#1-rho-vs-rho_sub_mean) |
| `ci_lo`, `ci_hi` | float | Bootstrap CI, **empty in 1,008 of 1,080 rows** |
| `n_subs` | int | 3 |
| `rho_sub_mean` | float | Per-subject scheme — see [§1](#1-rho-vs-rho_sub_mean) |
| `seed` | int | 42 / 123 / 456 / 789 / 1337 |
| `seed_idx` | int | 0–4 |

Grid: 6 conditions × 5 seeds × 6 resolutions × (4 layer→ROI pairs) = 1,080.

**On the empty CIs.** The sweep computes `ci_lo`/`ci_hi` off the GPU path
(`COMPUTE_BOOTSTRAP_CI = False`) because 500 Spearman calls over ~258k pairs per
cell dominated wall time, and no figure uses them. Only **72 rows are filled**:
`Random Weights` and `Backprop`, seed 42, all layers/ROIs/resolutions — the subset
that was backfilled by `recompute_bootstrap_ci_v2.py` as a spot check. Every other
row is empty by design, not by failure. The CIs are a pure function of RDMs saved on
the Modal volume and can be rebuilt for any subset with that script.

⚠️ `Random Weights (BN-calibrated)` in this file is the retired `A_aug` variant.
See [§2](#2-which-bn-calibration-variant-is-which).

---

### `bncal_rows.csv` — 2×2 BN-calibration factorial (§3.2, fig6)

Per-seed rows for the calibration design. **1,440 data rows.** Produced by
`code/bn_calibration_control_modal.py`.

| Column | Type | Notes |
|---|---|---|
| `variant` | str | 8 labels — **see the variant key in [§2](#2-which-bn-calibration-variant-is-which)** |
| `seed` | int | 42 / 123 / 456 / 789 / 1337 |
| `seed_idx` | int | 0–4 |
| `res` | int | Evaluation resolution |
| `layer` | str | `Conv1`, `Conv2`, `Conv3`, `FC1` |
| `roi` | str | `V1`, `V2`, `LOC`, `IT` |
| `rho` | float | RDM-averaged scheme |
| `n_subs` | int | 3 |
| `rho_sub_mean` | float | Per-subject scheme |

Grid: 8 variants × 5 seeds × 6 resolutions × 4 layer→ROI pairs = 1,440. There are no
`ci_lo`/`ci_hi` columns in this file at all; the paired contrasts in §3.2 are reported
as seed counts rather than SEM, because at *n* = 5 the paired SEM is itself unstable
(it ranges 0.001–0.009 across variants with comparable per-seed differences).

All eight variants share **bit-identical convolutional weights** with the plain
untrained condition at the same seed. Only the stored BN statistics differ. That is
what makes this an intervention rather than a comparison.

---

### `colour_statistic_control.csv` — global-statistic account, across rules (§3.4)

Tests whether global average pooling drives the descriptor toward a global image
statistic as resolution rises. **180 data rows.** Produced by
`code/colour_statistic_control.py`.

| Column | Type | Notes |
|---|---|---|
| `rule` | str | Same six labels as `bnfix_sweep.csv` |
| `res` | int | Evaluation resolution |
| `seed_idx` | int | 0–4 |
| `pooled_positions` | int | Spatial positions the Conv1 map is pooled over: `(res/2)²` → 256, 1024, 2304, 4096, 6400, 12544. The "dose" of the pooling account. |
| `rho_v1` | float | Conv1→V1 alignment, RDM-averaged scheme. Matches `bnfix_sweep.csv` for the same cell. |
| `rho_MEANRGB` | float | Model RDM vs per-image RGB channel means (3-dim), correlation distance |
| `rho_COLORHIST` | float | Model RDM vs per-image joint RGB histogram, 8×8×8 bins, correlation distance |
| `rho_MEANLUM` | float | Model RDM vs per-image Rec.709 luminance (**one scalar per image**), absolute difference |
| `rho_PIXEL` | float | Model RDM vs flattened 224×224×3 pixels, correlation distance |

All four reference RDMs are built **once, at 224 px** (`REF_PX = 224`) and are
resolution-invariant by construction — RDMs built at 32 px vs 224 px correlate at
ρ ≥ 0.958. Any resolution trend in these columns therefore comes from the model side.

`rho_PIXEL` here is **not** the same object as `sim_pixel` in
`rsa_lowlevel_control.csv`; it is recomputed inside this script for contrast, on the
same reference geometry as the other three.

⚠️ The `Random Weights (BN-calibrated)` rows are the retired `A_aug` variant and are
**not used** by the paper. That is why the next file exists.

---

### `colour_within_filter.csv` — the same test with filters held fixed (§3.4)

The rank correlation supporting the global-statistic account (ρ = 0.94, *n* = 6) runs
across the six *rules* of `bnfix_sweep.csv`, whose convolutional filters all differ —
so filters, training and normalisation vary together. This file repeats the test
across five conditions that share **bit-identical convolutional weights** (verified by
sha256 per seed) and identical pooling geometry, differing only in stored BN
statistics. **150 data rows.** Produced by `code/colour_within_filter.py` from RDMs
saved by `code/bn_calibration_rdms_modal.py`.

| Column | Type | Notes |
|---|---|---|
| `variant` | str | The **five sound variants only**: `Random (identity BN)`, `A CIFAR@32`, `B CIFAR@eval-res`, `C THINGS-disjoint@32`, `D THINGS-disjoint@res`. No `A_aug`, no leak probes. |
| `res` | int | Evaluation resolution |
| `seed_idx` | int | 0–4 |
| `rho_v1` | float | Conv1→V1 alignment, RDM-averaged scheme |
| `rho_MEANRGB`, `rho_COLORHIST`, `rho_MEANLUM`, `rho_PIXEL` | float | As above, identical reference RDMs and rank procedure (reused from `colour_statistic_control.py`) |

Grid: 5 variants × 5 seeds × 6 resolutions = 150.

`B CIFAR@eval-res` is the case that decides the section: it is the only calibration
variant whose V1 alignment *rises* with resolution while its luminance similarity
*falls* — the response rising as the putative dose falls.

---

### `rsa_resolution_sweep_upsample.csv` — content control (§3.5, fig7)

Separates the two things evaluation resolution changes at once: the number of pooled
positions, and the image detail available at each. **1,800 data rows.** Produced by
`code/learning_rules_v11_upsample_modal.py`.

Schema is `bnfix_sweep.csv` plus one column:

| Column | Type | Notes |
|---|---|---|
| `arm` | str | `NATIVE` — stimuli resized directly to `res`. `UPSAMPLED` — stimuli resized to **32 px first**, then upsampled to `res`, so image content is capped at the training resolution while the network still pools over the full number of positions. |

All other columns are as in `bnfix_sweep.csv`. `ci_lo`/`ci_hi` are **empty in all
1,800 rows** — they were never backfilled for this run.

Grid: 5 conditions (no BN-calibrated arm here) × 5 seeds × 6 resolutions × 4
layer→ROI pairs × 2 arms = 1,800.

Two properties that make the contrast exact:

- Both arms are evaluated on the **same model object in the same process**, with
  identical interpolation, antialiasing, cropping and normalization.
- At 32 px the second resize is a no-op, so the arms are identical by construction.
  This holds in the data (max |Δρ| = 0 across all cells) and is worth re-checking as
  a sanity test on any re-run.

Reproducibility caveat: convolution backward is nondeterministic on GPU, so the
gradient-trained conditions reproduce across separate runs only to about 10⁻³ at V1.
Conditions that never backpropagate through a convolution (random, predictive coding,
STDP) reproduce bit-identically.

---

### `training_dynamics_results.csv` — alignment over training (§3.6, fig3)

RSA at eight training milestones, at two resolutions. **23,760 data rows.** Produced
by `code/training_dynamics_rsa_modal_v2.py`.

| Column | Type | Notes |
|---|---|---|
| `rule` | str | `Random Weights`, `Backprop`, `Feedback Alignment`, `Predictive Coding`, `STDP` |
| `layer` | str | `Conv1`, `Conv2`, `Conv3`, `FC1` |
| `roi` | str | `V1`, `V2`, `V3`, `V4`, `LOC`, `IT` — six ROIs here, more than the sweep files |
| `subject` | str | `sub-01`, `sub-02`, `sub-03` — **one row per subject** |
| `seed_idx` | int | 0–4 |
| `epoch` | int | `0, 1, 2, 5, 10, 20, 30, 40`. Epoch 0 is the network at initialization. |
| `res` | int | `32` or `224` only — this file does not sweep all six |
| `rho` | float | **Per-subject** Spearman ρ against that subject's RDM. See [§1](#1-rho-vs-rho_sub_mean). |
| `pval` | float | The *p* returned by `scipy.stats.spearmanr` on the RDM upper triangles. It tests ρ ≠ 0 over ~258k stimulus pairs, which are not independent, so it is **near-zero almost everywhere and should not be used for inference.** It is recorded for completeness only. |

Grid: the four trained rules each contribute 4 layers × 6 ROIs × 3 subjects × 5 seeds
× 8 epochs × 2 resolutions = 5,760 rows. `Random Weights` is the network at
initialization and therefore exists **at epoch 0 only**, contributing 720 rows.
Total 4 × 5,760 + 720 = 23,760. A rule-by-epoch cross-tabulation of this file is
ragged for that reason, and that is expected, not missing data.

Because rows are per subject, this file uses the per-subject averaging scheme by
construction. To get a curve, average `rho` over `subject` and then over `seed_idx`.
`Random Weights` coincides exactly with each trained rule's own epoch-0 value where
initializations are shared — that is not a duplication error. (Feedback alignment and
STDP initialize with `kaiming_normal_` rather than PyTorch's default
`kaiming_uniform_`, so for those two the epoch-0 network is *not* the `Random Weights`
network; see Limitations item 6.)

Inference in §3.6 uses a paired, one-sided sign-flip permutation test over the five
seeds (2⁵ = 32 assignments, smallest attainable *p* = 1/32 ≈ 0.031), not `pval`.

---

### `rsa_resolution_sweep_resnet.csv` and `rsa_resolution_sweep_swin.csv` — additional architectures (§3.3, fig2)

Two ImageNet-pretrained models, evaluated without further training, against the same
720 THINGS stimuli. **24 and 18 data rows.** Produced by
`code/cross_species/run_resolution_sweep_resnet.py` and `..._swin.py`.

| Column | Type | Notes |
|---|---|---|
| `rule` | str | `resnet50` or `swin_t` (the column is named `rule` for schema compatibility; these are architectures, not learning rules) |
| `roi` | str | ResNet: `V1`, `V2`, `V4`, `IT`. Swin: `V1`, `V2`, `IT`. |
| `layer` | str | ResNet: `layer1`→V1/V2, `layer2`→V4, `layer4`→IT. Swin: `early`→V1/V2, `late`→IT. |
| `res` | int | Evaluation resolution |
| `rho` | float | RDM-averaged scheme |
| `ci_lo`, `ci_hi` | float | 95% bootstrap CI over stimulus pairs — **fully populated in these two files** |
| `n_stim` | int | 720 |

No `seed` column: these are single pretrained checkpoints, so there is one run per
cell and the error bars in fig2 are the bootstrap CIs, not seed SEM.

These two files are **not** affected by the `eval()` defect — torchvision models were
never subject to it — which is why they have no `_bnfix` counterpart. Both models
were trained at 224 px and both nonetheless align best at low resolution, which is
what rules out the train/eval-matching account.

---

### `rsa_resolution_control_macaque.csv` — macaque electrophysiology (§3.6)

Cross-species directional check via Brain-Score. **40 data rows.** Produced by
`code/cross_species/run_resolution_control_macaque_v2.py`.

| Column | Type | Notes |
|---|---|---|
| `rule` | str | `random`, `backprop`, `feedback_alignment`, `predictive_coding`, `stdp` — **lowercase with underscores here**, unlike the human files |
| `layer` | str | `Conv1`→V1/V2, `Conv2`→V4, `FC1`→IT |
| `region` | str | Column is named `region`, not `roi`. `V1`, `V2` (FreemanZiemba2013); `V4`, `IT` (MajajHong2015) |
| `res` | int | `32` or `224` only |
| `rho` | float | Spearman ρ against the neural RDM |
| `ci_lo`, `ci_hi` | float | 95% bootstrap CI over stimulus pairs — fully populated |
| `n_stim` | int | `135` (FZ textures) or `3200` (HVM objects) |

**Single seed.** There is no `seed` column: this is seed 42 only. The paper treats
the macaque result as directional support, not as an independently powered
replication — see Limitations items 1 and 2. The two datasets also use different
stimuli for different regions (textures for V1/V2, objects for V4/IT), a known
confound in that comparison.

---

### `rsa_lowlevel_control.csv` — Gabor and pixel references (§3.4, fig5)

Tests whether V1 alignment tracks low-level filter structure. **30 data rows.**
Produced by `code/cross_species/run_lowlevel_control_v2.py`.

| Column | Type | Notes |
|---|---|---|
| `model` | str | `cnn_random`, `cnn_backprop`, `resnet50`, and the two parameter-free references `GABOR` and `PIXEL` themselves |
| `res` | int | Evaluation resolution |
| `align_v1` | float | Model RDM vs V1 brain RDM, Spearman ρ |
| `sim_gabor` | float | Model RDM vs Gabor filterbank RDM (4 orientations × 3 spatial frequencies, energy). `1.0` on the `GABOR` rows by construction. |
| `sim_pixel` | float | Model RDM vs raw-luminance (pixel) RDM. `1.0` on the `PIXEL` rows by construction. |

Single seed (42), so no `seed` column. The `PIXEL` row's `align_v1` is constant at
0.011358 across resolutions because the pixel reference is resolution-invariant.

The counterexample this file carries: ResNet-50's early stage is roughly ten times
more Gabor-like than the untrained CNN (`sim_gabor` ≈ 0.22–0.31 vs ≈ 0.02–0.03) and
is nonetheless *less* V1-aligned.

---

## Superseded files

Two files ship in both a repaired and a pre-repair version, so the effect of the
`eval()` defect on the cross-species and low-level controls can be inspected
directly. **The unsuffixed name is always the repaired file, and is the one the
paper and the figures use.**

| Repaired (use this) | Superseded (reference only) |
|---|---|
| `rsa_lowlevel_control.csv` | `rsa_lowlevel_control_superseded.csv` |
| `rsa_resolution_control_macaque.csv` | `rsa_resolution_control_macaque_superseded.csv` |

Both superseded files begin with a `#` comment header repeating this warning, so that
someone who downloads a single file out of context still sees it. That header means
they need `comment='#'` to parse:

```python
pd.read_csv("rsa_resolution_control_macaque_superseded.csv", comment="#")
```

The repaired files have no comment header and parse with a bare `read_csv`.

What changes between them:

- **Macaque.** `predictive_coding` and `stdp` move substantially — these are the two
  conditions the defect touched. STDP at V1/224 px falls from ρ = 0.305 to 0.266;
  predictive coding at IT/224 px from 0.125 to 0.096. `random` and
  `feedback_alignment` are unchanged to ~10⁻³; `backprop` likewise.
- **Low-level.** Only the `cnn_backprop` rows differ, and they differ because the
  repaired run is the one consistent with the main sweep: `cnn_backprop` at 32 px is
  0.06285 in the repaired file, which matches `bnfix_sweep.csv` seed 42 exactly. The
  pre-repair value, 0.059652, does not.

The two files with no pre-repair counterpart in this directory are the ResNet-50 and
Swin-Tiny sweeps, which the defect could not affect. Everything else here
(`bnfix_sweep.csv`, `bncal_rows.csv`, both colour files, the upsample sweep, the
training dynamics) was generated only after the repair, so no pre-repair version
exists to ship.

The defect and its consequences are described in §4.1 of the paper. Correction notes
identifying the affected results accompany the current arXiv versions of the three
earlier preprints.
