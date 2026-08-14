# `code/` — what produced what

Every file in [`../results/`](../results/) is traceable to one script here. Files were
copied from their working trees unmodified, with a single exception noted below, so
the docstrings describe the runs as they were actually executed.

## Result → script

| Result file | Produced by |
|---|---|
| `bnfix_sweep.csv` | `learning_rules_v10_sweep_modal.py` (+ `spawn_runs.py`, CIs by `recompute_bootstrap_ci_v2.py`) |
| `rsa_resolution_sweep_upsample.csv` | `learning_rules_v11_upsample_modal.py` (+ `spawn_upsample.py`) |
| `training_dynamics_results.csv` | `training_dynamics_rsa_modal_v2.py` (+ `spawn_runs.py`) |
| `bncal_rows.csv` | `bn_calibration_control_modal.py` (+ `spawn_bncal.py`) |
| `colour_statistic_control.csv` | `colour_statistic_control.py` |
| `colour_within_filter.csv` | `colour_within_filter.py`, over RDMs saved by `bn_calibration_rdms_modal.py` (+ `spawn_bncal_rdms.py`) |
| `rsa_resolution_sweep_resnet.csv` | `cross_species/run_resolution_sweep_resnet.py` |
| `rsa_resolution_sweep_swin.csv` | `cross_species/run_resolution_sweep_swin.py` |
| `rsa_resolution_control_macaque.csv` | `cross_species/run_resolution_control_macaque_v2.py` |
| `rsa_lowlevel_control.csv` | `cross_species/run_lowlevel_control_v2.py` |
| `../figures/*.pdf` | `make_figures.py` |

## The one file that was edited

`make_figures.py` originally read from two separate working trees by absolute path.
Its path block now points at `../results/`; the plotting code is untouched. Running it
in a clean clone reproduces the seven PDFs in `../figures/` exactly, apart from the
`/CreationDate` timestamp matplotlib embeds (three bytes per file) — which is the check
that the shipped data is the data behind the paper:

```bash
cd code && python make_figures.py     # overwrites ../figures/ with identical output
```

## `bn_guard.py` — the assertion helper

The defect this paper corrects (§4.1) was a silent one: `PC_CNN` and `STDP_CNN`
overrode `eval()` with a no-op, so `model.eval()` did nothing for those two classes
and their BatchNorm layers stayed in training mode during feature extraction.

`bn_guard.py` is the check that would have caught it, and it covers both directions:

- `assert_bn_eval(model)` — call before feature extraction. Catches evaluation
  contaminated by evaluation-set batch statistics.
- `assert_bn_train(model)` — call before a training epoch. Catches the opposite
  failure, where a model that evaluated at a checkpoint never flipped back and its
  running statistics silently freeze for the rest of the run.

Scripts that evaluate at intermediate milestones (`training_dynamics_rsa_modal_v2.py`)
need both, on every train → eval → train cycle.

One detail is deliberate and worth preserving in any re-use: `iter_bn()` raises if it
finds **no** BatchNorm layers, rather than passing vacuously. A vacuous pass is
exactly how the original defect hid — if layer discovery misses the class with the
broken `eval()`, the check proves nothing. It also handles both model shapes in this
codebase, `nn.Module` subclasses and the plain class that merely holds `nn.Module`
attributes.

`cross_species/bn_guard.py` is the same module, vendored so that subtree runs
standalone.

## Layout

```
code/
├── bn_guard.py                          shared BatchNorm mode assertions
├── learning_rules_v10_sweep_modal.py    main 5-seed × 6-resolution sweep
├── learning_rules_v11_upsample_modal.py content control (NATIVE vs UPSAMPLED arms)
├── training_dynamics_rsa_modal_v2.py    8 milestones × 2 resolutions
├── bn_calibration_control_modal.py      2×2 BN-calibration factorial
├── bn_calibration_rdms_modal.py         same factorial, saving RDMs to disk
├── colour_statistic_control.py          global-statistic references, across rules
├── colour_within_filter.py              same test, convolutional weights held fixed
├── recompute_bootstrap_ci_v2.py         rebuilds ci_lo/ci_hi from saved RDMs, on CPU
├── spawn_runs.py                        detached submission for the two main runs
├── spawn_bncal.py / spawn_bncal_rdms.py / spawn_upsample.py
├── make_figures.py                      all seven figures from ../results/
└── cross_species/
    ├── models_v2.py                     model loading + feature extraction (BN-fixed)
    ├── data_loader.py                   THINGS-fMRI and Brain-Score loading
    ├── rsa_engine.py                    RDM construction, RSA, bootstrap CIs
    ├── bn_guard.py                      vendored copy
    ├── run_resolution_sweep_human_v2.py
    ├── run_resolution_control_macaque_v2.py
    ├── run_lowlevel_control_v2.py
    ├── run_resolution_sweep_resnet.py   ImageNet ResNet-50, eval only
    └── run_resolution_sweep_swin.py     Swin-Tiny, eval only
```

## Version and naming notes

- **`_v2` in `cross_species/` means BatchNorm-mode-fixed.** The pre-fix originals are
  not shipped. `run_resolution_sweep_resnet.py` and `run_resolution_sweep_swin.py`
  carry no `_v2` suffix because torchvision models were never subject to the defect,
  so there was nothing to repair.
- **`learning_rules_v10_sweep_modal.py` supersedes v9**, which is not shipped. v10
  differs in one respect: `finalize_sweep` no longer collapses a duplicated
  `(rule, seed_idx, res, layer, roi)` key by keeping one row. The run is seeded and
  deterministic, so two computations of the same cell must agree exactly; if they do
  not, a respawned shard changed a number, and that is a signal rather than noise.
  The merge now verifies every numeric column across duplicates and raises on
  disagreement. Everything else is identical to v9.
- **`spawn_*.py` exist because `modal run --detach` did not survive the host entering
  Modern Standby** — both jobs died two seconds after it, two hours in. Deploy +
  spawn removes the client relationship: the call is registered server-side and the
  local process can exit immediately.

## What is not here

Model checkpoints (`.pt`), saved RDMs (`.npy`), fMRI volumes, and the THINGS stimulus
images are excluded by `.gitignore`. See the data availability section of the
top-level [README](../README.md) for where to obtain each.
