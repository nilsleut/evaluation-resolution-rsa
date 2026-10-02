# Retrain report — 30 checkpoints, and the seed-42 determinism gate

Config `8163d8c4a146` (`config.lock`, `schema_version: 1`).
Run: `code/learning_rules_v12_allseeds_modal.py`, app `learning-rules-rsa-allseeds`,
30 shards, T4, all completed, **0 failures**, ~3.3 min wall clock.

Artifacts on the Modal volume `learning-rules-rsa`, under `outputs_allseeds/`:
`checkpoints/` (30 × `model_weights_{rule}_seed{n}.pt`), `shards/` (30 CSVs),
`rdms/`, `rsa_resolution_sweep.csv` (1,080 rows).

---

## 1. The gate: BRANCH 2 — not bit-identical

Per `PREREGISTRATION.md` §6, both branches were written before the run. The outcome is
branch 2, **but the split across rules is the whole story**:

| Rule | Checkpoint tensors differing | ρ deviation (all 180 cells) |
|---|---|---|
| Random Weights | **0 / 22** | **0.00e+00** |
| Random Weights (BN-calibrated) | **0 / 22** | **0.00e+00** |
| Predictive Coding | **0 / 24** | **0.00e+00** |
| STDP | **0 / 19** | **0.00e+00** |
| Feedback Alignment | 22 / 28 | 1.0e−05 |
| Backprop | 19 / 22 | 4.6e−03 (max) |

**The four rules the repo documents as reproducing bit-identically do reproduce
bit-identically** — in the weights *and* in ρ, exactly zero to the last digit, across all
180 cells each. The preregistration named a mismatch in those four as "the more serious
of the two possible findings". **It did not occur.**

The mismatch is confined to the two gradient-trained rules, which is the qualitative
pattern the documented nondeterminism of convolution backward on GPU predicts.

### Divergence by depth — corrected (P1)

**An earlier draft of this report cited Backprop's 63.6% `fc1.weight` divergence while
discussing the Conv1→V1 claim. That was a category error: `fc1` is downstream of Conv1
and cannot affect the measured features at all.** The per-tensor breakdown, which is what
should have been reported:

| Backprop tensor | rel-norm | | Conv1-block detail | rel-norm |
|---|---|---|---|---|
| `conv1.0.weight` | **4.9%** | | `conv1.1.weight` (γ) | 0.6% |
| `conv2.0.weight` | 23.0% | | `conv1.1.bias` (β) | 10.5% |
| `conv3.0.weight` | 41.5% | | `conv1.1.running_mean` | 15.4% |
| `fc1.weight` | 63.6% | | `conv1.1.running_var` | 13.9% |

Divergence grows monotonically with depth, and **the measured layer sits at the shallow
end of that gradient.** So neither of the two clean readings is right on its own: Conv1
does genuinely diverge (all 864 elements, max |Δ| 2.8e−2, and the BN statistics that
normalise its output diverge by ~14–15%), but it diverges 13× less than `fc1`.

The claim that survives is about ρ, and it does not depend on the fc1 number:

| Layer→ROI | Backprop weight rel-norm | max \|Δρ\| (seed 42) |
|---|---|---|
| Conv1→V1 | 4.9% (BN ~15%) | 1.30e−3 |
| Conv2→V1 | 23.0% | 2.23e−3 |
| Conv3→LOC | 41.5% | 5.78e−4 |
| FC1→IT | 63.6% | 6.54e−4 |

ρ agreement is ~1e−3 at every layer **with no depth trend** — FC1→IT, with the largest
weight divergence, agrees *better* than Conv2→V1 with a quarter of it. So "substantially
different weights, near-identical representational geometry" is supportable, but the
evidence for it is at the deep layers, not at Conv1, where the divergence is modest and
the near-identity is unremarkable.

Feedback Alignment diverges by ~1e−5 to 1e−4 throughout. One caveat on reading its table:
the `conv{n}.0.b` bias tensors show rel-norm up to 1.41 with max |Δ| of 2e−10 — the biases
initialise to zeros and barely move, so the denominator is ~0 and **relative norm is
uninformative for those tensors.** Absolute magnitude is the one to read there.

The repo's documented "~10⁻³" figure is about **ρ**, not weights. So the gate cannot judge
magnitude, and it now says so rather than labelling a 63% weight difference "consistent
with documented ~1e-3 drift", which is what it printed on the first run. Both that and a
second reporting defect — an incomplete run announced as `BRANCH 2: NOT IDENTICAL` when
nothing had mismatched — were fixed, with regression tests added for each.

## 2. The second gate: the published table reproduces

`code/ridge/check_csv_reproduction.py`, run against `results/bnfix_sweep.csv`. Because all
30 pairs were recomputed, this covers **the whole 1,080-row published table**, not only
seed 42. Structure confirmed: 1,080 rows, 30/30 complete (rule, seed) pairs, **6**
layer→ROI pairs (see `DOC_PATCHES_PENDING.md` patch 2).

**The headline finding reproduces.**

| res | published | retrain | shift | shift / SEM |
|---|---|---|---|---|
| 32 | −0.0009 ± 0.0073, 3/5 | −0.0006 ± 0.0073, 3/5 | +0.0003 | 0.05 |
| 64 | +0.0137 ± 0.0071, 3/5 | +0.0138 ± 0.0072, 3/5 | +0.0001 | 0.01 |
| 96 | +0.0250 ± 0.0067, 5/5 | +0.0249 ± 0.0069, 5/5 | −0.0002 | 0.02 |
| 128 | +0.0326 ± 0.0064, 5/5 | +0.0322 ± 0.0067, 5/5 | −0.0003 | 0.05 |
| 160 | +0.0370 ± 0.0062, 5/5 | +0.0366 ± 0.0066, 5/5 | −0.0004 | 0.07 |
| 224 | +0.0441 ± 0.0060, 5/5 | +0.0435 ± 0.0064, 5/5 | −0.0006 | 0.09 |

Sign pattern, seeds-positive counts and monotonicity all preserved; the largest shift is
0.0006, under a tenth of its own SEM and 1.4% of the effect.

### One number the repo should update

Backprop |Δρ|, retrain vs published:

- at Conv1→V1, the cell the claim rests on: mean 1.1–1.3e−3, **max 2.3e−3**
- across all 180 Backprop cells: median 8.7e−4, p95 3.0e−3, **max 4.6e−3**
- 77 of 180 cells exceed 1e−3

So "gradient-trained conditions reproduce across separate runs only to about 10⁻³ at V1"
(`README.md`, and the paper's Methods) is **accurate for the typical case and understates
the tail by roughly 2–4×**. A phrasing such as "to about 10⁻³ at V1, with excursions to
5×10⁻³ at other layer/ROI cells" would match what we measured. This is a
characterisation, not an error, and it does not touch the headline.

**What this does mean for the error bars — stated descriptively (P2).** An earlier draft
said the nondeterminism was "about a sixth of the reported seed-level SEM". That was an
overreach: **one re-run per rule is a single realization of |Δ|, not a variance estimate**,
and a ratio of a single |Δ| to an SEM is not a variance decomposition.

What was measured, and all that was measured: a single re-run produced ρ deviations up to
4.6e−3 across 1,080 cells, with 77/180 Backprop cells exceeding 1e−3 at Conv1→V1 and a
mean of 1.1–1.3e−3 there. The four deterministic rules deviated by exactly 0.

A variance component would need repeated Backprop runs at a fixed seed — three more would
support one. That is deferred, not done, and no variance claim is made here.

The seed remains a usable unit of inference for the four deterministic rules without
qualification, and for BP/FA with the above attached as a stated, unquantified second
source.

## 3. Checkpoint validation

- **30/30** load through the hardened `models_v2.load_model(rule, weights_dir, seed_idx)`
  and forward-pass to the documented Conv1 shape (2, 32).
- **30/30 pairwise distinct** by sha256 over all tensors — no silent duplicates, which
  would have been the quiet way for a seed to be missing.
- Loader hardening tests: **41/41 pass** (15 negative, deliberately triggering every new
  raise; 26 positive on real checkpoints).
- Determinism-gate self-test: passes, both branches plus the partial case exercised,
  15 key-naming cases.

## 4. Verdict against the preregistration

Branch 2 fired, so per §6 the instruction is **report per-tensor magnitudes and stop**.
Done, and stopping here.

The finding is narrower and less alarming than "branch 2" alone implies, and should be
reported as such:

> Four of the six conditions reproduce bit-identically in weights and in ρ. The two
> gradient-trained conditions do not, which is the documented behaviour; the resulting
> shift in the headline gap is at most 0.0006, under a tenth of its SEM, and the published
> table reproduces in sign, monotonicity and seeds-positive counts at every resolution.

**Not claimed:** that Backprop's weight-space divergence is *caused* by Adam's
amplification of conv-backward nondeterminism. That is the most plausible account and it
fits the FA contrast, but nothing in *this* run isolates it. It is tested separately by
`code/ridge/determinism_probe_modal.py` — see `PRECONDITIONS_P1_P4.md`.

## 5. Ready for Phase 2

- 30 checkpoints, all six conditions × five seeds, validated.
- `config.yaml` frozen and hashed; `PREREGISTRATION.md` written before compute.
- Loader raises instead of silently substituting a random-init or wrong-seed model.
- Both regression gates exist, are tested, and have been run.
