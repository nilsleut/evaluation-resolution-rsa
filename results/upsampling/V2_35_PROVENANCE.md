# Where the v2 claim of §3.5 came from

**Core.** v2 §3.5 used the v11 run (`code/learning_rules_v11_upsample_modal.py`, a separate
retrain of the five-seed sweep; `results/rsa_resolution_sweep_upsample.csv`), scored against
the RDM averaged over subjects on all stimulus pairs. It reported the *growth* of the
Random − Backprop gap between 64 and 224 px within each arm (+0.030 native, +0.003 upsampled)
and treated the upsampled arm's jump at the first step as a one-off "resize-chain" penalty,
so that only the flat 64–224 px stretch was read as the test; on that reading the gap does
not grow without detail, hence "detail above 32 px carries the effect". The v11 script's own
pre-stated prediction ("UPSAMPLED flat at its 32 px value → the effect follows image
content") is not what the data show: the upsampled gap jumps from −0.001 at 32 px to +0.047
at 64 px and ends at +0.050 at 224 px, above the native +0.044.

## Quantities in v2 §3.5 (v11 data, RDM averaged over subjects, all pairs)

| v2 phrase | definition | v2 value |
|---|---|---|
| "first step … backprop −0.033 at 64 px", "untrained nothing (−0.000)" | ρ(UPSAMPLED, 64 px) − ρ(NATIVE, 64 px), per condition | BP −0.033, Random −0.000 |
| "gap opens by +0.030 with content free to vary" | [gap(224) − gap(64)] in the NATIVE arm, gap = Random − BP at Conv1→V1 | +0.030 |
| "… and by +0.003 with it fixed" | the same in the UPSAMPLED arm | +0.003 |
| "backprop's decline is abolished (−0.023 → −0.000)" | ρ_BP(224) − ρ_BP(64), NATIVE vs UPSAMPLED | −0.023 / −0.000 |
| "removes about 90 % of the effect" | 1 − 0.003 / 0.030 | 90 % |

Gap levels behind them (v11, same convention):

| px | 32 | 64 | 96 | 128 | 160 | 224 |
|---|---|---|---|---|---|---|
| gap NATIVE | −0.0005 | +0.0140 | +0.0251 | +0.0325 | +0.0369 | +0.0438 |
| gap UPSAMPLED | −0.0005 | +0.0472 | +0.0520 | +0.0515 | +0.0512 | +0.0503 |
| Backprop NATIVE | 0.0650 | 0.0548 | 0.0463 | 0.0404 | 0.0371 | 0.0317 |
| Backprop UPSAMPLED | 0.0650 | 0.0214 | 0.0189 | 0.0195 | 0.0203 | 0.0213 |

## Why the reading does not hold

"The effect" in v2 §3.5 is the growth of the gap across 64–224 px, not the gap itself. With
content capped at 32 px the gap is present in full at 64 px and at 224 px exceeds the native
gap, so removing detail above the training resolution does not remove the gap; it enlarges
it. The evaluation-only upsampling analysis with a criterion fixed before the run (October
2026; `UPSAMPLING_REPORT.md`) gives, per subject on cross-run pairs with v12 checkpoints,
R = Gap_UP(224) / Gap_NAT(224) = 1.14 [1.03, 1.34]: verdict "input size / receptive-field
geometry", not "information content". The v11 numbers above show the same pattern in the v2
convention, so the v2 conclusion followed from the choice of quantity, not from a different
data set.
