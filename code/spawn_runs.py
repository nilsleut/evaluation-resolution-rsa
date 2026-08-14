"""
spawn_runs.py
=============
Client-independent submission for both BN-fixed runs.

`modal run --detach` did not survive the host entering Modern Standby (both jobs died
2 s after it, at 2 h in). deploy + spawn removes the client relationship entirely: the
call is registered server-side and this process can exit immediately.

Prereq:
  python -m modal deploy learning_rules_v9_sweep_modal.py
  python -m modal deploy training_dynamics_rsa_modal_v2.py

Run:
  python spawn_runs.py            # spawn everything, write call ids to spawned_calls.json
  python spawn_runs.py --poll     # poll the ids in spawned_calls.json, no new spawns
"""

import json
import sys
from pathlib import Path

import modal

RULES = ["Random Weights", "Backprop", "Feedback Alignment", "Predictive Coding", "STDP",
         "Random Weights (BN-calibrated)"]
SEEDS = [42, 123, 456, 789, 1337]

# Timeouts sized from the measured post-bootstrap pair time (0.64 min for an untrained
# 6-resolution pair) plus training, then doubled. Untrained pairs get a floor well above
# 2x because container cold start + THINGS/CIFAR volume reads dominate a sub-minute job.
# PC/STDP run T_INF=10 / T_SIM=10 inner loops per batch, so their 40-epoch training is
# roughly an order of magnitude heavier than BP/FA -- they get the widest window.
TIMEOUT_S = {
    "Random Weights":                 30 * 60,
    "Random Weights (BN-calibrated)": 30 * 60,
    "Backprop":                       60 * 60,
    "Feedback Alignment":             60 * 60,
    "Predictive Coding":            4 * 60 * 60,
    "STDP":                         4 * 60 * 60,
}

CALLS_FILE = Path(__file__).with_name("spawned_calls.json")


def spawn_all():
    sweep = modal.Function.from_name("learning-rules-rsa", "run_resolution_control")
    td = modal.Function.from_name("training-dynamics-rsa", "run_training_dynamics")

    calls = []

    # ── training-dynamics: one resumed call, no sharding (17/25 pairs already done) ──
    c = td.spawn()
    calls.append({"app": "training-dynamics-rsa", "fn": "run_training_dynamics",
                  "label": "training-dynamics (resume, all seeds)", "call_id": c.object_id})
    print(f"{'training-dynamics (resume)':<46} {c.object_id}")

    # ── sweep: 30 shards, one per (rule, seed) ──────────────────────────────────────
    for rule in RULES:
        f = sweep.with_options(timeout=TIMEOUT_S[rule])
        for si in range(len(SEEDS)):
            c = f.spawn(only_rule=rule, only_seed=si)
            label = f"sweep {rule} s{si}"
            calls.append({"app": "learning-rules-rsa", "fn": "run_resolution_control",
                          "label": label, "call_id": c.object_id,
                          "timeout_min": TIMEOUT_S[rule] // 60})
            print(f"{label:<46} {c.object_id}")

    CALLS_FILE.write_text(json.dumps(calls, indent=2))
    print(f"\n{len(calls)} calls spawned -> {CALLS_FILE.name}")
    return calls


def poll():
    calls = json.loads(CALLS_FILE.read_text())
    done = running = failed = 0
    for c in calls:
        fc = modal.FunctionCall.from_id(c["call_id"])
        try:
            r = fc.get(timeout=0)
            status = f"DONE  {r}"
            done += 1
        except TimeoutError:
            status = "running"
            running += 1
        except Exception as e:
            status = f"FAILED {type(e).__name__}: {str(e)[:70]}"
            failed += 1
        print(f"{c['label']:<46} {status}")
    print(f"\ndone={done}  running={running}  failed={failed}  total={len(calls)}")


if __name__ == "__main__":
    poll() if "--poll" in sys.argv else spawn_all()
