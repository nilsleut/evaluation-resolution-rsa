"""
spawn_upsample.py
=================
deploy + spawn for the NATIVE vs UPSAMPLED arms (learning_rules_v11_upsample_modal.py).

deploy + spawn rather than `modal run`, because the host has entered Modern Standby
mid-job twice and that kills a client-attached run.

  python -m modal deploy learning_rules_v11_upsample_modal.py
  python spawn_upsample.py
  python spawn_upsample.py --poll
"""

import json
import sys
from pathlib import Path

import modal

from learning_rules_v11_upsample_modal import RULES, SEEDS

CALLS_FILE = Path(__file__).with_name("spawned_upsample_calls.json")

# Measured on the bnfix sweep: Random 0.6-0.8 min, BP/FA 1.3-1.7, PC/STDP 1.5-2.0 per
# (rule, seed) with SIX evaluations. This run does TWELVE, so roughly double, plus the
# same training. These windows are ~15x that, not the 4 h guesses of the first run.
TIMEOUT_S = {
    "Random Weights":     20 * 60,
    "Backprop":           45 * 60,
    "Feedback Alignment": 45 * 60,
    "Predictive Coding":  60 * 60,
    "STDP":               60 * 60,
}


def spawn():
    f = modal.Function.from_name("learning-rules-upsample", "run_resolution_control")
    calls = json.loads(CALLS_FILE.read_text()) if CALLS_FILE.exists() else []
    known = {(c["rule"], c["seed_idx"]) for c in calls}
    for rule in RULES:
        fn = f.with_options(timeout=TIMEOUT_S[rule])
        for si in range(len(SEEDS)):
            if (rule, si) in known:
                print(f"{rule} s{si} already spawned, skipping"); continue
            c = fn.spawn(only_rule=rule, only_seed=si)
            calls.append({"rule": rule, "seed_idx": si, "call_id": c.object_id,
                          "timeout_min": TIMEOUT_S[rule] // 60})
            print(f"{rule:<22} s{si}  {c.object_id}")
    CALLS_FILE.write_text(json.dumps(calls, indent=2))
    print(f"\n{len(calls)} call(s) -> {CALLS_FILE.name}")


def poll():
    calls = json.loads(CALLS_FILE.read_text())
    done = running = failed = 0
    for c in calls:
        fc = modal.FunctionCall.from_id(c["call_id"])
        try:
            r = fc.get(timeout=0); status = f"DONE  {r}"; done += 1
        except TimeoutError:
            status = "running"; running += 1
        except Exception as e:
            status = f"FAILED {type(e).__name__}: {str(e)[:90]}"; failed += 1
        print(f"{c['rule']:<22} s{c['seed_idx']}  {status}")
    print(f"\ndone={done}  running={running}  failed={failed}  total={len(calls)}")


if __name__ == "__main__":
    poll() if "--poll" in sys.argv else spawn()
