"""
spawn_bncal.py
==============
Client-independent submission for the BN-calibration control, same pattern as
spawn_runs.py: the host has slept mid-job twice, and `modal run` does not survive
Modern Standby. deploy + spawn registers the call server-side so this process can exit.

Prereq:
  python -m modal deploy bn_calibration_control_modal.py

Run:
  python spawn_bncal.py --phase1     # Random + A_aug (the momentum=None re-run)
  python spawn_bncal.py --phase2     # A B C D C_leak D_leak
  python spawn_bncal.py --poll       # poll whatever is in spawned_bncal_calls.json
"""

import json
import sys
from pathlib import Path

import modal

from bn_calibration_control_modal import PHASE1, PHASE2, SEEDS, VARIANTS

CALLS_FILE = Path(__file__).with_name("spawned_bncal_calls.json")

# Forward passes only. Fixed-resolution variants calibrate once (50 batches) and run
# six 720-image evaluations; per-resolution variants calibrate six times. Measured
# neighbours put these at 1-5 min, so these windows are ~10x headroom, not 4 h.
TIMEOUT_S = {False: 20 * 60, True: 45 * 60}      # keyed by "recalibrates per res"


def spawn(variants):
    f = modal.Function.from_name("bn-calibration-control", "run_variant")
    calls = json.loads(CALLS_FILE.read_text()) if CALLS_FILE.exists() else []
    known = {(c["variant"], c["seed_idx"]) for c in calls}
    for v in variants:
        per_res = VARIANTS[v][1]
        fn = f.with_options(timeout=TIMEOUT_S[per_res])
        for si in range(len(SEEDS)):
            if (v, si) in known:
                print(f"{v} s{si:<3} already spawned, skipping"); continue
            c = fn.spawn(variant=v, seed_idx=si)
            calls.append({"variant": v, "seed_idx": si, "call_id": c.object_id,
                          "timeout_min": TIMEOUT_S[per_res] // 60})
            print(f"{v:<26} s{si}  {c.object_id}")
    CALLS_FILE.write_text(json.dumps(calls, indent=2))
    print(f"\n{len(calls)} call(s) total -> {CALLS_FILE.name}")


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
        print(f"{c['variant']:<26} s{c['seed_idx']}  {status}")
    print(f"\ndone={done}  running={running}  failed={failed}  total={len(calls)}")


if __name__ == "__main__":
    if "--poll" in sys.argv:
        poll()
    elif "--phase1" in sys.argv:
        spawn(PHASE1)
    elif "--phase2" in sys.argv:
        spawn(PHASE2)
    else:
        print(__doc__)
