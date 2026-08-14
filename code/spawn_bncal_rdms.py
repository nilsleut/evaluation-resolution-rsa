"""
spawn_bncal_rdms.py
===================
deploy + spawn for the RDM-saving re-run of the BN-calibration factorial.

  python -m modal deploy bn_calibration_rdms_modal.py
  python spawn_bncal_rdms.py          # 5 variants x 5 seeds
  python spawn_bncal_rdms.py --poll
"""

import json
import sys
from pathlib import Path

import modal

from bn_calibration_rdms_modal import RDM_VARIANTS, SEEDS, VARIANTS

CALLS_FILE = Path(__file__).with_name("spawned_bncal_rdm_calls.json")
TIMEOUT_S = {False: 20 * 60, True: 45 * 60}      # keyed by "recalibrates per res"


def spawn():
    f = modal.Function.from_name("bn-calibration-rdms", "run_variant")
    calls = json.loads(CALLS_FILE.read_text()) if CALLS_FILE.exists() else []
    known = {(c["variant"], c["seed_idx"]) for c in calls}
    for v in RDM_VARIANTS:
        fn = f.with_options(timeout=TIMEOUT_S[VARIANTS[v][1]])
        for si in range(len(SEEDS)):
            if (v, si) in known:
                print(f"{v} s{si} already spawned, skipping"); continue
            c = fn.spawn(variant=v, seed_idx=si)
            calls.append({"variant": v, "seed_idx": si, "call_id": c.object_id})
            print(f"{v:<26} s{si}  {c.object_id}")
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
        print(f"{c['variant']:<26} s{c['seed_idx']}  {status}")
    print(f"\ndone={done}  running={running}  failed={failed}  total={len(calls)}")


if __name__ == "__main__":
    poll() if "--poll" in sys.argv else spawn()
