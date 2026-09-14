#!/usr/bin/env python3
"""Collect per-case simulator statistics from msprof op simulator OPPROF dirs.

Reads core*.veccore0_instr_exe.csv (per-instruction call_count/cycles) and
simulator/trace.json (wall duration). Mechanism-signal analysis only: these
numbers are NOT CANNJudge timings.
"""
import csv
import glob
import json
import os
import sys


def analyze_run(run_dir):
    results = {}
    for opprof in sorted(glob.glob(os.path.join(run_dir, "OPPROF_*"))):
        instr_csvs = glob.glob(os.path.join(opprof, "simulator", "*", "*_instr_exe.csv"))
        trace = os.path.join(opprof, "simulator", "trace.json")
        if not instr_csvs:
            continue
        total_calls = 0
        pipe_calls = {}
        total_cycles = 0
        for ic in instr_csvs:
            core = os.path.basename(os.path.dirname(ic))
            with open(ic) as fp:
                for row in csv.DictReader(fp):
                    calls = int(row["call_count"])
                    total_calls += calls
                    pipe_calls[row["pipe"]] = pipe_calls.get(row["pipe"], 0) + calls
                    total_cycles += int(row["cycles"])
        duration_us = None
        if os.path.isfile(trace):
            with open(trace) as fp:
                t = json.load(fp)
            events = [e for e in t.get("traceEvents", []) if e.get("ph") == "X"]
            if events:
                end = max(e["ts"] + e.get("dur", 0) for e in events)
                start = min(e["ts"] for e in events)
                duration_us = round(end - start, 3)
        results[os.path.basename(opprof)] = {
            "total_instruction_calls": total_calls,
            "pipe_calls": pipe_calls,
            "total_cycles": total_cycles,
            "wall_duration_us": duration_us,
        }
    return results


if __name__ == "__main__":
    for run_dir in sys.argv[1:]:
        print("=" * 20, run_dir, "=" * 20)
        print(json.dumps(analyze_run(run_dir), indent=1, sort_keys=True))
