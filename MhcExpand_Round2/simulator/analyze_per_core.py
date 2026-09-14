#!/usr/bin/env python3
"""Per-core cycle/wall analysis for multi-core simulator runs."""
import csv
import glob
import json
import os
import re
import sys


def analyze(run_dir):
    opprofs = sorted(glob.glob(os.path.join(run_dir, "OPPROF_*")))
    for opprof in opprofs:
        per_core = {}
        for ic in glob.glob(os.path.join(opprof, "simulator", "core*", "*_instr_exe.csv")):
            core = os.path.basename(os.path.dirname(ic)).split(".")[0]
            calls = 0
            cycles = 0
            with open(ic) as fp:
                for row in csv.DictReader(fp):
                    calls += int(row["call_count"])
                    cycles += int(row["cycles"])
            per_core[core] = (calls, cycles)
        trace = os.path.join(opprof, "simulator", "trace.json")
        wall = None
        if os.path.isfile(trace):
            with open(trace) as fp:
                t = json.load(fp)
            events = [e for e in t.get("traceEvents", []) if e.get("ph") == "X"]
            if events:
                wall = round(max(e["ts"] + e.get("dur", 0) for e in events) -
                             min(e["ts"] for e in events), 3)
        cores = sorted(per_core)
        cyc = [per_core[c][1] for c in cores]
        print(os.path.basename(run_dir))
        print("  cores=%d wall_us=%s" % (len(cores), wall))
        if cyc:
            print("  cycles: min=%d max=%d max/min=%.3f sum=%d" %
                  (min(cyc), max(cyc), max(cyc) / max(min(cyc), 1), sum(cyc)))
            hist = {}
            for c in cyc:
                hist[c] = hist.get(c, 0) + 1
            top = sorted(hist.items(), key=lambda kv: -kv[0])[:4]
            print("  top cycle values (value:core_count):", top[:4])


if __name__ == "__main__":
    for d in sys.argv[1:]:
        analyze(d)
