#!/usr/bin/env python3
"""Print parent-vs-E3 instruction-call comparison from analyze_sim_run.py."""
import json
import re
import subprocess
import sys


def stats(run):
    out = subprocess.run(["python3", "analyze_sim_run.py", run], capture_output=True, text=True).stdout
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return None
    d = json.loads(m.group(0))
    return list(d.values())[0] if d else None


ROWS = [
    ("fp16 fwd S1D16M2 ", "run_parent/p_fp16_forward_small", "run_e3/e3_fp16_forward_small"),
    ("fp16 bwd S1D16M2 ", "run_parent/p_fp16_backward_small", "run_e3/e3_fp16_backward_small"),
    ("fp16 bwd S1D256M8", "run_parent/p_fp16_backward_m8", "run_e3/e3_fp16_backward_m8"),
    ("bf16 fwd S1D16M2 ", None, "run_e3/e3_bfloat16_forward_small"),
    ("bf16 bwd S1D16M2 ", None, "run_e3/e3_bfloat16_backward_small"),
    ("fp16 fwd S3D8208 ", None, "run_e3/e3_fp16_forward_d8208"),
    ("fp16 bwd S3D17M4 ", None, "run_e3/e3_fp16_backward_d17"),
]


def main():
    print("%-18s %12s %12s %8s %10s" % ("case", "parent_calls", "e3v2_calls", "delta", "e3v2_sclr"))
    for name, p, c in ROWS:
        cs = stats(c)
        if cs is None:
            print("%-18s MISSING %s" % (name, c))
            continue
        if p:
            ps = stats(p)
            if ps is None:
                print("%-18s PARENT MISSING %s" % (name, p))
                continue
            pc, cc = ps["total_instruction_calls"], cs["total_instruction_calls"]
            print("%-18s %12d %12d %+7.1f%% %10d" % (name, pc, cc, (cc - pc) / pc * 100,
                                                      cs["pipe_calls"]["SCALAR"]))
        else:
            print("%-18s %12s %12d %8s %10d" % (name, "n/a", cs["total_instruction_calls"], "",
                                                 cs["pipe_calls"]["SCALAR"]))


if __name__ == "__main__":
    main()
