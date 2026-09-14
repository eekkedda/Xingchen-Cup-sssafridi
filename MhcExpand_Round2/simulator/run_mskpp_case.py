#!/usr/bin/env python3
"""End-to-end CA-simulator validation via mskpp: real host tiling so + real
kernel .o + output readback, compared bit-exactly against golden bins.

Run from MhcExpand_Round2/simulator with the cann85 env active:
    python3 run_mskpp_case.py <case>
Cases: fp16_forward_small, fp16_backward_small, fp16_backward_m8,
       fp16_forward_d8208, fp16_backward_d17
"""
import os
import sys

import numpy as np

CANN = os.path.expanduser(
    "~/miniconda3/envs/cann85/Ascend/cann-8.5.0")
BUILD = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "mhc_expand_local_test", "build_round2_e5_bwd_raw_events"))
FP16_O = os.path.join(
    BUILD, "op_kernel/ascendc_kernels/binary/ascend910b/mhc_expand",
    "MhcExpand_3058266aa7f1f45d457dbde7c09d5333.o")

os.environ["LD_LIBRARY_PATH"] = ":".join([
    os.path.join(CANN, "tools/simulator/Ascend910B1/lib"),
    os.path.join(CANN, "tools/msopt/lib64"),
    os.path.join(CANN, "lib64"),
    os.environ.get("LD_LIBRARY_PATH", ""),
])
os.environ.setdefault("LD_PRELOAD", "")
if "libruntime_camodel.so" not in os.environ["LD_PRELOAD"]:
    os.environ["LD_PRELOAD"] = os.path.join(
        CANN, "tools/simulator/Ascend910B1/lib/libruntime_camodel.so")
sys.path.insert(0, os.path.join(CANN, "tools/msopt"))

import mskpp  # noqa: E402

# Shrink the 75MB simulated workspace that previously hung the camodel launch.
from mskpp.launcher.opgen_workflow import TilingOutput  # noqa: E402

_orig_tiling_init = TilingOutput.__init__


def _small_workspace_init(self, tiling_output):
    slim = dict(tiling_output)
    slim["workspace_size"] = min(int(slim.get("workspace_size", 0)), 4096)
    _orig_tiling_init(self, slim)


TilingOutput.__init__ = _small_workspace_init

TILING_SO = os.path.join(BUILD, "op_host/libcustom_ascendc_cust_optiling.so")

CASES = {
    "fp16_forward_small": dict(
        backward=False, m=2, s=1, d=16,
        in_file="forward_input.bin", golden="forward_golden.bin",
        in_shape=(1, 16), out_shape=(1, 2, 16)),
    "fp16_backward_small": dict(
        backward=True, m=2, s=1, d=16,
        in_file="backward_input.bin", golden="backward_golden.bin",
        in_shape=(1, 2, 16), out_shape=(1, 16)),
    "fp16_backward_m8": dict(
        backward=True, m=8, s=1, d=256,
        in_file="backward_m8_input.bin", golden="backward_m8_golden.bin",
        in_shape=(1, 8, 256), out_shape=(1, 256)),
    "fp16_forward_d8208": dict(
        backward=False, m=2, s=3, d=8208,
        in_file="forward_d8208_input.bin", golden="forward_d8208_golden.bin",
        in_shape=(3, 8208), out_shape=(3, 2, 8208)),
    "fp16_backward_d17": dict(
        backward=True, m=4, s=3, d=17,
        in_file="backward_d17_input.bin", golden="backward_d17_golden.bin",
        in_shape=(3, 4, 17), out_shape=(3, 17)),
}


def main():
    name = sys.argv[1]
    case = CASES[name]
    x = np.fromfile(case["in_file"], dtype=np.float16)
    golden = np.fromfile(case["golden"], dtype=np.float16)
    x = x.reshape(case["in_shape"])
    o = np.zeros(int(np.prod(case["out_shape"])), dtype=np.float16)

    tiling = mskpp.tiling_func(
        op_type="MhcExpand",
        inputs=[x], outputs=[o],
        lib_path=TILING_SO,
        attr={"mhc_mult": case["m"], "backward": case["backward"]},
        soc_version="Ascend910B1",
    )
    print("blockdim=%d tiling_key=%s tiling_bytes=%d"
          % (tiling.blockdim, tiling.tiling_key, tiling.tiling_data.size))
    print("tiling hex:", tiling.tiling_data.tobytes().hex())

    kernel = mskpp.get_kernel_from_binary(FP16_O, tiling_key=int(tiling.tiling_key))
    kernel[tiling.blockdim](x, o, tiling.workspace, tiling.tiling_data)

    ok = np.array_equal(o, golden)
    print("case=%s output_match=%s (%d/%d elements equal)"
          % (name, ok, int((o == golden).sum()), o.size))
    if not ok:
        bad = np.nonzero(o != golden)[0][:10]
        for i in bad:
            print("  idx %d got %04x want %04x" % (i, o[i].view(np.uint16),
                                                   golden[i].view(np.uint16)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
