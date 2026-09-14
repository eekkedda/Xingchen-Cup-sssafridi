#!/usr/bin/env python3
"""Generate mid-shape fixtures + configs for the R=1/R=2 load-balance study.

Shape: S=1024, D=4096, m=4, FP16, forward and backward.
Host tiling for this shape: W=4096, Rmax=min(32, 8192//4096, ceil(1024/48))=2.
R=2 (current rule): rowGroups=512, blockDim=48, cores get 10..11 tasks.
R=1 (candidate):    rowGroups=1024, blockDim=48, cores get 21..22 tasks.
"""
import json
import os
import struct

import numpy as np

SIM = os.path.dirname(os.path.abspath(__file__))
BUILD = "build_round2_e5_bwd_raw_events"
FP16 = "MhcExpand_3058266aa7f1f45d457dbde7c09d5333"

S, D, M = 1024, 4096, 4


def tiling(backward, row_block):
    # 40-byte struct: s,d (u64), mhcMult,backward,tileLength,rowBlock,blockDim (u32 x5) + 4 pad
    return struct.pack("<QQIIIII4x", S, D, M, 1 if backward else 0, D, row_block, 48)


def main():
    rng = np.random.default_rng(7)
    fwd_in = ((np.arange(S * D, dtype=np.int64) % 997) * 0.5).astype(np.float16)
    fwd_in.tofile(os.path.join(SIM, "mid_forward_input.bin"))
    bwd_in = ((np.arange(S * M * D, dtype=np.int64) % 991) * 0.25).astype(np.float16)
    bwd_in.tofile(os.path.join(SIM, "mid_backward_input.bin"))
    for backward, tag in ((0, "forward"), (1, "backward")):
        for rb in (1, 2):
            name = f"mid_{tag}_R{rb}_tiling.bin"
            with open(os.path.join(SIM, name), "wb") as fp:
                fp.write(tiling(backward, rb))

    def cfg(kernel_suffix, in_shape, in_file, out_shape, tiling_file, case_name):
        return {
            "kernel_name": f"{FP16}_{kernel_suffix}",
            "kernel_path": (f"../../mhc_expand_local_test/{BUILD}/op_kernel/"
                            f"ascendc_kernels/binary/ascend910b/mhc_expand/{FP16}.o"),
            "blockdim": 48, "mode": "ca", "device_id": 0,
            "magic": "RT_DEV_BINARY_MAGIC_ELF_AIVEC",
            "test_cases": [{"case_name": case_name, "param_desc": [
                {"param_type": "input", "type": "float16", "shape": in_shape,
                 "data_path": in_file, "name": "x"},
                {"param_type": "output", "type": "float16", "shape": out_shape, "name": "o"},
                {"param_type": "workspace", "user_workspace_size": 32},
                {"param_type": "tiling", "tiling_data_size": 40,
                 "tiling_data_path": tiling_file}]}],
        }

    configs = {
        "mid_forward_R1.json": cfg("1", [S, D], "./mid_forward_input.bin",
                                   [S, M, D], "./mid_forward_R1_tiling.bin",
                                   "MhcExpand_MID_FP16_forward_S1024_D4096_M4_R1"),
        "mid_forward_R2.json": cfg("1", [S, D], "./mid_forward_input.bin",
                                   [S, M, D], "./mid_forward_R2_tiling.bin",
                                   "MhcExpand_MID_FP16_forward_S1024_D4096_M4_R2"),
        "mid_backward_R1.json": cfg("257", [S, M, D], "./mid_backward_input.bin",
                                    [S, D], "./mid_backward_R1_tiling.bin",
                                    "MhcExpand_MID_FP16_backward_S1024_D4096_M4_R1"),
        "mid_backward_R2.json": cfg("257", [S, M, D], "./mid_backward_input.bin",
                                    [S, D], "./mid_backward_R2_tiling.bin",
                                    "MhcExpand_MID_FP16_backward_S1024_D4096_M4_R2"),
    }
    for name, c in configs.items():
        with open(os.path.join(SIM, name), "w") as fp:
            json.dump(c, fp, indent=2)
    print("fixtures + 4 configs written")


if __name__ == "__main__":
    main()
