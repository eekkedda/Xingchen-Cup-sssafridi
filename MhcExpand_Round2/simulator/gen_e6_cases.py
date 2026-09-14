#!/usr/bin/env python3
"""E6 fixtures: grouped (bwdGrouped=1, R=1) vs ungrouped (bwdGrouped=0, R=2)
tilings at 40 bytes each (bwdGrouped occupies the former padding slot at
offset 36), plus small-backward grouped regressions."""
import json
import os
import struct

import numpy as np

SIM = os.path.dirname(os.path.abspath(__file__))
BUILD = "build_round2_e6_bwd_grouped"
FP16 = "MhcExpand_3058266aa7f1f45d457dbde7c09d5333"
BF16 = "MhcExpand_2bc7e3aea9c70d4c9ddf23d85567b240"


def tiling(s, d, m, backward, tile_len, row_block, block_dim, grouped):
    return struct.pack("<QQIIIII I".replace(" ", ""), s, d, m,
                       1 if backward else 0, tile_len, row_block, block_dim, grouped)


def cfg(obj, suffix, blockdim, in_shape, in_file, out_shape, tiling_file, case_name, dtype="float16"):
    return {
        "kernel_name": f"{obj}_{suffix}",
        "kernel_path": (f"../../mhc_expand_local_test/{BUILD}/op_kernel/"
                        f"ascendc_kernels/binary/ascend910b/mhc_expand/{obj}.o"),
        "blockdim": blockdim, "mode": "ca", "device_id": 0,
        "magic": "RT_DEV_BINARY_MAGIC_ELF_AIVEC",
        "test_cases": [{"case_name": case_name, "param_desc": [
            {"param_type": "input", "type": dtype, "shape": in_shape,
             "data_path": in_file, "name": "x"},
            {"param_type": "output", "type": dtype, "shape": out_shape, "name": "o"},
            {"param_type": "workspace", "user_workspace_size": 32},
            {"param_type": "tiling", "tiling_data_size": 40,
             "tiling_data_path": tiling_file}]}],
    }


def main():
    S, D, M = 1024, 4096, 4
    # grouped: rowBlock=1, tileLength=D, bwdGrouped=1
    open(os.path.join(SIM, "mid_backward_G_tiling.bin"), "wb").write(
        tiling(S, D, M, 1, D, 1, 48, 1))
    # ungrouped R2 baseline: current host rule for this shape
    open(os.path.join(SIM, "mid_backward_R2_tiling.bin"), "wb").write(
        tiling(S, D, M, 1, D, 2, 48, 0))

    # small backward grouped (S=1,D=16,m=2): host grouped conditions hold
    open(os.path.join(SIM, "backward_G_small_tiling.bin"), "wb").write(
        tiling(1, 16, 2, 1, 16, 1, 1, 1))
    # m8 backward grouped (S=1,D=256,m=8)
    open(os.path.join(SIM, "backward_G_m8_tiling.bin"), "wb").write(
        tiling(1, 256, 8, 1, 256, 1, 1, 1))
    # non-aligned D=17 falls back to ungrouped (d%16!=0), R=1 pad path
    open(os.path.join(SIM, "backward_d17_tiling.bin"), "wb").write(
        tiling(3, 17, 4, 1, 32, 1, 3, 0))

    configs = {
        "e6_mid_backward_grouped.json": cfg(FP16, "257", 48, [S, M, D],
            "./mid_backward_input.bin", [S, D], "./mid_backward_G_tiling.bin",
            "MhcExpand_E6_FP16_backward_grouped_S1024_D4096_M4"),
        "e6_mid_backward_ungrouped.json": cfg(FP16, "257", 48, [S, M, D],
            "./mid_backward_input.bin", [S, D], "./mid_backward_R2_tiling.bin",
            "MhcExpand_E6_FP16_backward_ungrouped_S1024_D4096_M4"),
        "e6_backward_small_grouped.json": cfg(FP16, "257", 1, [1, 2, 16],
            "./backward_input.bin", [1, 16], "./backward_G_small_tiling.bin",
            "MhcExpand_E6_FP16_backward_grouped_S1_D16_M2"),
        "e6_backward_m8_grouped.json": cfg(FP16, "257", 1, [1, 8, 256],
            "./backward_m8_input.bin", [1, 256], "./backward_G_m8_tiling.bin",
            "MhcExpand_E6_FP16_backward_grouped_S1_D256_M8"),
        "e6_backward_d17_fallback.json": cfg(FP16, "257", 3, [3, 4, 17],
            "./backward_d17_input.bin", [3, 17], "./backward_d17_tiling.bin",
            "MhcExpand_E6_FP16_backward_fallback_S3_D17_M4"),
        "e6_bfloat16_backward_small_grouped.json": cfg(BF16, "283", 1, [1, 2, 16],
            "./backward_bfloat16_input.bin", [1, 16], "./backward_G_small_tiling.bin",
            "MhcExpand_E6_BF16_backward_grouped_S1_D16_M2", dtype="bfloat16"),
        "e6_forward_small.json": cfg(FP16, "1", 1, [1, 16],
            "./forward_input.bin", [1, 2, 16], "./forward_tiling_G0.bin",
            "MhcExpand_E6_FP16_forward_S1_D16_M2"),
    }
    # forward tiling with grouped=0 (reuse of small forward shape)
    open(os.path.join(SIM, "forward_tiling_G0.bin"), "wb").write(
        tiling(1, 16, 2, 0, 16, 1, 1, 0))
    for name, c in configs.items():
        with open(os.path.join(SIM, name), "w") as fp:
            json.dump(c, fp, indent=2)
    print("e6 fixtures + configs written")


if __name__ == "__main__":
    main()
