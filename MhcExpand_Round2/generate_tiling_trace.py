#!/usr/bin/env python3
"""Reproduce the Round 6 host tiling decisions for public, synthetic shapes."""

from __future__ import annotations

import configparser
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLATFORM_INI = Path(
    "/home/sssafridi/miniconda3/envs/cann85/Ascend/cann-8.5.0/"
    "x86_64-linux/data/platform_config/Ascend910B2C.ini"
)
MAX_TILE_ELEMENTS = 8192
MAX_ROWS_PER_TILE = 32
ELEMENTS_PER_BLOCK = 16
UINT16_MAX = (1 << 16) - 1


PUBLIC_CASES = (
    {"s": 1, "d": 1, "m": 2, "dtype": "float16", "direction": "forward"},
    {"s": 1, "d": 17, "m": 3, "dtype": "bfloat16", "direction": "backward"},
    {"s": 48, "d": 256, "m": 4, "dtype": "float16", "direction": "forward"},
    {"s": 49, "d": 257, "m": 5, "dtype": "bfloat16", "direction": "backward"},
    {"s": 1024, "d": 4096, "m": 4, "dtype": "float16", "direction": "forward"},
    {"s": 1024, "d": 4096, "m": 4, "dtype": "bfloat16", "direction": "backward"},
    {"s": 8192, "d": 7168, "m": 8, "dtype": "float16", "direction": "forward"},
    {"s": 8192, "d": 7168, "m": 8, "dtype": "bfloat16", "direction": "backward"},
)


def load_platform() -> tuple[str, int, int]:
    parser = configparser.ConfigParser()
    if not parser.read(PLATFORM_INI):
        raise FileNotFoundError(PLATFORM_INI)
    return (
        parser["version"]["SoC_version"],
        parser.getint("SoCInfo", "vector_core_cnt"),
        parser.getint("AICoreSpec", "ub_size"),
    )


def can_use_aligned_copy(offset: int, stride: int, width: int) -> bool:
    gap_blocks = (stride - width) // ELEMENTS_PER_BLOCK
    return (
        offset % ELEMENTS_PER_BLOCK == 0
        and width % ELEMENTS_PER_BLOCK == 0
        and stride % ELEMENTS_PER_BLOCK == 0
        and gap_blocks <= UINT16_MAX
    )


def trace_case(case: dict[str, int | str], num_cores: int, ub_size: int) -> dict:
    s = int(case["s"])
    d = int(case["d"])
    m = int(case["m"])
    backward = case["direction"] == "backward"
    aligned_d = (d + ELEMENTS_PER_BLOCK - 1) // ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK
    tile_length = min(MAX_TILE_ELEMENTS, aligned_d)
    row_block = min(MAX_ROWS_PER_TILE, MAX_TILE_ELEMENTS // tile_length)
    parallel_row_block = max(1, (s + num_cores - 1) // num_cores)
    row_block = min(row_block, parallel_row_block)
    if d * m > ((1 << 32) - 1) // 2:
        row_block = 1

    tiles_per_row = (d + tile_length - 1) // tile_length
    row_groups = (s + row_block - 1) // row_block
    tasks = row_groups * tiles_per_row
    block_dim = min(num_cores, tasks)
    padded_width = tile_length
    tile_elements = row_block * padded_width

    if backward:
        mode = "backward-ping-pong-input"
        buffers = {
            "input_queue_2x_16bit": 4 * tile_elements,
            "output_queue_1x_16bit": 2 * tile_elements,
            "value_fp32": 4 * tile_elements,
            "accumulator_fp32": 4 * tile_elements,
        }
    else:
        mode = "forward-single-pass" if tasks <= block_dim else "forward-ping-pong"
        buffers = {"forward_raw_2x_16bit": 4 * tile_elements}

    dma = {
        "mte2_api_calls": 0,
        "mte3_api_calls": 0,
        "aligned_mte2": 0,
        "padded_mte2": 0,
        "aligned_mte3": 0,
        "padded_mte3": 0,
    }
    for task in range(tasks):
        row_group = task // tiles_per_row
        row_start = row_group * row_block
        column_offset = (task % tiles_per_row) * tile_length
        valid_width = min(d - column_offset, tile_length)
        if backward:
            for replica in range(m):
                input_offset = row_start * m * d + replica * d + column_offset
                aligned = can_use_aligned_copy(input_offset, m * d, valid_width)
                dma["mte2_api_calls"] += 1
                dma["aligned_mte2" if aligned else "padded_mte2"] += 1
            output_offset = row_start * d + column_offset
            aligned = can_use_aligned_copy(output_offset, d, valid_width)
            dma["mte3_api_calls"] += 1
            dma["aligned_mte3" if aligned else "padded_mte3"] += 1
        else:
            input_offset = row_start * d + column_offset
            aligned = can_use_aligned_copy(input_offset, d, valid_width)
            dma["mte2_api_calls"] += 1
            dma["aligned_mte2" if aligned else "padded_mte2"] += 1
            for replica in range(m):
                output_offset = (row_start * m + replica) * d + column_offset
                aligned = can_use_aligned_copy(output_offset, m * d, valid_width)
                dma["mte3_api_calls"] += 1
                dma["aligned_mte3" if aligned else "padded_mte3"] += 1

    per_core = []
    for core in range(block_dim):
        count = 1 + (tasks - 1 - core) // block_dim
        per_core.append(
            {
                "core": core,
                "task_start": core,
                "task_step": block_dim,
                "task_count": count,
                "task_last": core + (count - 1) * block_dim,
            }
        )

    row_tail = s - (row_groups - 1) * row_block
    valid_tail_width = d - (tiles_per_row - 1) * tile_length
    aligned_tail_width = (
        (valid_tail_width + ELEMENTS_PER_BLOCK - 1)
        // ELEMENTS_PER_BLOCK
        * ELEMENTS_PER_BLOCK
    )
    return {
        **case,
        "mode": mode,
        "R": row_block,
        "W": tile_length,
        "padded_width": padded_width,
        "tile_elements_reserved": tile_elements,
        "tiles_per_row": tiles_per_row,
        "row_groups": row_groups,
        "tasks": tasks,
        "blockDim": block_dim,
        "per_core_task_range": per_core,
        "UB_bytes": {
            "buffers": buffers,
            "total_reserved": sum(buffers.values()),
            "platform_ub": ub_size,
            "headroom": ub_size - sum(buffers.values()),
        },
        "DMA_counts": {
            **dma,
            "per_tile_pattern": "m reads + 1 write" if backward else "1 read + m writes",
        },
        "tail": {
            "row_count": row_tail,
            "valid_width": valid_tail_width,
            "aligned_width": aligned_tail_width,
            "padding_elements_per_tail_row": aligned_tail_width - valid_tail_width,
        },
    }


def main() -> None:
    soc, num_cores, ub_size = load_platform()
    output = {
        "schema": 1,
        "implementation": "Round 6 baseline",
        "source": "B_easy/op_host/mhc_expand.cpp",
        "platform": {
            "config": str(PLATFORM_INI),
            "soc": soc,
            "num_cores_aiv": num_cores,
            "ub_bytes": ub_size,
            "note": (
                "Host tiling queries these values at runtime. This trace uses the local "
                "Ascend910B2C platform configuration; another 910B SKU may differ."
            ),
        },
        "cases": [trace_case(case, num_cores, ub_size) for case in PUBLIC_CASES],
    }
    destination = ROOT / "MhcExpand_Round2" / "tiling_trace.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
