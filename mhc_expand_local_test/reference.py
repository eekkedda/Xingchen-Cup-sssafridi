"""Local golden-data helper for the MhcExpand competition operator.

This directory is intentionally outside B_easy so it does not need to be
included in the CANNJudge submission archive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def forward(x: torch.Tensor, mhc_mult: int) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("forward input must have shape [S, D]")
    if mhc_mult <= 0:
        raise ValueError("mhc_mult must be positive")
    return x.unsqueeze(1).expand(-1, mhc_mult, -1).clone()


def backward(o_grad: torch.Tensor, mhc_mult: int) -> torch.Tensor:
    if o_grad.ndim != 3:
        raise ValueError("backward input must have shape [S, m, D]")
    if o_grad.shape[1] != mhc_mult:
        raise ValueError("input m dimension must equal mhc_mult")
    # Match the kernel baseline: accumulate in FP32 and cast once at the end.
    return o_grad.float().sum(dim=1).to(o_grad.dtype)


def write_raw(tensor: torch.Tensor, path: Path) -> None:
    # NumPy does not universally expose bfloat16, so preserve both supported
    # dtypes as their raw uint16 bit patterns.
    tensor.contiguous().view(torch.uint16).cpu().numpy().tofile(path)


def generate_case(args: argparse.Namespace) -> None:
    dtype = DTYPES[args.dtype]
    generator = torch.Generator().manual_seed(args.seed)
    shape = (args.s, args.m, args.d) if args.backward else (args.s, args.d)
    tensor_x = torch.randn(shape, dtype=torch.float32, generator=generator).to(dtype)
    golden = backward(tensor_x, args.m) if args.backward else forward(tensor_x, args.m)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_raw(tensor_x, args.output_dir / "input_x.bin")
    write_raw(golden, args.output_dir / "golden_o.bin")
    metadata = {
        "dtype": args.dtype,
        "input_shape": list(tensor_x.shape),
        "output_shape": list(golden.shape),
        "mhc_mult": args.m,
        "backward": args.backward,
        "seed": args.seed,
    }
    (args.output_dir / "case.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("case"))
    args = parser.parse_args()
    if args.s <= 0 or args.d <= 0 or args.m <= 0:
        parser.error("s, d and m must all be positive")
    return args


if __name__ == "__main__":
    generate_case(parse_args())
