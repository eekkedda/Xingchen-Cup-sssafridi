#!/usr/bin/env python3
"""Offline timing comparison. Python >=3.10, standard library only.

Input keeps original units/decimal precision. Display intervals assume rounding
at the last displayed digit; they are NOT statistical confidence intervals.
This tool neither accesses CANNJudge nor infers hidden shapes.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
from statistics import median
import sys
from typing import Any

D = Decimal
PATTERN = re.compile(r'^\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(us|µs|μs|ms|s)\s*$')
FACTORS = {'us': D(1), 'µs': D(1), 'μs': D(1), 'ms': D(1000), 's': D(1000000)}

@dataclass(frozen=True)
class Sample:
    us: Decimal
    quantum_us: Decimal

    @property
    def low(self) -> Decimal:
        return max(D(0), self.us - self.quantum_us / 2)

    @property
    def high(self) -> Decimal:
        return self.us + self.quantum_us / 2


def parse_time(text: str) -> Sample:
    if not isinstance(text, str):
        raise ValueError('Time must be a string with unit, e.g. "1.40 ms".')
    match = PATTERN.fullmatch(text)
    if not match:
        raise ValueError(f'Unsupported time: {text!r}')
    value = D(match.group(1))
    if not value.is_finite() or value <= 0:
        raise ValueError(f'Time must be finite and positive: {text!r}')
    factor = FACTORS[match.group(2)]
    quantum = D(10) ** value.as_tuple().exponent
    return Sample(value * factor, quantum * factor)


def load_data(path: Path) -> dict[str, list[dict[str, Sample]]]:
    raw: Any = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or not isinstance(raw.get('versions'), dict):
        raise ValueError('Expected object with a "versions" object.')
    result: dict[str, list[dict[str, Sample]]] = {}
    expected: set[str] | None = None
    for name, obj in raw['versions'].items():
        if not isinstance(obj, dict) or not isinstance(obj.get('runs'), list) or not obj['runs']:
            raise ValueError(f'{name}: requires a nonempty runs list.')
        parsed_runs = []
        for index, run in enumerate(obj['runs']):
            if not isinstance(run, dict) or not isinstance(run.get('times'), dict) or not run['times']:
                raise ValueError(f'{name} run {index}: missing times object.')
            times = run['times']
            if any(not isinstance(k, str) or not k.isdigit() or int(k) < 1 for k in times):
                raise ValueError(f'{name}: case keys must be positive integer strings.')
            keys = set(times)
            if expected is None:
                expected = keys
            if keys != expected:
                raise ValueError(f'{name} run {index}: missing/different case set.')
            parsed_runs.append({k: parse_time(v) for k, v in times.items()})
        result[name] = parsed_runs
    if not result:
        raise ValueError('No versions found.')
    return result


def med(values: list[Decimal]) -> Decimal:
    return D(median(values))


def summarize(samples: list[Sample]) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    center = med([x.us for x in samples])
    return (center, med([x.low for x in samples]), med([x.high for x in samples]),
            med([abs(x.us - center) for x in samples]))


def total_stats(runs: list[dict[str, Sample]]) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    # Median of coherent per-run totals, NOT sum of per-case medians.
    totals = [sum((x.us for x in run.values()), D(0)) for run in runs]
    lows = [sum((x.low for x in run.values()), D(0)) for run in runs]
    highs = [sum((x.high for x in run.values()), D(0)) for run in runs]
    center = med(totals)
    return center, med(lows), med(highs), med([abs(x - center) for x in totals])


def report(data: dict[str, list[dict[str, Sample]]], baseline: str, candidate: str) -> str:
    if baseline not in data or candidate not in data:
        raise ValueError('Unknown version. Available: ' + ', '.join(data))
    b_runs, c_runs = data[baseline], data[candidate]
    lines = [f'# {candidate} vs {baseline}', '',
             '单位：μs。正的节省量表示候选更快。以下不是官方评分。', '',
             f'基线记录数：{len(b_runs)}；候选记录数：{len(c_runs)}。', '',
             '| 点 | 基线中位数 | 候选中位数 | 节省 | 节省比例 | 候选 MAD |',
             '|---:|---:|---:|---:|---:|---:|']
    for case in sorted(b_runs[0], key=int):
        b, _, _, _ = summarize([r[case] for r in b_runs])
        c, _, _, mad = summarize([r[case] for r in c_runs])
        lines.append(f'| {case} | {b:.3f} | {c:.3f} | {b-c:+.3f} | {(b-c)/b*100:+.3f}% | {mad:.3f} |')
    bt, bl, bh, bm = total_stats(b_runs)
    ct, cl, ch, cm = total_stats(c_runs)
    lo, hi = bl-ch, bh-cl
    lines += ['', f'逐次合计的中位数：基线 {bt:.3f}；候选 {ct:.3f}。',
              f'合计节省：{bt-ct:+.3f} μs，{(bt-ct)/bt*100:+.4f}%。',
              f'合计 MAD：基线 {bm:.3f}；候选 {cm:.3f}。', '',
              f'仅由显示舍入传播得到的节省范围：[{lo:+.3f}, {hi:+.3f}] μs。',
              '假设各时间按最后显示位四舍五入；这些范围不包含运行噪声，不是统计置信区间。',
              '单次记录的 MAD=0 不表示没有噪声。重复记录时合计用每次总和的中位数，可能不等于表中中位数之和。']
    if lo <= 0 <= hi:
        lines += ['', '本次合计差异可被显示量化覆盖，不能据此单独宣布稳定改善。']
    else:
        lines += ['', '该合计差异未被所假设的显示量化覆盖；仍需重复测量排除运行波动。']
    lines += ['', 'PlatformBest 是逐点参考值的集合，未必对应同一提交或可同时达到；历史表转录不等于平台原始日志。', '']
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='JSON with versions/runs/times')
    parser.add_argument('--baseline', default='Baseline')
    parser.add_argument('--candidate', default='Round6')
    parser.add_argument('--output', type=Path, help='Optional output Markdown file')
    args = parser.parse_args()
    try:
        result = report(load_data(args.input), args.baseline, args.candidate)
        if args.output:
            args.output.write_text(result, encoding='utf-8')
        else:
            print(result)
        return 0
    except (OSError, ValueError, InvalidOperation, TypeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
