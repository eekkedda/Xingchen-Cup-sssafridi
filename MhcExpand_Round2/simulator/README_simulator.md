# msprof op simulator 实操记录（CANN 8.5，conda cann85）

本目录的 `msprof op simulator` 用法在 2026-09-14 重新打通并核实。此前记录称
"配置模式未导出结果 tensor"仍然成立：config 模式不会导出输出 tensor，
`--export` 与 `--config` 互斥。但**指令级执行统计可以稳定复现**。

## 1. 正确执行的前提

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cann85
source $CONDA_PREFIX/Ascend/cann-8.5.0/set_env.sh
export LD_LIBRARY_PATH=\
$CONDA_PREFIX/Ascend/cann-8.5.0/tools/simulator/Ascend910B1/lib:\
$CONDA_PREFIX/Ascend/cann-8.5.0/tools/msopt/lib64:$LD_LIBRARY_PATH
```

**关键坑**：缺 `tools/simulator/Ascend910B1/lib` 时，msprof 打印
`[ERROR] Failed to load simulator so`，但随后仍然输出 `All task success`
并生成**空的** OPPROF 目录（只有 device0 空目录）。判断是否真正执行：
`simulator/core0.veccore0/core0.veccore0_instr_exe.csv` 必须存在且非空；
日志中不得出现 `The size of file ... is not correct`（tiling bin 大小与
config 中 `tiling_data_size` 不一致时同样会静默跳过执行）。

## 2. 用例与 fixture

- `prepare_simulator_cases.cpp` 用真实 `MhcExpandTilingData` 生成
  input/golden/tiling bin；tiling 结构自 E3 起为 48 字节（含
  `tilesPerRow`、`taskCount`）。用
  `x86_64-conda-linux-gnu-g++ -O2 -std=c++17` 编译（系统无 g++）。
- 旧对象（40 字节 tiling 时代）复跑时要用截断的 `*_40.bin`
  （见 `p_*.json` 的生成方式）。
- `e3_*.json` 指向 `build_round2_e3_scalar_hoist_v2`；新增边界用例：
  FP16 forward S3 D8208 M2（tilesPerRow=2 通用路径）与
  FP16 backward S3 D17 M4（单 tile 快路径 + DataCopyPad）。

## 3. 运行与统计

```bash
for c in e3_fp16_forward_small ...; do
  msprof op simulator --config=$c.json --output=./run_e3/$c > run_e3/$c.log 2>&1
done
python3 analyze_sim_run.py run_e3/*        # 汇总指令数/流水分布/墙钟
python3 compare_e3.py                       # 父版本 vs E3 对照表
```

## 4. 结论边界

- 指令数与 cycles 是**机制信号**，不是 910B 真机耗时，更不是 CANNJudge 分数。
- config 模式不导出输出 tensor；`msopt_test_result/*/o.bin` 的来源未复现，
  在取得可复现导出方法前，不要把旧 o.bin 流程当作当前验证通道。
- 910B 标量 DIV 是硬件指令（每次除法约 1 条指令），“消除 64 位除法”
  的收益远小于消除每次 DMA 的重复对齐判定与多余 InitBuffer。
