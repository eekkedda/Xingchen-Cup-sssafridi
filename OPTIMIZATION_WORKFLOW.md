# MhcExpand 优化、编译、验证与跑分标准流程

> 适用工程：`kernel_optimization/B_easy`  
> 工具链：CANN 8.5.0，目标 `ascend910b`  
> 用途：供后续优化 agent 按可复现、可比较、可回退的方式继续实验。

## 1. 先理解三个版本概念

任何时候都要分别记录以下三个对象，不要混用：

| 对象 | 含义 | 当前记录（2026-09-14） |
|---|---|---|
| 平台稳定版 | 已在 CANNJudge 通过正确性且有真实分数的版本 | `E1_pipe_entry`，8/8 Pass，74.84 分 |
| 工作候选 | 当前 `B_easy/` 中正在开发或等待平台验证的源码 | `E3_scalar_hoist`（父版本 `E2_on_E1_fp16_mixed_axpy`） |
| 提交包 | 某次候选对应的不可变 7 文件压缩包 | 必须以路径和 SHA-256 唯一标识 |

状态的唯一入口是：

- `MhcExpand_Round2/round2_results.json`
- `MhcExpand_Round2/experiments/*.json`
- `MhcExpand_Round2/历史跑分.json`

不要只根据 `B_easy` 文件时间、压缩包名字或聊天记忆判断当前版本。工作区没有 Git
元数据，开始工作前必须检查源码哈希。

## 2. 不可改变的接口和验证边界

提交接口必须保持：

- 输入 `x`，输出 `o`；
- 属性 `mhc_mult`，默认 2；
- 属性 `backward`，默认 false；
- 支持 FP16、BF16；
- 前向 `[S,D] -> [S,m,D]`；
- 反向 `[S,m,D] -> [S,D]`，按 m 顺序使用 FP32 累加并最终转换，除非独立实验已经
  证明替代路径与契约一致；
- 目标芯片 `ascend910b`；
- 提交包严格只有模板规定的 7 个文件。

禁止按隐藏测试点编号、猜测 shape、数据指纹或 golden 值写特化。Host 只能做 shape、
属性和 tiling 计算，不能代替 Kernel 做数值计算。

本机没有 NPU，各验证层能证明的内容不同：

| 验证层 | 能证明 | 不能证明 |
|---|---|---|
| `binary + install` | Host/Kernel API、模板和两个 dtype 可以由 CANN 8.5 编译安装 | 数值正确、真机性能 |
| Python reference | 公式和本地 golden 生成逻辑正确 | 当前 Ascend Kernel 正确 |
| CA simulator | 对象能执行、部分指令/流水变化 | CANNJudge 分数、真实 910B 带宽 |
| CANNJudge | 隐藏平台用例的正确性和真实计时 | 所有合法输入已穷尽验证 |

## 3. 每一轮的标准生命周期

```text
读取状态 → 固化父版本 → 写单一假设 → 修改源码 → fresh 编译
        → 本地分层验证 → 打包并校验 → 平台 A/B → 记录 → 合入或回退
```

一次实验只改变一个主要机制。若同时调整 tile、任务分配、算术和同步，即使分数变化，
也无法判断是哪项导致，后续很难可靠合并。

## 4. Step 0：进入工程并读取上下文

```bash
cd /home/sssafridi/kernel_optimization
source /home/sssafridi/miniconda3/etc/profile.d/conda.sh
conda activate cann85

sed -n '1,220p' OPTIMIZATION_WORKFLOW.md
sed -n '1,220p' MhcExpand_Round2/round2_results.json
sed -n '1,220p' MhcExpand_Round2/tips.txt
sed -n '1,220p' MhcExpand_Round2/给CodingAgent的执行指令.md

sha256sum \
  B_easy/op_host/mhc_expand.cpp \
  B_easy/op_kernel/mhc_expand.cpp \
  B_easy/op_kernel/mhc_expand_tiling.h \
  B_easy/op_kernel/tiling_key_mhc_expand.h
```

如果环境中存在 Git，再执行 `git status --short`；当前工作区没有 Git 时，不要把
`git reset/clean/checkout` 当作回退手段。

同时检查是否有其他未完成候选：

```bash
find MhcExpand_Round2/experiments -maxdepth 1 -type f -name '*.json' -print | sort
find mhc_expand_local_test -maxdepth 1 -type f -name '*.zip' -print | sort
```

## 5. Step 1：建立实验身份并保护父版本

为实验取描述机制的名字，例如：

```bash
MHC_EXP_ID=e3_forward_tile_16384
```

不要用“final2”“new”“try”这类无法追踪的名字。先从
`MhcExpand_Round2/实验记录模板.json` 新建实验记录，至少填好：

- `experiment_id`；
- 父版本名称及四个源码 SHA-256；
- 单一假设 `single_hypothesis`；
- 唯一主要改动 `changed_mechanism`；
- 明确保持不变的内容；
- 生效域和通用 fallback；
- 预计风险和判定标准。

在没有 Git 的情况下，父版本至少要满足下面两种保护方式之一：

1. 已存在完整 snapshot；或
2. 修改前先运行打包脚本并记录包的 SHA-256。

Round 6 的不可变快照位于：

```text
MhcExpand_Round2/snapshots/round2_start_20260913/
```

平台稳定的 E1 包位于：

```text
mhc_expand_local_test/mhc_expand_round2_e1_pipe_entry.zip
SHA-256: 149e013989bbbd90877bed24c8af12242476f49cf73e6dc9c24d10d71b0b4f5b
```

## 6. Step 2：实施前审计

修改前回答以下问题，并写入实验 JSON 或审计文档：

- 改动属于 forward、backward、FP16、BF16 还是全部路径？
- Host 会生成什么 `R/W/tasks/blockDim`？
- 每个活跃 UB buffer 的类型、份数和总字节数是多少？
- 每 tile 的 MTE2、MTE3、Cast、Add/Axpy 和 barrier 数量如何变化？
- 新增或删除的 event 保护哪条 RAW/WAR/WAW 依赖？
- 对 `S=1`、`D=1`、非对齐 D、尾行、尾 tile、`m=1` 和 generic m 是否仍有真
  fallback？
- 变化是否会降低任务数，使小 shape 只占少数 AIV 核？

若改 Host tiling，应同步更新并运行：

```bash
python3 MhcExpand_Round2/generate_tiling_trace.py
```

注意：该脚本当前镜像 Round 6 规则。Host 规则变化时应先同步脚本，再把生成 trace
复制为带实验 ID 的独立记录，不能用旧模型证明新 tiling。

## 7. Step 3：修改源码

提交范围内只有以下 7 个文件：

```text
B_easy/CMakeLists.txt
B_easy/op_host/CMakeLists.txt
B_easy/op_host/mhc_expand.cpp
B_easy/op_kernel/CMakeLists.txt
B_easy/op_kernel/mhc_expand.cpp
B_easy/op_kernel/mhc_expand_tiling.h
B_easy/op_kernel/tiling_key_mhc_expand.h
```

通常只修改 Host/Kernel 实现和 tiling 数据。不要为了“清理”重写 CMake、接口注册或
无关代码。若修改 `MhcExpandTilingData`：

- Host 和 Kernel 必须同时更新；
- simulator 的 C++ fixture、`tiling_data_size` 和 `.bin` 必须重新生成；
- 不要在 Python 中手工猜 C++ 结构体 padding；
- fresh 编译后检查生成 JSON 的 `opParaSize`，但不要把它与结构体 `sizeof` 混为一谈。

修改后先检查差异。存在 Git 时使用 `git diff -- B_easy`；无 Git 时与父 snapshot 做：

```bash
diff -ru \
  MhcExpand_Round2/snapshots/round2_start_20260913/B_easy \
  B_easy
```

如果实验父版本不是 Round 6，应改用该实验真实父快照，不能用错误基线解释 diff。

## 8. Step 4：使用 fresh 目录编译 CANN 8.5

这台机器的标准环境：

```bash
cd /home/sssafridi/kernel_optimization
source /home/sssafridi/miniconda3/etc/profile.d/conda.sh
conda activate cann85

MHC_EXP_ID=e3_forward_tile_16384
bash mhc_expand_local_test/compile_cann85.sh \
  "mhc_expand_local_test/build_${MHC_EXP_ID}"
```

脚本固定执行：

```text
cmake configure → binary → install
CANN 8.5.0 → ascend910b
```

不要复用父版本 build 目录。fresh 目录可以避免 CMake/生成对象缓存让候选出现“假编译”。

成功的硬标准是日志最后出现：

```text
CANN 8.5 binary build and install targets succeeded.
```

日志必须保存为：

```text
mhc_expand_local_test/build_<experiment_id>/build.log
```

再确认两个 dtype 对象与 JSON 都存在：

```bash
find "mhc_expand_local_test/build_${MHC_EXP_ID}/op_kernel" \
  -path '*/ascend910b/mhc_expand/MhcExpand_*' \
  -type f -print | sort

MHC_KERNEL_JSON_DIR="mhc_expand_local_test/build_${MHC_EXP_ID}/op_kernel/ascendc_kernels/binary/ascend910b/mhc_expand"
rg -n '"dtype"|"opParaSize"|"kernelName"' \
  "${MHC_KERNEL_JSON_DIR}"/*.json
```

编译失败时只处理日志中第一条真实 `error:`，不要从最后的 `make: Error 2` 猜根因：

```bash
rg -n -m 5 'error:|\[ERROR\]' \
  "mhc_expand_local_test/build_${MHC_EXP_ID}/build.log"
```

自动生成文件中的 `backslash and newline separated by space` warning 已在成功构建中
出现过；它不是 Kernel 编译失败。仍应以完整日志和退出码为准。

## 9. Step 5：运行本地验证

以下命令默认已按 Step 0 激活 `cann85`。如果新开了 shell，先重新执行：

```bash
source /home/sssafridi/miniconda3/etc/profile.d/conda.sh
conda activate cann85
```

### 9.1 公式参考测试（必跑）

```bash
cd /home/sssafridi/kernel_optimization/mhc_expand_local_test
python -m unittest -v test_reference.py
```

通过标准是 3 项测试全部 `ok`。记录时必须写：

```text
reference: pass（只验证公式，没有执行当前 Ascend Kernel）
```

不能简写成“精度测试通过”。

### 9.2 离线结果分析工具测试（改记录工具时必跑）

```bash
cd /home/sssafridi/kernel_optimization/MhcExpand_Round2
python -m unittest -v test_analyze_results.py
```

### 9.3 shape/tiling 覆盖检查（改 tiling、offset、stride 时必跑）

至少覆盖：

- dtype：FP16、BF16；
- direction：forward、backward；
- m：1、2、3、4、5、7、8、9；
- S：1、2、3，以及 rowBlock/核数边界附近；
- D：1、15、16、17、255、256、257、8191、8192、8193；
- 尾行与尾 D tile 同时存在；
- 题面典型 `(64,256,2)`、`(1024,4096,4)`、`(8192,7168,8)`；
- 每个有效输出恰好写一次，无跨核覆盖、漏写和越界；
- UB 总量不超过平台查询值，所有 blockLen/count/stride 可由 API 字段表示。

### 9.4 CA simulator（推荐但不是平台性能证明）

已有配置和 fixture 位于 `MhcExpand_Round2/simulator/`。使用新对象时必须复制配置并
更新真实 `kernel_path`、`kernel_name`、`blockdim` 和 tiling 数据，不能直接把 E1/E2
配置当作新候选配置。

fixture 必须由包含真实 tiling 头文件的 C++ 程序生成。目前可参考：

```text
MhcExpand_Round2/simulator/prepare_simulator_cases.cpp
```

配置确认后，从 simulator 目录运行：

```bash
cd /home/sssafridi/kernel_optimization/MhcExpand_Round2/simulator
MHC_EXP_ID=e3_forward_tile_16384
MHC_SIM_CONFIG=e3_forward_tile_16384
msprof op simulator \
  --config="./${MHC_SIM_CONFIG}.json" \
  --output="./profile_${MHC_EXP_ID}"
```

当前 CA 模式已能执行 FP16/BF16 前后向对象并给出指令统计，但没有可靠导出结果 tensor。
因此只能记录“对象执行/机制信号”，不能记录为完整 golden 通过，也不能用 simulator
微秒数预测 CANNJudge 分数。

## 10. Step 6：生成不可变提交包

回到工程根目录，用新的输出名打包：

```bash
cd /home/sssafridi/kernel_optimization
MHC_EXP_ID=e3_forward_tile_16384

bash mhc_expand_local_test/package_submission.sh \
  "mhc_expand_local_test/mhc_expand_${MHC_EXP_ID}.zip"

sha256sum "mhc_expand_local_test/mhc_expand_${MHC_EXP_ID}.zip"
python3 -m zipfile -l \
  "mhc_expand_local_test/mhc_expand_${MHC_EXP_ID}.zip"
```

打包脚本拒绝覆盖已有文件，这是预期保护。需要重打时使用新的候选名；不要删除旧包后
复用同名路径。

压缩包必须恰好包含 `code/` 下的 7 个模板文件，不得包含 README、build、golden、
分析脚本或实验记录。把包 SHA-256 写入对应实验 JSON。

若平台采用在线编辑而不是上传 zip，只粘贴本实验实际修改的文件；修改过 tiling struct
时必须成对更新 Host、Kernel 和 tiling header。提交前再将在线内容与本地候选比较。

## 11. Step 7：平台提交时必须保存的数据

每次提交保存同一组完整信息：

- experiment ID；
- 父版本；
- submission ID（若平台提供）；
- 平台实际总分；
- 8/8 正确性状态和每点输出错误占比；
- 八个原始耗时字符串，保留页面单位与小数位，如 `1.38 ms`；
- 当次页面显示的八个 TBest；
- 提交包路径和 SHA-256；
- 四个源码 SHA-256；
- 提交时间和备注。

不要先把 `1.38 ms` 改写成 `1380.00 us` 再保存；那会伪造并不存在的显示精度。

记录示例：

```json
{
  "submission_id": null,
  "platform_score": 74.84,
  "correctness": "8_of_8_pass_zero_error",
  "times": {
    "1": "3.94 us",
    "2": "15.74 us",
    "3": "1.38 ms"
  },
  "tbest_at_submission": {
    "1": "1.36 us",
    "2": "14.02 us",
    "3": "1.36 ms"
  }
}
```

实际记录必须补全 8 点。

## 12. Step 8：分析结果并决定合入

先把新 run 追加到 `MhcExpand_Round2/历史跑分.json`，再运行：

```bash
cd /home/sssafridi/kernel_optimization/MhcExpand_Round2
MHC_BASELINE=E1_pipe_entry
MHC_CANDIDATE=E2_on_E1_fp16_mixed_axpy
python analyze_results.py 历史跑分.json \
  --baseline "${MHC_BASELINE}" \
  --candidate "${MHC_CANDIDATE}" \
  --output "历史对比_${MHC_BASELINE}至${MHC_CANDIDATE}.md"
```

决策优先级：

1. 任一点错误、异常或编译失败：不合入，先定位正确性。
2. 平台实际分数：主要版本选择依据。
3. 与实验假设对应的逐点变化：用于判断机制是否生效及是否存在负交互。
4. 重复提交的中位数/MAD：区分收益和噪声。
5. 八点显示耗时总和、相对 TBest、模拟器耗时：只作诊断，不代替官方分数。

当前没有证据确认官方评分公式。不要假定每点等权、算术平均、几何平均或固定每点
12.5 分。排行榜已经表明小点的评分价值不能按其绝对微秒占比判断。

建议使用以下结论状态：

| 状态 | 条件 |
|---|---|
| `promote` | 8/8 正确，平台分数明确提高，逐点变化与机制基本一致 |
| `retest` | 正确，但变化可被显示量化或单次噪声覆盖，或存在明显正负点交换 |
| `reject` | 分数下降、目标路径无收益，或出现不可接受回退 |
| `blocked` | 缺少平台权限/结果，不能继续完成该候选判定 |

合入后更新：

- `MhcExpand_Round2/round2_results.json`；
- 对应 `experiments/<id>.json`；
- `MhcExpand_Round2/历史跑分.json`；
- 根目录 `README.md` 的性能历史；
- 稳定提交包路径与 SHA-256。

## 13. 回退流程

发生错误时不要直接覆盖或删除当前候选。优先：

1. 保留失败候选源码、build.log、包、平台日志和结论；
2. 从对应父版本 snapshot 或已校验压缩包恢复到新目录；
3. 比较 7 个文件和 SHA-256；
4. 再将确认后的父版本复制回 `B_easy`；
5. 使用新的 build 目录重新完成 `binary + install`。

Round 6 可从 `MhcExpand_Round2/snapshots/round2_start_20260913/B_easy` 恢复；平台稳定
E1 应以 `mhc_expand_round2_e1_pipe_entry.zip` 及其已记录哈希为准。

禁止使用 `git reset --hard`、递归清空工程根目录或删除整个 `mhc_expand_local_test`。

## 14. 常见问题速查

| 现象 | 首要检查 |
|---|---|
| CANN 环境未激活 | 是否激活 `cann85`，`ASCEND_CANN_PACKAGE_PATH` 是否指向 8.5.0 |
| 找不到 `stdio.h` | 编译脚本是否正确加入 CANN HCC sysroot 的 `CPATH` |
| 两个 dtype 一起失败 | 共用 Kernel/API、tiling struct 或入口 ABI |
| 仅 BF16 失败 | API 是否支持 BF16、Cast RoundMode、模板分支是否真为编译期分支 |
| Host 成功但 binary 失败 | 从 Kernel opc 日志第一条 `error:` 开始处理 |
| 平台全部精度错 | Host/Kernel tiling ABI、shape 推导、方向或属性顺序 |
| 只有非对齐点错 | `DataCopyPad`、32B 对齐、GM/UB stride 单位、有效尾长 |
| 小点变慢 | TPipe/queue/event 固定开销、blockDim、行合并是否压低并行度 |
| 中等点变慢 | DMA 命令数、单双 buffer、MTE2/MTE3 是否被强制串行 |
| 大点变慢 | HBM 流量、tile 容量、每核循环、反向 Cast/Add 流水、负载均衡 |
| 总耗时更低但分数下降 | 官方计分并非已知简单总和；查看逐点相对表现和真实总分 |

## 15. 一轮实验的完成定义

只有同时满足以下项目，才算完成一轮：

- [ ] 已确认真实父版本和源码/包哈希；
- [ ] 实验只验证一个主要假设；
- [ ] 修改范围和保持不变项已记录；
- [ ] 使用 fresh build 目录完成 CANN 8.5 `binary + install`；
- [ ] FP16、BF16 两个对象均生成；
- [ ] reference 测试通过且没有夸大其含义；
- [ ] 涉及 tiling/stride 时完成边界覆盖审计；
- [ ] 可选 simulator 结果按“机制信号”记录；
- [ ] 7 文件提交包和 SHA-256 已保存；
- [ ] 平台 submission ID、真实分数、八点原始耗时和 TBest 已记录；
- [ ] 已作 `promote/retest/reject` 决策；
- [ ] README、results JSON 和实验记录已同步。

## 16. 给下一位 agent 的最短开工指令

```text
先读 OPTIMIZATION_WORKFLOW.md、MhcExpand_Round2/round2_results.json、
MhcExpand_Round2/tips.txt 和目标 experiments JSON。不要假设当前 B_easy 是稳定版；
先核对四个源码 SHA-256。以平台已验证父版本为基线，一次只改一个机制，用全新 build
目录跑 CANN 8.5 binary+install，再跑 reference/边界审计，生成唯一命名的 7 文件包并
记录 SHA-256。平台结果必须保存真实分数、8 点原始单位时间和当次 TBest；按真实分数
优先决策，显示耗时总和和 simulator 只作诊断。不得按隐藏点编号或猜测 shape 特化。
```
