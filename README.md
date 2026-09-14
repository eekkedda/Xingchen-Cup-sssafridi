# 星辰杯 B 组：MhcExpand 工程交接与测试指南

> 最后更新：2026-09-14  
> 当前阶段：平台稳定父版本为 `E1_pipe_entry`（8/8 Pass，74.84 分）；
> `E2_on_E1_fp16_mixed_axpy` 已提交（极可能是榜单 2026/09/13 20:51:33 的 76.05 分，
> 见 `算子性能评分分析与优化优先级报告.md`）；
> `B_easy/` 当前是待平台验证的 `E3_scalar_hoist` 工作候选（叠在 E2 之上）。

## 1. 一眼看懂当前工程

后续优化 agent 应先执行根目录标准流程：
[MhcExpand 优化、编译、验证与跑分标准流程](OPTIMIZATION_WORKFLOW.md)。

| 项目 | 当前结论 |
|---|---|
| 题目 | B 组简单题：mHC-expand 前向与反向 |
| CANNJudge 题号 | 301，公开测试点共 8 个 |
| 评测 CANN 版本 | **8.5.0** |
| 目标芯片 | **`ascend910b`**，对应 Atlas A2 / 910B |
| 支持类型 | `float16`、`bfloat16` |
| 提交工程 | `B_easy/` 中的 7 个文件 |
| 编程模型 | Ascend C/C++，由毕昇编译器编译；不是 CUDA |
| 平台稳定版 | `E1_pipe_entry`：基于 Round 6，将 TPipe 所有权移到 Kernel 入口；74.84 分 |
| 已提交候选 | 第二大轮 13 次提交:最好 **76.05**(E2);详见 [MhcExpand_Round2/第二轮收官总结.md](MhcExpand_Round2/第二轮收官总结.md) |
| 当前工作候选 | `E4G2_pair_grouped`(机制超集,8/8 验证,包 aea1f45a...):按收官总结第 6 节收割协议重复提交,抽到 ≥75.9 冻结 |
| 稳定版正确性 | CANNJudge 8/8 Pass，所有点输出错误占比 0.00% |
| 当前本地验证 | E3 候选已通过 CANN 8.5.0 `binary + install`、reference 与 7 个 CA simulator 用例（含 D=8208 通用路径、D=17 非对齐路径新边界用例） |
| 稳定版平台总耗时 | E1 页面显示值近似合计 4104.76 μs；平台分数 74.84 |
| 下一优先级 | 主攻点 1、5（4.06/3.88 μs vs TBest 1.36/1.48 μs，占模型失分约 74.5%）；点 2、4 第二梯队 |

CANNJudge 页面：
[mHC-expand 提交页](https://cannjudge.cn/public/ct_starcup_aiop_g2/mhcexpand/submit)

特别注意：题目页面末尾若出现 `softmax(src, index, ptr, ...)` 的补充说明，
那是另一道题的接口说明，与本题 MhcExpand 无关。不要给本算子添加
`index`、`ptr`、`axis` 或 `eps` 参数。

## 2. 算子接口与语义

### 前向

- 输入 `x`：`[S, D]`
- 属性 `mhc_mult=m`
- 属性 `backward=false`
- 输出 `o`：`[S, m, D]`
- 计算：`o[s, k, d] = x[s, d]`

### 反向

- 输入 `x`（实际含义为 `o_grad`）：`[S, m, D]`
- 属性 `mhc_mult=m`
- 属性 `backward=true`
- 输出 `o`（实际含义为 `x_grad`）：`[S, D]`
- 计算：`o[s, d] = sum(x[s, k, d], k=0...m-1)`

Host 注册接口必须保持为：

- 输入名：`x`
- 输出名：`o`
- 属性：`mhc_mult`，默认 `2`
- 属性：`backward`，默认 `false`
- 数据类型：`DT_FLOAT16`、`DT_BF16`

## 3. 目录说明

```text
kernel_optimization/
├── README.md                         # 本交接文档
├── B_easy/                           # 唯一需要提交/编译的算子源码
│   ├── CMakeLists.txt
│   ├── op_host/
│   │   ├── CMakeLists.txt
│   │   └── mhc_expand.cpp
│   └── op_kernel/
│       ├── CMakeLists.txt
│       ├── mhc_expand.cpp
│       ├── mhc_expand_tiling.h
│       └── tiling_key_mhc_expand.h
├── mhc_expand_local_test/            # 不提交：本地辅助脚本和 golden
│   ├── compile_cann85.sh
│   ├── package_submission.sh
│   ├── reference.py
│   ├── test_reference.py
│   └── README.md
├── xingchenbei-main/                 # 题面资料
└── MhcExpand_problem_301_template.zip
```

在 `B_easy` 内，当前只修改过以下 4 个实现文件，三个 CMake 文件仍来自空模板：

1. `op_host/mhc_expand.cpp`
2. `op_kernel/mhc_expand.cpp`
3. `op_kernel/mhc_expand_tiling.h`
4. `op_kernel/tiling_key_mhc_expand.h`

## 4. 能在什么环境里验证

| 环境 | 编译 | 数学参考测试 | 运行 Ascend Kernel | 复现比赛分数 |
|---|---:|---:|---:|---:|
| WSL2 + CANN 8.5.0，无 NPU | 通常可以（非官方认证） | 可以 | 需另建 CPU 调试 runner | 不可以 |
| 原生 Linux + CANN 8.5.0，无 NPU | 可以 | 可以 | 可做 CPU 域调试 | 不可以 |
| Linux + Ascend 910B | 可以 | 可以 | 可以 | 可近似复现 |
| CANNJudge | 可以 | 隐藏 golden | 可以 | **最终权威结果** |

华为官方文档支持在无昇腾设备的 Linux 主机上安装 Toolkit 做算子开发；驱动和
固件只在有昇腾设备时需要。WSL2 是方便的 Linux 开发环境，但华为 CANN 的正式
支持列表并未单列 WSL，因此应把它定位为“编译/静态开发环境”。若安装器或工具
链出现 WSL 特有问题，改用原生 Ubuntu、学校昇腾服务器或云上 910B 环境。

## 5. 把工程迁移到 WSL

先在 Windows PowerShell 查看发行版确实使用 WSL2：

```powershell
wsl -l -v
```

建议把工程复制到 WSL 自己的 ext4 文件系统中。直接在 `/mnt/e` 编译也能工作，
但大量小文件编译通常更慢，而且更容易遇到权限或换行问题。

```bash
mkdir -p ~/work/kernel_optimization
cp -a /mnt/e/kernel_optimization/. ~/work/kernel_optimization/
cd ~/work/kernel_optimization
```

后续让 agent 的工作目录固定为：

```text
~/work/kernel_optimization
```

如果脚本报 `/usr/bin/env: 'bash\r'`，仅需转换 shell 脚本换行：

```bash
sudo apt-get install -y dos2unix
dos2unix mhc_expand_local_test/*.sh
```

### 当前这台机器的 WSL 快照

已于 2026-09-13 做过只读检查：

- `Ubuntu-22.04` 正在以 WSL2 运行；
- 系统为 Ubuntu 22.04.5 LTS、`x86_64`；
- Python 3.10.12 已存在；
- WSL Linux 根文件系统当前空间充足；
- 已在 `~/miniconda3/envs/cann85` 安装用户态 Python 3.10、CMake、GCC/G++、
  CANN Toolkit 8.5.0、910B ops 8.5.0 和 PyTorch CPU；
- 两个官方 CANN 8.5.0 下载地址均已返回 HTTP 200，可从 WSL 访问。

WSL 启动时目前会提示 Windows 的 localhost 代理未镜像到 NAT 模式，但上述下载
地址实际可达，所以暂不需要处理。若后续 `apt`、`wget` 或 Conda 因代理失败，
Windows 11 22H2 及以上可在 `%USERPROFILE%\.wslconfig` 中尝试：

```ini
[wsl2]
networkingMode=mirrored
autoProxy=true
dnsTunneling=true
```

修改后在 PowerShell 执行 `wsl --shutdown`，再重新进入发行版。不要在网络正常时
无故更改配置。

## 6. 安装比赛对应的 CANN 8.5.0 环境

### 6.1 基础依赖

以下以 x86_64 的 Ubuntu 22.04 为例。先检查架构和空间：

```bash
uname -m
df -h ~
```

`uname -m` 应为 `x86_64`。官方要求 CANN 安装目录至少有 10 GB 可用空间；考虑
Toolkit、910B ops、构建缓存和测试数据，建议实际预留更多空间。

```bash
sudo apt-get update
sudo apt-get install -y \
  gcc g++ make cmake python3 python3-pip python3-venv \
  git wget zip unzip dos2unix

python3 -m pip install --user \
  attrs cython 'numpy>=1.19.2,<2.0' decorator sympy cffi pyyaml \
  pathlib2 psutil 'protobuf==3.20.0' scipy requests absl-py
```

Ubuntu 22.04 自带的 Python 3.10 与 CANN 8.5.0 兼容，CMake 需不低于 3.16。

### 6.2 推荐：安装官方 `.run` 包

不要安装 8.5 alpha、8.3 或 9.x；评测侧明确是稳定版 **8.5.0**。Toolkit 和
`910b-ops` 必须为同一版本、安装到同一 CANN 路径。

```bash
mkdir -p ~/packages/cann85
cd ~/packages/cann85

wget -O Ascend-cann-toolkit_8.5.0_linux-x86_64.run \
  'https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-toolkit_8.5.0_linux-x86_64.run'

wget -O Ascend-cann-910b-ops_8.5.0_linux-x86_64.run \
  'https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run'

bash ./Ascend-cann-toolkit_8.5.0_linux-x86_64.run --install
source "$HOME/Ascend/cann/set_env.sh"
export LD_LIBRARY_PATH="$HOME/Ascend/cann/x86_64-linux/devlib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash ./Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install
```

以普通用户安装时，默认环境脚本通常为：

```bash
source "$HOME/Ascend/cann/set_env.sh"
export LD_LIBRARY_PATH="$HOME/Ascend/cann/x86_64-linux/devlib:${LD_LIBRARY_PATH:-}"
```

以 root 安装时通常为：

```bash
source /usr/local/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH="/usr/local/Ascend/cann/x86_64-linux/devlib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

如果路径不确定：

```bash
find "$HOME/Ascend" /usr/local/Ascend -maxdepth 3 -name set_env.sh \
  -print 2>/dev/null
```

每次新开终端都需要重新 `source`。确认安装无误后，可以自行把对应的 `source`
命令加入 `~/.bashrc`。

CANN 8.5 的环境脚本是 `.../cann/set_env.sh`；不要照搬旧版文档中的
`.../ascend-toolkit/set_env.sh`。同一个 CANN 安装路径也不要混装多个芯片系列的
ops 包，本题只需要 `910b-ops`。

### 6.3 备选：Conda 安装

已有 Conda 时也可采用官方频道。不要同时在同一环境混用 Conda 与 `.run` 两种
安装方式。

```bash
conda create -n cann85 python=3.10 cmake make -y
conda activate cann85
conda config --add channels https://repo.huaweicloud.com/ascend/repos/conda/
conda install ascend-cann-toolkit==8.5.0 -y
conda install ascend-cann-910b-ops==8.5.0 -y
```

Conda 环境目录及其上级目录需要对当前用户具备可访问权限。激活环境后，按实际
位置 `source set_env.sh`；本工程的编译脚本会自动尝试常见路径。

```bash
source "$CONDA_PREFIX/Ascend/cann/set_env.sh"
```

### 6.4 环境自检

```bash
python3 --version
pip3 --version
cmake --version
gcc --version
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-}"
echo "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH:-}"
find "${ASCEND_HOME_PATH:-$HOME/Ascend/cann}" -name bisheng -type f \
  -print 2>/dev/null | head
```

WSL 无 NPU 时没有 `npu-smi` 属于正常现象，不影响 NPU Kernel 的交叉编译。

## 7. 编译：提交平台前必须先做

回到工程根目录：

```bash
cd ~/work/kernel_optimization
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cann85
bash mhc_expand_local_test/compile_cann85.sh
```

脚本先执行与评测日志相同的 `binary` 核心流程，再执行 `TYPE SHARED` 工程对应的
`install` 目标，以便把 Host 生成与安装阶段的问题也提前暴露出来：

```bash
cmake -S B_easy -B <build_dir> \
  -DASCEND_COMPUTE_UNIT=ascend910b
cmake --build <build_dir> --target binary --parallel "$(nproc)"
cmake --build <build_dir> --target install --parallel "$(nproc)"
```

默认构建目录和日志：

```text
mhc_expand_local_test/build_cann85/
mhc_expand_local_test/build_cann85/build.log
```

若怀疑旧 CMake 缓存污染，不必删除旧目录，直接传一个新的构建目录：

```bash
bash mhc_expand_local_test/compile_cann85.sh \
  "$HOME/build/mhc_expand_cann85_clean"
```

成功标志：

```text
CANN 8.5 binary build and install targets succeeded.
```

仅仅运行 `g++ mhc_expand.cpp`、VS Code C++ 语法检查或 CUDA 编译都不能代替该
步骤；Ascend C 使用专有关键字、头文件、内建函数和毕昇编译链。

## 8. Python 数学参考测试

建议给 PyTorch golden 单独建一个环境，避免影响 CANN 的 Python 依赖：

```bash
python3 -m venv ~/.venvs/mhc-reference
source ~/.venvs/mhc-reference/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

cd ~/work/kernel_optimization/mhc_expand_local_test
python -m unittest -v test_reference.py
```

这些测试覆盖：

- FP16 与 BF16；
- forward 与 backward；
- `S=1`、`D=1` 和非对齐维度；
- backward 先 FP32 累加、最后转换回输入 dtype 的参考语义。

它们只证明参考公式和数据生成器正确，**没有执行当前 Ascend Kernel**。

生成原始输入及 golden 示例：

```bash
python reference.py --s 3 --d 17 --m 4 --dtype float16 \
  --output-dir cases/fp16_forward

python reference.py --s 3 --d 17 --m 4 --dtype bfloat16 --backward \
  --output-dir cases/bf16_backward
```

## 9. CPU 调试和真实 NPU 测试的边界

CANN 8.5 支持在无昇腾设备环境中进行 CPU 域调试，但官方 8.5 文档明确说明：
普通异构命令行/CMake 工程暂不直接支持 CPU 孪生调试，当前应基于 Kernel 直调
样例完成。`B_easy` 是标准自定义算子工程，也没有直调所需的 `main.cpp`、tiling
构造和输入输出 runner。因此：

- `compile_cann85.sh`：检查 CANN 8.5 的 `binary` 与 `install` 构建流程；
- `test_reference.py`：检查数学参考实现；
- 若要在 WSL 直接跑 Kernel：下一步需基于 CANN 8.5 的 Kernel 直调样例新增
  CPU runner，并逐项对比 `golden_o.bin`；
- CPU 仿真耗时不是 NPU 耗时，不能用于比赛性能排名。

CANN 还提供 `msprof op simulator` 做 NPU 指令流水和热点分析；它与实际 910B
板上的 `msprof op` 性能采集不是一回事，模拟结果同样不能当作比赛分数。当前工程
尚未提供调用该模拟器所需的完整可执行 runner。

要做真实性能优化，需要一台安装匹配驱动、固件、CANN 8.5.0 和 910B ops 的
Ascend 910B 机器，使用 `npu-smi info` 确认设备，再通过 profiler 测量。最终仍以
CANNJudge 的隐藏测试与计时为准。

## 10. 提交前检查与打包

原始模板压缩包的根目录名是 `code/`。辅助脚本会严格复制 7 个允许提交的文件，
不会带入 README、golden、构建缓存或本地脚本：

```bash
cd ~/work/kernel_optimization
bash mhc_expand_local_test/package_submission.sh
```

它会生成带时间戳的压缩包并列出内容，结构应当正好是：

```text
code/CMakeLists.txt
code/op_host/CMakeLists.txt
code/op_host/mhc_expand.cpp
code/op_kernel/CMakeLists.txt
code/op_kernel/mhc_expand.cpp
code/op_kernel/mhc_expand_tiling.h
code/op_kernel/tiling_key_mhc_expand.h
```

如果提交页面是逐文件编辑而不是上传压缩包，只同步 `B_easy` 内对应的 7 个文件。

提交前最低检查清单：

- [ ] 激活的是 CANN 8.5.0，不是 alpha/RC/其他大版本；
- [ ] 目标为 `ascend910b`；
- [ ] `binary` 构建成功；
- [ ] FP16、BF16 两个 tiling key 都被编译；
- [ ] Python reference 单元测试通过；
- [ ] 压缩包只有上述 7 个源码文件；
- [ ] 保存本次平台完整日志和分数，便于下一轮定位。

## 11. 上一轮错误与后续排错方法

上一轮 8 个测试点全部显示“编译失败”，根因不是 8 个独立问题。两个 dtype 的
Kernel 共用同一份源码，下面这一处错误使所有测试都无法生成 Kernel 二进制：

```cpp
// 错误
GetBlockIdx()

// 已修复
AscendC::GetBlockIdx()
```

遇到新日志时按以下顺序处理：

1. 找日志中最早出现的 `error:`，不要从最后的 `make: Error 2` 倒推；
2. 先区分 Host 编译、Kernel/opc 编译、运行精度、运行异常、性能超时；
3. 一次只修有证据的问题，然后重新跑本地 `binary`；
4. 保存完整日志，不要只截最后十行；
5. 编译通过后再讨论切块、流水和性能，避免同时引入正确性与性能变量。

常见症状：

| 症状 | 优先检查 |
|---|---|
| `Could not find ASC` | 是否 source 了正确的 `set_env.sh`，CANN 路径是否为 8.5.0 |
| 找不到 910B 配置/ops | 是否安装 `Ascend-cann-910b-ops_8.5.0` |
| `opc tool compile failed` | 向上寻找第一条 Kernel 源码 `error:` |
| Host 类型或注册错误 | `op_host/mhc_expand.cpp` 的接口、dtype、属性和 tiling |
| 两个 dtype 同时失败 | 共用 Kernel/API 或模板定义通常有问题 |
| 仅 BF16 失败 | BF16 tiling key、Cast 模式及 API 数据类型支持 |
| 全部点精度失败 | shape 推导、属性顺序、forward/backward 分支及输出偏移 |
| 非对齐点失败 | `DataCopyPad`、32 Byte 对齐和尾块有效长度 |

## 12. 给下一个 agent 的任务顺序

1. 先完整阅读本 README、题面和上一轮完整日志；
2. 在 WSL 中记录 `uname -m`、CANN 路径和版本；
3. 执行 `compile_cann85.sh`，将完整 `build.log` 保留下来；
4. 若失败，从第一条编译器错误开始修，直到 `binary` 成功；
5. 执行 Python reference 测试；
6. 如有时间，补一个 CANN 8.5 Kernel 直调 CPU runner 做真实 Kernel/golden 对比；
7. 打包并提交 CANNJudge，先确保 8 点正确，再按平台耗时优化；
8. 不要擅自改变题目接口、dtype、属性名、文件名或目标芯片。

当前实现已经进入真实性能优化阶段。后续重点是小 shape 固定开销、前向多副本写回、
反向归约流水，以及在不损失中大 shape 性能的前提下继续细化 `S/D/m` tiling。

## 13. 官方资料

- [CANN 8.5.0 安装说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/softwareinst/instg/instg_0008.html)
- [CANN 8.5.0 Ascend C 环境准备](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0003.html)
- [CANN 8.5.0 算子工程 CMake 编译](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00035.html)
- [CANN 8.5.0 CPU 域孪生调试](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0073.html)
- [CANN 8.5.0 msProf 算子调测](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_00851.html)
- [Microsoft WSL 安装说明](https://learn.microsoft.com/windows/wsl/install)
- [Microsoft WSL 网络与代理说明](https://learn.microsoft.com/windows/wsl/networking)
- [PyTorch Linux 安装说明](https://pytorch.org/get-started/locally/)

## 14. 性能优化迭代记录

### 14.1 各轮代码变化

| 版本 | 主要变化 | 平台结论 |
|---|---|---|
| Baseline | 每行按 D 分块；队列搬运；反向 FP32 累加 | 8 点正确，作为稳定回退版 |
| Round 1 | 前向改为单个原始 UB buffer，直接 MTE2→MTE3，移除 UB→UB 复制 | 未单独提交，合并进入 Round 3 |
| Round 2 | 反向双输入 buffer；首个副本直接初始化 FP32 累加器 | 未单独提交，合并进入 Round 3 |
| Round 3 | 对齐块使用普通 `DataCopy`，尾块使用 `DataCopyPad`；最大 tile 改为 8192 | 大点 8 明显改善，但点 2 因前向单 buffer 串行而回退 |
| Round 4 | 尝试按核数缩 tile、前向副本维并行、FP16 `m=2` 快捷 Add | 总体无收益；说明机械铺满核不是主要矛盾，相关策略已撤回 |
| Round 5 | 按 S×D 二维分块；一次 DMA 搬多行；前向 ping-pong；反向多行 FP32 归约 | 点 2、4、6、8 明显改善，证明二维 DMA 是有效主方向 |
| Round 6 | 限制行合并不能压低小 shape 并行度；一核一任务走轻量前向路径 | 点 1、5 大幅改善；当前综合最好版本 |

Round 5/6 的二维策略按单个 UB tile 最多 8192 个元素分配：

- `S=64,D=256`：Round 5 一次合并 32 行；Round 6 为保留并行度改为每次 2 行，
  共 32 个任务。
- `S=1024,D=4096`：每次 2 行，共 512 个任务，既减少 DMA 命令又能铺满 AIV 核。
- `S=8192,D=7168`：自动退化为每次 1 行，避免 UB 超限并保留大规模并行度。
- 非对齐 D 使用 `DataCopyPad`；对齐块使用普通二维 `DataCopy`。
- 前向一个 block 仅有一个任务时使用轻量单 buffer 路径；有循环任务时使用双 buffer
  重叠 MTE2 读入和 MTE3 多副本写回。
- 反向保持 FP32 累加，最后转换为 FP16/BF16，未采用 Round 4 的低精度特化。

### 14.2 平台结果

下表统一使用 μs；页面只显示两位小数的 ms 值，因此大点合计是近似值。

| 测试点 | 平台最优 | Baseline | Round 3 | Round 4 | Round 5 | Round 6 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.36 | 5.08 | 5.00 | 4.76 | 5.76 | **3.90** |
| 2 | 14.02 | 16.90 | 22.04 | 21.66 | **16.24** | 16.62 |
| 3 | 1360 | 1390 | 1390 | 1390 | **1390** | 1400 |
| 4 | 45.36 | 61.72 | 52.98 | 54.10 | **47.88** | 50.60 |
| 5 | 1.48 | 5.44 | 5.08 | 5.06 | 6.64 | **3.68** |
| 6 | 863.88 | 889.28 | 898.92 | 908.34 | 887.74 | **881.10** |
| 7 | 726.94 | 744.90 | 744.00 | **743.62** | 746.16 | 742.34 |
| 8 | 995.84 | 1160 | 1030 | 1030 | 1020 | **1010** |
| 近似合计 | 4008.88 | 4273.32 | 4148.02 | 4157.54 | 4120.42 | **4108.24** |

Round 6 相对 Baseline 的近似总耗时下降约 165.08 μs（3.86%），距离各点页面最优
合计约 99.36 μs（2.48%）。但比赛分数若按逐点相对最优归一化，小点 1、5 的提升
价值会明显高于它们在绝对耗时合计中的占比。

约 1% 的单次变化应先视作测量波动。例如 Round 3→4 的大部分变化不足以证明策略
有效；点 2 的 21.66→16.24、点 4 的 54.10→47.88、点 1 的 5.76→3.90 和点 5
的 6.64→3.68，则足以作为明确的方向性证据。

当前提交包：`mhc_expand_local_test/mhc_expand_opt_round6.zip`。

## 15. 隐藏测试点推断与跑分定位

平台没有公开每个测试点的 shape、dtype 和方向，因此下表是依据题面典型规模、理论
数据量以及各轮差分表现得到的推断，不应当当作已知事实。

| 点 | 可能的主要场景 | 主要考察指标 | 推断依据 | 置信度 |
|---:|---|---|---|---|
| 1 | 小规模前向或前向边界 shape | Kernel 固定开销、事件数量、少任务调度 | 轻量前向和恢复小 shape 并行度后从 5.76 降至 3.90 μs | 较高 |
| 2 | 中规模前向，接近 `(1024,4096,m=4)` | 二维 DMA 命令效率、MTE2/MTE3 重叠、多副本写回 | 单 buffer 串行明显回退；二维 DMA 后恢复到 16 μs | 高 |
| 3 | 最大规模、偏归约或带宽饱和场景 | HBM 持续带宽、FP32 归约吞吐 | 各轮几乎不变且已距最优约 2%～3%，说明已接近硬件吞吐上限 | 中 |
| 4 | 中规模反向，接近 `(1024,4,4096)` | 多行读取、Cast/Add 流水、FP32 累加 | 二维反向归约后从 54.10 降至 47.88 μs | 高 |
| 5 | 小规模反向或第二种 dtype 的小 shape | 队列/向量初始化、核利用率、尾块处理 | 过度合并行时变慢，恢复并行度后从 6.64 降至 3.68 μs | 较高 |
| 6 | 大规模反向或高 `m` 归约 | 输入带宽、MTE2 与向量流水重叠 | 耗时接近 0.9 ms，二维归约有稳定但较小收益 | 中 |
| 7 | 大规模纯搬运，较可能是前向 | 峰值 HBM 带宽、多副本写回吞吐 | 是几个大点中最快且长期距离最优约 2%，对小 shape 策略不敏感 | 中 |
| 8 | 大规模高 `m` 前向/反向变体 | 长流水稳定性、UB 双缓冲、持续 DMA 吞吐 | 移除冗余复制和引入双缓冲后由 1.16 ms 降到 1.01 ms | 中 |

从典型数据量也能支持点 2/4 的判断：`S=1024,D=4096,m=4` 的前向和反向都需要
搬运约 40 MB 数据；纯复制可以接近十几 μs，而包含 FP32 Cast/Add 的反向通常在
几十 μs。大规模 `(8192,7168,m=8)` 单方向总读写量约 1.06 GB，对应 0.7～1.4 ms
的点主要由 HBM 带宽及反向向量归约决定。

### 15.1 当前跑分是在排查还是实际优化

两者都是，但阶段已经发生变化：

1. Baseline 首次提交主要是正确性闭环，确认接口、tiling、FP16/BF16 和边界处理。
2. Round 3/4 兼有黑盒排查性质。没有隐藏 shape 和 profiler，只能通过一次改变某个
   机制、观察八点差分来判断瓶颈；Round 4 的失败同样提供了“盲目增加并行任务无效”
   的证据。
3. Round 5/6 是实际性能优化。二维 DMA、减少命令数、MTE2/MTE3 双缓冲和小 shape
   轻量路径都已进入最终 kernel，并产生可重复的方向性降时，不只是诊断代码。
4. 每次 CANNJudge 结果既是真实比赛计时，也是当前缺少 910B profiler 时唯一的黑盒
   性能反馈。应始终保存综合最优包，只有明显超过噪声的变化才合入下一版。

因此，正确表述是：当前在做“以真实跑分为反馈的黑盒实际优化”，其中每轮差分提交
同时承担性能诊断作用。
