# MhcExpand Round 2 基线审计

审计对象是第二轮开始时的 `B_easy` Round 6 源码，基线 kernel SHA-256 为
`355e42c14dec6d06e4b2a13d3253ea8c35b84d872702c093cb0ec10cff9dc926`。工作区没有
Git 元数据，完整起点哈希见 `round2_baseline_sha256.txt`，源码与 Round 5/6 包已复制到
`snapshots/round2_start_20260913/`。本文不把 Python reference 当作实际 kernel 验证。

## 1. 分派与入口

- Host 只把输入 dtype 做成编译期模板参数，生成 FP16 与 BF16 两个 AIV 对象。方向、
  对齐/尾块、任务数量以及前向单双任务模式均是运行时分支。
- `MhcExpandTilingData` 共 40 字节：`s,d` 为 64 位；`m,backward,W,R,blockDim`
  为 32 位。生成 JSON 的 `opParaSize=48` 是运行时参数区信息，不应与结构体大小混为一谈。
- Round 6 的 `TPipe` 是 `KernelMhcExpand` 的成员，在 kernel 入口构造计算对象时一起构造；
  `Init` 再按方向初始化 UB。未走方向的 buffer 不会 `InitBuffer`。
- 前向的轻量分支条件是 `taskCount <= blockDim`。因 Host 令
  `blockDim=min(AIV,taskCount)`，该条件等价于 `taskCount<=AIV`。此分支不取四个显式
  ping-pong event，只取一个 `MTE2_MTE3` event。
- 已具备、跳过重做：二维多行 DMA、自适应 R、前向 raw UB、前向轻量/多任务执行分流、
  多任务 ping-pong、反向 FP32 顺序累加、按方向初始化。

## 2. Host tiling 与任务分配

记 `P=ceil(D/16)*16`，Host 取 `W=min(8192,P)`，初始
`R=min(32,floor(8192/W))`，再以 `ceil(S/AIV)` 限制 R，避免小工作量只占少数核。
若反向 GM 行跨度 `D*m*2` 不能由 `uint32_t` 字节间隔表示，R 强制为 1。

`tilesPerRow=ceil(D/W)`、`rowGroups=ceil(S/R)`、`T=tilesPerRow*rowGroups`，
`blockDim=min(AIV,T)`。每核使用跨步分配：核 `c` 执行
`c,c+blockDim,c+2*blockDim,...<T`。不存在连续区间分配。公开自构造形状的具体结果见
`tiling_trace.json`；trace 使用本机 CANN 8.5 的 Ascend910B2C 配置（48 AIV、196608 B
UB），真实 Host 仍通过平台接口查询，未在提交代码中写死这些数值。

## 3. UB 分配与生命周期

令 `N=R*W`。W 是 16 元素/32 字节对齐值，故各 `InitBuffer` 大小也按 32 字节对齐。

| 路径 | Buffer | 份数与类型 | 保留字节 | 生命周期 |
|---|---|---|---:|---|
| 前向 | `forwardBuf_` | 2×DT_X raw TBuf | `4N` | 单槽从 MTE2 写入至该槽最后一份 MTE3 读完；两槽交替 |
| 反向 | `inQueue_` | 深度 2×DT_X | `4N` | 每份从 MTE2 写入至 Vector Cast/Add/Axpy 读取完成 |
| 反向 | `outQueue_` | 深度 1×DT_X | `2N` | 最终 Cast 写入至 MTE3 读取完成 |
| 反向 | `valueFp32Buf_` | 1×FP32 | `4N` | 后续每个 m 流的 Cast 临时值，Add 后复用 |
| 反向 | `accFp32Buf_` | 1×FP32 | `4N` | 首流 Cast 建立，随后原位累加，最终输出 Cast 后结束 |

因此前向保留 `4N`，反向保留 `14N`；N 最大 8192 时分别为 32768 B 和
114688 B。轻量前向虽然只访问第一个槽，当前仍保留完整 `4N`，不是物理单缓冲分配。
队列/TPipe 的非 UB 元数据及编译器内部保留量未从 API 暴露，未计入上表，也未发现
shared temporary buffer。

非对齐尾块由 `DataCopyPad` 把每行补到 `align_up(validLength,16)`，padding 值为 0。
Vector 计算会覆盖 padding；MTE3 每行只写 `validLength*2` 字节，所以 padding 不进入有效输出。

## 4. DMA 路径与调用数

每个前向 tile 调用一次 MTE2 读、m 次 MTE3 写；每个反向 tile 调用 m 次 MTE2 读、
一次 MTE3 写，总调用数均为 `(m+1)*T`。这只是 API/命令数量，不代表物理 burst 数。

- 对齐路径使用 `DataCopyParams{blockCount=rowCount, blockLen=validLength/16,
  srcStride/dstStride=(gmRowStride-validLength)/16}`，块长和间隔单位为 32 B block。
- 非对齐路径使用 `DataCopyExtParams`，`blockLen=validLength*2` 字节，GM 间隔也是字节；
  UB 间隔为 0。读入额外传 `DataCopyPadExtParams`，写出仅写有效字节。
- 是否对齐同时检查 GM 首地址、有效宽、GM 行距的 16 元素对齐，以及 gap 不超过
  `uint16_t`。二维 `blockCount` 为 `uint16_t`，Host 的 R 上限 32，安全。
- 尾宽、尾行及每类 DMA 的实际调用数量已在 `tiling_trace.json` 分案例展开。

## 5. 反向向量计算与求和顺序

每 tile 保持 k=0 到 m-1 的顺序。第 0 流直接 `Cast(DT_X->float)` 到 accumulator；
其余每流先 Cast 到 `valueFp32`，执行一次 `PipeBarrier<PIPE_V>`，再执行 FP32 Add 原位累加。
循环尾还有一次 Vector barrier。最终执行一次 `Cast(float->DT_X)`：BF16 使用
`CAST_RINT`，FP16 使用 `CAST_NONE`。

基线每 tile 的调用数为：输入 Cast m 次、FP32 Add `m-1` 次、输出 Cast 1 次，
Vector barrier `2m-1` 次。首流之后没有改变归约树，也没有 half 累加。

## 6. 同步事件

前向多任务路径每核从全局 `GetTPipePtr()` 各取两个 `MTE2_MTE3` 和两个
`MTE3_MTE2` event ID。每槽先用初始 `MTE3_MTE2` flag 解锁；MTE2 完成后以
`MTE2_MTE3` 保护 MTE3 的 RAW 依赖；最后一个 replica 发出后，以 `MTE3_MTE2`
保护下一轮 MTE2 覆盖同槽的 WAR/WAW 依赖。循环退出前等待两个槽，保证最后写出消费完 UB。

前向轻量路径只有一次读和一组写，只需一个 `MTE2_MTE3` RAW event；kernel 结束承担最终
MTE3 完成边界。反向显式源码只含 Vector barrier，MTE2→Vector、Vector→MTE3 及队列槽复用
由 `TQue` 的 Alloc/EnQue/DeQue/FreeTensor 管理；其内部 event ID 未从当前源码直接导出，记为
未核实，不能据此删除 barrier。

## 7. 构建产物与验证边界

- CANN 8.5.0、`ascend910b` 的 fresh `binary + install` 已通过，日志为
  `mhc_expand_local_test/build_round2_e0_baseline/build.log`。
- 生成两个 14024 B AIV 对象：FP16 key 后缀 `_1`，BF16 key 后缀 `_27`；对象 JSON
  标记 `RT_DEV_BINARY_MAGIC_ELF_AIVEC`、动态 blockDim、1 个 workspace（Host 设置为 0）。
- Conda `cann85` 环境中的 Python reference 3 项通过。它只验证公式，未执行上述对象。
- 当前 WSL 无 NPU；实际 kernel 输出、910B profiling、CANNJudge 第二轮分数均未验证。

## 8. 第二轮首批可测假设

1. E1：只把 `TPipe` 移到入口并传引用，保持所有数据路径不变，观察小工作量固定开销；
2. E2：仅 FP16 反向把后续流 Cast+Add 换为 mixed Axpy，保持 BF16、tile 与 k 顺序；
3. E3/E4：在独立实验中用查询到的 UB 容量分别扩大前向/反向 N，不共享全局常数；
4. 不以八点耗时总和单独决定合入。每次必须绑定 submission ID、平台实际分数、八点原始
   时间、当时 TBest 与源码哈希；当前公式未知，不假定算术/几何平均或每点等权。
