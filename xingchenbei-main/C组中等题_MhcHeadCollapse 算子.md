# 【C组中等题】MhcHeadCollapse 算子（mHC四路归一）

---

## 1. 算子说明

MhcHeadCollapse 算子（又称"mHC 四路归一"）是 Manifold-Constrained Hyper-Connections（mHC）架构中位于 Transformer 解码器末端的可学习折叠算子。其功能是将 `n` 路并行的打包残差流（packed residual streams）通过 RMS 归一化、可学习门控线性投影与加权求和，折叠为单路隐藏态输出。

该算子接收输入打包张量 `x`（形状 `(S, B, n*H)` 或 `(T, n*H)`），内部先对 `n*H` 维做 RMS 归一化，再通过一个无偏置线性层 `W ∈ ℝ^{n×(nH)}` 投影得到 `n` 个混合系数，经 sigmoid 门控激活后，与 `n` 个 H 维残差流逐路加权求和，最终输出 `(S, B, H)` 或 `(T, H)` 的单路隐藏态。默认配置下 `n=4`，故常被称为"四路归一"。

> **注意**：当前生产代码仅支持 3 维 `(S, B, n*H)` 输入（`s, _, _ = hidden_states.shape` 解包 3 值），2 维 `(T, n*H)` 为备选扩展形态，需在算子实现中同时兼容两种维度。

该算子是可学习的（含权重 `W`、偏置 `hc_base`、标量 `hc_scale`）。当 `weight = 0`、`hc_base = 0`、`hc_scale = 0` 时，算子退化为参数无关的算术均值归一（成固定比例 `n/2`），保证训练起步数值一致；优化器更新后逐步引入可学习偏置以提升多步 MTP 损失对齐能力。注意：`reset_parameter()` 仅重置 `hc_base` 与 `hc_scale`，`weight` 由外部初始化方法（如 `config.init_method`）独立控制，若需等价于均值归一，必须确保 `weight` 也为零矩阵。

---

## 2. 输入输出说明

### 输入规格

| 输入 | 类型 | 形状 | 数据类型 | 含义 |
|---|---|---|---|---|
| x | 张量 | (S, B, n*H) 或 (T, n*H) | float32/bf16/fp16 | 输入打包残差流，最后一维为 n 路残差流的拼接；前序维度（T 或 S、B）为批量维度 |
| weight | 张量 | (n, n*H) | float32 | 可学习门控线性投影权重矩阵，无偏置 |
| hc_base | 张量 | (n,) | float32 | 可学习门控偏置向量 |
| hc_scale | 张量 | (1,) | float32 | 可学习门控缩放标量 |
| eps_norm | 属性 | 标量 | float32 | RMS 归一化防除零参数，建议值 1e-6 |
| eps_hc | 属性 | 标量 | float32 | sigmoid 输出防零参数，建议值 1e-6 |

### 输出规格

| 输出 | 形状 | 数据类型 | 含义 |
|---|---|---|---|
| y | 与 x 对应：(S, B, H) 或 (T, H) | 与 x 相同 | 加权折叠后的单路隐藏态 |

### 形状约束

- 输入 `x` 仅支持 2 维 `(T, n*H)` 或 3 维 `(S, B, n*H)`，其他维度数不支持。
- 残差流路数 `n` 为任意正整数，建议取值 **2、4、8**（`n*H` 需能被 `n` 整除，即 `nH % n == 0`）。
- 单路隐藏维度 `H` 建议为 64 的整数倍（典型值 1024、7168），无硬性上限。
- 仅支持 `FLOAT32`、`BFLOAT16`、`FLOAT16` 数据类型与 `ND` 数据格式；内部计算统一在 FLOAT32 下进行，输入输出类型可低于 FLOAT32。
- 输入含 `-inf/inf/nan` 时，对应位置输出 `nan`。
- 算子默认采用确定性实现，相同输入多次调用结果一致。

### 输出形状计算公式

- `y` 形状：`x` 的前序维度（除了最后一维）不变，最后一维由 `n*H` 变为 `H`。
- 例如 `x ∈ (S, B, n*H)` → `y ∈ (S, B, H)`；`x ∈ (T, n*H)` → `y ∈ (T, H)`。

---

## 3. 算子逻辑说明

### 3.1 初始化阶段：RMS 归一化

1. 将输入 `x` 提升至 FLOAT32，记为 `flat`（形状 `(…, n*H)`）。
2. 对 `flat` 沿最后一维（`dim=-1`，即 `n*H` 维）计算均方根（RMS）逆：

   $$ \text{rms\_inv} = \frac{1}{\sqrt{\frac{1}{nH}\sum_{k=1}^{nH} flat_k^2 + \varepsilon_{\text{norm}}}} $$

   其中 `flat` 的平方逐元素计算，`mean` 沿最后一维 keepdim，`rsqrt` 为倒数平方根。`rms_inv` 形状为 `(…, 1)`。

### 3.2 门控打分阶段：线性投影与 RMS 逆加权

1. 用权重矩阵 `weight ∈ ℝ^{n×(nH)}` 对 `flat` 做线性投影（无偏置），得到 logits（推导用中间变量，实际代码中直接与 `rms_inv` 相乘得到 `mixes`）：

   $$ \text{mixes\_logits}_{i} = \sum_{k=1}^{nH} W_{i,k} \cdot flat_k,\quad i=1,\dots,n $$

2. 将 logits 与 RMS 逆逐元素相乘（广播至 `(…, n)`），得到混合系数 `mixes`：

   $$ \text{mixes}_i = \text{mixes\_logits}_i \cdot \text{rms\_inv} $$

   此步等价于：对 RMS 归一化后的特征做线性投影，避免显式存储归一化特征。

### 3.3 门控激活阶段：sigmoid + 可学习参数

1. 对 `mixes` 施加可学习门控：

   $$ \text{pre}_i = \sigma(\text{mixes}_i \cdot s_{\text{scale}} + b_{\text{base},i}) + \varepsilon_{\text{hc}} $$

   其中 `σ` 为 sigmoid 函数；`hc_scale ∈ ℝ^1`、`hc_base ∈ ℝ^n` 为可学习参数；`hc_eps` 防除零。`pre` 形状为 `(…, n)`。

### 3.4 加权归一阶段：逐路加权求和

1. 将 `flat` reshape 为 `(…, n, H)` 的按路视图：`streams ∈ ℝ^{…×n×H}`。
2. 将 `pre` 升维为 `(…, n, 1)`，与 `streams` 逐元素相乘，再沿 `dim=-2`（即 `n` 维）求和：

   $$ \text{output} = \sum_{i=1}^{n} \text{pre}_i \cdot \text{streams}_i \in \mathbb{R}^{…×H} $$

3. 将 `output` 从 FLOAT32 回退至输入数据类型，作为最终输出 `y`。

### 3.5 特殊说明

- **延迟初始化**：`hc_base` 与 `hc_scale` 可被 `reset_parameter()` 置零。当 `weight` 也为零矩阵时，`mixes ≡ 0`，`pre_i = σ(0) + ε_hc ≈ 0.5 + 1e-6`，输出 `≈ 0.5 · Σ_i streams_i = (n/2) · mean(streams)`，与参数无关的算术均值归一成固定比例。此设计保证训练起步时与参数-free 路径数值一致。注意：`reset_parameter()` 仅重置 `hc_base` 与 `hc_scale`，`weight` 由外部初始化方法独立控制，需确保 `weight` 为零矩阵方满足等价性。
- **RMS 归一化**：gamma 固定为 1，不可学习，仅做 scale-invariant 归一化。
- **精度约束**：RMS 归一化与门控线性投影必须在 FLOAT32 下计算，过早转 BF16/FP16 会显著偏移训练 loss 与梯度。

### 3.6 边界条件与异常处理

| 条件 | 期望行为 |
|------|----------|
| 输入 `x` 最后一维的大小不能被 `n` 整除 | reshape 阶段抛形状错误，算子入口应前置校验 `x.shape[-1] % n == 0` |
| `s=0` 或 `b=0`（空输入） | reshape 时 `-1` 推断可能产生非预期形状，建议前置校验 `s>0` 且 `b>0` |
| 输入含 `NaN` 或 `Inf` | 输出对应位置为 `NaN`（不静默屏蔽，便于上游定位） |
| `weight` 为全零矩阵 | 所有 `mixes ≡ 0`；此时 `pre_i = σ(hc_base_i) + ε_hc`，各路权重由 `hc_base` 决定，仅当 `hc_base=0` 时才退化为均匀权重 `≈ 0.5` |
| `hc_scale` 或 `hc_base` 为 DTensor（分布式场景） | 需通过 `to_local()` 取本地分片后再参与计算，算子本身不感知分布式 |

### 3.7 反向梯度计算

本算子为全可微算子，反向传播时各参数梯度计算如下：

- **对 `weight` 的梯度**：由 `mixes = (flat @ W^T) * rms_inv`，`rms_inv` 对 `W` 无梯度依赖（`rms_inv` 仅由 `flat` 决定，`flat` 不依赖 `W`），故 `∂L/∂W = (∂L/∂mixes ⊙ rms_inv)^T @ flat`，对 batch 维度求和后形状 `(n, n*H)`。
- **对 `hc_base` 的梯度**：`∂L/∂hc_base = Σ_batch ∂L/∂pre ⊙ σ'(mixes·hc_scale + hc_base)`，对 batch 维度 reduce 求和，结果形状 `(n,)`。
- **对 `hc_scale` 的梯度**：`∂L/∂hc_scale = Σ_{batch,i} ∂L/∂pre_i · σ'(…) · mixes_i`，对 batch 和 n 维度同时 reduce 求和，结果形状 `(1,)`。
- **对 `x` 的梯度**：需同时计算 (a) RMS 逆分支 `∂rms_inv/∂x`、(b) 线性投影分支 `∂mixes/∂x`、(c) reshape→streams 分支 `∂output/∂streams` 的梯度，三项求和后链式回传。
- 实现时可将 `sigmoid` 输出 `pre` 与 `rms_inv` 保存为中间结果复用，避免反向重算。

---

## 4. 决赛任务要求

- **精度保障**：针对不同 shape 维度（2 维/3 维）与不同 `n` 取值（2/4/8），设计算子逻辑，保证 RMS 归一化与门控投影精度正确，输出与 NumPy/PyTorch 参考实现最大绝对误差 ≤ 1e-5（FP32）。
- **性能优化**：充分发挥系统带宽能力，将 RMS 归一化、线性投影、sigmoid 门控、加权 reduce 四步融合为单 kernel，消除中间张量（`rms_inv`、`mixes`、`pre`、`streams`）的显存落盘，算子性能更优。
- **切分最优**：探索输入 tensor 在 `(T, n*H)` 与 `(S, B, n*H)` 两种场景下的切分方式，找到不同输入 shape 场景下的最优解。`weight` 可预加载至 L1 缓存复用。

---

## 功能示例

```python
import numpy as np

def rms_norm_inv(x, eps=1e-6):
    """计算 RMS 归一化的倒数（gamma=1），仅返回 rms_inv"""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return 1.0 / rms

def sigmoid(x):
    """Sigmoid 激活"""
    return 1.0 / (1.0 + np.exp(-x))

def impl(x, weight, hc_base, hc_scale, eps_norm=1e-6, eps_hc=1e-6):
    """
    x: (T, n*H) 或 (S, B, n*H)，最后一维为 n 路残差流拼接
    weight: (n, n*H) 门控线性投影权重
    hc_base: (n,) 门控偏置
    hc_scale: (1,) 门控缩放标量
    """
    orig_dtype = x.dtype
    x = x.astype(np.float32)
    prefix_shape = x.shape[:-1]   # 前序维度，如 (S,B) 或 (T,)
    nH = x.shape[-1]

    # 从 weight 推断 n（或从外部传入）
    n = weight.shape[0]
    H = nH // n

    # 1. RMS 归一化逆（gamma=1）
    rms_inv = rms_norm_inv(x, eps=eps_norm)   # 形状 (..., 1)

    # 2. 线性投影 + 乘 RMS 逆
    mixes = np.dot(x, weight.T) * rms_inv    # 形状 (..., n)

    # 3. sigmoid 门控
    pre = sigmoid(mixes * hc_scale + hc_base) + eps_hc   # 形状 (..., n)

    # 4. 按路加权求和
    streams = x.reshape(prefix_shape + (n, H))           # 形状 (..., n, H)
    pre_exp = pre[..., :, np.newaxis]         # 形状 (..., n, 1)
    output = np.sum(pre_exp * streams, axis=-2)  # 形状 (..., H)

    return output.astype(orig_dtype)


np.set_printoptions(precision=4, suppress=True)

# 示例1：延迟初始化（hc_base=0, hc_scale=0），输出应与均值归一成比例
n, H = 4, 4
x = np.ones((1, 2, n * H), dtype=np.float32)  # (S=1, B=2, n*H=16)
x[:, 0, :] = np.arange(n*H, dtype=np.float32)          # stream 0
x[:, 1, :] = np.arange(n*H, dtype=np.float32) * 2      # stream 1

weight = np.zeros((n, n * H), dtype=np.float32)
hc_base = np.zeros((n,), dtype=np.float32)
hc_scale = np.zeros((1,), dtype=np.float32)

y = impl(x, weight, hc_base, hc_scale)
# 输出形状: (1, 2, 4)
# 延迟初始化下 pre_i ≈ 0.5，输出 ≈ (n/2) * mean(streams)
# y[0,0,:] ≈ (4/2) * mean(arange(16).reshape(4,4), axis=0) = 2 * [6,7,8,9] = [12,14,16,18]
# 实际值因 eps_hc 略有偏差
print("延迟初始化输出:", y)
# 延迟初始化输出: [[[12.0000 14.0000 16.0000 18.0000]
#                   [24.0000 28.0000 32.0000 36.0000]]]

# 示例2：可学习门控（非零权重），n=4, H=4
x2 = np.array([[[
    1, 2, 3, 4,  5, 6, 7, 8,  9,10,11,12,  13,14,15,16
]]], dtype=np.float32)  # shape (1, 1, 16)

# 构造门控权重：单位块对角矩阵，每路通道独立投影
weight2 = np.zeros((n, n*H), dtype=np.float32)
for i in range(n):
    weight2[i, i*H:(i+1)*H] = 1.0   # 每路取自身通道
hc_base2 = np.array([2.0, 1.0, 0.0, -1.0], dtype=np.float32)
hc_scale2 = np.array([1.0], dtype=np.float32)

y2 = impl(x2, weight2, hc_base2, hc_scale2)
# 输出形状: (1, 1, 4)
# 各通道 mixes_logits = [10, 26, 42, 58]，sigmoid(·) 单调递增，
# 第4路 sigmoid 最大（mixes=5.998 最大，即使 hc_base=-1.0 衰减后仍最大）→ 输出偏向 streams[3]=[13,14,15,16]
# 第1路 sigmoid 最小（mixes=1.034 最小，hc_base=2.0 使 sigmoid 进入饱和区）→ streams[0]=[1,2,3,4] 权重最小
print("可学习门控输出:", y2)

# 示例3：2 维批量输入（T=2, n=4, H=4），对每个样本独立计算
x3 = np.stack([
    np.ones((4 * 4,), dtype=np.float32),          # 全 1
    np.arange(16, dtype=np.float32),              # 递增
], axis=0)  # shape (2, 16)
y3 = impl(x3, weight, hc_base, hc_scale)
# 输出形状: (2, 4)
# y3[0] 为 4 个 2.0（均值归一成比例）
# y3[1] 为 [12, 14, 16, 18]
print("2 维批量输出:", y3)
```