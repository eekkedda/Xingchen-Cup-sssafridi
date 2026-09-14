# 【C组简单题】LightningIndexer 算子

---

## 1. 算子说明

LightningIndexer 是稀疏注意力（Sparse Attention）场景下的**索引选择算子**：为每个 query token 从上下文（key 序列）中选出相关度最高的 Top-k 个位置索引，供后续稀疏注意力算子仅对这些位置做注意力计算，从而降低长序列场景下的计算量。

### 背景术语（题目内自解释）

- **多头注意力与 GQA**：注意力计算将隐藏状态拆分为多个"头"（Head）。Grouped Query Attention（GQA）中，若干个 query 头共享同一组 key/value 头，共享的一组称为一个 group，其包含的 query 头数记为 group size `g`。本算子中 key 的头数 `N2=1`，故 group size `g = N1`（query 头数）。
- **维度符号**：`B`=Batch Size，`S1/Q_S`=query 序列长度，`S2/K_S`=key 序列长度，`N1`=query 头数，`N2`=key 头数（本算子恒为 1），`D`=每个头的维度（HeadDim=128）。
- **数据排布**：当前仅实现 `BSND` 排布即可，即 `(B, S, N, D)`。
- **变长序列**：`actual_seq_lengths` 给出每个 batch 实际有效的 token 数（超出部分为 padding，不参与计算），为一维长度 B 的 int32 张量；也可传 None 表示与对应序列长度 S 相同。

### 计算公式

对某个 query token，设其 Index Query `Q_index ∈ R^{g×d}`（group 内 g 个头的向量），上下文 Index Key `K_index ∈ R^{S_k×d}`，权重 `W ∈ R^{g×1}`（group 内每个头一个权重），`S_k` 为上下文长度：

```
Indices = Top-k{ [1]_{1×g} @ [ (W @ [1]_{1×S_k}) ⊙ ReLU(Q_index @ K_index^T) ] }
```

即：先算 query 与 key 的相似度矩阵 `Q_index @ K_index^T`（g×S_k），ReLU 激活；用 W 沿 group 维加权；再用全 1 行向量 `[1]_{1×g}` 对 group 维求和，得到该 token 对每个 key 位置的得分（1×S_k）；最后在 S_k 维取 Top-k 位置的索引。

---

## 2. 输入输出说明

### 输入规格

| 输入 | 必选/可选 | 类型 | 形状 | 数据类型 | 含义 |
|---|---|---|---|---|---|
| query | 必选 | 张量 | (B, S1, N1, D) | float16/bfloat16 | 输入 Q，公式中的 Q_index。不支持空 tensor 与非连续 |
| key | 必选 | 张量 | (B, S2, N2, D) | float16/bfloat16 | 输入 K，公式中的 K_index。N2 恒为 1 |
| weights | 必选 | 张量 | (B, S1, N1) | float16/bfloat16/float32 | 输入 W，group 内每个头的权重。不支持空 tensor 与非连续 |
| actual_seq_lengths_query | 可选 | 张量 | (B,) | int32 | 每个 batch 中 query 的有效 token 数。传 None 表示与 S1 相同 |
| actual_seq_lengths_key | 可选 | 张量 | (B,) | int32 | 每个 batch 中 key 的有效 token 数。传 None 表示与 S2 相同 |
| sparse_count | 必选 | 属性 | 标量 | int32 | Top-k 选取的位置数（输出索引个数）。支持 [1, 2048]，以及 3072、4096、5120、6144、7168、8192；默认 2048 |
| sparse_mode | 必选 | 属性 | 标量 | int32 | 掩码模式：0=defaultMask（全 attend，不屏蔽）；3=rightDownCausal（query 序列右端对齐 key 序列右端的下三角掩码）。默认 3 |
| return_values | 必选 | 属性 | 标量 | bool | 是否输出 sparse_values。True 输出、False 不输出；默认 False |

> 说明：属性 `pre_tokens`、`next_tokens` 用于稀疏计算的关联 token 数，本算子仅支持默认最大值 `2^63-1`，参赛者无需修改。

### 输出规格

| 输出类型 | 必选/可选 | 形状 | 数据类型 | 含义 |
|---|---|---|---|---|
| sparse_indices | 必选 | (B, S1, N2, sparse_count) | int32 | 每个 query token 选中的 Top-k 个 key 位置索引 |
| sparse_values | 可选 | 与 sparse_indices 一致 | float16/bfloat16 | 被选索引对应位置的得分值，与 sparse_indices 一一对应；return_values=False 时不输出 |

### 形状约束

- query 头数 `N1 ≤ 64`（Atlas A2/A3 系列为 1~64 连续范围），key 头数 `N2 = 1`（故 group size `g = N1`）。
- HeadDim `D = 128`。
- query、key 数据类型须一致；weights 不为 float32 时，query、key、weights 三者数据类型须一致。
- 仅支持 ND 数据格式（任意多维非结构化连续格式）。
- `sparse_count` 取值范围：[1, 2048] 及 3072、4096、5120、6144、7168、8192。
- Ascend 950PR/950DT：query N1 仅支持枚举值 **8、16、24、32、64**（离散取值，非连续范围），且 weights 不支持 float32。

### 输出形状计算公式

- `sparse_indices`：在 query 形状的基础上去掉 D 维、末尾追加 sparse_count 维，即 `(B, S1, N2, sparse_count)`，其中 `N2=1`。
- `sparse_values`：形状与 `sparse_indices` 完全一致。

---

## 3. 算子逻辑说明

### 3.1 相似度计算与激活

1. 取当前 query token 在 group 内 g 个头的向量 `Q_index ∈ R^{g×D}`，与上下文 key `K_index ∈ R^{S_k×D}` 做矩阵乘：`score = Q_index @ K_index^T`，得到 `g × S_k` 的相似度矩阵。
2. 对 score 逐元素做 ReLU 激活（负相似度置 0）：`score = ReLU(score)`。

### 3.2 权重加权与 group 聚合

1. 用 group 内权重 `W ∈ R^{g×1}` 对 score 沿 group 维加权（W 广播为 `g × S_k`，逐元素相乘）：`weighted = W ⊙ score`。
2. 用全 1 行向量 `[1]_{1×g}` 对 group 维求和，得到该 query token 对每个 key 位置的聚合得分 `agg ∈ R^{S_k}`（长度为上下文长度）。即 `agg = Σ_{group} weighted`。

### 3.3 掩码与 Top-k 选择

1. 按 `sparse_mode` 应用掩码（屏蔽不可见位置，使其得分置 0 不参与选择）：
   - `sparse_mode=0`（defaultMask）：不屏蔽，所有 key 位置可见。
   - `sparse_mode=3`（rightDownCausal）：query 序列右端对齐 key 序列右端的下三角掩码。query 位置 `i` 仅可见 key 位置 `j`，当 `j ≤ S_k − S1 + i`（其中 S1 为 query 序列长度）。
2. 在掩码后的 `agg` 上取最大的 `sparse_count` 个位置，输出这些位置的索引（即 `sparse_indices`）。若开启 `return_values`，同时输出这些位置对应的得分（即 `sparse_values`）。

---

## 4. 决赛任务要求

- **精度保障**：针对不同 `N1`、变长序列与不同 `sparse_mode` 掩码，设计算子逻辑，保证 Top-k 索引选择正确。
- **性能优化**：充分发挥系统带宽与算力，优化 `QK^T` 矩阵乘与 Top-k 选择的数据搬运与并行度，算子性能更优。
- **切分最优**：探索 query 序列维度、group 维度与 key 上下文维度在多 batch / 变长场景下的切分方式，找到不同输入 shape 场景下的最优解。

---

## 功能示例

```python
import numpy as np

def lightning_indexer(query, key, weights, sparse_count, sparse_mode=3,
                      act_seq_k=None):
    # query:(B,S1,N1,D)  key:(B,S2,N2,D), N2=1  weights:(B,S1,N1)
    # 输出 sparse_indices:(B,S1,N2,sparse_count) int32
    q = query.astype(np.float32); k = key.astype(np.float32); w = weights.astype(np.float32)
    B, S1, N1, D = q.shape
    _, S2, N2, _ = k.shape
    indices = np.full((B, S1, N2, sparse_count), -1, dtype=np.int32)
    values   = np.full((B, S1, N2, sparse_count), 0.0, dtype=np.float32)
    for b in range(B):
        Klen = S2 if act_seq_k is None else int(act_seq_k[b])   # 当前 batch 有效 key 长度
        K = k[b, :Klen, 0, :]                                    # (Klen, D)
        for s in range(S1):
            Q = q[b, s, :, :]                                    # (N1, D) = (g, D)
            W = w[b, s, :].reshape(-1, 1)                        # (N1, 1) = (g, 1)
            score = Q @ K.T                                      # (g, Klen) 相似度
            score = np.maximum(score, 0.0)                      # ReLU
            if sparse_mode == 3:                                # rightDownCausal 掩码
                j = np.arange(Klen)
                mask = (j <= (Klen - S1 + s)).astype(np.float32)
                score = score * mask[None, :]
            agg = (W * score).sum(axis=0)                       # (Klen,) group 维加权求和
            kk = min(sparse_count, Klen)
            order = np.argsort(-agg, kind='stable')[:kk]        # 降序 Top-k，同分按索引升序
            order = np.sort(order)
            indices[b, s, 0, :kk] = order
            values[b, s, 0, :kk] = agg[order]
    return indices, values.astype(np.float16)

np.set_printoptions(precision=4, suppress=True)

# 示例1：rightDownCausal 掩码的作用（BSND，B=1, S1=2, N1=2, D=128, S2=4, sparse_count=2）
# key 位置 j 全为 (j+1)，query 全 1，weights 全 1 => QK^T = 128*(j+1)，得分随 j 递增
B, S1, N1, D, S2 = 1, 2, 2, 128, 4
query   = np.ones((B, S1, N1, D), dtype=np.float16)
key     = np.tile(np.arange(1, S2+1, dtype=np.float16).reshape(1, S2, 1, 1), (1, 1, 1, D))  # (1,4,1,128)
weights = np.ones((B, S1, N1), dtype=np.float16)
idx, val = lightning_indexer(query, key, weights, sparse_count=2, sparse_mode=3)
# 输出形状：(1, 2, 1, 2)；squeeze 后索引矩阵 (S1, sparse_count)
# [[1 2]   <- s=0：j<=4-2+0=2 可见 key 0/1/2，得分 [256,512,768]，Top-2 选 [1,2]
#  [2 3]]  <- s=1：j<=4-2+1=3 全可见，得分 [256,512,768,1024]，Top-2 选 [2,3]
# 对比 sparse_mode=0（全 attend）：s=0 因全可见会选 [2,3]，体现了 rightDownCausal 的屏蔽作用

# 示例2：多 batch 变长序列（BSND，B=2, S1=1, N1=2, D=128, S2=4, sparse_count=2, sparse_mode=0）
B, S1, N1, D, S2 = 2, 1, 2, 128, 4
query   = np.ones((B, S1, N1, D), dtype=np.float16)
key     = np.tile(np.arange(1, S2+1, dtype=np.float16).reshape(1, S2, 1, 1), (B, 1, 1, D))  # (2,4,1,128)
weights = np.ones((B, S1, N1), dtype=np.float16)
idx, val = lightning_indexer(query, key, weights, sparse_count=2, sparse_mode=0,
                             act_seq_k=[3, 4])
# 输出形状：(2, 1, 1, 2)
# batch0（有效 key 长度 3，得分 [256,512,768]）-> 选 [1,2]
# batch1（有效 key 长度 4，得分 [256,512,768,1024]）-> 选 [2,3]
```
