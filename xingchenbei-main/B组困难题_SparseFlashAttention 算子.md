# 【B组困难题】SparseFlashAttention 算子

---

## 1. 算子说明

SparseFlashAttention（SFA）是针对长序列推理场景的**稀疏注意力计算算子**：根据预先选定的稀疏索引，仅对 query 与少量重要的 key/value 位置做注意力计算（而非全部上下文），从而大幅降低计算量。它通常与索引选择算子（如 `lightning_indexer`）配合使用——后者选出每个 query token 重要的 key 位置索引，本算子依据这些索引完成注意力计算，并针对稀疏访问带来的离散访存做了搬运聚合优化。

### 背景术语（题目内自解释）

- **注意力计算**：标准注意力为 `softmax(Q @ K^T / √d_k) @ V`，其中 Q 为 query、K 为 key、V 为 value，`d_k` 为每个头的维度，`√d_k` 用于缩放防止点积过大。
- **稀疏注意力**：只取少量重要 key/value 位置（记为 `K̃, Ṽ`）参与计算，公式为 `softmax(Q @ K̃^T / √d_k) @ Ṽ`，降低计算量与访存量。
- **GQA 与 KV 头数**：Grouped Query Attention 中多个 query 头共享一组 key/value 头。本算子 key/value 头数 `KV_N=1`，即所有 query 头共享同一组 KV。
- **RoPE（旋转位置编码）**：将位置信息以旋转矩阵方式注入 query/key，使注意力具备相对位置感知。本算子 query/key 各带一份 RoPE 向量（维度 `Dr`），必传不可为空。
- **MLA-absorb 模式**：Multi-head Latent Attention 的"吸收"模式（`attentionMode=2`），将部分投影矩阵吸收到 query/value 中以减少在线计算量，RoPE 部分单独处理并融合到注意力输出。
- **sinks**：注意力中可学习的偏置项（仅 Ascend 950PR/950DT 支持）。本题目基于 aclnn V1 接口，不涉及该参数。
- **维度符号**：`B`=Batch Size，`Q_S`=query 序列长度，`KV_S`=key/value 序列长度，`Q_N`=query 头数，`KV_N`=key/value 头数（恒为 1），`Q_D/KV_D`=每个头的维度（=512），`Dr`=RoPE 维度（=64），`sparse_size`=每个 query token 选取的 key 位置数。
- **数据排布**：本算子仅支持 `BSND` 排布，即 `(B, S, N, D)`。
- **变长序列**：`actual_seq_lengths` 给出每个 batch 实际有效 token 数（超出部分为 padding，不参与计算），为一维长度 B 的 int32 张量；也可传 None 表示与对应序列长度 S 相同。

### 计算公式

```
AttentionOut = softmax( Q @ K̃^T / √d_k ) @ Ṽ
```

其中 `K̃, Ṽ` 为依据 `sparseIndices` 从 KV 缓存中离散选取的重要性较高的 Key/Value，`d_k = Q_D`。

---

## 2. 输入输出说明

### 输入规格

| 输入 | 必选/可选 | 类型 | 形状 | 数据类型 | 含义 |
|---|---|---|---|---|---|
| query | 必选 | 张量 | (B, Q_S, Q_N, Q_D) | float16/bfloat16 | 输入 Q。Q_D=512。不支持空 tensor 与非连续 |
| key | 必选 | 张量 | (B, KV_S, KV_N, Q_D) | float16/bfloat16 | 输入 K。KV_N=1。不支持空 tensor 与非连续 |
| value | 必选 | 张量 | (B, KV_S, KV_N, Q_D) | float16/bfloat16 | 输入 V，shape 与 key 一致。不支持空 tensor 与非连续 |
| sparseIndices | 必选 | 张量 | (B, Q_S, KV_N, sparse_size) | int32 | 离散选取 KV 缓存的索引。要求每行有效值在前半部分、无效值在后半部分，sparse_size>0 |
| actual_seq_lengths_query | 可选 | 张量 | (B,) | int32 | 每个 batch 中 query 的有效 token 数。传 None 表示与 Q_S 相同 |
| actual_seq_lengths_kv | 可选 | 张量 | (B,) | int32 | 每个 batch 中 key/value 的有效 token 数。传 None 表示与 KV_S 相同 |
| queryRope | 必选 | 张量 | (B, Q_S, Q_N, Dr) | float16/bfloat16 | query 的 RoPE 信息，Dr=64。不支持为空 |
| keyRope | 必选 | 张量 | (B, KV_S, KV_N, Dr) | float16/bfloat16 | key 的 RoPE 信息，Dr=64。不支持为空 |
| scaleValue | 必选 | 属性 | 标量 | float16 | 注意力缩放系数，对应公式中的 `1/√d_k`。接口传入为 double，内部按 float16 精度处理 |
| sparseBlockSize | 必选 | 属性 | 标量 | int64 | 稀疏选择的块大小：=1 为 Token-wise（逐 token 独立选取）；>1 且 ≤128 为 Block-wise（块内共享选择决策） |
| sparseMode | 必选 | 属性 | 标量 | int64 | 掩码模式：0=全部计算（不屏蔽）；3=rightDownCausal（query 序列右端对齐 key 序列右端的下三角掩码）。常用取值 3 |
| attentionMode | 必选 | 属性 | 标量 | int64 | 注意力模式，仅支持 2（MLA-absorb 模式） |
| returnSoftmaxLse | 必选 | 属性 | 标量 | bool | 是否输出 softmaxMaxOut/softmaxSumOut。True 输出、False 不输出；默认 False |

> 说明：属性 `pre_tokens`、`next_tokens` 用于稀疏计算的关联 token 数，本算子仅支持默认最大值 `2^63-1`，参赛者无需修改。

### 输出规格

| 输出类型 | 必选/可选 | 形状 | 数据类型 | 含义 |
|---|---|---|---|---|
| attentionOut | 必选 | (B, Q_S, Q_N, Q_D) | float16/bfloat16 | 注意力计算最终结果 |
| softmaxMaxOut | 可选 | (B, KV_N, Q_S, Q_N/KV_N) | float | 每行 Q@K̃^T 的最大值（softmax 数值稳定用）。returnSoftmaxLse=False 时不输出 |
| softmaxSumOut | 可选 | (B, KV_N, Q_S, Q_N/KV_N) | float | 每行 Q@K̃^T 减去 max 后取 exp 的求和（softmax 分母）。returnSoftmaxLse=False 时不输出 |

### 形状约束

- 仅支持推理场景，支持图模式。
- query 头数 `Q_N`：Ascend 950PR/950DT 支持 1~128；Atlas A2/A3 系列仅支持枚举值 **1、2、4、8、16、32、64、128**（离散取值，非连续范围）。
- key/value 头数 `KV_N = 1`。
- HeadDim `Q_D = KV_D = 512`，RoPE 维度 `Dr = 64`。
- query、key、value 数据类型必须一致。
- RoPE（queryRope/keyRope）必传，不支持为空。
- `sparseBlockSize`：Ascend 950PR/950DT 只支持 1；Atlas A2/A3 系列支持 [1,128] 且为 2 的幂次方。
- 仅支持 ND 数据格式（任意多维非结构化连续格式）。

### 输出形状计算公式

- `attentionOut`：与 query 形状一致，即 `(B, Q_S, Q_N, Q_D)`。
- `softmaxMaxOut` / `softmaxSumOut`：`(B, KV_N, Q_S, Q_N/KV_N)`，其中 `KV_N=1`，故为 `(B, 1, Q_S, Q_N)`，逐 query 头记录 softmax 的最大值与求和。

---

## 3. 算子逻辑说明

### 3.1 稀疏索引 Gather

对每个 query token，根据 `sparseIndices`（形状 `(B, Q_S, KV_N, sparse_size)`）中记录的 key 位置索引，从 KV 缓存中离散 gather 出对应的 `K̃` 与 `Ṽ`。要求每行有效索引集中在前半部分、无效值在后半部分，以便聚合访存。`sparseBlockSize` 决定选择粒度：1 为逐 token，>1 为按块共享决策。

### 3.2 注意力主流程

1. 计算 query 与稀疏 key 的相似度并缩放：`score = Q @ K̃^T * scaleValue`（`scaleValue = 1/√d_k`）。
2. 数值稳定 softmax：先取每行最大值 `softmaxMaxOut = max(score)`，再做 `exp(score - max)` 与求和 `softmaxSumOut = Σ exp(...)`，最后归一化得注意力权重 `attn = exp(...) / Σ exp(...)`。
3. 加权求和：`attentionOut = attn @ Ṽ`。

### 3.3 RoPE 与 MLA-absorb 融合

在 `attentionMode=2`（MLA-absorb）模式下，RoPE 采用 **"content 与 rope 沿特征维拼接"** 的方式融合进 score 计算（而非旋转后相加）：

```
score = concat(query, queryRope) @ concat(K̃, keyRopẽ)^T * scaleValue
      = (query @ K̃^T + queryRope @ keyRopẽ^T) * scaleValue   # 拼接 matmul 等价于分段内积之和
attentionOut = softmax(score) @ Ṽ
```

即 query 的 content 部分（`Q_D=512`）与 queryRope（`Dr=64`）沿特征维拼接为 576 维作为左矩阵，稀疏 gather 出的 K̃ 与 keyRopẽ 同样拼接为 576 维作为右矩阵，做拼接 matmul 得 score。关键点：
- `queryRope/keyRope` 已是应用旋转位置编码后的结果（输入即"位置编码的输出"，算子内不再做旋转，直接与 content 拼接）；
- `value`（Ṽ）仅用 content 部分（512 维，无 rope）；
- 数学上等价于 `query@K̃^T + queryRope@keyRopẽ^T`，rope 段承载相对位置信息。

### 3.4 掩码处理

按 `sparseMode` 应用掩码：`sparseMode=0` 不屏蔽；`sparseMode=3`（rightDownCausal）为 query 右端对齐 key 右端的下三角掩码，屏蔽不可见位置使其不参与 softmax。

---

## 4. 决赛任务要求

- **精度保障**：针对不同 `Q_N`、变长序列、不同 `sparseBlockSize` 与 `sparseMode` 掩码，设计算子逻辑，保证稀疏注意力计算精度正确。
- **性能优化**：针对稀疏索引带来的离散访存，优化 gather 的搬运聚合与 `Q@K̃^T` 矩阵乘的并行度，充分发挥系统带宽与算力。
- **切分最优**：探索 query 序列维度、query 头维度与稀疏索引维度在多 batch / 变长场景下的切分方式，找到不同输入 shape 场景下的最优解。

---

## 功能示例

```python
import numpy as np

def sparse_flash_attention(query, key, value, sparse_indices, scale_value,
                           act_seq_kv=None):
    # query:(B,Q_S,Q_N,Q_D)  key/value:(B,KV_S,KV_N,Q_D), KV_N=1
    # sparse_indices:(B,Q_S,KV_N,sparse_size)  输出 attentionOut:(B,Q_S,Q_N,Q_D)
    # 为突出稀疏索引选择与注意力主流程，本参考实现聚焦核心公式 softmax(Q@K̃^T/√d)@Ṽ，
    # 省略 3.3 节 RoPE 拼接融合（即 score 中的 queryRope@keyRopẽ^T 段），本示例 score 仅含 query@K̃^T。
    q = query.astype(np.float32); k = key.astype(np.float32); v = value.astype(np.float32)
    B, S1, N1, D = q.shape
    _, S2, N2, _ = k.shape
    out = np.zeros((B, S1, N1, D), dtype=np.float32)
    for b in range(B):
        Klen = S2 if act_seq_kv is None else int(act_seq_kv[b])   # 当前 batch 有效 KV 长度
        K = k[b, :Klen, 0, :]                                     # (Klen, D)
        V = v[b, :Klen, 0, :]                                      # (Klen, D)
        for s in range(S1):
            idx = sparse_indices[b, s, 0]                         # (sparse_size,) 位置索引
            valid = idx[(idx >= 0) & (idx < Klen)]                # 过滤无效索引
            Ksel = K[valid]; Vsel = V[valid]                       # 稀疏 gather 出 K̃, Ṽ
            Q = q[b, s]                                           # (N1, D)
            score = (Q @ Ksel.T) * scale_value                    # (N1, m) 相似度并缩放
            mx = score.max(-1, keepdims=True)                     # softmax 数值稳定
            exp = np.exp(score - mx)
            attn = exp / exp.sum(-1, keepdims=True)               # (N1, m) 注意力权重
            out[b, s] = attn @ Vsel                                # (N1, D) 加权求和
    return out.astype(np.float16)

np.set_printoptions(precision=4, suppress=True)

# 示例1：稀疏索引 gather 的作用（BSND，B=1, Q_S=2, Q_N=2, Q_D=512, KV_S=4, sparse_size=2）
# query 全 1，key 全 0（=> Q@K̃^T=0，softmax 均匀），value 位置 j 全为 (j+1)
B, S1, N1, D, S2 = 1, 2, 2, 512, 4
query   = np.ones((B, S1, N1, D), dtype=np.float16)
key     = np.zeros((B, S2, 1, D), dtype=np.float16)
value   = np.tile(np.arange(1, S2+1, dtype=np.float16).reshape(1, S2, 1, 1), (1, 1, 1, D))  # (1,4,1,512)
sparse_indices = np.array([[[[0, 1]], [[2, 3]]]], dtype=np.int32)  # (1,2,1,2)：s0 选 key 0/1，s1 选 key 2/3
scale = 1.0 / np.sqrt(D)
out = sparse_flash_attention(query, key, value, sparse_indices, scale)
# 输出形状：(1, 2, 2, 512)
# s=0：选 key 0/1（value=1/2），softmax 均匀 => 输出每维均值 1.5
# s=1：选 key 2/3（value=3/4），softmax 均匀 => 输出每维均值 3.5
print('s=0 输出均值:', out[0, 0].mean())   # 1.5
print('s=1 输出均值:', out[0, 1].mean())   # 3.5

# 示例2：多 batch 变长序列（BSND，B=2, Q_S=1, Q_N=2, Q_D=512, KV_S=4, sparse_size=2）
B, S1, N1, D, S2 = 2, 1, 2, 512, 4
query   = np.ones((B, S1, N1, D), dtype=np.float16)
key     = np.zeros((B, S2, 1, D), dtype=np.float16)
value   = np.tile(np.arange(1, S2+1, dtype=np.float16).reshape(1, S2, 1, 1), (B, 1, 1, D))  # (2,4,1,512)
sparse_indices = np.array([[[[0, 1]]], [[[2, 3]]]], dtype=np.int32)  # (2,1,1,2)
out = sparse_flash_attention(query, key, value, sparse_indices, scale, act_seq_kv=[3, 4])
# 输出形状：(2, 1, 2, 512)
# batch0（有效 KV 长度 3，选 key 0/1）=> 输出均值 1.5
# batch1（有效 KV 长度 4，选 key 2/3）=> 输出均值 3.5
print('batch0 输出均值:', out[0, 0].mean())  # 1.5
print('batch1 输出均值:', out[1, 0].mean())  # 3.5
```
