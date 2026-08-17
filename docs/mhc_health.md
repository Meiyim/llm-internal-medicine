# mHC Health Monitor

监控 mHC (Manifold-Constrained Hyper-Connections) 层的健康状况。mHC 用如下传播替换普通残差：

```
x_{l+1} = H_res @ x_l + H_post^T · F(H_pre @ x_l)
```

每个 token、每个 hyper-connection 模块学习三个映射（`n = num_residual_streams`）：

- `h_pre`  `[s, b, n]`     — 多流聚合门（sigmoid）
- `h_post` `[s, b, n]`     — 多流扩展门（2·sigmoid）
- `h_res`  `[s, b, n, n]`  — Sinkhorn 双随机（doubly-stochastic）残差混合矩阵

每个 `HyperConnectionTransformerLayer` 含两个 hyper-connection 模块：`self_attention_hyper_connection`
（`attn`）与 `mlp_hyper_connection`（`mlp`），forward 中 attn 先于 mlp 执行。

---

## 开关与 no-op 保证

通过在 `internal_medicine_monitors` 中加入 `mhc_health`（或 `all`）启用。启用它在**任何**模型上都是安全的——
以下任一情况下它彻底 no-op（不 wrap 任何模块、不声明/产生任何指标、不抛异常）：

1. **mHC 类无法 import**：`setup_mhc_monitor` 在 import 失败时将类名绑定为 `None`，直接返回 `model`。
2. **模型未使用 mHC 层**：discovery 用 `isinstance(layer, HyperConnectionTransformerLayer)` 与
   `isinstance(mod, HyperConnectionModule)` 精确匹配（不做 duck-typing），普通 `TransformerLayer` 或
   `IdentityOp` 占位符都不会被匹配。

```yaml
internal_medicine_monitors:
    mhc_health: true
```

---

## 采集方式：wrap `compute_mappings`（非重算）

`h_pre` 不在 `HyperConnectionModule.forward` 的返回中，普通 forward hook 看不到它。因此本 monitor **包裹
（wrap）每个 mHC 模块的 `compute_mappings` 绑定方法**，直接捕获其真实返回的 `(h_pre, h_post, h_res)` —— 不重算。

- `compute_mappings` 是普通 Python 方法（仅 `@nvtx_decorator`，非 `@torch.compile`），在 `_forward_normal` 中以
  `self.compute_mappings(...)` 调用，且**不被 checkpoint**，因此在 grad-enabled 的正向中恰好执行一次；实例属性
  wrapper 干净地遮蔽类方法。
- 整个捕获逻辑受 `_should_monitor()` 门控（含 grad 门），非监控步只是 `orig(x)` + 一次布尔判断。
- `remove_hooks()` 恢复原始方法并清空所有状态。

### VRAM 安全（无泄漏）

跨调用状态仅有 `self._composite`（每 chunk 一个小 `[s*b, n, n]` 张量）与固定的 0 维累加器。规则：

- 捕获后立即对 `h_pre/h_post/h_res` `.detach()`，并在 `torch.no_grad()` 下做全部指标/复合计算——否则一个仍带梯度的
  张量会通过反向把整层 autograd graph 钉住（大泄漏）。
- wrapper 原样返回 `out`，除 0 维标量外不保留任何对它/其视图的引用。
- 复合映射 seed 用 `hb.clone()`（而非 `h_res` 的 reshape 视图），使 slot 不会 alias 模型的 `h_res` 存储。
- `step()` 每步清空 `self._composite`，`remove_hooks()` 一并清空。

热路径纪律见 `.claude/skills/monitor-hook-perf-rules`：hook 内无 D2H 同步、无集合通信，schema 在 `allocate_buffers`
前声明。TP 不沿 `n` 切分映射，故无需 hook 内通信；跨 rank 归约在 flush 时由 `gather_and_aggregate` 完成（mean）。

---

## 监控指标

每个 hc 模块产出 13 个指标，指标名以 `attn_` / `mlp_` 前缀区分，全部按 token/batch 求均值
（并在 flush 时对 microbatch/rank 求均值）。日志键形如 `mhc_health/layer_{i}/{c}_{name}`，`{c}` ∈ `{attn, mlp}`；
对应的 `mhc_health/global_{c}_{name}` 由逐层累加器在 flush 时自动派生。

### 门控统计（h_pre / h_post）

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `{c}_h_pre_mean`  | `mean(h_pre)`  | 聚合门均值 |
| `{c}_h_pre_std`   | `std(h_pre)`   | 聚合门离散度 |
| `{c}_h_post_mean` | `mean(h_post)` | 扩展门均值 |
| `{c}_h_post_std`  | `std(h_post)`  | 扩展门离散度 |

### amax-gain（h_res 与复合映射）

paper 定义的最坏情况增益：矩阵的**最大绝对行和**界定前向传播的最坏放大，**最大绝对列和**界定反向传播的最坏放大。
对每个 token 的 `n×n` 矩阵计算，再对 token 求均值：

```
amax_gain_fwd = mean_t( max_i | Σ_j  M_ij | )      # 行和（forward）
amax_gain_bwd = mean_t( max_j | Σ_i  M_ij | )      # 列和（backward）
```

| 指标 | `M` 取值 | 诊断意义 |
|------|----------|----------|
| `{c}_amax_gain_fwd` | 本层 `h_res` | 单层前向最坏放大（双随机 → ≈1.0） |
| `{c}_amax_gain_bwd` | 本层 `h_res` | 单层反向最坏放大（≈1.0） |
| `{c}_composite_amax_gain_fwd` | 复合映射 `M_k = h_res_k @ M_{k-1}` | 跨层累积前向放大 |
| `{c}_composite_amax_gain_bwd` | 复合映射 | 跨层累积反向放大 |

单层 `h_res` 经 Sinkhorn 投影为双随机矩阵（行/列和 ≈ 1），故单层 amax-gain ≈ 1.0；复合映射的增益随深度偏离 1.0，
正是残差流放大/收缩的信号。

### 流形与结构（h_res 的形状诊断）

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `{c}_h_res_orth_dev` | `mean_t( \|\| h_resᵀ h_res − I \|\|_F )` | 偏离正交（等距）的程度：0 = 精确保范 |
| `{c}_composite_h_res_orth_dev` | 同上，作用于复合映射 | 跨层累积的非等距程度 |
| `{c}_h_res_beta_mean` | `mean_t( n − tr(h_res) )` | rank-1 擦除强度 β |
| `{c}_h_res_beta_std` | `std_t( n − tr(h_res) )` | β 的 token 级离散度：→0 说明退化成逐层常数 |
| `{c}_h_res_outer_dev` | `mean_t( \|\| h_res − h_post ⊗ h_pre \|\|_F )` | 残差混合中「读写外积」解释不掉的部分 |

**β = `n − tr(h_res)`。** 对 `H_res = I − β·ûûᵀ`（`‖û‖ = 1`）这个迹亏损**精确等于** β，所以 rank-1 擦除消融的核心判据
（β 收敛到 0 / (0,2) / 2）可以直接读这条曲线，不必再从 `orth_dev = 2β − β²` 反推 —— 后者在 β=0 与 β=2 处都是 0，
单看它分不出「擦除无效」和「Householder 反射」。其他构型下它就是普通的迹亏损：恒等 = 0，双随机 = `n` 减对角质量。

**外积偏差的下标方向。** `apply_h_res` 算的是 `out_i = Σ_j h_res[i,j]·x_j`，而子层通路写回的是
`h_post_i · Σ_j h_pre_j·x_j`，所以与 `h_res` 下标同序的 rank-1 矩阵是 `h_post ⊗ h_pre`（元素 `[i,j] = h_post_i·h_pre_j`），
本指标即残差混合与这个读写外积的 Frobenius 距离。注意它未减掉恒等分量，所以近似恒等的 `h_res` 会有一个 `√n` 量级的底；
对称的 `h_res`（如 rank-1 擦除）两个下标方向给出同一个值，方向只在 dense/正交构型下才有区别。

### 复合映射（composite mapping）

复合映射是本 pipeline stage / VPP chunk 内、按 forward 执行顺序（attn→mlp，逐层递增）对 `h_res` 的累乘，每次 forward
在本 stage 首个 hc 模块（最低层 attn）处重置。

**局限**：在流水并行（PP>1）下，复合映射只跨越本 stage 局部的层，并非整网的全局累乘；不同 stage/chunk 的逐层键因
层号不同不会冲突，但自动派生的 `global_*` 复合均值会混合深浅复合值——因此 composite 的**逐层视图**更有意义。PP=1 时精确。
每 chunk 独立 slot，避免后一 chunk 的层污染前一 chunk 的累乘。

---

## 与 microbatch / 激活重算的交互

- 每个梯度累积 microbatch 都会按序重新调用所有 hc 模块；chunk root 先触发并重置复合映射，故复合映射按 microbatch 正确、
  不跨 microbatch 泄漏。
- 整个捕获受 `_should_monitor()` 门控；监控步内所有 wrapper 都触发（root 重置 → 同一 forward 内有序 bmm 累积），故复合
  状态自洽；非监控步不触发，下一监控步的 root 重置会重新 seed。
- `_should_monitor()` 要求 grad enabled，故若外层 layer 被整体激活重算，仅 grad-enabled 的那次正向记录；`compute_mappings`
  本身不被 checkpoint，不会重复触发。
