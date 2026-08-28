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

## 采集方式：wrap `compute_mappings` 与 `fused_h_res_h_post_bda`（非重算）

`h_pre` 不在 `HyperConnectionModule.forward` 的返回中，普通 forward hook 看不到它。因此本 monitor **包裹
（wrap）每个 mHC 模块的 `compute_mappings` 绑定方法**，直接捕获其真实返回的 `(h_pre, h_post, h_res)` —— 不重算。

- `compute_mappings` 是普通 Python 方法（仅 `@nvtx_decorator`，非 `@torch.compile`），在 `_forward_normal` 中以
  `self.compute_mappings(...)` 调用，且**不被 checkpoint**，因此在 grad-enabled 的正向中恰好执行一次；实例属性
  wrapper 干净地遮蔽类方法。
- 整个捕获逻辑受 `_should_monitor()` 门控（含 grad 门），非监控步只是 `orig(x)` + 一次布尔判断。
- wrapper 自己的入参 `x`（`[s, b, n*C]`，聚合前的多流隐状态）也在作用域内，多流几何指标即由它算出，无需额外 hook。
- 能量分解还需要 sublayer 输出 `o` 与更新后的残差，只有 `fused_h_res_h_post_bda` 同时持有它们，故该绑定方法也被包裹。
  它是外层 dispatcher（内部再分派 native / checkpoint 两条路径），在 grad-enabled 正向中每模块恰好调用一次；其入参
  `original_residual` 与 `compute_mappings` 收到的 `x` 逐位相同（`transformer_layer` 在聚合前捕获 `residual = hidden_states`）。
- `remove_hooks()` 恢复原始方法并清空所有状态。

### VRAM 安全（无泄漏）

跨调用状态只有 `self._h_res`（本 microbatch 在飞的 detached `h_res` 引用）与固定的 0 维累加器。规则：

- 捕获后立即对 `h_pre/h_post/h_res` `.detach()`，并在 `torch.no_grad()` 下做全部指标/复合计算——否则一个仍带梯度的
  张量会通过反向把整层 autograd graph 钉住（大泄漏）。
- wrapper 原样返回 `out`，除 0 维标量与 detached `h_res` 外不保留任何对它/其视图的引用。
- `self._h_res` 存的是已 detach 的 `h_res` 引用，那块 storage 在反向前本来就被 autograd graph 持有，故驻留成本≈0；
  真正新增的只有累乘中间量，每条链一个 `[s*b, n, n]`（l12 / mbs4 / s4096 / n=4 下约 1 MiB）。
- 一个 chunk 的模块集齐即在 forward 内立刻 drain（见下），**不跨 microbatch**；梯度累积下不会攒成
  `ga_steps × 模块数` 份。`step()` 兜底 drain 残留的不完整 stash，`remove_hooks()` 一并清空。

热路径纪律见 `.claude/skills/monitor-hook-perf-rules`：hook 内无 D2H 同步、无集合通信，schema 在 `allocate_buffers`
前声明。TP 不沿 `n` 切分映射，故无需 hook 内通信；跨 rank 归约在 flush 时由 `gather_and_aggregate` 完成
（mean，三个 `*_orth_dev*_max_med_ratio`、`*_stream_norm_max_min_ratio`、`*_stream_gram_offdiag_max` 走 max）。
序列并行下 `x` 按 token 切分但隐藏维完整，逐 token 的多流几何在本 rank 就是完备的，同样不需要 hook 内通信。

---

## 监控指标

每个 hc 模块产出 `35 + n` 个指标（`n` 条逐流 norm）；`n = 4` 时另加两条 `SO(4)` 转角序列
（`h_res_theta_lo` / `h_res_theta_hi`），共 `37 + n`。指标名以 `attn_` / `mlp_` 前缀区分。除三个
`*_orth_dev*_max_med_ratio`、`*_stream_norm_max_min_ratio`、`*_stream_gram_offdiag_max`、
`*_mix_write_cos_abs_max` 按 **max** 合成外，
其余全部按 token/batch 求均值（并在 flush 时对 microbatch/rank 求均值）。日志键形如
`mhc_health/layer_{i}/{c}_{name}`，`{c}` ∈ `{attn, mlp}`；
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
| `{c}_composite_amax_gain_fwd` | **前缀积** `M_l = H_l ⋯ H_0` | 从 stage 入口累积到本层的前向放大 |
| `{c}_composite_amax_gain_bwd` | **后缀积** `S_l = H_N ⋯ H_l` | 梯度从 stage 出口传到本层的反向放大 |

单层 `h_res` 经 Sinkhorn 投影为双随机矩阵（行/列和 ≈ 1），故单层 amax-gain ≈ 1.0；复合映射的增益随深度偏离 1.0，
正是残差流放大/收缩的信号。

> **Sinkhorn 下这两条恒为 1。** 双随机矩阵之积仍是双随机矩阵，实测 12 层累乘后仍是 1.0000001。所以在 stock mHC
> 上它们只是 Sinkhorn 收敛哨兵；真正有信息量的是 signed chart（`quat_pair` / `cayley` / `orth` / `erase` 等
> variant），那里 `h_res` 不再非负，累积增益才会张开。

### 流形与结构（h_res 的形状诊断）

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `{c}_h_res_orth_dev` | `mean_t( \|\| h_resᵀ h_res − I \|\|_F )` | 偏离正交（等距）的程度：0 = 精确保范 |
| `{c}_composite_h_res_orth_dev_fwd` | 同上，作用于**前缀积** `M_l` | 从 stage 入口累积到本层的非等距程度 |
| `{c}_composite_h_res_orth_dev_bwd` | 同上，作用于**后缀积** `S_l` | 本层以上累积的非等距程度 |
| `{c}_h_res_orth_dev_max_med_ratio` | `(max_t d + ε) / (med_t d + ε)`，`d` 同上，`ε = 1e-6` | 逐 token 尾部集中度：1.0 = 无尾 |
| `{c}_composite_h_res_orth_dev_fwd_max_med_ratio` | 同上，作用于前缀积 | 前向累积后的尾部集中度 |
| `{c}_composite_h_res_orth_dev_bwd_max_med_ratio` | 同上，作用于后缀积 | 反向累积后的尾部集中度 |
| `{c}_h_res_beta_mean` | `mean_t( n − tr(h_res) )` | rank-1 擦除强度 β |
| `{c}_h_res_beta_std` | `std_t( n − tr(h_res) )` | β 的 token 级离散度：→0 说明退化成逐层常数 |
| `{c}_h_res_outer_dev` | `mean_t( \|\| h_res − h_post ⊗ h_pre \|\|_F )` | 残差混合中「读写外积」解释不掉的部分 |
| `{c}_h_res_sigma_min` | `mean_t( min_k σ_k )`，`σ = svdvals(h_res \|_{1^⊥})` | `1^⊥` 上最弱方向的增益：1 = 等距，→0 = `h_res → J` |
| `{c}_h_res_sigma_mean` | `mean_t( mean_k σ_k )` | 同上取均值；必须与 `sigma_min` 对读 |
| `{c}_composite_h_res_sigma_min_fwd` | 同上，作用于**前缀积** `M_l` | 入口到本层累积后 `1^⊥` 的最弱方向：**正交 vs 双随机的主曲线** |
| `{c}_composite_h_res_sigma_mean_fwd` | 同上取均值 | 与 `_min_fwd` 对读 |
| `{c}_composite_h_res_sigma_min_bwd` | 同上，作用于**后缀积** `S_l` | 梯度路径上是否保持流的可分性 |
| `{c}_composite_h_res_sigma_mean_bwd` | 同上取均值 | 与 `_min_bwd` 对读 |
| `{c}_h_res_theta_lo` | `min(α, γ)`，`spec(h_res) = {e^{±iα}, e^{±iγ}}`（闭式，无特征分解；**仅 `n = 4`**） | `SO(4)` 的共轭类是这**一对**转角 |
| `{c}_h_res_theta_hi` | `max(α, γ)`，同上 | `h_res` 实际用掉多少 `SO(4)`；两角接近时差值有 ~3e-2 rad 噪声底 |

**max/median 比值：均值抓不到的尾巴。** `orth_dev` 是 token 均值，所以「少数 token 完全不正交」和「全体精确正交」
在日志上可以长得一模一样：R3 的 15 步 Schulz 迭代掉出收敛盆的 token 占比约 1e-3，均值全程显示 `0.00000`，
而那批 token 主导了反向传播。比值把这件事变成可读的：分母是 median（对尾部免疫的「典型 token」水平），
分子是 max，`ε = 1e-6` 是视作「与精确正交不可区分」的 fp32 舍入底 —— 精确正交时读数是 **1.0**，不是 0/0。
跨 microbatch / rank 按 **max** 合成（取均值会抹平尾部），所以全局值是各 rank 比值的最大者，不是全局 max/median。
Cayley 参数化下构造上恒为 1.0；迭代式正交化（Schulz）与 Sinkhorn/擦除构型下才有信息量。

**β = `n − tr(h_res)`。** 对 `H_res = I − β·ûûᵀ`（`‖û‖ = 1`）这个迹亏损**精确等于** β，所以 rank-1 擦除消融的核心判据
（β 收敛到 0 / (0,2) / 2）可以直接读这条曲线，不必再从 `orth_dev = 2β − β²` 反推 —— 后者在 β=0 与 β=2 处都是 0，
单看它分不出「擦除无效」和「Householder 反射」。其他构型下它就是普通的迹亏损：恒等 = 0，双随机 = `n` 减对角质量。

**外积偏差的下标方向。** `apply_h_res` 算的是 `out_i = Σ_j h_res[i,j]·x_j`，而子层通路写回的是
`h_post_i · Σ_j h_pre_j·x_j`，所以与 `h_res` 下标同序的 rank-1 矩阵是 `h_post ⊗ h_pre`（元素 `[i,j] = h_post_i·h_pre_j`），
本指标即残差混合与这个读写外积的 Frobenius 距离。注意它未减掉恒等分量，所以近似恒等的 `h_res` 会有一个 `√n` 量级的底；
对称的 `h_res`（如 rank-1 擦除）两个下标方向给出同一个值，方向只在 dense/正交构型下才有区别。

**Σ：保均值混合在 `1^⊥` 上的谱。** 任何保均值的混合（`H·1 = 1`、`1ᵗH = 1ᵗ` —— R8 的谱球面**和** Sinkhorn 基线都满足）
都保持 `1^⊥` 不变，于是 `QᵗHᵗHQ = Σ²`（`Q` 为 `1^⊥` 的任一正交基），等价地 `svdvals(H) = {1} ∪ {|σ_i|}`：
`1` 方向被固定、不含信息，`Σ` 才是这个算子全部可学的部分。**这是 R8a 唯一的判据** —— `σ → 1` 是模型自己在要求
一个保均值的**等距**（R8b 直接硬编码了它，因此 R8b 必须读到平坦的 1.0）；`σ → 0` 则是 `h_res → J`，
每条流被流均值取代，即比普通残差**更弱**的 rank-1 塌缩。同时给 `min` 与 `mean` 是 `amax_gain` 的教训：
`n−1` 个方向里塌掉 1 个，均值仍然读得很健康。非仿射的 `h_res`（R1 恒等、R3-Cayley、R4 擦除）下 `1^⊥` 并不不变，
此时读到的是 `HᵗH` 压缩到 `1^⊥` 上的谱 —— 仍被 `‖h_res‖₂` 界住，但不再是算子的分解。
`n = 4`（即 `m = 3`）走闭式三次特征值解（Cardano），因为 `eigvalsh`/`svdvals` 会同步 host，hook 内不允许。

**复合 σ：正交优于双随机的主曲线。** 单层 σ 只说明这一层压了多少，真正的问题是**跨深度累积**之后还剩多少。
双随机 `H_res` 固定流**均值**方向（σ_max = 1）但可以压扁流**差**子空间 `1^⊥`，累乘之后把 `n` 条残差流压到一条方向上
—— 信息损失。保均值的正交 `H_res`（R3-Cayley / R9 quat_pair）在 `1^⊥` 上是等距，任何深度都保持 σ_min = σ_max = 1。
实测（`n = 4`，逐层随机）：

| depth | DS（Sinkhorn）σ_min | 正交（保均值 Cayley）σ_min |
|---|---|---|
| 1 | 0.144 | 1.000000 |
| 3 | 8.35e-4 | 1.000000 |
| 6 | 0.000000 | 1.000000 |
| 12 | 0.000000 | 1.000000 |

`fwd`（前缀积）与 `bwd`（后缀积）都报：前者答「入口到本层累积后流还剩多少可分性」，与 `stream_cv` 配对；
后者答「梯度路径上是否保持可分性」。两条链共用同一个累乘循环，故这四条序列的额外开销为零。

**两个转角：`SO(4)` 的共轭类不是一个角。** `SO(4)` 的元素不是「一个转动」——它分裂成两个不变 2-平面上各自
按 `α`、`γ` 的独立转动，谱为 `{e^{±iα}, e^{±iγ}}`，**这一对**才是共轭类。而 `β = n − tr(h_res) = 4 − 2(cos α + cos γ)`
只读到两个余弦的**和**：`(α, γ) = (1.0, 2.0)` 与 `(0.2, 2.5981)` 的 `β` 同为 3.7517，前者两个平面都在中等角度上转，
后者近乎「一个平面几乎不动、另一个几乎翻转」，是完全不同的算子。`theta_lo/hi` 把这两种情形分开（相差 > 0.5 rad），
也直接量出等倾（`α = γ`）的偏离程度 —— R3-Cayley 的 β ∈ [0.253, 0.291] 一直被当成等倾 `θ ≈ 0.36` 读，这条曲线
是该假设第一次可验证。两个不变量都来自逐元素代数（`tr Q` 与 `tr Q² = (Q ⊙ Qᵗ).sum`），无 matmul、无特征分解。
**只有 `h_res` 正交时**（R3-Cayley / R9 / R8b）这两个数才是共轭类角；其他混合上 clamp 保证读数有限，但不描述
任何 `SO(4)` 元素。分开两角要过 `sqrt((cos α − cos γ)²)`，两角接近时有效位折半，故**差值**有 ~3e-2 rad 的 fp32
噪声底且系统性偏大（等倾 `θ ≈ 0.36` 处降到 ~9e-4，`(1.0, 2.0)` 处 ~2e-7）；均值恒稳（它是 `tr Q` 的重参数化）。
用 fp64 累加没用 —— 精度丢在 `h_res` 自己的 fp32 舍入里。

### 多流几何（stream 本身，而非映射）

上面所有指标测的都是**映射**（`h_pre`/`h_post`/`h_res`），且都沿 stream 轴做了归约。这一组测的是映射所作用的
**那 n 条流本身**，输入是 wrapper 入参 `x` reshape 成 `[T, n, C]` 后的 Gram 矩阵（`T = s·b`）：

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `{c}_stream_norm_{i}` | `mean_t( ‖x_i‖₂ )`，`i = 0..n−1` | 第 `i` 条流的量级，**逐流**给出 |
| `{c}_stream_norm_max_min_ratio` | `mean_t( (max_i ‖x_i‖ + ε) / (min_i ‖x_i‖ + ε) )` | 流间量级失衡：1.0 = 均衡，↑ = 出现主导流 |
| `{c}_stream_gram_offdiag_mean` | `mean_t mean_{i≠j} cos(x_i, x_j)`，**带符号** | 流间方向关系 ∈ [−1,1]：0 = 相互正交，+1 = 塌缩到共同均值，负 = 反向对齐 |
| `{c}_stream_gram_offdiag_max` | `mean_t max_{i≠j} \|cos(x_i, x_j)\|` | 同上的逐 token 尾部（少数 token 塌缩，均值看不见）；保持 `\|cos\|` |
| `{c}_stream_cv` | `mean_t( sqrt(Var) / ‖m‖ )`，`m = mean_i x_i`，`Var = (1/n)Σ_i‖x_i − m‖²` | 跨流变异系数：→0 = 流塌缩到共同均值，大 = 各流携带独立内容 |
| `{c}_stream_eff_rank` | `mean_t( (tr G)² / ‖G‖_F² )`，`G = X Xᵗ` 为逐 token 流 Gram | 参与比（Rényi-2）有效秩 ∈ [1, n]：→1 = 只剩一个方向承载能量（秩 1 塌缩），→n = 各流正交且能量均衡 |

**为什么必须逐流。** 其余每一条指标都沿 stream 轴归约（`mean(h_pre)`、逐 token 行和 …），所以「1 条流承载全部信号、
另外 `n−1` 条衰减成噪声」与「n 条流均衡工作」在它们身上读数相同 —— `stream_norm_mean` 这种把 stream 轴也平均掉的
写法同样看不见主导流。多流残差退化成单流有两条路：量级失衡（由 `max_min_ratio` 抓）和方向塌缩（由 Gram 非对角
余弦与 `stream_cv` 抓），这组指标就是这两条路的读数。

**非对角均值为什么带符号。** `|cos|` 把「塌缩到共同均值」（带符号 → +1）和「反向对齐的旋转」（带符号 → −1）
读成同一个 1.0，而这是两种相反的状态：实测 `+v/−v/+v/−v` 的 `|cos|` 均值是 1.000、带符号是 −1/3。所以**均值**用带
符号，`_max` 仍用 `|cos|`（它是尾部/塌缩探测器，带符号的 max 会漏掉反向对齐的尾巴）。保均值的正交 `H_res` 不改变
这个读数；一般（会旋转均值方向的）正交映射可以把它推到负值。

**`stream_cv` / `stream_eff_rank` 与复合 σ 是一对。** `composite_h_res_sigma_min_*` 测的是**算子**（机制：DS 把
`1^⊥` 压掉），`stream_cv` 与 `stream_eff_rank` 测的是**表示**（后果：流真的塌缩了）。「正交优于双随机」的证据是
这两侧同步 —— 算子 σ_min → 0 伴随 CV → 0 / 有效秩 → 1 / 带符号 cos → +1，而正交栈下四者都随深度保持不变。

`stream_cv` 与 `stream_eff_rank` 抓的塌缩不完全重合：CV 是**一阶**量（流偏离共同均值多远），有效秩是**二阶谱**量
（几个方向真正承载能量）。「4 条流两两正交但其中 2 条量级近零」这种状态 CV 不小，有效秩却已经掉到 2；反过来
`n` 条流等距张开时两者都饱和。有效秩额外的用处是它有**绝对刻度**：读数直接就是「实际用了几条流」，不需要和
基线比。

**实现上的三点。** Gram 由原始（bf16）流直接 `bmm` 得到、只把 `[T, n, n]` 的输出升到 fp32：先把 `x` 升 fp32 会在
forward 里产生一个 `[T, n, C]` 的临时拷贝（s=8192、n=4、C=1024 时约 134 MB）。norm 取 Gram 对角线的平方根，
不另算一遍；`ε = 1e-6` 是比值/余弦分母的下界，使某条流恰好为 0 时读数仍是有限值而不是 nan。
`stream_cv` 完全从同一个 `gram` 导出（平行轴定理：`Σ_i‖x_i‖² = tr(gram)`、`‖m‖² = Σ_ij gram_ij / n²`，故
`Var = tr/n − ‖m‖²`），不新增张量也不多一次 bmm；`‖m‖²` 的下限是**相对**的（`1e-6 · tr/n`）而非绝对 eps ——
各流相消的 token 上 CV 本身无意义，绝对 eps 会让它贡献巨大值而毁掉均值，与 `residual_energy_split` 的
`rel_floor` 同一 pattern。`stream_eff_rank` 同样只用 `gram` 的两个迹（`tr G` 取对角线之和、`‖G‖_F²` 取全元素
平方和），**不做 `eigvalsh`** —— 参与比形式恰好绕开特征分解，因此没有 host sync（`n = 4` 的逐 token
`eigvalsh` 会在 forward hook 里同步主机，违反热路径规则）。

### 能量分解（更新的交叉项）

上面所有指标测的是映射与流的几何，没有一条能回答「这次更新给残差加了多少能量」。把单次更新写成
`out = Q x + w`（`Q = h_res`，`w = h_post oᵗ`，即 `w_i = h_post_i · o`），在 `n×C` 上取 Frobenius 内积，
逐指标展开后得到**精确**恒等式：

```
‖out‖² = ‖Qx‖² + W + X
    R = ‖x‖²,              W = ‖h_post‖²‖o‖²,
    X = 2⟨Qx, w⟩ = 2 (Qᵗh_post)ᵗ d,     d_j = ⟨x_j, o⟩ ∈ R^n
```

| 指标 | 公式 | 诊断意义 |
|------|------|----------|
| `{c}_write_over_resid` | `mean_t( W/R )` | 单次写入相对残差的能量占比 |
| `{c}_cross_over_resid` | `mean_t( X/R )` | **有符号**交叉项；正交 `h_res` + 球面 `h_post` 都约束不到它 |
| `{c}_cross_over_write` | `mean_t( X/W′ )`，`W′ = max(W, 1e-6·R)` | 交叉项相对写入能量；`< −1` ⟹ 本模块净减少残差能量 |
| `{c}_mix_write_cos` | `mean_t( X / (2√(R·W′)) )` | `cos θ ∈ [−1,1]`，尺度无关、可跨层比 |
| `{c}_mix_write_cos_abs_max` | `max_t \|cos θ\|` | 对齐写入的逐 token 尾部（**max** 合成） |
| `{c}_resid_write_cos` | `mean_t( ⟨x, w⟩ / (‖x‖‖w‖) ) ` | 混合**前**的读写对齐；反 Hermite 写入会把它结构性归零 |
| `{c}_resid_gain` | `mean_t( ‖out‖²/R )` | 单模块能量增益 |

**为什么需要它。** `h_res` 正交（R3-Cayley）保证混合等距，`h_post` 球面化（R5）保证写入能量恒定，两者合起来的
隐含前提是「残差能量只能按写入能量单调增长」。但 R5 实跑在 `attn3 → mlp3` 上残差范数 295.7 → 258.2
（`ΔR ≈ −2.08e4 < 0`）—— 在 `Q` 正交、`W > 0` 下这只能来自 `X`。`X` 是残差范数能够**变小**的唯一途径，
而在这组序列之前它完全不可观测。形式化与三个修法见训练仓 `conf/mai_ladder/mhc/R7_NORM_CONTROL.md`。

**`resid_gain` 是记账自检。** 未加写入修正时它必须等于 `1 + W/R + X/R`；若开启 cross-free 写入（R7c）则必须变成
`1 + (W/R)·sin²θ`。两个 regime 都对得上，才说明监控与 patch 都没写错。它同时给出增长是加性还是乘性。

**实现上的三点。** `d` 是唯一需要 `C` 长度的计算（`n` 个内积，一次 `[T,n,C]×[T,C,1]` bmm），其余全是已经是
fp32 的 `[T,n]` 小张量代数，`n*C × n*C` 算子从不构造；`R` 用 `‖x‖²` 代替 `‖Qx‖²`（`h_res` 正交时相等，
Sinkhorn 基线下偏差等于其非正交度，精确算需要 `[T,n,n]` 流 Gram，4 倍 flops）；`o` 取 dropout **前**的值，
`hidden_dropout > 0` 时 `resid_gain` 自检会按被丢弃的质量偏移（当前 mHC recipe 全为 0）。
`X/W` 与 `cos` 的分母用**相对**下限 `1e-6·R` 而非绝对 eps：写入能量可忽略的 token 上这两个比值本身无意义，
绝对 eps 会让它们贡献 ~1e17 而毁掉均值。

### 复合映射（composite mapping）

设本 pipeline stage / VPP chunk 内的 hc 模块按 forward 执行顺序编号 `0..N`（逐层递增，层内 attn→mlp），
第 `k` 个做的是 `r ← H_k r + w_k`。两条链分别是：

```
前缀   M_l = H_l ⋯ H_0      = d(第 l 层输出) / d(stage 入口)
后缀   S_l = H_N ⋯ H_l      = d(stage 出口)   / d(第 l 层输入)
```

`fwd` 系列读前缀，`bwd` 系列读后缀。**为什么 bwd 必须用后缀**：到达第 `l` 层输入的梯度是 `S_lᵀ g`，
所以「梯度从 loss 传到第 l 层被放大了多少」只有后缀积能回答，前缀积看的是反方向（一个扰动从入口传到第 l 层）。
`amax_gain(S, dim=-2)`（最大绝对**列**和）就是 `Sᵀ` 的最大绝对行和，即 `‖Sᵀ‖_∞` 那个反向界。

**延迟计算。** `S_l` 需要第 `l` 层**以上**的所有 `h_res`，而那些层在第 `l` 层 hook 触发时还没跑。所以 wrapper 只把
detached `h_res` 按 `(layer, component)` 暂存，等一个 chunk 的模块集齐后在 forward 内 drain：升序建前缀、降序建后缀。
两条链都按**静态层序**遍历，不依赖 wrapper 触发顺序。

> **这同时修掉了一个 full-recompute 下的静默错误。** `_should_monitor()` 要求 grad enabled，而
> `recompute_granularity=full` 时 `CheckpointFunction.forward` 在 `no_grad` 下跑、只有反向重放 `enable_grad`。
> 也就是说那种配置下 wrapper **只在反向触发，且顺序是降序**。旧的增量累乘（`M ← h_res @ M`，在 chunk root 处重置）
> 因此会静默变成降序累乘、且 root 层被截断成单个因子。改成静态序遍历后，`full` 与 `selective` 给出相同结果。
> Sinkhorn chart 下这个 bug 完全看不出来——双随机之积恒为 1.0，两种顺序都是 1.0。

**局限**：在流水并行（PP>1）下，两条链都只跨越本 stage 局部的层，并非整网的全局累乘；不同 stage/chunk 的逐层键因
层号不同不会冲突，但自动派生的 `global_*` 复合均值会混合深浅复合值——因此 composite 的**逐层视图**更有意义。PP=1 时精确。
每 chunk 独立 stash，避免后一 chunk 的层污染前一 chunk 的累乘。

---

## 与 microbatch / 激活重算的交互

- 每个梯度累积 microbatch 都会按序重新调用所有 hc 模块；一个 chunk 的模块集齐即 drain，故复合映射按 microbatch
  正确、不跨 microbatch 泄漏（否则梯度累积下会攒成 `ga_steps × 模块数` 份 `h_res` 引用）。
- 整个捕获受 `_should_monitor()` 门控；非监控步不触发，stash 保持为空。
- `_should_monitor()` 要求 grad enabled。`recompute_granularity=full` 时唯一 grad-enabled 的 pass 是**反向重放**，
  wrapper 于是只在反向按降序触发——两条链走静态层序，故结果与正向触发一致（见上）。`compute_mappings` 本身不被
  checkpoint，不会重复触发。
- `fused_h_res_h_post_bda` 是外层 dispatcher，重算发生在它内部的 `CheckpointWithoutOutput` 里而非 wrapper 层，
  故 `recompute_modules` 含 `"mhc"` 时能量分解仍每模块每 microbatch 只记录一次。
