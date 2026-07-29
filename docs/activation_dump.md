# Activation Dump (`act_dump`)

按 monitor 间隔把**残差流 hidden states** `[s, b, h]` 采样落盘，供离线结构分析
（有效秩 / 各向异性 / 聚类 / massive-outlier 检测——见 `spec_entropy_explorer.py`）。

与其它 monitor 不同，`act_dump` **不上报任何 `training_logs` 标量指标**：它在 forward
hook 里把一份随机 token 采样从 GPU 拷到 pinned CPU，在 step 末尾（cold path）写成
safetensors 文件。

## 为什么不复用 Megatron 的 activation offload

Megatron 的 `fine_grained_activation_offloading`（以及 TE 的 CPU offload）只在
**GPU ↔ pinned CPU-RAM** 之间搬运 saved-for-backward 张量，与 autograd 耦合，同一个
step 的反向里 reload + free（pool 复用），**没有落盘路径**，也没有"周期性快照"的概念。
复用它的 manager 会与其 pool-reuse / once-per-step 契约冲突。因此本工具只借用它的
**传输技巧**（side CUDA stream 上的 non-blocking D2H 拷入 pinned buffer，用 CUDA event
定序），把阻塞的 sync + 落盘推迟到 flush（cold path）——这遵守
`monitor-hook-perf-rules`：hot path 上无 `.item()/.cpu()` sync，无 collectives。

## 随机 token 位置采样（不是前 K 个）

默认采样 `n_sample_tokens` 个**随机 token 位置**，而不是前 K 个。前导 token 受
BOS / attention-sink 的 massive activation 影响，量级与统计特性异常，前 K 切片**不能
代表**残差流。索引在 CPU 上按 `token_sample_seed + step_count` 生成（因此可复现，
且同一 step 内所有采样层用**相同**位置，跨层可比），再搬到 device；CPU 索引同时作为
文件元数据，无需 hot-path 的 D2H sync。

## 激进采样默认值（控制磁盘占用）

默认只落盘：被监控的 step × 采样层 × 第一个 microbatch × `n_sample_tokens` 个位置 ×
一个 DP/TP rank。所有开关可覆盖。

此外 `max_dump_steps`（默认 `20`）做**轮转（rotation）**：每次写盘后，只保留最近
`max_dump_steps` 个 `step_*` 目录，删除更旧的，使磁盘占用有上界，跑多久都不会撑爆。
设为 `null` 关闭轮转（无上限）。

粗略磁盘占用（每 step）≈ `n_sample_tokens × hidden_size × dtype_size × 采样层数`；
配合 `max_dump_steps` 后总占用 ≈ 上式 × `max_dump_steps`。

## 配置（必须用 dict 形式传 per-monitor kwargs）

```yaml
internal_medicine_monitor_interval: 50
internal_medicine_monitors:
    act_dump:
        dump_dir: "./outputs/act_dumps"
        which: "output"          # "output"=层输出残差 | "input"=层输入残差
        sample_layers: [0, 6, 11] # global 层索引；null=全部层
        n_sample_tokens: 512      # 随机 token 位置数；null=全部 token
```

### 全部 kwargs

| 参数 | 默认 | 说明 |
|---|---|---|
| `dump_dir` | `"./outputs/act_dumps"` | 落盘根目录 |
| `which` | `"output"` | `"output"` 取层输出残差，`"input"` 取层输入残差 |
| `sample_layers` | `None` | global 层索引列表；`None`=全部层 |
| `n_sample_tokens` | `512` | 随机 token 位置数；`None`=不采样（全部 token） |
| `token_sample_seed` | `0` | 基础种子；每 step 位置用 `seed + step_count` |
| `first_microbatch_only` | `True` | 每 step 每层只落盘第一个 microbatch |
| `dump_dp_ranks` | `[0]` | 哪些 DP rank 落盘 |
| `dump_tp_ranks` | `[0]` | 哪些 TP rank 落盘 |
| `dump_dtype` | `None` | 写盘前 cast（`float32`/`float16`/`bfloat16`）；`None`=保留源 dtype |
| `max_dumps_per_step` | `None` | 每 step 文件数安全上限 |
| `max_dump_steps` | `20` | 轮转：只保留最近 N 个 `step_*` 目录；`None`=不轮转 |

## 文件布局与元数据

```
{dump_dir}/step_{step:07d}/rank{grank}_pp{pp}_tp{tp}_dp{dp}_layer{L}_{which}.safetensors
```

每个文件含两个张量：
- `hidden`：`[min(n_sample_tokens, s*b), hidden_size]`（采样后的残差，可能已 cast）
- `token_index`：`[n_sample_tokens]` int64，采样的扁平 token 位置（`s*b` 行主序）

safetensors metadata（全部为字符串）：`step`、`layer_idx`、`which`、`global_rank`、
`pp_rank`、`tp_rank`、`dp_rank`、`seq`、`batch`、`hidden_size`、`n_tokens`、
`n_sample_tokens`、`token_sample_seed`、`src_dtype`、`sequence_parallel`、
`token_layout`（`seq_major_flattened_s_times_b`）。

## 并行语义

- **TP / Sequence Parallel**：每个 TP rank 只持有序列的一个分片；默认只 dump `tp_rank 0`
  的分片，`sequence_parallel=True` 时 metadata 记录该状态（跨 rank 拼全序列不在范围内）。
- **PP / VPP**：每个 PP stage dump 自己的层；三段式 setup（prepare 跨所有 chunk →
  allocate 一次 → 逐 chunk attach）处理 VPP 多 chunk。
- **step 编号 off-by-one**：文件名里的 `step` 是 hook 触发时 `step()` **自增前**的
  `step_count`，与标量指标的归属方式一致。

## 离线加载

```python
from safetensors import safe_open

with safe_open(path, framework="pt") as f:
    meta = f.metadata()
    hidden = f.get_tensor("hidden")          # [T, hidden_size]
    token_index = f.get_tensor("token_index")
```

或直接喂给探针工具：

```bash
python spec_entropy_explorer.py <file>.safetensors --key hidden
```
