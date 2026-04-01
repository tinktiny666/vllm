# vLLM 性能调优特性分析

本文档系统梳理 vLLM v1 当前支持的所有性能调优特性，包括编译优化、CUDA 图、注意力后端、
量化、推测解码、并行策略、调度优化、内存管理等，帮助用户根据场景做出最优配置。

---

## 1. 编译与 CUDA 图

### 1.1 优化级别（Optimization Level）

vLLM 提供 O0–O3 四个优化级别，作为所有编译特性的总开关：

| 级别 | 编译模式 | CUDA 图 | 融合 Pass | FlashInfer Autotune | 适用场景 |
| ---- | -------- | ------- | --------- | ------------------- | -------- |
| O0 | 无 | 无 | 全部关闭 | 关闭 | 调试 / 快速启动 |
| O1 | VLLM_COMPILE | PIECEWISE | 条件启用 norm_quant / act_quant | 开启 | 开发迭代 |
| O2 | VLLM_COMPILE | FULL_AND_PIECEWISE | 条件启用全部 | 开启 | **默认**，生产推荐 |
| O3 | 同 O2 | 同 O2 | 同 O2 | 开启 | 预留扩展 |

O1 比 O0 快约 10–15%，O2 比 O1 再快 5–10%（具体取决于模型和硬件）。

### 1.2 torch.compile（VLLM_COMPILE 模式）

vLLM 的自定义编译后端基于 Dynamo + Inductor，核心流程：

1. **Dynamo 追踪**模型 `forward()`，生成 FX 图。
2. **图拆分**：在注意力算子处将图拆成多个子图（piecewise）。
3. **Inductor 编译**每个子图，运行自定义 fusion pass。
4. **缓存**编译产物到磁盘（`~/.cache/vllm/torch_compile_cache/`）。

关键配置字段（`CompilationConfig`）：

| 字段 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `mode` | `None`(O2→3) | 编译模式：0=无，1=标准 torch.compile，3=VLLM_COMPILE |
| `backend` | `"inductor"` | Inductor 编译后端 |
| `compile_sizes` | `None` | 指定编译的 batch size（启用 Inductor autotune） |
| `compile_ranges_endpoints` | `None` | 编译范围端点，如 `[8, 64]` 生成 `[1,8]`, `[9,64]`, `[65,max]` |
| `custom_ops` | `[]` | 自定义 CUDA kernel 控制：`"all"`, `"none"`, `"+op"`, `"-op"` |
| `splitting_ops` | 注意力算子 | 图拆分点，空列表 = 不拆分（全图 cudagraph） |
| `dynamic_shapes_config.type` | `BACKED` | 动态形状策略：BACKED / UNBACKED / BACKED_SIZE_OBLIVIOUS |
| `use_inductor_graph_partition` | `False` | Inductor 级别拆分（需 torch>=2.9，支持全图 fusion） |

### 1.3 融合 Pass

vLLM 在 Inductor 编译阶段运行多个自定义 fusion pass，消除中间张量读写：

| 融合 Pass | 配置字段 | 说明 | 前置条件 |
| --------- | -------- | ---- | -------- |
| RMSNorm + Quant | `fuse_norm_quant` | RMSNorm + FP8/FP4 量化 | 自定义 kernel 激活 |
| SiLU+Mul + Quant | `fuse_act_quant` | 激活函数 + 量化 | 自定义 kernel 激活 |
| Attention + Quant | `fuse_attn_quant` | 注意力输出 + 量化 | 全图模式 |
| AllReduce + RMSNorm | `fuse_allreduce_rms` | TP all-reduce 与 RMSNorm 融合 | Hopper/Blackwell, TP>1, FlashInfer |
| RoPE + KV-Cache | `fuse_rope_kvcache` | 旋转位置编码 + KV 缓存写入 | ROCm/AITER |
| QK Norm + RoPE | `enable_qk_norm_rope_fusion` | Q/K RMSNorm + RoPE 单 kernel | CUDA sm80+ |
| 序列并行 | `enable_sp` | ReduceScatter + local RMSNorm + AllGather | TP>1, 高 batch |
| AsyncTP GEMM + Collective | `fuse_gemm_comms` | GEMM 与 reduce-scatter/all-gather 融合 | 需 enable_sp |
| RMSNorm + Padding | `fuse_act_padding` | 残差 + RMSNorm + padding | ROCm/AITER |

**性能影响**：
- `fuse_allreduce_rms`：低 batch 场景提升显著（减少一次 all-reduce kernel launch）。
- `enable_sp` + `fuse_gemm_comms`：高 batch 场景可提升 7–10%。
- `fuse_norm_quant` / `fuse_act_quant`：FP8 模型消除全精度往返，提升 3–5%。

### 1.4 CUDA 图

CUDA 图通过预捕获 GPU kernel 序列并重放来消除 Python 开销。

#### CUDA 图模式

| 模式 | 说明 |
| ---- | ---- |
| `NONE` | 无 CUDA 图，纯 eager |
| `PIECEWISE` | 子图级别捕获，注意力 eager 执行 |
| `FULL` | 全图捕获（prefill + decode） |
| `FULL_DECODE_ONLY` | 仅 decode 阶段全图，prefill/mixed eager |
| `FULL_AND_PIECEWISE` | **O2 默认**。decode 用 FULL，mixed/prefill 用 PIECEWISE |

#### 捕获大小

默认自动生成模式：`[1, 2, 4] + range(8, 256, 8) + range(256, max+1, 16)`。
`max_cudagraph_capture_size` 默认为 `min(max_num_seqs * 2, 512)`。

可通过 `cudagraph_capture_sizes` 显式指定，减少捕获启动时间和显存占用。

#### 注意力后端兼容性

| 后端 | CUDA 图支持级别 |
| ---- | --------------- |
| FlashAttention v3 | ALWAYS |
| Triton Attention | ALWAYS |
| FlashAttention v2 | UNIFORM_BATCH |
| FlashInfer | UNIFORM_SINGLE_TOKEN_DECODE |
| FlashMLA | UNIFORM_BATCH |
| Mamba | UNIFORM_SINGLE_TOKEN_DECODE |

#### Performance Mode

| 模式 | CUDA 图捕获策略 | 适用场景 |
| ---- | --------------- | -------- |
| `balanced` | 步长 8/16 捕获 | **默认** |
| `interactivity` | 1–32 逐个捕获 | 低延迟优先 |
| `throughput` | 更大粒度捕获 | 高吞吐优先 |

---

## 2. 注意力后端

### 2.1 后端注册表

vLLM v1 注册了 25+ 个注意力后端，按模型类型分组：

**Dense 注意力**：

| 后端 | 说明 | 硬件要求 |
| ---- | ---- | -------- |
| `FLASH_ATTN` | FlashAttention 2/3/4 | SM 80+（Ampere+） |
| `FLASHINFER` | FlashInfer（TRTLLM kernel on Blackwell） | SM 75+ |
| `TRITON_ATTN` | Triton 全面退路 | 通用 |
| `FLEX_ATTENTION` | PyTorch flex_attention | 通用 |
| `CPU_ATTN` | CPU 退路 | CPU |

**MLA（Multi-Latent Attention）**：`FLASHMLA`, `FLASHINFER_MLA`, `FLASH_ATTN_MLA`, `TRITON_MLA`, `CUTLASS_MLA`

**SSM**：`MAMBA1`, `MAMBA2`, `SHORT_CONV`, `LINEAR`, `GDN_ATTN`

### 2.2 后端选择逻辑

vLLM 根据硬件和模型配置自动选择后端，优先级如下：

**Dense 模型**：
- **Blackwell (SM 100)**：FlashInfer > FlashAttention > Triton
- **Hopper/Ampere (SM < 100)**：FlashAttention > FlashInfer > Triton

**MLA 模型**：
- SM 100 + FP8 KV：FlashInfer MLA > FlashMLA
- SM 100 + BF16 KV：按 head 数选择 FlashInfer MLA 或 FlashMLA
- SM < 100：FlashAttention MLA > FlashMLA > FlashInfer MLA

可通过 `AttentionConfig.backend` 强制指定后端。

### 2.3 FlashInfer vs FlashAttention 对比

| 维度 | FlashAttention | FlashInfer |
| ---- | -------------- | ---------- |
| FP8 KV Cache | FA3+ 支持 | 全面支持（E4M3/E5M2） |
| Blackwell 支持 | 降级 | 默认首选（TRTLLM kernel） |
| Autotune | 不支持 | 支持（`enable_flashinfer_autotune`） |
| Block Size | 16 的倍数 | 16/32/64 |
| CUDA 图 | FA3 全支持，FA2 有限 | UNIFORM_SINGLE_TOKEN_DECODE |
| MLA | FlashAttention MLA | FlashInfer MLA |

### 2.4 FlashInfer Autotune

`KernelConfig.enable_flashinfer_autotune`：
- O0：关闭
- O1/O2/O3：开启
- 在 warmup 阶段运行 kernel autotune，选择最优 kernel 配置

---

## 3. 量化

### 3.1 支持的量化方法

vLLM 支持 22 种量化方法，覆盖重量级和轻量级场景：

**权重+激活量化**：

| 方法 | 精度 | 说明 |
| ---- | ---- | ---- |
| `fp8` | W8A8 FP8 | NVIDIA 原生 FP8，支持 static/dynamic，block-quant |
| `modelopt` | W8A8 FP8 | NVIDIA ModelOpt FP8 |
| `modelopt_fp4` | W4A8 FP4 | ModelOpt NVFP4（Blackwell） |
| `modelopt_mxfp8` | MXFP8 | Microscaling block-32 FP8 |
| `mxfp8` | MXFP8 | vLLM 原生 MXFP8 |
| `mxfp4` | MXFP4 | Microscaling block-32 FP4 |
| `compressed-tensors` | W4A4/W4A8/W8A8 | Neural Magic，支持多种组合 |

**权重量化**：

| 方法 | 精度 | 说明 |
| ---- | ---- | ---- |
| `awq` / `awq_marlin` | W4 | Activation-Aware Weight Quantization |
| `gptq` / `gptq_marlin` | W2/W3/W4/W8 | Post-training，Marlin kernel 加速 |
| `gguf` | 多种 | GGML 格式（Q4_0, Q5_K 等） |
| `bitsandbytes` | W4/W8 | NF4/FP4/INT8 |
| `torchao` | W4/W8 | PyTorch AO |
| `quark` | 多种 | AMD Quark |
| `moe_wna16` | WNA16 | MoE 专用权重量化 |

### 3.2 配置方式

```bash
# 自动检测（从 HF config 的 quantization_config 读取）
vllm serve meta-llama/Llama-3.1-8B-Instruct

# 手动指定
vllm serve model --quantization fp8
```

### 3.3 FP8 KV Cache

通过 `CacheConfig.cache_dtype` 启用，大幅减少 KV cache 显存占用：

```bash
vllm serve model --kv-cache-dtype fp8
```

- `fp8` / `fp8_e4m3`：高精度，NVIDIA 默认
- `fp8_e5m2`：更大动态范围
- `fp8_ds_mla`：DeepSeek MLA 专用
- 支持按层跳过量化（`kv_cache_dtype_skip_layers`）

**效果**：KV cache 显存减半，attention 后端支持 FP8 直接计算。

---

## 4. KV Cache 管理

### 4.1 前缀缓存（Prefix Caching）

默认开启（`enable_prefix_caching=True`），通过 Merkle 链 hash 实现：

```
block_hash = hash(parent_hash, token_ids, extra_keys)
```

- 相同前缀的请求共享 KV cache block（`ref_cnt` 引用计数）。
- 新请求到来时查找 hash 匹配，跳过冗余计算。
- 支持 multimodal hash、LoRA name、cache salt。

**效果**：多轮对话、系统 prompt 共享场景下显著减少 prefill 开销。

### 4.2 分页 KV Cache

- 固定大小物理 block（默认 16 tokens/block）。
- 逻辑 KV cache 是 block ID 列表（block table）。
- LRU 驱逐：`FreeKVCacheBlockQueue` 双链表，front = 最先驱逐。

### 4.3 KV Cache 卸载

| 配置 | 说明 |
| ---- | ---- |
| `kv_offloading_size` | 卸载到 CPU 的 buffer 大小（GiB） |
| `kv_offloading_backend` | `"native"`（vLLM 原生）或 `"lmcache"` |

### 4.4 模型权重卸载

vLLM 支持 sleep/wake 机制：

| 模式 | 说明 |
| ---- | ---- |
| `sleep(level=1)` | 权重卸载到 CPU，保留 KV cache |
| `sleep(level=2)` | 权重 + KV cache 全部丢弃 |
| `wake_up(tags=["weights"])` | 恢复权重 |
| `wake_up(tags=["kv_cache"])` | 恢复 KV cache |

**场景**：多模型轮询服务，空闲模型 sleep 释放显存。

### 4.5 CPU 卸载（UVA / Prefetch）

| 后端 | 配置 | 说明 |
| ---- | ---- | ---- |
| UVA | `cpu_offload_gb=N` | N GiB 权重通过 UVA 零拷贝卸载到 CPU |
| Prefetch | `offload_group_size`, `offload_prefetch_step` | 按层组卸载 + 预取，隐藏传输延迟 |

---

## 5. 调度优化

### 5.1 统一调度模型

vLLM v1 没有独立的 prefill/decode 阶段。每个请求只有 `num_computed_tokens` 和
`num_tokens_with_spec`，调度器每步分配 token 使两者对齐。

### 5.2 Chunked Prefill

默认开启（`enable_chunked_prefill=True`）：

- 长 prompt 按 token budget 分批处理。
- 每步处理 `max_num_batched_tokens`（默认 2048）个 token。
- 请求在多步中完成 prefill（`is_prefill_chunk=True`），之后进入 decode。

### 5.3 调度参数

| 参数 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `max_num_batched_tokens` | 2048 | 每步最大 token 数 |
| `max_num_seqs` | 128 | 最大并发序列数 |
| `max_num_scheduled_tokens` | = max_num_batched_tokens | 调度器实际调度上限 |
| `max_num_partial_prefills` | 1 | 最大并发部分 prefill 数 |
| `long_prefill_token_threshold` | 0 | "长"请求的阈值 |
| `policy` | `"fcfs"` | 调度策略：`"fcfs"` 或 `"priority"` |
| `stream_interval` | 1 | 流式输出间隔（token 数） |
| `scheduler_reserve_full_isl` | `True` | 入队前检查完整 ISL 是否能放入 KV cache |

### 5.4 异步调度

`SchedulerConfig.async_scheduling`：

- 消除 GPU 利用率间隙，提升延迟和吞吐。
- 仅兼容 EAGLE/MTP/Draft Model/NGram GPU 推测解码。
- 与 `enable_dbo` 不兼容。

### 5.5 调度策略

| 策略 | 说明 |
| ---- | ---- |
| `fcfs` | 先到先服务 |
| `priority` | 按 `(priority, arrival_time)` 最小堆排序 |

---

## 6. 推测解码

### 6.1 支持的方法

| 方法 | 原理 | 适用场景 |
| ---- | ---- | -------- |
| **ngram** | CPU 字符串匹配，Numba JIT | 代码生成、重复文本 |
| **ngram_gpu** | GPU 向量化 ngram 匹配 | 同上，避免 CPU-GPU 同步 |
| **EAGLE** | 轻量 transformer head，共享 hidden states | 通用文本，高质量 |
| **EAGLE-3** | EAGLE 扩展，使用中间层 aux hidden states | 更高质量 |
| **DFlash** | 交叉注意力 + 并行 drafting | 最大吞吐 |
| **Medusa** | 独立残差 block head | 已有 Medusa checkpoint |
| **draft_model** | 独立小模型 | 有合适小模型 |
| **MTP** | 模型内置 MTP 层（DeepSeek-V3 等） | 内置 MTP 的模型 |
| **suffix** | 后缀树缓存历史请求 | 有重复响应模式 |

### 6.2 拒绝采样

| 方法 | 说明 |
| ---- | ---- |
| `strict` | 严格匹配，简单但接受率低 |
| `probabilistic` | 基于概率比接受，接受率高但需额外显存 |
| `synthetic` | 几何衰减接受率，基准测试用 |

### 6.3 关键配置

| 配置 | 说明 |
| ---- | ---- |
| `num_speculative_tokens` | 每步推测 token 数（越多→潜在加速越高，但拒绝浪费也越多） |
| `prompt_lookup_max/min` | ngram 匹配窗口大小 |
| `parallel_drafting` | 并行生成所有推测 token |
| `draft_tensor_parallel_size` | draft 模型 TP 度 |
| `use_local_argmax_reduction` | 减少 TP all-gather 通信 |

### 6.4 选择建议

| 场景 | 推荐方法 |
| ---- | -------- |
| 代码/重复文本 | `ngram_gpu` |
| 通用文本 | `eagle` / `eagle3` |
| 超大模型（70B+） | `eagle` / `mtp` |
| DeepSeek-V3 等 | `mtp`（内置） |
| 最大吞吐 | `dflash` + `parallel_drafting` |

---

## 7. 并行策略

### 7.1 并行维度

| 维度 | 配置 | 说明 |
| ---- | ---- | ---- |
| 张量并行（TP） | `tensor_parallel_size` | 分片 attention head / MLP 维度 |
| 流水线并行（PP） | `pipeline_parallel_size` | 分片模型层 |
| 数据并行（DP） | `data_parallel_size` | 复制完整模型 |
| 专家并行（EP） | `enable_expert_parallel` | MoE 专家分片 |
| Prefill 上下文并行 | `prefill_context_parallel_size` | 分片 prefill 上下文 |
| Decode 上下文并行 | `decode_context_parallel_size` | 分片 decode 上下文（复用 TP GPU） |

### 7.2 专家并行（EP）与负载均衡（EPLB）

| 配置 | 说明 |
| ---- | ---- |
| `enable_expert_parallel` | EP 替代 TP 用于 MoE 层 |
| `expert_placement_strategy` | `"linear"`（连续）或 `"round_robin"`（交替） |
| `enable_eplb` | 动态负载均衡 |
| `eplb.window_size` | 负载记录窗口大小 |
| `eplb.step_interval` | 专家重排列间隔 |
| `eplb.num_redundant_experts` | 冗余专家数量 |
| `all2all_backend` | All2All 通信后端 |

**All2All 后端**：`allgather_reducescatter`（默认）, `deepep_high_throughput`, `deepep_low_latency`, `mori`, `nixl_ep`, `flashinfer_nvlink_two_sided`, `flashinfer_nvlink_one_sided`

### 7.3 双批重叠（DBO）

`enable_dbo=True`：微批处理重叠计算和通信。

| 配置 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `dbo_decode_token_threshold` | 32 | decode-only 批次阈值 |
| `dbo_prefill_token_threshold` | 512 | 含 prefill 批次阈值 |

仅支持 `deepep_low_latency` / `deepep_high_throughput` all2all 后端。

---

## 8. 显存管理

### 8.1 KV Cache 显存

| 配置 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `gpu_memory_utilization` | 0.9 | GPU 显存用于模型执行的比例 |
| `block_size` | 16 | KV cache block 大小（token 数） |
| `cache_dtype` | `"auto"` | KV cache 数据类型 |
| `kv_cache_memory_bytes` | `None` | 显式指定 KV cache 显存（优先于 gpu_memory_utilization） |

### 8.2 编译缓存

- 编译产物缓存到 `~/.cache/vllm/torch_compile_cache/{hash}/`。
- 跨进程复用，避免重复编译。
- `compile_cache_save_format`：`"binary"`（多进程安全）或 `"unpacked"`（调试用）。

### 8.3 CUDA 图显存

CUDA 图捕获需要额外显存，`max_cudagraph_capture_size` 越大、捕获大小列表越长，显存占用越高。

减少显存的策略：
- 减小 `max_cudagraph_capture_size`
- 使用 `cudagraph_capture_sizes` 精确指定
- 使用 `FULL_DECODE_ONLY` 替代 `FULL_AND_PIECEWISE`
- 使用 `PIECEWISE` 替代 `FULL`

---

## 9. 高级特性速查

### 9.1 Enforce Eager

`--enforce-eager`：完全禁用 `torch.compile` 和 CUDA 图，等价于 `-cc.mode=none -cc.cudagraph_mode=none`。

**用途**：调试、测试、快速启动。

### 9.2 快速 Detokenization

vLLM 使用 Rust 级别 `DecodeStream`（`FastIncrementalDetokenizer`）进行增量 detokenize，
比 Python 级别快数倍。需要 `tokenizers >= 0.22.0` + `PreTrainedTokenizerFast`。

### 9.3 MoE Cold Start

`fast_moe_cold_start`（默认开启，推测解码时关闭）：避免 MoE 模型冷启动期间的重编译。

### 9.4 自定义 All-Reduce

`disable_custom_all_reduce=False`（默认）：使用自定义 CUDA all-reduce kernel 替代 NCCL，
在 intra-node 通信中延迟更低。可通过 `--disable-custom-all-reduce` 关闭。

---

## 10. 常见场景配置建议

### 场景 1：低延迟在线服务

```bash
vllm serve model \
  -O2 \
  -cc.performance_mode=interactivity \
  -cc.cudagraph_capture_sizes=$(seq -s, 1 1 32)
```

### 场景 2：高吞吐离线推理

```bash
vllm serve model \
  -O2 \
  -cc.performance_mode=throughput \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 256
```

### 场景 3：多轮对话（共享系统 prompt）

```bash
vllm serve model \
  --enable-prefix-caching \
  -O2
```

### 场景 4：FP8 量化 + KV Cache 压缩

```bash
vllm serve model \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  -O2 \
  -cc.pass_config.fuse_norm_quant=true \
  -cc.pass_config.fuse_act_quant=true
```

### 场景 5：大模型 + 推测解码

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --speculative-model meta-llama/Llama-3.1-8B-Instruct \
  --num-speculative-tokens 5 \
  -tp 4
```

### 场景 6：快速启动（开发调试）

```bash
vllm serve model -O0
```
