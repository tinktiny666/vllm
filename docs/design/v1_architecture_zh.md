# vLLM v1 内部架构

本文档描述了 vLLM v1 的内部架构——请求如何在系统中流转、调度器和 KV 缓存如何工作，以及模型执行如何在 GPU 之间编排。关于进程级别的概览和类层次结构，请参阅[架构概览](arch_overview.md)。

## 系统层次

vLLM v1 组织为五个逻辑层：

```
┌──────────────────────────────────────────────┐
│  入口点                                    │
│  LLM 类、OpenAI API 服务器、CLI            │
├──────────────────────────────────────────────┤
│  引擎层                                    │
│  LLMEngine / AsyncLLM                       │
│  InputProcessor、OutputProcessor、Detokenizer│
├──────────────────────────────────────────────┤
│  引擎核心                                  │
│  Scheduler、KVCacheManager、Executor        │
├──────────────────────────────────────────────┤
│  Worker / 模型运行器                        │
│  GPUModelRunner、InputBatch、Attention      │
├──────────────────────────────────────────────┤
│  平台 / 内核                               │
│  CUDA、ROCm、TPU、CPU 后端                  │
│  FlashAttention、FlashInfer、Triton 操作    │
└──────────────────────────────────────────────┘
```

每一层通过明确定义的数据结构（`EngineCoreRequest`、`SchedulerOutput`、`ModelRunnerOutput` 等）与下一层通信。引擎和引擎核心层在通过 ZMQ 套接字连接的独立进程中运行，使 API 服务器能够在不阻塞模型执行的情况下处理网络 I/O。

## 进程架构

对于典型的 `vllm serve -tp=4` 部署，六个进程协同工作：

| 进程                 | 数量                    | 职责                                               |
| -------------------- | ----------------------- | -------------------------------------------------- |
| API 服务器           | `A`（默认 `DP`）        | HTTP 处理、分词、结果流式传输                      |
| 引擎核心             | `DP`（默认 1）          | 调度、KV 缓存管理、执行分发                        |
| GPU Worker           | `DP x PP x TP`         | 模型加载、前向传播、GPU 内存管理                    |
| DP 协调器            | 1（如果 `DP > 1`）      | MoE DP 的负载均衡和波次同步                        |

**ZMQ 通信拓扑**：每个 API 服务器通过 ZMQ ROUTER/PUSH 套接字连接到每个引擎核心。每个引擎核心通过多进程管道连接到其工作进程。多部分 ZMQ 消息携带序列化的 `EngineCoreRequest`/`EngineCoreOutput` 对象，大张量通过带外传输以实现零拷贝效率。

## 引擎层

### LLMEngine（离线模式）

`vllm/v1/engine/llm_engine.py` — `LLM` 类使用的同步外观。它将三个子系统连接在一起：

| 组件                | 角色                                               |
| ------------------- | -------------------------------------------------- |
| `InputProcessor`    | 对提示词进行分词、验证参数、生成 `EngineCoreRequest` |
| `EngineCoreClient`  | 将请求传输到/从引擎核心（ZMQ 或进程内）             |
| `OutputProcessor`   | 对输出进行去分词、聚合并行样本、生成 `RequestOutput` |

`step()` 方法是核心迭代：

1. `engine_core.get_output()` — 从引擎核心拉取 `EngineCoreOutputs`。
2. `output_processor.process_outputs()` — 去分词、检查停止字符串、计算对数概率、创建 `RequestOutput` 对象。
3. 终止在去分词过程中检测到停止字符串的请求。
4. 记录指标。

### AsyncLLM（服务器模式）

`vllm/v1/engine/async_llm.py` — API 服务器使用的异步引擎。它使用 `AsyncMPClient`（始终是多进程的），并运行一个 `output_handler` 异步任务，该任务持续从引擎核心拉取并将 `RequestOutput` 对象推送到每个请求的 `RequestOutputCollector` 队列中。`generate()` 异步生成器通过从这些队列中拉取来产生输出。

### 引擎核心客户端

`EngineCoreClient` 工厂在四种传输模式之间选择：

| 模式             | 客户端类            | 引擎核心位置                    | 用例                 |
| ---------------- | ------------------- | ------------------------------- | -------------------- |
| 进程内           | `InprocClient`      | 同一进程，无忙循环              | 测试/调试            |
| 同步多进程       | `SyncMPClient`      | 后台进程，ZMQ                   | `LLM` 类             |
| 异步多进程       | `AsyncMPClient`     | 后台进程，ZMQ                   | `AsyncLLM`（服务器） |
| DP 异步          | `DPLBAsyncMPClient` | 多进程，负载均衡                | 数据并行服务         |

多进程客户端管理 ZMQ ROUTER 套接字用于输入，PULL 套接字用于输出，序列化由 `vllm/v1/serial_utils.py` 中的 `MsgpackEncoder`/`MsgpackDecoder` 处理。

## 引擎核心

`vllm/v1/engine/core.py` 包含内部执行循环。

### EngineCore

基本迭代（`core.py` 中的 `step()`）：

```
scheduler.has_requests()
  → scheduler.schedule()           → SchedulerOutput
  → model_executor.execute_model() → Future[ModelRunnerOutput]
  → scheduler.get_grammar_bitmask()
  → future.result()                → ModelRunnerOutput
  → model_executor.sample_tokens()（如果需要）
  → scheduler.update_from_output() → EngineCoreOutputs
```

### EngineCoreProc

多进程包装器运行一个忙循环：

```
while not shutdown:
    _process_input_queue()      # 从 ZMQ 排空 ADD/ABORT/UTILITY
    _process_engine_step()      # step() + 将输出放入输出队列
```

输入和输出由专用线程（`process_input_sockets`、`process_output_sockets`）处理，这些线程在 ZMQ 套接字和线程安全队列之间架起桥梁，使主线程专注于调度和执行。

### DPEngineCoreProc

为数据并行 MoE 模型扩展了 `EngineCoreProc`。增加：
- 一个用于所有规约未完成请求状态的 DP 进程组。
- 基于波次的同步（步骤计数器、波次完成通知）。
- 发布请求计数以供 DP 协调器执行负载均衡。

## 调度器

`vllm/v1/core/sched/scheduler.py`

### 统一调度模型

v1 调度器**没有**单独的预填充和解码阶段。每个请求都有 `num_computed_tokens` 和 `num_tokens_with_spec`。每个调度步骤只是为请求分配令牌，以便 `num_computed_tokens` 赶上 `num_tokens_with_spec`。这自然地处理了分块预填充（多步处理大型提示）、连续批处理（流中接纳新请求）和解码（每步一个令牌）。

### 调度阶段

1. **调度 RUNNING 请求** — 遍历活动请求，计算令牌缺口，分配 KV 块。如果分配失败，则抢占优先级最低的请求。
2. **调度 WAITING 请求** — 如果没有发生抢占，则接纳等待队列中的新请求。检查阻塞条件（语法、远程 KV 传输、流式输入），计算前缀缓存命中，分配块。

输出是 `SchedulerOutput`，包含 `scheduled_new_reqs`、`scheduled_cached_reqs`、每个请求的令牌计数、完成/抢占的请求 ID 以及需要清零的块。

### 请求队列

- **FCFS**（`FCFSRequestQueue`）：普通双端队列，先进先出。
- **Priority**（`PriorityRequestQueue`）：按 `(priority, arrival_time)` 排序的最小堆。

两个独立的队列跟踪等待请求 — 普通 `waiting` 和 `skipped_waiting`（被语法/远程-KV/流式阻塞）。调度器根据活动策略交错它们。

### 请求生命周期

```
WAITING  →  RUNNING  →  FINISHED_*
              ↕
           PREEMPTED
```

可能的完成状态：`STOPPED`（EOS/停止字符串）、`LENGTH_CAPPED`、`ABORTED`、`ERROR`、`REPETITION`。

## KV 缓存管理

### 分层架构

```
Scheduler
  → KVCacheManager              （请求级逻辑）
    → KVCacheCoordinator        （跨 KV 缓存组协调）
      → SingleTypeKVCacheManager（FullAttention、SlidingWindow、Mamba 等）
        → BlockPool             （物理块分配、空闲列表、前缀哈希映射）
          → KVCacheBlock        （每块元数据：id、ref_cnt、hash）
```

### 分页 KV 缓存

每个请求的逻辑 KV 缓存是一个固定大小物理块的列表（"块表"）。块由 `BlockPool` 管理，它拥有所有 `KVCacheBlock` 对象并提供：

- `free_block_queue` — LRU 驱逐顺序的双向链表（前端 = 先驱逐）。
- `cached_block_hash_to_block` — 用于前缀缓存的哈希索引。

当请求完成或被抢占时，其块的 `ref_cnt` 递减。`ref_cnt == 0` 的块返回空闲队列。通过前缀缓存共享的块在有请求引用它们时保持存活。

### 前缀缓存

完整块被哈希为 `hash(parent_hash, token_ids, extra_keys)`，形成 Merkle 链。当新请求到达时，查找其块哈希以找到匹配的缓存块，跳过冗余计算。哈希包括多模态特征哈希、LoRA 名称和缓存盐以确保正确性。

### 混合 KV 缓存（全注意力 + 滑动窗口）

混合全注意力和滑动窗口层的模型使用 `HybridKVCacheCoordinator`，它通过定点算法跨组迭代以找到最长的公共缓存命中。每个组维护自己的 `SingleTypeKVCacheManager`，但共享相同的物理 `BlockPool`。

### 抢占

当 `allocate_slots()` 失败（没有空闲块）时，调度器：
1. 选择优先级最低的运行请求。
2. 释放其所有 KV 缓存块。
3. 将 `num_computed_tokens` 重置为 0。
4. 将其移回等待队列的前端。

被抢占的请求稍后重新调度，如果其前缀块仍驻留，则可能受益于前缀缓存。

## 模型执行

### Worker

每个 GPU 有一个专用的工作进程（`vllm/v1/worker/gpu_worker.py`）：

1. **初始化**：设置 CUDA 设备、初始化分布式环境、创建 `GPUModelRunner`。
2. **加载模型**：委托给模型加载器策略（safetensors、GGUF、bitsandbytes 等）。
3. **内存分析**：运行虚拟前向传播以测量峰值内存，然后计算可用的 KV 缓存内存。
4. **初始化 KV 缓存**：基于分析的内存分配 GPU 块。
5. **编译/预热**：在特定批次大小下对模型进行 `torch.compile`，捕获 CUDA 图。
6. **执行**：每个调度迭代调用 — 接收输入、运行前向传播、返回输出令牌/对数概率。

### GPUModelRunner

`execute_model()` 中的热路径：

```
_update_states()                 — 更新持久化 InputBatch
_execute_mm_encoder()            — 如果需要，运行多模态编码器
_prepare_inputs()                — 构建 logits_indices、spec_decode 元数据
构建注意力元数据                — 块表、seq_lens、slot_mapping
model(input_ids, positions)      — 通过解码器层前向传播
lm_head                          — logits 投影
采样 / 返回 ModelRunnerOutput
```

### CUDA 图

对于支持的批次大小，前向传播重放预捕获的 CUDA 图，而不是启动单独的内核，从而消除 Python 开销。每个注意力后端通过 `AttentionCGSupport`（ALWAYS、UNIFORM_BATCH、UNIFORM_SINGLE_TOKEN_DECODE、NEVER）声明其 CUDA 图支持级别。

### torch.compile

模型通过 `VLLMCompileBackend` 进行编译，实现算子融合：norm+quant、act+quant、allreduce+rms、rope+kvcache。编译结果通过影响计算图的所有配置的哈希进行缓存。

## 注意力后端

### 三部分协议

`vllm/v1/attention/` 中的每个注意力后端实现：

1. **`AttentionBackend`** — 类级别的能力：支持的 dtype、KV 缓存形状、块大小、CUDA 图支持。
2. **`AttentionMetadataBuilder`** — 从 `CommonAttentionMetadata` 构建每批次元数据。声明 CUDA 图支持级别和批次重排序策略。
3. **`AttentionImpl`** — 在 `forward()` 中执行内核。

### 已注册后端

系统支持 25+ 个后端，包括：

| 后端                | 备注                                            |
| ------------------- | ----------------------------------------------- |
| `FLASH_ATTN`        | NVIDIA GPU 的默认选择（CC >= 8.0）              |
| `FLASHINFER`        | 替代高性能后端                                  |
| `TRITON_ATTN`       | 基于 Triton 的回退方案                          |
| `ROCM_ATTN`         | AMD GPU 支持                                    |
| `FLASHMLA`          | DeepSeek 风格的多潜在注意力                     |
| `MAMBA` / `MAMBA2`  | 基于 SSM 的模型                                 |
| `FLEX_ATTENTION`    | PyTorch FlexAttention                           |
| `CPU_ATTN`          | CPU 回退方案                                    |
| `TREE_ATTN`         | 投机解码树注意力                                |

后端选择由 `AttentionConfig` 和平台的 `get_attn_backend_cls()` 驱动，结果缓存以供重用。

## 分布式并行

### 并行组

vLLM 管理 5D 排名布局：`ExternalDP x DP x PP x PCP x TP`。每个维度是一个 `GroupCoordinator`，包装了 PyTorch `ProcessGroup`：

| 组     | 目的                                               |
| ------ | -------------------------------------------------- |
| TP     | 张量并行 — 分割注意力头、MLP 维度                  |
| PP     | 流水线并行 — 跨 GPU 分割层                         |
| DP     | 数据并行 — 复制模型                                |
| EP     | 专家并行 — 分布 MoE 专家                           |
| PCP    | 预填充上下文并行 — 并行化预填充                    |
| DCP    | 解码上下文并行 — 并行化解码                        |

### GroupCoordinator

包装 `ProcessGroup` 并提供集体操作（`all_reduce`、`all_gather`、`reduce_scatter`、`send_tensor_dict`、`recv_tensor_dict`）。支持共享内存消息队列广播以实现高效的节点内通信，并将集体操作注册为 `torch.ops.vllm.*` 以实现 `torch.compile` 兼容性。

## 配置系统

所有配置类都是 pydantic 数据类（通过 `@config` 装饰器），使用 `extra="forbid"` 以确保严格性。`VllmConfig` 聚合了约 25 个子配置（`ModelConfig`、`CacheConfig`、`ParallelConfig`、`SchedulerConfig` 等），并运行约 600 行的 `__post_init__` 来交叉验证所有内容并应用优化级别默认值（`O0`–`O3`）。

关键工具：`set_current_vllm_config()` 将活动配置存储在线程局部全局变量中，让任何模块无需通过每个函数调用传递即可访问 `get_current_vllm_config()`。

## 模型注册表

`vllm/model_executor/models/registry.py` 将 HuggingFace 架构名称（例如 `"LlamaForCausalLM"`）映射到 `("module_name", "ClassName")` 元组以进行惰性导入。注册表涵盖 260+ 种架构，包括文本生成、多模态、嵌入和投机解码模型。在模型加载期间通过 `ModelRegistry.resolve_model_cls()` 执行一次解析。

## 序列化和 IPC

`vllm/v1/serial_utils.py` 实现了 `MsgpackEncoder`/`MsgpackDecoder` 用于进程间通信：

- 小数据在 msgpack 消息中内联序列化。
- 大于等于 256 字节的张量作为单独的 ZMQ 帧附加（零拷贝）。
- CUDA 张量可以通过共享内存或 IPC 句柄进行带外传输。
- `OOBTensorConsumer`/`OOBTensorProvider` 机制使控制平面自包含，同时大负载绕过序列化。

## 指标和可观测性

`vllm/v1/metrics/` 提供分层统计信息：

- **`SchedulerStats`** — 每迭代：运行/等待计数、KV 缓存使用率、前缀缓存命中率、CUDA 图利用率。
- **`IterationStats`** — 每批次：生成令牌、提示令牌来源（计算、缓存命中、KV 传输）、TTFT 和令牌间延迟。
- **`RequestStateStats`** — 每请求：到达时间、排队/调度/首令牌时间戳。

日志后端包括 `LoggingStatLogger`（Python 日志记录）和 `PrometheusStatLogger`（支持多进程的 Prometheus 计数器/仪表/直方图）。可以通过 `STAT_LOGGER_PLUGINS_GROUP` 入口点注册自定义统计记录器。