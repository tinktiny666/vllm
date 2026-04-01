# vLLM `serve` 命令关键代码解读

本文档从 `vllm serve` 命令入手，深入解读 vLLM 的服务化架构和核心原理。

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [vllm serve 命令入口](#2-vllm-serve-命令入口)
3. [API Server 启动流程](#3-api-server-启动流程)
4. [Engine 核心原理](#4-engine-核心原理)
5. [调度器 (Scheduler)](#5-调度器-scheduler)
6. [KV Cache 管理](#6-kv-cache-管理)
7. [请求处理流程](#7-请求处理流程)

---

## 1. 整体架构概览

vLLM 采用多进程架构，主要包含以下组件：

```
┌─────────────────────────────────────────────────────────────┐
│                      API Server (FastAPI)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ /v1/chat     │  │ /v1/completions│  │  /health    │       │
│  │  /completions │  │              │  │             │       │
│  └──────┬───────┘  └──────┬───────┘  └─────────────┘       │
│         │                 │                                  │
│         └────────┬────────┘                                  │
│                  ▼                                           │
│         ┌────────────────┐                                   │
│         │ AsyncLLM Engine│  (前端进程)                       │
│         └────────┬───────┘                                   │
│                  │ ZMQ IPC                                   │
├──────────────────┼───────────────────────────────────────────┤
│                  ▼                                           │
│         ┌────────────────┐                                   │
│         │  EngineCore    │  (后端进程)                       │
│         │  ┌───────────┐ │                                   │
│         │  │ Scheduler │ │                                   │
│         │  └─────┬─────┘ │                                   │
│         │        ▼       │                                   │
│         │  ┌───────────┐ │                                   │
│         │  │Executor   │ │                                   │
│         │  │(Workers)  │ │                                   │
│         │  └───────────┘ │                                   │
│         └────────────────┘                                   │
└─────────────────────────────────────────────────────────────┘
```

**关键设计思想：**
- **前后端分离**：API Server (AsyncLLM) 与 EngineCore 通过 ZMQ IPC 通信
- **异步处理**：支持高并发请求处理
- **PagedAttention**：高效管理 KV Cache

---

## 2. vllm serve 命令入口

### 2.1 CLI 入口

命令入口文件：`vllm/entrypoints/cli/serve.py`

```python
class ServeSubcommand(CLISubcommand):
    name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # 模型参数优先级处理
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        # 根据 api_server_count 决定运行模式
        if args.api_server_count < 1:
            run_headless(args)          # 无头模式（仅引擎）
        elif args.api_server_count > 1:
            run_multi_api_server(args)  # 多API Server模式
        else:
            # 单API Server模式（默认）
            uvloop.run(run_server(args))
```

### 2.2 运行模式

vLLM 支持三种运行模式：

| 模式 | 说明 | 使用场景 |
|------|------|----------|
| 单API Server | 默认模式，单进程处理 | 单GPU/简单部署 |
| 多API Server | 多进程处理 | 高并发/数据并行 |
| Headless | 仅引擎，无HTTP服务 | 分布式推理worker节点 |

---

## 3. API Server 启动流程

### 3.1 OpenAI 兼容服务器

主要文件：`vllm/entrypoints/openai/api_server.py`

```python
async def run_server(args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server."""
    decorate_logs("APIServer")
    listen_address, sock = setup_server(args)  # 创建socket
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)
```

### 3.2 核心启动步骤

```python
@instrument(span_name="API server setup")
def setup_server(args):
    """Validate API server args, set up signal handler, create socket"""

    log_version_and_model(logger, VLLM_VERSION, args.model)
    log_non_default_args(args)

    # 1. 创建服务端socket（在引擎启动前绑定端口，避免Ray竞争条件）
    if args.uds:
        sock = create_server_unix_socket(args.uds)
    else:
        sock_addr = (args.host or "", args.port)
        sock = create_server_socket(sock_addr)

    # 2. 设置文件描述符限制
    set_ulimit()

    # 3. 注册信号处理器
    signal.signal(signal.SIGTERM, signal_handler)

    return listen_address, sock
```

### 3.3 引擎客户端构建

```python
@asynccontextmanager
async def build_async_engine_client(args) -> AsyncIterator[EngineClient]:
    """创建EngineClient，支持两种模式：
    1. 进程内直接使用 AsyncLLM
    2. 多进程使用 AsyncLLM RPC
    """
    engine_args = AsyncEngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    # 使用 V1 引擎
    from vllm.v1.engine.async_llm import AsyncLLM

    async_llm = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=usage_context,
        ...
    )

    yield async_llm
    async_llm.shutdown()
```

---

## 4. Engine 核心原理

### 4.1 AsyncLLM 架构

主要文件：`vllm/v1/engine/async_llm.py`

AsyncLLM 是 vLLM 的异步引擎前端，负责：

1. **请求接收与转换**：将用户请求转换为 EngineCoreRequest
2. **输出处理**：异步处理引擎输出并返回给用户
3. **生命周期管理**：管理 EngineCore 的生命周期

```python
class AsyncLLM(EngineClient):
    def __init__(self, vllm_config: VllmConfig, executor_class: type[Executor], ...):
        # 1. 渲染器（处理 prompt 模板）
        self.renderer = renderer_from_config(self.vllm_config)

        # 2. IO 处理器
        self.io_processor = get_io_processor(self.vllm_config, self.renderer, ...)

        # 3. 输入处理器：EngineInput -> EngineCoreRequest
        self.input_processor = InputProcessor(self.vllm_config, renderer)

        # 4. 输出处理器：EngineCoreOutputs -> RequestOutput
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            ...
        )

        # 5. EngineCore 客户端（启动后端进程）
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            ...
        )
```

### 4.2 请求处理核心方法

```python
async def generate(
    self,
    prompt: PromptType,
    sampling_params: SamplingParams,
    request_id: str,
    ...
) -> AsyncGenerator[RequestOutput, None]:
    """
    主要的生成方法：
    1. 创建 AsyncStream 对应请求
    2. 处理输入
    3. 添加请求到 OutputProcessor
    4. 添加请求到 EngineCore（独立进程）

    后台 output_handler 循环：
    - 从 EngineCore 拉取输出
    - 推送到对应的 AsyncStream

    调用者迭代返回的 AsyncGenerator，获取 RequestOutput
    """
    # 添加请求
    q = await self.add_request(request_id, prompt, sampling_params, ...)

    # 从队列中拉取输出
    finished = False
    while not finished:
        out = q.get_nowait() or await q.get()
        finished = out.finished
        if out is not STREAM_FINISHED:
            yield out
```

### 4.3 Output Handler 后台循环

```python
async def output_handler():
    """后台循环：从 EngineCore 拉取输出并推送到 AsyncStreams"""
    while True:
        # 1. 从 EngineCore 拉取输出
        outputs = await engine_core.get_output_async()

        # 2. 分块处理（避免阻塞事件循环）
        for start in range(0, len(outputs.outputs), chunk_size):
            outputs_slice = outputs.outputs[start:end]

            # 3. 处理输出（由 OutputProcessor 完成）
            processed_outputs = output_processor.process_outputs(
                outputs_slice, outputs.timestamp, iteration_stats
            )
            # 注意：RequestOutput 被推送到各自的队列中

        # 4. 日志记录
        if logger_manager:
            logger_manager.record(...)
```

---

## 5. 调度器 (Scheduler)

主要文件：`vllm/v1/core/sched/scheduler.py`

### 5.1 调度算法核心思想

vLLM 的调度器采用统一的 token 级调度：

```python
def schedule(self) -> SchedulerOutput:
    """
    调度算法核心思想：

    没有独立的 "decoding phase" 或 "prefill phase"。
    每个请求有：
    - num_computed_tokens: 已计算的 token 数
    - num_tokens_with_spec: prompt_tokens + output_tokens + spec_tokens

    每步调度器尝试为请求分配 token，使 num_computed_tokens
    追上 num_tokens_with_spec。

    这种设计足够通用，可以覆盖：
    - chunked prefills（分块预填充）
    - prefix caching（前缀缓存）
    - speculative decoding（推测解码）
    - jump decoding（未来优化）
    """
```

### 5.2 请求状态管理

```python
# 请求队列
self.waiting: RequestQueue          # 等待队列
self.skipped_waiting: RequestQueue  # 被跳过的等待队列（异步依赖/约束）
self.running: list[Request]         # 运行队列
```

### 5.3 调度流程

```python
# 1. 首先调度 RUNNING 请求
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]

    # 计算需要调度的新 token 数
    num_new_tokens = (
        request.num_tokens_with_spec
        + request.num_output_placeholders
        - request.num_computed_tokens
    )

    # 分配 KV Cache 块
    new_blocks = self.kv_cache_manager.allocate_slots(
        request, num_new_tokens, ...
    )

    if new_blocks is None:
        # 内存不足，抢占低优先级请求
        preempted_req = self.running.pop()
        self._preempt_request(preempted_req)

# 2. 然后调度 WAITING 请求
while (self.waiting or self.skipped_waiting) and token_budget > 0:
    # 检查前缀缓存命中
    new_computed_blocks, num_new_local_computed_tokens = (
        self.kv_cache_manager.get_computed_blocks(request)
    )

    # 分配新块
    new_blocks = self.kv_cache_manager.allocate_slots(
        request, num_new_tokens, ...
    )

    # 将请求移到运行队列
    self.running.append(request)
```

### 5.4 抢占机制

```python
def _preempt_request(self, request: Request, timestamp: float) -> None:
    """抢占请求并放回等待队列"""
    # 释放 KV Cache
    self.kv_cache_manager.free(request)
    self.encoder_cache_manager.free(request)

    # 重置请求状态
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0
    request.num_preemptions += 1

    # 放回等待队列
    self.waiting.prepend_request(request)
```

---

## 6. KV Cache 管理

主要文件：`vllm/v1/core/kv_cache_manager.py`

### 6.1 PagedAttention 原理

vLLM 使用 PagedAttention 管理 KV Cache，核心思想类似操作系统虚拟内存：

```
逻辑视图（连续）           物理存储（分块）
┌──────────────┐         ┌──────────────┐
│ Token 0-15   │ ──────► │ Block 0      │
├──────────────┤         ├──────────────┤
│ Token 16-31  │ ──┐     │ Block 1      │
├──────────────┤   │     ├──────────────┤
│ Token 32-47  │ ──┼───► │ Block 2      │
├──────────────┤   │     ├──────────────┤
│ Token 48-63  │ ──┘     │ Block 3      │
└──────────────┘         └──────────────┘
```

### 6.2 KVCacheManager 核心方法

```python
class KVCacheManager:
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,  # 前缀缓存命中
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,     # 推测解码
        ...
    ) -> KVCacheBlocks | None:
        """
        为请求分配新的 KV Cache 槽位。

        块布局：
        | <comp> | <new_comp> | <ext_comp> | <new> | <lookahead> |
                 |           < to be computed >                  |
                 |              < to be allocated >              |
        """
        # 1. 计算需要分配的块数
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(...)

        # 2. 检查是否有足够的空闲块
        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            return None  # 内存不足

        # 3. 分配前缀缓存块
        if new_computed_blocks:
            self.coordinator.allocate_new_computed_blocks(...)

        # 4. 分配新块
        new_blocks = self.coordinator.allocate_new_blocks(...)

        # 5. 缓存块（启用前缀缓存时）
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """获取请求的已缓存块（前缀缓存查找）"""
        if not self.enable_caching:
            return self.empty_kv_cache_blocks, 0

        # 查找最长的缓存命中
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )
        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens
```

### 6.3 前缀缓存 (Prefix Caching)

前缀缓存允许共享相同前缀的请求复用 KV Cache：

```
请求1: "Translate the following text to French: Hello"
请求2: "Translate the following text to French: World"

两个请求共享前缀 "Translate the following text to French:" 的 KV Cache
```

---

## 7. 请求处理流程

### 7.1 完整请求生命周期

```
┌─────────────────────────────────────────────────────────────┐
│  1. 用户请求                                                 │
│     POST /v1/chat/completions                               │
│         │                                                   │
│         ▼                                                   │
│  2. OpenAI Serving 层                                       │
│     - 解析请求参数                                          │
│     - 应用 chat template                                    │
│         │                                                   │
│         ▼                                                   │
│  3. AsyncLLM.generate()                                     │
│     - input_processor: 转换为 EngineCoreRequest              │
│     - 添加到 OutputProcessor                                │
│     - 添加到 EngineCore (ZMQ)                               │
│         │                                                   │
│         ▼                                                   │
│  4. EngineCore.add_request()                                │
│     - 添加到 Scheduler 等待队列                             │
│         │                                                   │
│         ▼                                                   │
│  5. Scheduler.schedule()                                    │
│     - 检查前缀缓存                                          │
│     - 分配 KV Cache 块                                      │
│     - 构建 SchedulerOutput                                  │
│         │                                                   │
│         ▼                                                   │
│  6. Executor.execute_model()                                │
│     - 模型前向传播                                          │
│     - 采样 token                                            │
│         │                                                   │
│         ▼                                                   │
│  7. Scheduler.update_from_output()                          │
│     - 更新请求状态                                          │
│     - 检查停止条件                                          │
│     - 生成 EngineCoreOutput                                 │
│         │                                                   │
│         ▼                                                   │
│  8. AsyncLLM.output_handler()                               │
│     - 从 EngineCore 拉取输出                                │
│     - OutputProcessor: detokenize                           │
│     - 推送到 RequestOutputCollector                         │
│         │                                                   │
│         ▼                                                   │
│  9. 用户接收响应                                            │
│     - 流式：逐 token 返回                                   │
│     - 非流式：等待完整响应                                  │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 核心数据结构

```python
# EngineCoreRequest: 前端 -> 后端
@dataclass
class EngineCoreRequest:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    block_hashes: list[BlockHash]  # 用于前缀缓存
    ...

# SchedulerOutput: 调度器 -> 模型执行器
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]      # 新请求
    scheduled_cached_reqs: list[CachedRequestData] # 已缓存请求
    num_scheduled_tokens: dict[str, int]           # 每个请求的 token 数
    scheduled_spec_decode_tokens: dict[str, list[int]]  # 推测解码
    ...

# EngineCoreOutput: 后端 -> 前端
@dataclass
class EngineCoreOutput:
    request_id: str
    new_token_ids: list[int]           # 新生成的 token
    finish_reason: FinishReason | None
    new_logprobs: LogprobsLists | None
    ...
```

---

## 关键文件索引

| 组件 | 文件路径 | 说明 |
|------|----------|------|
| CLI 入口 | `vllm/entrypoints/cli/serve.py` | serve 命令解析 |
| API Server | `vllm/entrypoints/openai/api_server.py` | OpenAI 兼容服务器 |
| AsyncLLM | `vllm/v1/engine/async_llm.py` | 异步引擎前端 |
| EngineCore | `vllm/v1/engine/core.py` | 引擎核心（后端） |
| Scheduler | `vllm/v1/core/sched/scheduler.py` | 请求调度器 |
| KVCacheManager | `vllm/v1/core/kv_cache_manager.py` | KV Cache 管理 |
| Executor | `vllm/v1/executor/` | 模型执行器 |

---

## 总结

vLLM 的核心设计特点：

1. **前后端分离**：API Server 和 EngineCore 通过 ZMQ IPC 通信，支持多进程部署
2. **异步处理**：基于 asyncio 的异步架构，支持高并发
3. **统一调度**：token 级别的统一调度，支持 chunked prefill、prefix caching、speculative decoding
4. **PagedAttention**：高效的 KV Cache 管理，减少内存碎片
5. **可扩展性**：支持多种调度策略、多种执行器、多种模型架构
