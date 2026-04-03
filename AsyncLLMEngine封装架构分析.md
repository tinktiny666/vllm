# AsyncLLMEngine 封装 LLMEngine 架构分析

## 1. 概述

`AsyncLLMEngine`（现为 `AsyncLLM` 的别名）是 vLLM V1 架构中的异步引擎封装层。它**不直接封装**传统的 `LLMEngine`，而是采用全新的架构设计，通过进程间通信与 `EngineCore` 交互。

```
┌─────────────────────────────────────────────────────────────┐
│                    AsyncLLM (AsyncLLMEngine)                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  EngineClient Protocol              │   │
│  └─────────────────────────────────────────────────────┘   │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │   Input      │   │   Output     │   │   Engine     │   │
│  │  Processor   │   │  Processor   │   │  Core Client │   │
│  └──────────────┘   └──────────────┘   └──────────────┘   │
│                                              │              │
└──────────────────────────────────────────────┼──────────────┘
                                               │ ZMQ
                                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 EngineCore (独立进程)                        │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │  Scheduler   │   │    Model     │   │   Executor   │   │
│  │              │   │  Executor    │   │              │   │
│  └──────────────┘   └──────────────┘   └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 2. 核心组件

### 2.1 AsyncLLM 类

**位置**: `vllm/v1/engine/async_llm.py`

**关键属性**:
```python
class AsyncLLM(EngineClient):
    # 核心组件
    self.engine_core: EngineCoreClient      # 与 EngineCore 通信的客户端
    self.input_processor: InputProcessor    # 处理输入请求
    self.output_processor: OutputProcessor  # 处理输出结果
    self.renderer: BaseRenderer             # 渲染器（聊天模板等）
    self.io_processor: IOProcessor          # IO 处理器
```

### 2.2 EngineCoreClient（通信层）

**位置**: `vllm/v1/engine/core_client.py`

EngineCoreClient 是一个抽象基类，定义了与 EngineCore 通信的接口：

```python
class EngineCoreClient(ABC):
    """子类实现不同的通信方式"""
    
    # 子类:
    # - InprocClient: 进程内（V0 风格）
    # - SyncMPClient: ZMQ + 同步多进程（用于 LLM）
    # - AsyncMPClient: ZMQ + 异步多进程（用于 AsyncLLM）
```

### 2.3 AsyncMPClient（异步多进程客户端）

**位置**: `vllm/v1/engine/core_client.py:859`

AsyncLLM 使用 AsyncMPClient 通过 ZMQ 与 EngineCore 通信：

```python
class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""
    
    def __init__(self, ...):
        self.outputs_queue = asyncio.Queue[EngineCoreOutputs]()
        # 启动后台任务接收输出
        self._ensure_output_queue_task()
    
    async def get_output_async(self) -> EngineCoreOutputs:
        """异步获取输出"""
        outputs = await self.outputs_queue.get()
        return outputs
    
    def _send_input(self, request_type, request, engine=None):
        """发送请求到 EngineCore"""
        message = (request_type.value, *self.encoder.encode(request))
        return self.input_socket.send_multipart(msg, copy=False)
```

### 2.4 EngineCore（核心引擎）

**位置**: `vllm/v1/engine/core.py:87`

EngineCore 是实际执行推理的核心组件：

```python
class EngineCore:
    """Inner loop of vLLM's Engine."""
    
    def __init__(self, vllm_config, executor_class, ...):
        # 模型执行器
        self.model_executor = executor_class(vllm_config)
        
        # 调度器
        self.scheduler = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            ...
        )
        
        # 结构化输出管理器
        self.structured_output_manager = StructuredOutputManager(vllm_config)
```

## 3. 数据流

### 3.1 请求处理流程

```
用户请求
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ AsyncLLM.generate()                                     │
│   1. 调用 add_request()                                  │
│   2. 创建 RequestOutputCollector                         │
│   3. 返回 AsyncGenerator                                 │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ InputProcessor.process_inputs()                         │
│   - 将 PromptType 转换为 EngineCoreRequest               │
│   - 处理分词、多模态输入等                                │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ AsyncMPClient.add_request_async()                       │
│   - 通过 ZMQ 发送请求到 EngineCore 进程                  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ EngineCore (独立进程)                                    │
│   1. Scheduler 调度请求                                  │
│   2. Executor 执行模型推理                               │
│   3. 生成 EngineCoreOutput                               │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ AsyncMPClient.get_output_async()                        │
│   - 从 ZMQ 接收输出                                      │
│   - 放入 outputs_queue                                   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ OutputProcessor.process_outputs()                       │
│   - 反序列化输出                                         │
│   - 增量去分词                                           │
│   - 构建 RequestOutput                                   │
│   - 放入 RequestOutputCollector                          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ AsyncLLM.generate()                                     │
│   - 从 RequestOutputCollector 获取输出                   │
│   - yield RequestOutput                                  │
└─────────────────────────────────────────────────────────┘
```

### 3.2 输出处理循环

AsyncLLM 在后台运行一个 `output_handler` 协程，持续从 EngineCore 获取输出：

```python
async def output_handler():
    while True:
        # 1. 从 EngineCore 获取输出
        outputs = await engine_core.get_output_async()
        
        # 2. 处理输出（分块处理避免阻塞事件循环）
        for start in range(0, num_outputs, chunk_size):
            outputs_slice = engine_core_outputs[start:end]
            processed_outputs = output_processor.process_outputs(
                outputs_slice, outputs.timestamp, iteration_stats
            )
            
            # 3. 处理需要中止的请求
            if processed_outputs.reqs_to_abort:
                await engine_core.abort_requests_async(
                    processed_outputs.reqs_to_abort
                )
```

## 4. 关键设计

### 4.1 进程隔离

```
┌─────────────────────┐     ZMQ      ┌─────────────────────┐
│   API Server 进程   │◄────────────►│   EngineCore 进程   │
│                     │              │                     │
│  - AsyncLLM         │              │  - Scheduler        │
│  - InputProcessor   │              │  - Model Executor   │
│  - OutputProcessor  │              │  - KV Cache         │
│  - AsyncMPClient    │              │  - GPU Workers      │
└─────────────────────┘              └─────────────────────┘
```

**优势**:
- API Server 可以处理高并发请求，不受 GPU 推理阻塞影响
- EngineCore 进程崩溃不会导致 API Server 崩溃
- 支持多客户端连接同一个 EngineCore

### 4.2 异步设计

AsyncLLM 使用 asyncio 实现全异步架构：

```python
# 生成请求是异步的
async def generate(self, prompt, sampling_params, request_id, ...):
    # 添加请求到队列
    q = await self.add_request(request_id, prompt, sampling_params, ...)
    
    # 异步迭代输出
    while not finished:
        out = q.get_nowait() or await q.get()
        finished = out.finished
        yield out
```

### 4.3 流式处理

支持两种流式模式：

1. **输出流式**: 通过 `AsyncGenerator` 返回增量输出
2. **输入流式**: 支持 `StreamingInput` 用于多轮对话

```python
# 输出流式
async for output in async_llm.generate(...):
    print(output)

# 输入流式（多轮对话）
async def input_stream():
    yield StreamingInput(prompt="Hello")
    yield StreamingInput(prompt="How are you?")

async for output in async_llm.generate(
    ..., 
    prompt=input_stream()
):
    print(output)
```

## 5. 与 V0 架构的对比

| 特性 | V0 (LLMEngine) | V1 (AsyncLLM) |
|------|---------------|---------------|
| 进程模型 | 单进程/多线程 | 多进程 |
| 通信方式 | 直接函数调用 | ZMQ IPC |
| 异步支持 | 有限 | 完全异步 |
| 扩展性 | 较差 | 好 |
| 故障隔离 | 无 | 进程级隔离 |

## 6. 总结

AsyncLLM 不是简单地封装 LLMEngine，而是采用了全新的架构：

1. **进程分离**: API Server 和 EngineCore 运行在不同进程
2. **ZMQ 通信**: 使用 ZeroMQ 进行高效的进程间通信
3. **全异步**: 基于 asyncio 的完全异步架构
4. **组件化**: InputProcessor、OutputProcessor、Renderer 等组件解耦
5. **可扩展**: 支持多客户端、数据并行等高级特性

这种设计使得 vLLM V1 能够更好地处理高并发请求，同时保持高效的 GPU 利用率。
