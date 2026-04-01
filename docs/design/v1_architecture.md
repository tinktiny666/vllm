# vLLM v1 Internal Architecture

This document describes the internal architecture of vLLM v1 — how requests flow
through the system, how the scheduler and KV cache work, and how model execution
is orchestrated across GPUs. For the process-level overview and class hierarchy,
see [Architecture Overview](arch_overview.md).

## System Layers

vLLM v1 is organized into five logical layers:

```
┌──────────────────────────────────────────────┐
│  Entrypoints                                 │
│  LLM class, OpenAI API server, CLI           │
├──────────────────────────────────────────────┤
│  Engine                                      │
│  LLMEngine / AsyncLLM                        │
│  InputProcessor, OutputProcessor, Detokenizer│
├──────────────────────────────────────────────┤
│  Engine Core                                 │
│  Scheduler, KVCacheManager, Executor         │
├──────────────────────────────────────────────┤
│  Worker / Model Runner                       │
│  GPUModelRunner, InputBatch, Attention       │
├──────────────────────────────────────────────┤
│  Platform / Kernels                          │
│  CUDA, ROCm, TPU, CPU backends              │
│  FlashAttention, FlashInfer, Triton ops      │
└──────────────────────────────────────────────┘
```

Each layer communicates with the layer below through well-defined data
structures (`EngineCoreRequest`, `SchedulerOutput`, `ModelRunnerOutput`, etc.).
The engine and engine-core layers run in separate processes connected via ZMQ
sockets, enabling the API server to handle network I/O without blocking model
execution.

## Process Architecture

For a typical `vllm serve -tp=4` deployment, six processes cooperate:

| Process              | Count                 | Responsibility                                      |
| -------------------- | --------------------- | --------------------------------------------------- |
| API Server           | `A` (default `DP`)    | HTTP handling, tokenization, result streaming        |
| Engine Core          | `DP` (default 1)      | Scheduling, KV cache management, execution dispatch  |
| GPU Worker           | `DP x PP x TP`        | Model loading, forward pass, GPU memory management   |
| DP Coordinator       | 1 (if `DP > 1`)       | Load balancing and wave synchronization for MoE DP   |

**ZMQ communication topology**: Each API server connects to every engine core
via ZMQ ROUTER/PUSH sockets. Each engine core connects to its worker processes
via multiprocessing pipes. Multi-part ZMQ messages carry serialized
`EngineCoreRequest`/`EngineCoreOutput` objects, with large tensors transferred
out-of-band for zero-copy efficiency.

## Engine Layer

### LLMEngine (Offline)

`vllm/v1/engine/llm_engine.py` — the synchronous facade used by the `LLM`
class. It wires together three subsystems:

| Component           | Role                                              |
| ------------------- | ------------------------------------------------- |
| `InputProcessor`    | Tokenize prompt, validate params, produce `EngineCoreRequest` |
| `EngineCoreClient`  | Transport requests to/from engine core (ZMQ or in-process)    |
| `OutputProcessor`   | Detokenize outputs, aggregate parallel samples, produce `RequestOutput` |

The `step()` method is the core iteration:

1. `engine_core.get_output()` — pull `EngineCoreOutputs` from engine core.
2. `output_processor.process_outputs()` — detokenize, check stop strings,
   compute logprobs, create `RequestOutput` objects.
3. Abort requests whose stop strings were detected during detokenization.
4. Record metrics.

### AsyncLLM (Server)

`vllm/v1/engine/async_llm.py` — the async engine used by the API server. It
uses `AsyncMPClient` (always multiprocess) and runs an `output_handler`
asyncio task that continuously pulls from the engine core and pushes
`RequestOutput` objects into per-request `RequestOutputCollector` queues. The
`generate()` async generator yields outputs by pulling from these queues.

### Engine Core Clients

The `EngineCoreClient` factory selects between four transport modes:

| Mode           | Client Class    | Engine Core Location               | Use Case             |
| -------------- | --------------- | ---------------------------------- | -------------------- |
| In-process     | `InprocClient`  | Same process, no busy loop         | Testing / debugging  |
| Sync multiproc | `SyncMPClient`  | Background process, ZMQ            | `LLM` class          |
| Async multiproc| `AsyncMPClient`  | Background process, ZMQ            | `AsyncLLM` (server)  |
| DP async       | `DPLBAsyncMPClient` | Multiple processes, load balanced  | Data-parallel serving|

The multiprocess clients manage ZMQ ROUTER sockets for input and PULL sockets
for output, with serialization handled by `MsgpackEncoder`/`MsgpackDecoder`
from `vllm/v1/serial_utils.py`.

## Engine Core

`vllm/v1/engine/core.py` contains the inner execution loop.

### EngineCore

The fundamental iteration (`step()` in `core.py`):

```
scheduler.has_requests()
  → scheduler.schedule()           → SchedulerOutput
  → model_executor.execute_model() → Future[ModelRunnerOutput]
  → scheduler.get_grammar_bitmask()
  → future.result()                → ModelRunnerOutput
  → model_executor.sample_tokens() (if needed)
  → scheduler.update_from_output() → EngineCoreOutputs
```

### EngineCoreProc

The multiprocess wrapper runs a busy loop:

```
while not shutdown:
    _process_input_queue()      # drain ADD/ABORT/UTILITY from ZMQ
    _process_engine_step()      # step() + put outputs to output queue
```

Input and output are handled by dedicated threads (`process_input_sockets`,
`process_output_sockets`) that bridge between ZMQ sockets and thread-safe
queues, keeping the main thread focused on scheduling and execution.

### DPEngineCoreProc

Extends `EngineCoreProc` for data-parallel MoE models. Adds:
- A DP process group for all-reduce of unfinished-request status.
- Wave-based synchronization (step counter, wave completion notifications).
- Request count publishing for the DP coordinator to perform load balancing.

## Scheduler

`vllm/v1/core/sched/scheduler.py`

### Unified Scheduling Model

The v1 scheduler does **not** have separate prefill and decode phases. Every
request has `num_computed_tokens` and `num_tokens_with_spec`. Each scheduling
step simply assigns tokens to requests so `num_computed_tokens` catches up to
`num_tokens_with_spec`. This naturally handles chunked prefill (large prompts
processed in multiple steps), continuous batching (new requests admitted
mid-stream), and decode (one token per step).

### Schedule() Phases

1. **Schedule RUNNING requests** — iterate active requests, compute token
   deficits, allocate KV blocks. If allocation fails, preempt the
   lowest-priority request.
2. **Schedule WAITING requests** — if no preemptions occurred, admit new
   requests from the waiting queue. Check blocking conditions (grammar,
   remote KV transfer, streaming input), compute prefix cache hits, allocate
   blocks.

The output is a `SchedulerOutput` containing `scheduled_new_reqs`,
`scheduled_cached_reqs`, per-request token counts, finished/preempted request
IDs, and blocks to zero.

### Request Queues

- **FCFS** (`FCFSRequestQueue`): plain deque, first-in-first-out.
- **Priority** (`PriorityRequestQueue`): min-heap by `(priority, arrival_time)`.

Two separate queues track waiting requests — normal `waiting` and
`skipped_waiting` (blocked on grammar/remote-KV/streaming). The scheduler
interleaves them based on the active policy.

### Request Lifecycle

```
WAITING  →  RUNNING  →  FINISHED_*
              ↕
           PREEMPTED
```

Possible finish states: `STOPPED` (EOS/stop-string), `LENGTH_CAPPED`,
`ABORTED`, `ERROR`, `REPETITION`.

## KV Cache Management

### Layered Architecture

```
Scheduler
  → KVCacheManager              (request-level logic)
    → KVCacheCoordinator        (coordinates across KV cache groups)
      → SingleTypeKVCacheManager (FullAttention, SlidingWindow, Mamba, ...)
        → BlockPool             (physical block allocation, free list, prefix hash map)
          → KVCacheBlock        (per-block metadata: id, ref_cnt, hash)
```

### Paged KV Cache

Each request's logical KV cache is a list of fixed-size physical blocks (the
"block table"). Blocks are managed by `BlockPool`, which owns all
`KVCacheBlock` objects and provides:

- `free_block_queue` — doubly linked list in LRU eviction order (front =
  evict first).
- `cached_block_hash_to_block` — hash index for prefix caching.

When a request finishes or is preempted, its blocks' `ref_cnt` is decremented.
Blocks with `ref_cnt == 0` return to the free queue. Blocks shared via prefix
caching stay alive as long as any request references them.

### Prefix Caching

Full blocks are hashed as `hash(parent_hash, token_ids, extra_keys)` forming
a Merkle chain. When a new request arrives, its block hashes are looked up to
find matching cached blocks, skipping redundant computation. The hash includes
multimodal feature hashes, LoRA names, and cache salts to ensure correctness.

### Hybrid KV Cache (Full + Sliding Window)

Models mixing full attention and sliding window layers use
`HybridKVCacheCoordinator`, which iterates across groups with a fixed-point
algorithm to find the longest common cache hit. Each group maintains its own
`SingleTypeKVCacheManager` but shares the same physical `BlockPool`.

### Preemption

When `allocate_slots()` fails (no free blocks), the scheduler:
1. Selects the lowest-priority running request.
2. Frees all its KV cache blocks.
3. Resets `num_computed_tokens` to 0.
4. Moves it back to the front of the waiting queue.

The preempted request is re-scheduled later, potentially benefiting from prefix
caching if its prefix blocks are still resident.

## Model Execution

### Worker

Each GPU has a dedicated worker process (`vllm/v1/worker/gpu_worker.py`):

1. **Init**: set CUDA device, initialize distributed environment, create
   `GPUModelRunner`.
2. **Load model**: delegate to the model loader strategy (safetensors, GGUF,
   bitsandbytes, etc.).
3. **Profile memory**: run a dummy forward pass to measure peak memory, then
   compute available KV cache memory.
4. **Init KV cache**: allocate GPU blocks based on the profiled memory.
5. **Compile/warmup**: `torch.compile` the model at specific batch sizes,
   capture CUDA graphs.
6. **Execute**: called every scheduling iteration — receive inputs, run forward
   pass, return output tokens/logprobs.

### GPUModelRunner

The hot path in `execute_model()`:

```
_update_states()                 — update persistent InputBatch
_execute_mm_encoder()            — run multimodal encoders if needed
_prepare_inputs()                — build logits_indices, spec_decode metadata
build attention metadata         — block tables, seq_lens, slot_mapping
model(input_ids, positions)      — forward through decoder layers
lm_head                          — logits projection
sample / return ModelRunnerOutput
```

### CUDA Graphs

For supported batch sizes, the forward pass replays pre-captured CUDA graphs
instead of launching individual kernels, eliminating Python overhead. Each
attention backend declares its CUDA graph support level via
`AttentionCGSupport` (ALWAYS, UNIFORM_BATCH, UNIFORM_SINGLE_TOKEN_DECODE,
NEVER).

### torch.compile

The model is compiled through `VLLMCompileBackend`, enabling operator fusions:
norm+quant, act+quant, allreduce+rms, rope+kvcache. Compilation is cached by
a hash of all configs that affect the computation graph.

## Attention Backends

### Three-Part Protocol

Every attention backend in `vllm/v1/attention/` implements:

1. **`AttentionBackend`** — class-level capabilities: supported dtypes, KV
   cache shapes, block sizes, CUDA graph support.
2. **`AttentionMetadataBuilder`** — builds per-batch metadata from
   `CommonAttentionMetadata`. Declares CUDA graph support level and batch
   reordering strategy.
3. **`AttentionImpl`** — executes the kernel in `forward()`.

### Registered Backends

The system supports 25+ backends including:

| Backend              | Notes                                         |
| -------------------- | --------------------------------------------- |
| `FLASH_ATTN`         | Default for NVIDIA GPUs (CC >= 8.0)           |
| `FLASHINFER`         | Alternative high-performance backend          |
| `TRITON_ATTN`        | Triton-based fallback                         |
| `ROCM_ATTN`          | AMD GPU support                               |
| `FLASHMLA`           | DeepSeek-style Multi-Latent Attention          |
| `MAMBA` / `MAMBA2`   | SSM-based models                              |
| `FLEX_ATTENTION`     | PyTorch FlexAttention                         |
| `CPU_ATTN`           | CPU fallback                                  |
| `TREE_ATTN`          | Speculative decoding tree attention            |

Backend selection is driven by `AttentionConfig` and the platform's
`get_attn_backend_cls()`, with results cached for reuse.

## Distributed Parallelism

### Parallel Groups

vLLM manages a 5D rank layout: `ExternalDP x DP x PP x PCP x TP`. Each
dimension is a `GroupCoordinator` wrapping a PyTorch `ProcessGroup`:

| Group | Purpose                                            |
| ----- | -------------------------------------------------- |
| TP    | Tensor parallel — splits attention heads, MLP dims  |
| PP    | Pipeline parallel — splits layers across GPUs       |
| DP    | Data parallel — replicates the model                |
| EP    | Expert parallel — distributes MoE experts           |
| PCP   | Prefill context parallel — parallelizes prefill      |
| DCP   | Decode context parallel — parallelizes decode        |

### GroupCoordinator

Wraps `ProcessGroup` with collective ops (`all_reduce`, `all_gather`,
`reduce_scatter`, `send_tensor_dict`, `recv_tensor_dict`). Supports
shared-memory message queue broadcast for efficient intra-node communication
and registers collectives as `torch.ops.vllm.*` for `torch.compile`
compatibility.

## Configuration System

All config classes are pydantic dataclasses (via the `@config` decorator) with
`extra="forbid"` for strictness. `VllmConfig` aggregates ~25 sub-configs
(`ModelConfig`, `CacheConfig`, `ParallelConfig`, `SchedulerConfig`, etc.) and
runs a ~600-line `__post_init__` that cross-validates everything and applies
optimization-level defaults (`O0`–`O3`).

Key utility: `set_current_vllm_config()` stores the active config in a
thread-local global, letting any module access `get_current_vllm_config()`
without threading it through every function call.

## Model Registry

`vllm/model_executor/models/registry.py` maps HuggingFace architecture names
(e.g., `"LlamaForCausalLM"`) to `("module_name", "ClassName")` tuples for lazy
import. The registry covers 260+ architectures including text generation,
multimodal, embedding, and speculative decoding models. Resolution is performed
once during model loading via `ModelRegistry.resolve_model_cls()`.

## Serialization and IPC

`vllm/v1/serial_utils.py` implements `MsgpackEncoder`/`MsgpackDecoder` for
inter-process communication:

- Small data is serialized inline in the msgpack message.
- Tensors >= 256 bytes are attached as separate ZMQ frames (zero-copy).
- CUDA tensors can be transferred out-of-band via shared memory or IPC handles.
- The `OOBTensorConsumer`/`OOBTensorProvider` mechanism keeps the control plane
  self-contained while large payloads bypass serialization.

## Metrics and Observability

`vllm/v1/metrics/` provides hierarchical stats:

- **`SchedulerStats`** — per-iteration: running/waiting counts, KV cache usage,
  prefix cache hit rate, CUDA graph utilization.
- **`IterationStats`** — per-batch: generation tokens, prompt token sources
  (computed, cache-hit, KV-transfer), TTFT and inter-token latencies.
- **`RequestStateStats`** — per-request: arrival time, queued/scheduled/first-
  token timestamps.

Logging backends include `LoggingStatLogger` (Python logging) and
`PrometheusStatLogger` (Prometheus counters/gauges/histograms with multiprocess
support). Custom stat loggers can be registered via the
`STAT_LOGGER_PLUGINS_GROUP` entry point.
