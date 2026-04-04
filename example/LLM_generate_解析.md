# LLM.generate() 深度解析

本文档深入剖析 vLLM 离线推理的核心入口 `LLM.generate()` 方法，帮助你理解 vLLM 如何在单进程内高效执行批量推理。

---

## 1. 概述

`LLM.generate()` 是 vLLM 为**离线批处理**场景设计的同步阻塞 API。它的核心职责是：
1. 接收一批 Prompts 和采样参数
2. 将它们加入推理引擎队列
3. 驱动引擎逐步执行推理，直到所有请求完成
4. 一次性返回所有生成结果

> **关键特征**：同步阻塞、内置 Continuous Batching、只返回最终结果（`FINAL_ONLY`）。

---

## 2. 代码执行链路

```text
用户调用: llm.generate(prompts, sampling_params)
      │
      ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 1: 入口校验 (LLM.generate)                                  │
│ - 验证 runner_type == "generate"                                 │
│ - 若未提供 sampling_params，使用默认配置                          │
│ - 调用 _run_completion(...)                                      │
└──────────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 2: 请求入队 (_run_completion -> _add_completion_requests)   │
│ - 遍历所有 prompts                                               │
│ - 调用 _add_request(prompt, params)                              │
│   ├─ 设置 output_kind = FINAL_ONLY (离线推理只关心最终结果)      │
│   ├─ 生成唯一 request_id                                         │
│   └─ llm_engine.add_request(request_id, prompt, params)          │
│       └─ 将请求加入 LLMEngine 内部调度队列                        │
└──────────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 3: 循环执行 (_run_engine)                                   │
│ - 初始化 tqdm 进度条                                             │
│ - while llm_engine.has_unfinished_requests():                    │
│     ├─ step_outputs = llm_engine.step()  <-- 核心：执行一步推理  │
│     │     (调度 -> GPU计算 -> 处理输出)                          │
│     └─ for output in step_outputs:                               │
│           ├─ if output.finished:                                 │
│           │     └─ outputs.append(output)  <-- 收集完成的结果    │
│           └─ 更新进度条 (输入/输出 token 速度)                   │
└──────────────────────────────────────────────────────────────────┘
      │
      ▼
返回: list[RequestOutput] (所有生成结果)
```

---

## 3. 核心方法拆解

### 3.1 `generate()` - 入口网关
```python
def generate(self, prompts, sampling_params=None, ...):
    # 1. 验证模型类型
    if self.model_config.runner_type != "generate":
        raise ValueError("...")
    
    # 2. 默认参数
    if sampling_params is None:
        sampling_params = self.get_default_sampling_params()
    
    # 3. 委托给内部方法
    return self._run_completion(prompts=prompts, params=sampling_params, ...)
```

### 3.2 `_run_completion()` - 协调器
```python
def _run_completion(self, prompts, params, ...):
    # 第一步：将所有请求加入队列
    self._add_completion_requests(prompts, params, ...)
    
    # 第二步：驱动引擎执行直到完成
    return self._run_engine(output_type=RequestOutput, ...)
```

### 3.3 `_add_request()` - 请求包装
```python
def _add_request(self, prompt, params, ...):
    # 离线推理的关键设置：只关心最终输出
    params.output_kind = RequestOutputKind.FINAL_ONLY
    
    request_id = str(next(self.request_counter))
    
    # 交给底层引擎
    return self.llm_engine.add_request(
        request_id, prompt, params, ...
    )
```

### 3.4 `_run_engine()` - 推理心脏
```python
def _run_engine(self, output_type, use_tqdm=True):
    outputs = []
    
    # 核心循环：持续执行直到所有请求完成
    while self.llm_engine.has_unfinished_requests():
        # 执行一步推理 (对应模型生成一个 token)
        step_outputs = self.llm_engine.step()
        
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                
    return outputs
```

---

## 4. 与 AsyncLLM.generate() 的对比

| 特性 | `LLM.generate()` (离线) | `AsyncLLM.generate()` (在线) |
|:---|:---|:---|
| **执行模型** | 同步阻塞 | 异步非阻塞 |
| **控制权** | 用户线程显式调用 `step()` | 后台 `output_handler` 协程隐式循环 |
| **输出模式** | `FINAL_ONLY` (只返回最终文本) | `DELTA` (流式增量输出) |
| **通信方式** | 进程内直接调用 `LLMEngine` | 通过 ZMQ 与独立 `EngineCore` 进程通信 |
| **进度反馈** | 内置 `tqdm` 进度条 | 无 (由 API Server 处理 SSE 流) |
| **适用场景** | 离线批处理、评测、数据生成 | 在线服务、Chat 接口、实时推理 |

---

## 5. 关键设计解析

### 5.1 为什么离线推理用 `FINAL_ONLY`？
在线服务需要实时返回中间结果（打字机效果），所以用 `DELTA` 模式。离线推理通常用于批量生成数据，用户只关心最终结果，不需要中间状态。这可以显著减少内存拷贝和对象创建开销。

### 5.2 Continuous Batching 在离线推理中的体现
虽然 `LLM.generate()` 是同步的，但它**依然使用了 Continuous Batching**：
```python
while self.llm_engine.has_unfinished_requests():
    step_outputs = self.llm_engine.step()
```
每次 `step()` 调用时，底层的 `Scheduler` 会：
1. 检查哪些请求已经生成结束（遇到 EOS 或达到 max_tokens）
2. 将这些请求移出 Batch
3. 从队列中插入新的请求
4. 执行 GPU 计算

这意味着即使你传入了 1000 个 prompt，vLLM 也不会傻等最长的那个跑完，而是动态调整 Batch 内容，最大化 GPU 利用率。

### 5.3 进度条是如何工作的？
`_run_engine()` 使用 `tqdm` 实时显示推理进度：
```python
pbar = tqdm(total=num_requests, desc="Processed prompts")
# ...
pbar.postfix = f"est. speed input: {in_spd:.2f} toks/s, output: {out_spd:.2f} toks/s"
pbar.update(n)
```
它通过统计已完成的请求数和耗时，动态计算输入/输出吞吐量。

---

## 6. 完整示例

```python
from vllm import LLM, SamplingParams

# 1. 初始化引擎
llm = LLM(model="facebook/opt-125m")

# 2. 准备数据
prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, max_tokens=32)

# 3. 执行推理 (同步阻塞)
# 内部流程:
#   a. prompts 加入队列
#   b. while 循环调用 step()
#   c. 收集结果并返回
outputs = llm.generate(prompts, sampling_params)

# 4. 处理结果
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
```

---

## 7. 总结

`LLM.generate()` 是 vLLM 离线推理的“一站式”接口。它通过：
1. **批量入队**：一次性提交所有请求
2. **显式循环**：在用户线程中驱动 `step()` 循环
3. **Continuous Batching**：底层调度器动态优化 Batch 内容
4. **同步返回**：等待所有完成后一次性返回结果

这种设计牺牲了实时性，但换取了极高的吞吐量和简单的 API 使用体验，非常适合离线数据处理和模型评测场景。
