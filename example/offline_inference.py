# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
vLLM Offline Inference Example

This example demonstrates how to use vLLM for offline batch inference
without starting an HTTP server.
"""

from vllm import LLM, SamplingParams

# 1. Prepare input prompts
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is",
]

# 2. Configure sampling parameters
# These control the generation behavior (randomness, length, etc.)
sampling_params = SamplingParams(
    temperature=0.8,   # Higher values = more random (0.0 - 1.0)
    top_p=0.95,        # Top-p sampling: cumulative probability threshold
    max_tokens=32,     # Maximum number of tokens to generate
)

# 3. Initialize the LLM engine
# On first run, this will automatically download the model.
# Using a lightweight model (opt-125m) for demonstration.
print("Loading model...")
llm = LLM(model="facebook/opt-125m")

# 4. Run inference
# generate() is a synchronous blocking method that waits for all requests to complete.
print("Generating...")
outputs = llm.generate(prompts, sampling_params)

# 5. Process and print results
print("\n--- Results ---")
for output in outputs:
    # output.prompt: the original input prompt
    # output.outputs: a list containing n generated results (default n=1)
    generated_text = output.outputs[0].text
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {generated_text!r}")
    print("-" * 40)
