# Optimization log

## P01 - Batch CUDA projections during prefill

- Bottleneck and hypothesis. One-token projection launches repeatedly read
  weight matrices during prefill. Reusing each matrix across up to 128 prompt
  tokens should reduce TTFT with no greedy output changes.
- Change. `cuda/qwen_gpu.cpp`, `cuda/qwen_gpu.hpp`,
  `cuda/kernels/qwen_ops.cuh`, and `engine/runtime/cli.cpp` add an opt-in
  prefill path with FP32 cuBLAS GEMM. Packed matrices expand in a reusable
  device scratch buffer. Attention and DeltaNet updates stay causal. The
  existing single-token path still processes the final prompt token and
  subsequent tokens.
- Control. SPARK `1fa4a53`, packed GGUF, FP32 KV and no graphs.
- Correctness. Packed decoder, recurrent and CUDA operation unit tests passed.
  All 20 golden prompts matched all 64 continuation IDs. Maximum absolute
  first-step relative logit-gap error was 0.040231 over the full vocabulary,
  against the FP32-expanded artifact running on CUDA. All ten measured
  pairs against the original GGUF matched every generated token.
- Primary workload. 512 prompt and 128 generated tokens, 5 warm-ups and
  10 measurements. Values below are medians except where specified.

| Metric | Serial control | Batched prefill | Change |
| --- | ---: | ---: | ---: |
| TTFT | 6,969.26 ms | 785.80 ms | -88.7% |
| Prefill | 73.48 tok/s | 653.09 tok/s | +789% |
| Decode | 50.29 tok/s | 50.35 tok/s | +0.1% |
| Total latency | 9,504.01 ms | 3,309.40 ms | -65.2% |
| Peak board VRAM | 683 MiB | 717 MiB | +34 MiB |
| Peak process working set | 384.97 MiB | 408.88 MiB | +23.91 MiB |
| GPU board efficiency | 0.163 tok/J | 0.499 tok/J | +206% |

Decision **KEEP**. The first-token improvement is large and passes the
generation gate. Treat the decode difference as unchanged within run
variation. The same-session llama.cpp medians were 136.64 ms TTFT,
3,821.83 tok/s prefill, 121.64 tok/s decode and 761 MiB peak VRAM.

[Control](baseline.json) |
[Raw candidate trials](experiments/batch-prefill-primary.json) |
[Golden evidence](experiments/batch-prefill-golden.jsonl)
