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

## D01 - Replay packed decode with bucketed CUDA Graphs

- Bottleneck and hypothesis. The audited decoder submits hundreds of kernels
  per token. Replay should reduce decode latency with packed weights and
  unchanged prefill behavior.
- Change. Enable the existing bucket capture path for the persistent CUDA
  worker. Worker and runner entry points changed in `engine/runtime/cli.cpp`
  and `benchmark/comparison/compare_cuda.cjs`. The batch-prefill control
  and candidate use the same single-token math and prompt path.
- Correctness. CUDA operation tests passed. All 1,280 golden continuation
  IDs matched. Largest first-step logit-gap error 0.040231. All ten scored
  pairs against the original GGUF matched every generated token.
- Results. 50.35 to 51.83 median decode tok/s (+2.94%). P95 decode latency
  2,529.41 to 2,467.12 ms (-2.46%). Total median latency 3,309.40 to
  3,258.02 ms (-1.55%). TTFT 785.80 to 805.32 ms (+2.48%).
  Peak board VRAM rose from 717 to 753 MiB. Peak process working set rose
  from 408.88 to 427.01 MiB.
- Decision **KEEP** as a selectable mode. The primary decode change improved
  across measured trials at a 36 MiB board-memory cost. Continue tracking
  tail latency and memory at the other two prompt lengths before treating
  the gain as general.

[Raw trials](experiments/packed-graph-primary.json) |
[Golden evidence](experiments/packed-graph-golden.jsonl)

## D02 - Reuse Q6_K block bytes in packed GEMV

- Bottleneck and hypothesis. The baseline reads the tied vocabulary matrix
  every token and spends 4.45 ms in its final norm and output projection
  on a short decode step. Reusing Q6_K low and high nibble stripes and
  the block multiplier across consecutive products should reduce decode
  time without changing reduction order.
- Change. `cuda/kernels/qwen_ops.cuh` adds a Q6_K-specific warp kernel.
  `cuda/qwen_gpu.cpp` dispatches Q6_K matrices to it when enabled.
  The route uses identical packed storage, FP32 arithmetic and launch
  geometry. Other GGUF types use the original kernel. Both arms use
  batched prefill and bucketed graphs.
- Correctness. CUDA tests passed, including Q6_K projection versus decoded
  CPU weights. Golden output matched 1,280/1,280 IDs, maximum logit-gap
  error 0.040231. All ten matched GGUF benchmark pairs agreed exactly.
- Results. Median decode 51.83 to 69.56 tok/s (+34.2%). P95 decode latency
  2,467.12 to 1,834.88 ms (-25.6%). Total latency 3,258.02 to 2,624.71 ms
  (-19.4%). TTFT 805.32 to 794.56 ms (-1.3%). Peak VRAM 753 MiB on both arms.
  Peak working set 427.01 to 427.02 MiB.
- Decision **KEEP**. The pinned llama.cpp comparison measured 121.72 tok/s,
  140.44 ms TTFT and 761 MiB peak board memory on this run.

[Raw trials](experiments/q6-block-primary.json) |
[Golden evidence](experiments/q6-block-golden.jsonl) |
[Packed baseline CUDA event trace](packed-baseline-profile.jsonl)

## P02 - FP16 operands for batched prefill GEMM

- Bottleneck and hypothesis. Nsight measured 137.85 ms in the dominant
  SGEMM kernel and 67.95 ms in packed expansion before the first token.
  FP16 matrix and activation scratch with FP32 accumulation could make
  the projection fraction faster and lower end-to-end TTFT.
- Change. `cuda/qwen_gpu.cpp`, `cuda/kernels/qwen_ops.cuh`, and
  `engine/runtime/cli.cpp` isolate lower precision to packed prefill
  GEMM operands. The mixed scratch, FP32 output, and separate cuBLAS
  handle leave the packed decode path unchanged.
- Correctness. All 1,280 golden decode IDs matched. Maximum first-step
  logit-gap error was 0.035286. All ten scored comparison pairs matched.
- Results. TTFT 794.56 to 815.15 ms (+2.59%). Prefill 646.16 to 629.70
  tok/s (-2.55%). Decode 69.56 to 69.57 tok/s. P95 decode latency
  1,834.88 to 1,836.36 ms. Peak board VRAM 753 MiB on both arms.
  Peak process working set increased from 427.02 to 526.77 MiB.
- Decision **REVERT from the default**. Keep the opt-in path for
  reproducing the experiment. Additional conversion and library state
  could offset savings from GEMM. The wall-clock result does not support
  changing the default precision.

[Raw trials](experiments/fp16-prefill-primary.json) |
[Golden evidence](experiments/fp16-prefill-golden.jsonl) |
[FP32 prefill CUDA trace totals](prefill-profile-kernels.csv)
