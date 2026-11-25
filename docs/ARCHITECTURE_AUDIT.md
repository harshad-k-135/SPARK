# Architecture audit

Reviewed at SPARK commit `1fa4a53` on Windows build 26300, Intel Core
i5-13450HX (16 logical CPUs), and RTX 3050 6GB Laptop GPU under WDDM.
The current scheme is Balanced. NVIDIA driver 596.08 advertises CUDA 13.2.
The installed toolkit is 12.8.61. SPARK builds with MinGW GCC and compiles
kernels for compute 8.6 through NVRTC. CMake and the MSVC compiler are absent
from the shell PATH. The pinned llama.cpp b10159 package enumerates the same
GPU as CUDA0. The resident SPARK worker appeared in the NVIDIA compute list.

## Pipeline and ownership

| Stage | CPU responsibilities | CUDA responsibilities |
| --- | --- | --- |
| Load | Validate GGUF metadata, dimensions, offsets and encodings, build byte-BPE tables, read each tensor through a staging vector | Compile and load kernels, allocate one arena, upload packed tensors |
| Prefill | Tokenize, reset sequence, enqueue the decoder once per prompt token | Gather embedding, run 24 causal layers, produce logits for the last prompt token |
| Decode | Submit token, wait at sampling boundary, validate and sample logits, detokenize and stream | Run recurrent and attention layers, apply SwiGLU, project to 248,320 logits |
| Benchmarks | Node drives SPARK over local pipes and pinned llama.cpp over loopback SSE. FastAPI serializes dashboard runs and stores results in SQLite | One engine at a time, with NVIDIA board telemetry sampled externally |

The standalone CPU engine expands GGUF weights to floats and provides an
independent numerical path. The CUDA worker selects its backend explicitly
and fails on initialization errors. Packed projections handle Q4_K, Q5_K,
Q6_K and Q8_0 directly, with FP32 activations and accumulation. F32 projections
use cuBLAS on the same non-default stream. Qwen uses 18 Gated DeltaNet layers
with convolution and matrix state, plus six gated grouped-query attention
layers. The optional MTP block is outside the forward pass.

The device arena takes a 512 MiB free-memory reserve, bumps on 256-byte
boundaries, and holds weights, scratch, recurrent state and all KV storage.
At 2,048 tokens, the six FP32 KV caches reserve 48 MiB in
[position][KV head][feature] order. Reset clears convolution and DeltaNet
state, and limits attention visibility with a fresh position counter.
Transfer uses a pinned host logits buffer. Single-token decode uses the
preallocated scratch; the optional graph path captures and allocates each
new bucket lazily. The dashboard uses packed weights with graphs disabled
in the audited revision.

## Bottlenecks and prioritized experiments

1. **Serial prefill.** The existing matched 512-token run took 6.93 s to first
   token versus 151 ms. SPARK invokes all 24 layers for each prompt token and
   rereads packed projection weights. Batch projections across positions in a
   chunk while preserving causal attention and recurrent update order.
2. **Packed projections and final vocabulary projection.** Existing
   projections decode one weight per lane per loop iteration, including
   repeated scale and nibble work. The tied output matrix is about 199 MiB
   for this artifact and is read for every emitted token. Profile the packed
   path before selecting changes. Test shape-specific reuse of packed blocks
   and scale metadata with full-logit parity.
3. **Kernel dispatch and sampling boundary.** A decode step submits hundreds
   of dependent operations. Test bucketed graphs with packed weights as a
   single controlled change. The sampling boundary transfers 993,280 bytes
   per emitted token and scans the vector on the CPU. Test deterministic GPU
   greedy selection separately if measurements justify it.

The comparison logs expose sampled board utilization, power and VRAM.
These measure the whole GPU and can miss short spikes. Use CUDA events and an
unprofiled control for stage attribution. The old Nsight Systems report
covers the FP32 path only. Nsight Compute previously returned
`ERR_NVGPUCTRPERM`, so occupancy and hardware bandwidth remain unverified.
A fused attention candidate should be evaluated at longer contexts after
the projection and launch measurements. Batch-one prompt processing has
priority over multi-user scheduling.

See [design](DESIGN.md), [historical results](RESULTS.md), and
[benchmark protocol](BENCHMARK_PROTOCOL.md).

## Packed baseline stage trace

A representative short-prompt decode step emitted CUDA event times of
7.48 ms MLP, 5.43 ms recurrent, 4.45 ms final norm and vocabulary
projection, and 1.45 ms full-attention blocks. This is one profiled
step and includes the effects of dispatch between markers. It is distinct
from protocol throughput. High board utilization (97.66% median on the
primary baseline) indicates a busy device over the sample windows.
Achieved memory bandwidth, instruction occupancy, and Tensor Core usage
require kernel counters or disassembly to establish.

[Raw packed CUDA events](../benchmarks/packed-baseline-profile.jsonl)
