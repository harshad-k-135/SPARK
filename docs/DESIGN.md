# Design

SPARK targets Qwen3.5-0.8B text inference at batch one. The supplied GGUF v3
contains an embedding, a final norm, 24 text-decoder blocks, and an auxiliary
multi-token prediction block. Text generation uses the 24 main blocks.
Eighteen blocks use Gated DeltaNet with a causal convolution and recurrent
state. Six use gated grouped-query attention with Q/K normalization and
partial rotary embedding. Both paths feed an MLP and residual connection.
[Upstream model configuration](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)

The native C++ reader validates GGUF metadata, tensor geometry, alignment,
payload extents, and overlaps. GGUF dimensions are ordered from the contiguous
axis outward. For a dense projection, the first dimension is the number of
inputs and the second is the number of output rows. The loader records
absolute payload offsets and preserves packed tensors in their file layout.
[GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

The artifact uses F32, Q4_K, Q5_K, Q6_K, and Q8_0. SPARK decodes packed
rows to FP32 for CPU verification. Its byte BPE tokenizer reads vocabulary,
token classes, and ranked merges from the GGUF. The pre-tokenizer attaches
combining marks to words and splits digits individually. Ordinary prompts
keep template markers literal unless special-token parsing is requested.

CPU parity covers project-and-normalize operations, both attention paths,
and full-model autoregressive output. Device
execution will hold the packed weights plus recurrent state, FP16 KV for the
six full-attention blocks, activations, and scratch. The planner must check
actual free VRAM with a 512 MiB reserve and cap context at 4,096 tokens.

The CPU DeltaNet recurrence stores each head's state as a matrix with
contiguous value rows. A step decays the previous state, measures the value
prediction error at the key, and adds the gated key–error outer product. The
query then reads the updated state. The recurrence takes pre-normalized query
and key vectors and precomputed scalar gates.

## CUDA baseline

The Windows MinGW build uses NVRTC to compile the bundled CUDA C++ kernels
and launches them on one explicit non-default CUDA stream. The FP32 baseline
widens the same packed GGUF weights used by the CPU parity path. It keeps
those weights, Gated DeltaNet state, causal convolution history, the six
FP32 KV caches, and scratch in one load-time device arena. The planner checks
`cudaMemGetInfo` with a 512 MiB margin. KV rows are indexed by position,
KV head, and feature. DeltaNet state is indexed by head, key feature, and
value feature.

cuBLAS performs dense projections. CUDA kernels handle normalization,
residuals, convolution, Gated DeltaNet state, gated attention, and SwiGLU.
The runtime copies logits to pinned host memory and waits on its stream
only when sampling produces a host-visible token. Repeated benchmark
requests reset recurrent state between trials and reuse the arena.
