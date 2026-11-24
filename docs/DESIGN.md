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

The artifact uses F32, Q4_K, Q5_K, Q6_K, and Q8_0. CPU parity will check
dequantized weights, project-and-normalize operations, both attention paths,
and the tokenizer before checking full-model autoregressive output. Device
execution will hold the packed weights plus recurrent state, FP16 KV for the
six full-attention blocks, activations, and scratch. The planner must check
actual free VRAM with a 512 MiB reserve and cap context at 4,096 tokens.
