# Design

SPARK targets a single Llama-family dense decoder, one sequence at a time.

## Phase 1 weights

An export contains `manifest.json` and `weights.bin`. The manifest records the
format identifier, upstream model ID and commit, the source configuration, and
each tensor's shape, dtype, and byte offset from the start of the file. Matrices
are row-major, with dimensions [output, input]. Embeddings use [token, hidden].
Normalization weights are one-dimensional. Floats are IEEE 754 little-endian
FP32. The exporter widens FP16 and BF16 values exactly. For tied embeddings, the
LM head can reference the same file range as the embeddings.

With the actual model configuration, SPARK derives the KV footprint as
`2 * layers * kv_heads * head_dim * context * bytes_per_element`. The two
buffers are keys and values. The calculation uses the KV heads and head
dimension from the model export.
