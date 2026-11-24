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

## Tokenization

Export `tokenizer.json` with `tools/export/export_tokenizer.py`. The binary
table contains byte strings indexed by token ID, listed added special tokens,
ranked BPE merges, and Unicode character-class ranges derived from the export
interpreter's Unicode data. C++ applies the Llama 3 pre-tokenization rules,
then byte BPE within each piece. Decoding concatenates original token bytes.
The runtime supplies the model's BOS before encoding raw prompts. Model files
and tokenizer data stay outside the source repository.

## CPU execution

`Session` handles one sequence and preallocates FP32 keys, values, and scratch.
Both KV buffers are indexed [layer][KV head][position][head dimension]. Each
query head reads its associated KV head over the causal prefix. The forward
pass processes prefill one token at a time and evaluates the LM head only for
the final input token. Independent projection rows can run in parallel using
OpenMP with the same FP32 reduction order in each row.
