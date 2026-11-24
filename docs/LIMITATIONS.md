# Limitations

The native executable inspects, tokenizes, and generates text on CPU from the
supplied Qwen3.5 GGUF. GPU execution, the API, and the frontend are pending.
The CPU model expands the trunk weights to FP32 and processes prefill one
token at a time.
The model scope is text-only, batch one, one RTX 3050, and up to 4,096 tokens.
Vision and the auxiliary multi-token prediction block are outside the
initial implementation scope.

There are no matched GPU benchmark measurements. The external CPU smoke
test provides compatibility evidence for the model file.
