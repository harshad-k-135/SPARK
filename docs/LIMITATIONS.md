# Limitations

The native executable inspects, tokenizes, and generates text on CPU from the
supplied Qwen3.5 GGUF. The CUDA backend also generates natively and is
under full-model parity testing. The API and frontend are pending. Both
backends expand the trunk weights to FP32 and process prefill one token at
a time.
The model scope is text-only, batch one, one RTX 3050, and up to 4,096 tokens.
Vision and the auxiliary multi-token prediction block are outside the
initial implementation scope.

There are no matched GPU benchmark measurements. The external CPU smoke
test provides compatibility evidence for the model file.
