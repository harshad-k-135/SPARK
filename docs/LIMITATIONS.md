# Limitations

The native executable inspects, tokenizes, and generates text on CPU from the
supplied Qwen3.5 GGUF. The CUDA backend also generates natively and is
verified against the FP32 reference. The local dashboard streams the
packed CUDA worker and displays native benchmark results. The baseline
expands the trunk weights to FP32 and processes prefill one token at a time.
The model scope is text-only, batch one, one RTX 3050, and up to 4,096 tokens.
Vision and the auxiliary multi-token prediction block are outside the
initial implementation scope.

The staged llama.cpp build passed GPU enumeration and full offload. Matched
external GPU trials are pending. Windows denied browser launch inside the
sandbox, so live dashboard screenshots are unavailable there.
