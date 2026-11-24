# Limitations

The official Phase 1 parity test needs the approved Meta Llama 3.2 1B Instruct
snapshot and a local PyTorch/Transformers environment. The synthetic test
covers the export and CPU execution path on a small fixture. The official
model's logits, generated text, and tokenizer behavior still need verification.

The FP32 CPU build loads the full widened weights into host RAM and processes
the prompt one token at a time. It accepts raw prompts and returns generated
token bytes as text. The CMake preset has yet to be exercised here because
CMake was unavailable on PATH.

CUDA inference, INT4, GPU profiling, external benchmarks, the control API, and
stretch items await their prerequisite gates. No comparative performance
measurements are available.
