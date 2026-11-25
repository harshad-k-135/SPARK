# Limitations

The verified configuration uses the supplied Qwen3.5-0.8B text GGUF,
one sequence, and one RTX 3050 Laptop GPU. Native CLI context is capped
at 4,096 tokens. The streaming worker uses 2,048 tokens. The local
dashboard uses the single GGUF configured at startup. Other Qwen sizes,
vision, speculative multi-token prediction, and cross-request batching
remain outside this release's validated scope.

llama.cpp led prefill, TTFT, and decode throughput at every tested
prompt length. SPARK's closest measured decode result was 117.77 versus
124.65 tokens/sec at 128 input tokens. SPARK had lower sampled board
VRAM and process working set than the pinned comparison. These results
cover the recorded model, device, power mode, and transport boundaries.
They do not establish a hardware ceiling.

The FP16 prefill experiment passed generation parity with slower TTFT
and higher sampled process RAM, so FP32 remains the default. The current
tuned engine still runs recurrent state updates and causal attention
per token within a prefill chunk. The final CUDA event snapshot places
recurrent layers ahead of the MLP and output projection on one profiled
decode step. A steady-state kernel-counter trace is required to separate
memory stalls from occupancy and instruction pressure.

System monitoring sampled the whole GPU board and the selected runtime
process working set. The CUDA sanitizer result covers the operation suite.
See [measurement boundaries](BENCHMARKING.md) and the
[remaining performance work](ROADMAP.md).
