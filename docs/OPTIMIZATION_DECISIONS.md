# Optimization decisions

- Keep packed CUDA projections for the target GGUF. The previous controlled
  test improved decode from 40.46 to 51.21 tokens/sec and reduced the device
  arena from 3,088,575,744 to 625,908,992 bytes. The packed representation
  avoids allocating a permanent FP32 copy of the full model.
- Keep single and bucketed CUDA Graph options to reproduce launch scheduling
  experiments. Graphs are independent of the weight encoding and prefill
  path. A new bucket incurs a capture and instantiation cost.
- Preserve the single-token projection and FP32 CPU paths as correctness
  controls. The head-to-head uses one sequence, so prompt batching operates
  across the positions of that sequence.

- Use FP32 chunked prefill for the tuned path. Timed FP16 GEMM operands
  regressed median TTFT by 2.59% and increased sampled process RAM.
- Use Q4_K and Q6_K block GEMV for the tuned path. Both preserve the
  original on-device weight layout and passed full-vocabulary checks
  and exact greedy continuation parity.
- Enable bucketed decode replay in the tuned worker. Serial launch remains
  available to separate launch overhead from matrix math.

[Historical experiments](EXPERIMENTS.md) |
[Current experiments](../benchmarks/optimization_log.md)
