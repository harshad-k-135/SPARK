# Performance roadmap

- Reduce prefill launch count and tile the two GQA kernels. The 512-token
  Nsight trace attributed 157.38 ms to those kernels and 102.51 ms to
  recurrent update and convolution. Preserve causal state and exact greedy
  output before timing any replacement.
- Profile sustained throughput and memory stalls of Q4_K and Q6_K GEMV.
  CUDA event and Nsight Systems traces are available. Nsight Compute
  hardware counters require permissions that were unavailable in this
  environment.
- Evaluate GPU greedy selection separately from projection changes.
  Each decode step still returns the full FP32 logits vector to the host.
- Evaluate tiled attention and KV precision independently on long context.
  Attach golden parity and memory data to each experiment.
- Extend model support and multi-user scheduling after the batch-one
  Qwen path has reproducible correctness and performance data.

[Measured baseline](../benchmarks/baseline.md) |
[Audit priorities](ARCHITECTURE_AUDIT.md)
