# Experiments

## CUDA Graph decode

- Baseline `733e641` measured 40.46 median decode tokens/sec on a 512-token
  prompt with 128 forced output tokens. The unprofiled P95 decode latency
  was 3,433.72 ms.
- A representative CUDA event trace attributed 20.70 ms to recurrent
  layers, attention layers, MLP, and the final projection combined.
  The CUDA API trace shows repeated dispatches through the 24-layer decoder.
- Hypothesis. Replaying the decode launches as a graph improves median
  throughput on the primary workload without changing generated IDs.
- Change. Compare per-operation dispatch with CUDA Graph replay using either
  one fixed extent or power-of-two attention extents. Both pass the current
  position through a device-side scalar. Prefill stays on the baseline path.
- Bucket result. 46.08 median decode tokens/sec, 2,760.61 ms P95 decode
  latency, and a 3,088,575,744-byte device arena. All 20 parity cases passed.
  Keep the bucket variant for this workload. [Raw trials](../benchmark/results/spark-qwen-graph-bucket.json)
- Single-graph result. 45.92 median decode tokens/sec, 2,774.40 ms P95 decode
  latency, and all 20 parity cases passed. Keep this mode available for
  reproducing the comparison. [Raw trials](../benchmark/results/spark-qwen-graph-single.json)

## Packed GGUF projections

- Baseline `733e641` measured 40.46 median decode tokens/sec with a
  3,088,575,744-byte arena. The representative CUDA event profile spent
  6.23 ms in the output projection and final norm.
- Hypothesis. Reading the original Q4_K, Q5_K, Q6_K, and Q8_0 blocks from
  device memory improves median decode throughput by reducing projection
  weight traffic.
- Change. Each warp decodes one packed row and accumulates directly in FP32.
  F32 tensors still use cuBLAS when projected. Graphs were disabled.
- Result. 51.21 median decode tokens/sec, 3,270.22 ms P95 decode latency, and
  a 625,908,992-byte arena. All 1,280 continuation IDs matched. The maximum
  first-step logit-gap error was 0.001083.
- Decision. Keep the packed path for this GGUF and use it in the dashboard.
  [Raw trials](../benchmark/results/spark-qwen-packed.json)
