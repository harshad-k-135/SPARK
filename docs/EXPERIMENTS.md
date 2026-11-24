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
  Keep the bucket variant available for this workload. The single-graph
  comparison is still running. [Raw trials](../benchmark/results/spark-qwen-graph-bucket.json)
