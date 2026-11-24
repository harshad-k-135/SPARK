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
- Status. Unit tests and a 16-token smoke check passed for both variants.
  Full correctness and timings are pending. Graph replay is opt-in.
