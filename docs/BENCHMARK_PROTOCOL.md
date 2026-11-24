# Qwen benchmark protocol

Freeze the GGUF SHA-256, upstream revision when verifiable, runtime revisions,
and trial configuration before scoring. Use the same Qwen3.5-0.8B Q4_K_M file,
token IDs, 2,048-token context, single sequence, greedy sampling, and 128
forced output tokens on both runtimes. Use prompts of exactly 128, 512, and
1,024 tokens. The 512-token case is the primary decode workload.

Verify CPU and GPU correctness before collecting GPU timings. Fully offload
to the RTX 3050 and document CUDA, driver, power mode, runtime settings, and
offloaded layer counts. Plug in the laptop and close other GPU applications.
Run five warm-ups and ten measured trials per configuration. Record cold load
separately and reset each trial's sequence state.

Report TTFT from warmed-runtime request arrival through the first sampled
token. Report prefill tokens/sec through first-logits readiness. Divide 127
by the elapsed time from first to last sampled token for decode tokens/sec.
Sample GPU utilization, board memory, power, temperature, and clocks at
100 ms intervals. Report median, IQR, and P95 using linear interpolation at
sorted index `(n - 1) * q`. Save raw data under `benchmark/results/`.
