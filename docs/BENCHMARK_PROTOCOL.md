# Benchmark protocol

Freeze the workload, runtime revisions, and metrics before collecting GPU
performance data. Keep the configuration and raw trial data with the results.

## Workload

- Run one sequence on one NVIDIA GeForce RTX 3050 6 GB Laptop GPU, with a
  2,048-token context and FP16 KV. Set greedy sampling and force exactly 128 new
  tokens, including on EOS. Disable prefix caching and reuse the model across
  trials with an empty KV cache for each trial.
- Use fixed inputs of exactly 128, 512, and 1,024 tokens, including BOS. Store the
  prompt text, token IDs, and hash in the raw results. Confirm exact token IDs
  after each runtime's tokenization. Use identical raw prompts, with no
  runtime-supplied chat template.
- The primary decode workload is the 512-token prompt followed by 128 new
  tokens. The 128- and 1,024-token cases show sensitivity to context.
- Compare SPARK FP16 with llama.cpp using FP16 GGUF converted from the same
  source revision. Record the conversion command, artifact hashes, and exact
  llama.cpp commit. Build llama.cpp with CUDA and fully offload the model.
  Record batch, microbatch, CPU threads, CUDA Graph and attention settings.
  If tuning settings, keep the calibration results and freeze the chosen
  settings before collecting scored trials.
- Use the official Meta Llama 3.2 1B Instruct snapshot at
  `9213176726f574b556790deb65791e0c5aa438b6`. For Phase 3, also record the
  official 3B Instruct revision, quantization format and conversion details.
  A cross-model throughput comparison cannot establish a same-model speedup.

## Execution

Plug in the laptop, select the performance power mode, and close other GPU
applications. Record OS, driver, CUDA toolkit and runtime, CPU, GPU, compute
capability, power mode, VRAM at idle, and runtime build revisions. Record
temperature and clocks to expose thermal throttling.

Use five warm-up requests followed by ten measured requests per runtime,
model, and prompt length. Rotate runtime order between independent sessions.
Measure cold load separately with one new process per runtime per session and
document filesystem cache state. Warm-up timings are reported separately.
Run the acceptance gates before timing a build. Use the same configuration
for profiling and record profiler-assisted timings separately.

## Timings and telemetry

- Prefill tokens/sec divides input tokens by elapsed prefill time through
  first logits readiness. TTFT starts when the warmed runtime receives the
  request and ends at availability of the first sampled token, including
  tokenization, prefill, and sampling. Exclude cold load and API transport.
- Decode tokens/sec divides 127 by elapsed time from first sampled token to
  last sampled token. Record decode latency over the same boundary.
- Sample board utilization, framebuffer used, power, temperature, and graphics
  and memory clocks via NVML at 100 ms intervals. Record raw timestamps, idle
  usage, peak sampled board VRAM, and mean GPU utilization per trial. Mark
  unavailable counters and timings explicitly.
- Report median, IQR (Q3 minus Q1), and P95 for each metric. Use linear
  interpolation at sorted index `(n - 1) * q`. Preserve every trial, label
  invalid trials with a reason, and retain them when repeating the workload.
- External tools without equivalent token IDs or metric boundaries can be
  reported as secondary measurements with their own definitions. Omit them
  from claims about matched workloads.

Commit raw results under `benchmark/results/` and summarize them in
`docs/RESULTS.md`. Report quality alongside performance and link each
optimization trial to its quality check and baseline commit.
