# Reproduce the CUDA comparison

The baseline uses the local Qwen3.5-0.8B Q4_K_M GGUF, the RTX 3050 Laptop GPU,
SPARK's native CUDA worker and llama.cpp b10159 (`f95de9776`). The same
artifact, prompt IDs, 2,048-token context, FP32 KV, one sequence and 128
forced greedy output IDs are used throughout. The three prompts have 128,
512 and 1,024 tokens. The 512-token workload is the primary comparison.
The verified artifact SHA-256 is recorded in each JSON report.

Use AC power, record the Windows power scheme, and close other GPU workloads.
The recorded baseline used Balanced with the machine on AC. The comparison
requires llama.cpp to enumerate the RTX 3050 and confirm full CUDA offload.
It uses batch cap 1,024, microbatch 512 and Flash Attention auto where
compatible with the chosen precision.

From the repository root, build SPARK and prepare the CUDA comparison
binary using [the setup steps](../README.md). Then run:

```powershell
$Model = 'C:\path\to\Qwen_Qwen3.5-0.8B-Q4_K_M.gguf'
node benchmark/comparison/compare_cuda.cjs build/cpu/spark.exe build/llama-b10159-cuda13/llama-server.exe $Model build/comparison-new.json --batch-prefill --bucket-graph --q4-block-gemv --q6-block-gemv
node benchmarks/summarize.cjs build/comparison-new.json
```

Add `--primary` for only the 512-token case. Add `--batch-prefill` to
evaluate the 128-token chunk path and `--bucket-graph` to evaluate packed
decode graph replay. Add `--q4-block-gemv` and `--q6-block-gemv` for the
specialized packed projection paths. Omit all four flags for the serial
control.
Use a fresh output filename for each run. If benchmarking an executable
from another checkout, set `SPARK_SOURCE_COMMIT` to its source commit.

The driver runs five warm-ups and ten measured trials per engine and
workload. Only one inference process holds the GPU during timing. TTFT and
total latency are measured by the same Node client, from request submission
to receipt of token 1 and token 128 respectively. Decode throughput counts
the 127 intervals from the first to last delivered token. The transports
are local pipes for SPARK and SSE for llama.cpp. Prefill uses each engine's
internal timing boundary. The report stores per-launch process-ready time
separately. It includes model loading and initialization with the filesystem
cache in its observed state. It is a single observation per workload and
runtime.

NVIDIA board utilization, memory, temperature, power and clocks are sampled
at nominal 100 ms intervals. Windows process working set and cumulative CPU
time are sampled at nominal 200 ms intervals. CPU utilization is normalized
to all logical processors. Peak RAM and VRAM are sampled peaks. Board
tokens/joule integrates successive power measurements over at least 80% of
the trial and scales by time coverage. It includes total GPU board power.
CPU, laptop and wall power are outside that measurement.

JSON retains trial-level data, outcome IDs and telemetry. Summaries contain
median, mean, sample standard deviation, IQR, P95 and valid sample count.
Review exact token matches, temperature, clocks and distributions before
making comparisons across independent runs.

[Baseline summary](../benchmarks/baseline.md) |
[Raw baseline](../benchmarks/baseline.json) |
[Correctness and experiment history](../benchmarks/optimization_log.md)

The final CUDA reference gate can be rerun with the FP32-expanded local
artifact produced by the CPU reference setup.

```powershell
powershell -File scripts/verify_qwen_cuda.ps1 -Model $Model -ReferenceModel build/qwen-reference-f32.gguf -Spark build/cpu/spark.exe -BatchPrefill -Graph bucket -Q4BlockGemv -Q6BlockGemv
node scripts/stress_qwen_cuda.cjs build/cpu/spark.exe $Model --bucket-graph --q4-block-gemv --q6-block-gemv
```

## Prepare the external CUDA comparison

Stage the official b10159 Windows x64 CUDA 13.3 package and its companion
runtime archive with the verification script. The native worker builds
and runs independently of these files.

```powershell
powershell -File benchmark/comparison/prepare_llama_cuda.ps1 -ReferenceBin 'PATH\TO\llama-b10159-bin-win-cuda-13.3-x64' -RuntimeZip 'PATH\TO\cudart-companion.zip'
```
