# SPARK on the RTX 3050

These 1,440 x 1,080 Chromium captures come from the local FastAPI dashboard.
The dashboard read benchmark records captured on the RTX 3050 Laptop GPU.
The inference capture came from a native SPARK CUDA worker loading the local
Qwen3.5-0.8B Q4_K_M GGUF.

| View | Capture |
| --- | --- |
| Dashboard | [Dashboard](dashboard.png) |
| High-level UML component diagram | [PNG](architecture-uml.png) and [editable SVG](architecture-uml.svg) |
| Engine execution flow | [Architecture](architecture.png) |
| Live generation | [Streaming](live_inference.png) and [completed](live_inference_complete.png) |
| Sampled GPU utilization and board VRAM | [GPU monitoring](gpu_monitoring.png) |
| Matched llama.cpp comparison and decode throughput | [Benchmark comparison](benchmark_comparison.png) |
| Prompt throughput | [Prefill](prefill_throughput.png) |
| Archived optimization runs | [Optimization history](optimization_history.png) |
| End-to-end latency | [Performance summary](performance_summary.png) |

[Benchmark protocol](../BENCHMARKING.md) |
[Measured results](../RESULTS.md) |
[Raw bench files](../../benchmarks/)

To recapture, run the dashboard and use a local Chromium or Chromium
headless shell executable.

```powershell
node scripts/capture_showcase.cjs 'PATH\TO\chrome-headless-shell.exe' 'http://127.0.0.1:8765/' 'docs/showcase'
```
