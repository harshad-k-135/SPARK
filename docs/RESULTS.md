# Results

| Gate | Status | Evidence |
| --- | --- | --- |
| Bootstrap GPU allocation | Passed | CUDA 12.8 allocated and freed 4,096 bytes on the target GPU. See [environment](ENVIRONMENT.md). |
| Phase 1 offline CPU smoke test | Passed | Three synthetic prompts, 24/24 continuation tokens, and maximum first-logit absolute error of `3.78087e-7`. [Raw test output](../tests/golden/smoke-results.jsonl) |
| 1 — Official CPU parity | Blocked | The pinned official 1B snapshot is absent and direct downloads from the workspace failed. There is no Transformers golden reference JSON yet. |
| 2 — CUDA FP16 parity and baseline | Awaiting Gate 1 | |
| 3 — Optimized decode | Awaiting Gate 2 | |
| 4 — External comparison | Awaiting Gate 3 | |

The offline test exported a two-layer fixture containing F32, F16, and BF16
weights and checked the C++ runtime against a standalone mathematical
reference. It also checked byte-exact tokenization, greedy CLI output,
fixed-seed sampling repeatability, and context overflow rejection. The binaries
were built with MinGW GCC 14.2 and warnings treated as errors.

The Gate 1 target is `meta-llama/Llama-3.2-1B-Instruct` at
`9213176726f574b556790deb65791e0c5aa438b6`. Official-model top-1 agreement,
logit error, and 64-token continuation agreement remain pending.

| Runtime | Decode tokens/sec | TTFT | Prefill tokens/sec | Peak VRAM | GPU utilization |
| --- | --- | --- | --- | --- | --- |
| spark-fp16-v0 | Pending | Pending | Pending | Pending | Pending |
| spark-optimized | Pending | Pending | Pending | Pending | Pending |
| llama.cpp | Pending | Pending | Pending | Pending | Pending |

GPU throughput, latency, profiling, and perplexity measurements await the
corresponding earlier gates. See [hardware and toolkit](ENVIRONMENT.md) and
the [benchmark protocol](BENCHMARK_PROTOCOL.md).
