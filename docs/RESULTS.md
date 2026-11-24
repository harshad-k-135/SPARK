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

## Local Qwen compatibility smoke test

The supplied Qwen3.5-0.8B Q4_K_M GGUF ran through the external llama.cpp server
at build b10159 (`f95de9776`). The upstream model revision for this GGUF is
unknown. Its file SHA-256 is
`fb044e93939a70469c905781334f5de1e6c8b608ced6cbc8c9249bd4127d9526`.
The server used one slot and a 2,048-token context. The health check and
model-list endpoint succeeded. A greedy chat request asking for the sum
of 2 + 2 returned `4`. [API evidence](../benchmark/results/qwen-local-smoke/02-qwen-api.png)

The bundled browser UI served its HTML, JavaScript, and stylesheet with HTTP
200. Browser rendering and interaction remain unverified after the Windows
desktop capture and headless browser attempts failed.
[Frontend HTTP evidence](../benchmark/results/qwen-local-smoke/03-frontend-http.png)

The CUDA 12.8 runtime saw the RTX 3050 and allocated 4,096 bytes. The external
CUDA 13.3 build listed zero GPU devices and its CUDA backend imports
`cublas64_13.dll`. The Qwen chat request ran on CPU.
[Device evidence](../benchmark/results/qwen-local-smoke/01-cuda-device.png)

Native SPARK rejected the GGUF and passed its existing CPU fixture again.
Gate 1 remains blocked pending the official Llama snapshot and reference.
[Native evidence](../benchmark/results/qwen-local-smoke/04-spark-native.png)

The sandbox rerun rebuilt the CPU executable and reproduced all three synthetic
logit errors. The Qwen chat request again returned `4`. A separate LibreWolf
headless capture attempt failed to map its browser framebuffer. Gate 1 still
reports a missing reference JSON.

- [Synthetic first-logit error comparison](../benchmark/results/qwen-local-smoke/native-cpu-logit-error.svg)
- [Coverage across the available execution paths](../benchmark/results/qwen-local-smoke/sandbox-test-coverage.svg)
