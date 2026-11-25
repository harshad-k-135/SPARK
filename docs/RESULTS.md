# Results

| Gate | Status | Evidence |
| --- | --- | --- |
| CUDA runtime bootstrap | Passed | RTX 3050 detected. CUDA 12.8 allocated and freed 4,096 bytes. |
| Qwen GGUF import | Passed | Native C++ inspection validated the supplied file layout and classified all blocks. |
| Packed weight decoding | Passed | Fixed FP32, FP16, BF16, Q4_K, Q5_K, Q6_K, and Q8_0 cases passed. The native inspector read and decoded the first real embedding row. |
| Qwen tokenizer smoke | Passed | Eight prompts matched the external reference token for token and round-tripped byte for byte. [Raw output](../benchmark/results/qwen-local-smoke/native-tokenizer.jsonl) |
| Gated DeltaNet recurrence | Passed | A two-step CPU test checked state decay, the delta update, and query readout against hand-computed values. |
| Native Qwen CPU generation | Smoke passed | Native output matched the independent FP32 comparison on 3/3 prompts and 33/33 generated IDs. |
| 1 — Qwen CPU parity | Passed | All 20 first tokens and 1,280 greedy continuation IDs matched. Maximum first-step logit-gap error with FP32 KV on both sides was 0.001041. |
| Native Qwen CUDA generation | Smoke passed | CUDA generated `4` on the RTX 3050 from the supplied GGUF. The arena used 3,088,575,744 bytes at 2,048 context. Fixed-seed CUDA operation tests passed. |
| 2 — Qwen CUDA parity and baseline | Passed | All 20 first tokens and 1,280 continuation IDs matched. Maximum first-step logit-gap error was 0.001081. Baseline and profiler evidence recorded below. |
| 3 — Optimized Qwen decode | Passed | Packed GGUF CUDA raised median native decode throughput from 40.46 to 51.21 tokens/sec. The bucketed graph variant reached 46.08. Both passed full parity. |
| 4 — GPU comparison and control API | Passed | The one-command harness compared both runtimes at 128, 512, and 1,024 prompt tokens. The dashboard loads all three workloads and links the raw trials. |

Target file `Qwen_Qwen3.5-0.8B-Q4_K_M.gguf`, SHA-256
`fb044e93939a70469c905781334f5de1e6c8b608ced6cbc8c9249bd4127d9526`.
The source revision of the supplied GGUF is unverified. The file is 579,615,840
bytes and contains 335 tensors in 25 blocks, including 18 recurrent blocks,
six attention blocks, and one auxiliary block. Hidden width is 1,024 and the
stored vocabulary has 248,320 entries.
The first Q6_K embedding row decoded to finite values from `-0.0759277` to
`0.0695343`. A deliberately truncated copy of the GGUF header was rejected.

The original [external Qwen API smoke test](../benchmark/results/qwen-local-smoke/02-qwen-api.png)
ran on CPU. The matched GPU comparison is recorded below.

The staged b10159 build (`f95de9776`) detected the RTX 3050 as `CUDA0`
after installing the official CUDA 13.3 runtime bundle beside its binaries.
`llama-cli.exe` offloaded all 26 reported layers, returned `4` for the
arithmetic prompt, and exited with code zero.
[GPU smoke output](../benchmark/results/qwen-local-smoke/llama-cuda-enumeration.json)

[Sandbox verification chart](../benchmark/results/qwen-local-smoke/native-qwen-coverage.svg)

The native CPU smoke used the supplied GGUF as SPARK's input. The external
reference used a separate FP32 GGUF created from the same packed weights with
`llama-quantize --allow-requantize --pure ... F32 6` at b10159. Each runtime
received identical prompt IDs. The maximum first-step logit-gap difference
over the reference's top eight tokens was 0.004061. The packed CPU reference
diverged on the multilingual prompt and had a higher first-step error.
[Raw smoke output](../benchmark/results/qwen-local-smoke/native-generation-f32.jsonl)
· [Logit gap chart](../benchmark/results/qwen-local-smoke/native-generation-f32.svg)

The first full run matched all 64 greedy IDs on each of 20 prompts from 16
through 1,500 tokens, satisfying the CPU gate. The maximum first-step gap
difference across the 248,320 logits exceeded 0.05 at 768 and 1,500 tokens
when the external comparison used FP16 KV. Repeating the 768-token reference
with FP32 K and V and flash attention disabled reduced that difference from
0.125093 to 0.000602 using the same native logits.
[Initial full-run output](../benchmark/results/qwen-local-smoke/native-generation-fp16-kv-reference.jsonl)

With FP32 K and V on both sides, all 20 continuations again matched and the
maximum first-step logit-gap error was 0.001041 over 248,320 logits per prompt.
[Matched-precision full-run output](../benchmark/results/qwen-local-smoke/native-generation-fp32-kv-reference.jsonl)

Native CUDA also matched every continuation with FP32 KV. Its largest
first-step logit-gap error was 0.001081 across the full vocabulary.
[CUDA full-run output](../benchmark/results/qwen-local-smoke/native-cuda-generation-fp32-kv-reference.jsonl)

## Native CUDA baseline

RTX 3050 6 GB Laptop GPU, driver 596.08, CUDA 12.8, Windows Balanced power
scheme on AC. The session used 512 fixed prompt IDs, forced 128 output tokens,
2,048-token context, five warm-ups, and ten measured requests. First-token
timing starts at prefill because the harness passes pretokenized input.

| Metric | Median | IQR | P95 |
| --- | ---: | ---: | ---: |
| Decode tokens/sec | 40.46 | 0.75 | 42.27 |
| Prefill tokens/sec | 70.18 | 7.45 | 70.81 |
| First token from prefill (ms) | 7,296.86 | 858.56 | 8,865.95 |
| Decode latency (ms) | 3,139.06 | 58.13 | 3,433.72 |

The FP32 arena used 3,088,575,744 bytes. The short-prompt CUDA event profile
recorded 5.76 ms for recurrent layers, 1.54 ms for attention layers, 7.17 ms
for MLP, and 6.23 ms for the output projection and final norm. Nsight Systems
saved a CUDA trace. Nsight Compute returned `ERR_NVGPUCTRPERM` because GPU
performance counters were unavailable to the current account.
[Raw baseline](../benchmark/results/spark-qwen-v0.json)
· [Nsight Systems report](profiles/spark-qwen-v0.nsys-rep)

## Bucketed CUDA Graph result

On the same model, device, prompt, and trial protocol, median decode
throughput rose to 46.08 tokens/sec (+13.90%). Median prefill was 70.85
tokens/sec. P95 decode latency was 2,760.61 ms. All 1,280 generated IDs
matched, with a maximum first-step logit-gap error of 0.001081.
This comparison is against the native SPARK baseline.
[Raw trials](../benchmark/results/spark-qwen-graph-bucket.json)
· [CUDA Graph parity](../benchmark/results/qwen-local-smoke/native-cuda-bucket-parity.jsonl)

The single-graph variant also passed all 20 prompts. Median decode was
45.92 tokens/sec, and P95 decode latency was 2,774.40 ms. The bucketed
variant had the higher measured median of the two graph modes.
Both graph measurements used the executable built from `b87af6d`.
The single-graph raw metadata also records the later documentation-only
checkout commit present at measurement time.
[Single-graph raw trials](../benchmark/results/spark-qwen-graph-single.json)
· [Native throughput chart](../benchmark/results/spark-qwen-native-throughput.svg)

## Packed CUDA result

With graph replay disabled, the packed GGUF path reached 51.21 median
decode tokens/sec (+26.57% versus the expanded FP32 baseline), 74.43 median
prefill tokens/sec, and 3,270.22 ms P95 decode latency. Its arena was
625,908,992 bytes. All 1,280 continuation IDs matched and the maximum
first-step logit-gap error was 0.001083.
[Raw trials](../benchmark/results/spark-qwen-packed.json)
· [Packed CUDA parity](../benchmark/results/qwen-local-smoke/native-cuda-packed-parity.jsonl)

## Dashboard smoke run

The local server returned HTTP 200 for the dashboard and JavaScript.
The native worker returned `4` for the arithmetic prompt and produced
16 tokens on a second request using the same loaded worker. The runs API
returned the baseline and graph data and accepted a new benchmark request.
The dashboard-triggered benchmark completed and stored all ten measured
trials. Median decode throughput was 46.33 tokens/sec. Windows denied
launching the interactive browser from the sandbox, so the frontend was
verified through its HTTP assets and API.
[Raw generation events](../benchmark/results/qwen-local-smoke/dashboard-generation.jsonl)
· [Dashboard-triggered raw trials](../benchmark/results/spark-qwen-dashboard-repeat.json)

The dashboard was restarted with the packed CUDA worker and again streamed
`4` from the supplied Qwen model. The status endpoint confirmed that the
packed model remained loaded and the runs endpoint returned the packed
benchmark results. [Packed dashboard events](../benchmark/results/qwen-local-smoke/dashboard-packed-generation.jsonl)

## Matched CUDA comparison with llama.cpp

Both engines used the same Qwen Q4_K_M GGUF and pretokenized requests, a
2,048-token context, 128 forced output tokens, greedy sampling, and FP32
KV. Each runtime ran alone on the RTX 3050. Each workload used five
warm-ups and ten measured trials per runtime. SPARK was built from
`8e6c1b2` with the CUDA 12.8 toolkit. The b10159 llama.cpp binary
(`f95de9776`) used its CUDA 13.3 runtime and offloaded all 26 reported
layers. The NVIDIA driver was 596.08 and Windows used the Balanced power
scheme.

| Input tokens | SPARK decode tok/s | llama.cpp decode tok/s | SPARK first token | llama.cpp first token | SPARK peak VRAM | llama.cpp peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 52.79 | 130.12 | 1,682.68 ms | 101.32 ms | 683 MiB | 761 MiB |
| 512 | 50.39 | 123.76 | 6,929.95 ms | 150.64 ms | 683 MiB | 761 MiB |
| 1,024 | 48.49 | 119.88 | 14,377.17 ms | 246.59 ms | 683 MiB | 761 MiB |

The table shows medians. llama.cpp led on decode, first-token latency,
and prefill in every measured workload. SPARK used 78 MiB less sampled
peak board memory in each workload. All 3,840 generated token IDs
matched across the 30 measured pairs. The raw report includes IQR,
P95, prefill throughput, GPU utilization, clocks, power, temperature,
full outputs, and each runtime's cold-load timing.
[Raw trials](../benchmark/results/spark-vs-llama-cuda.json)
· [Decode dashboard capture](../benchmark/results/dashboard/spark-dashboard-decode.png)
· [VRAM dashboard capture](../benchmark/results/dashboard/spark-dashboard-vram.png)
