# CUDA baseline

Measured on NVIDIA GeForce RTX 3050 6GB Laptop GPU, 596.08, 6144 MiB.
Host 13th Gen Intel(R) Core(TM) i5-13450HX, Windows 10.0.26300, 15.69 GiB exposed RAM.
SPARK commit 1fa4a5306701486ac13d6cd103d41e702e254e9e. llama.cpp version: 10159 (f95de9776).
GGUF SHA-256 fb044e93939a70469c905781334f5de1e6c8b608ced6cbc8c9249bd4127d9526.

5 warm-ups, 10 measured runs per runtime and workload.
Each trial uses a single sequence, 128 forced greedy outputs and a 2048-token context. See [methodology](../docs/BENCHMARKING.md).

## 128 prompt tokens

Exact measured output matches 10/10.

**SPARK** cold process-ready latency was 2963.66 ms (one observation). Packed Q4_K_M CUDA, FP32 KV, graphs off.

| Metric | Median | Mean | Sample SD | IQR | P95 | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode (tok/s) | 52.18 | 51.73 | 1.45 | 0.39 | 52.59 | 10 |
| TTFT (ms) | 1713.33 | 1715.46 | 11.05 | 12.71 | 1733.40 | 10 |
| Prefill (tok/s) | 74.77 | 74.71 | 0.45 | 0.54 | 75.22 | 10 |
| Decode latency (ms) | 2433.72 | 2456.94 | 73.71 | 18.23 | 2572.91 | 10 |
| Total latency (ms) | 4143.73 | 4172.40 | 79.86 | 33.92 | 4297.72 | 10 |
| Peak board VRAM (MiB) | 683.00 | 683.00 | 0.00 | 0.00 | 683.00 | 10 |
| Peak process RAM (MiB) | 385.12 | 385.08 | 0.06 | 0.06 | 385.12 | 10 |
| Board GPU utilization (%) | 94.84 | 94.08 | 2.99 | 0.72 | 95.71 | 10 |
| Engine CPU utilization (%) | 4.84 | 4.83 | 0.24 | 0.17 | 5.12 | 10 |
| Board-only tok/J | 0.38 | 0.38 | 0.01 | 0.01 | 0.39 | 10 |

**llama.cpp** cold process-ready latency was 2352.69 ms (one observation). --device CUDA0 --n-gpu-layers 99 --ctx-size 2048 --parallel 1 --batch-size 1024 --ubatch-size 512 --flash-attn auto --cache-type-k f32 --cache-type-v f32 --log-verbosity 4.

| Metric | Median | Mean | Sample SD | IQR | P95 | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode (tok/s) | 125.15 | 124.08 | 4.77 | 4.73 | 128.34 | 10 |
| TTFT (ms) | 97.71 | 103.08 | 13.97 | 3.73 | 129.00 | 10 |
| Prefill (tok/s) | 1361.43 | 1305.47 | 150.97 | 70.96 | 1419.49 | 10 |
| Decode latency (ms) | 1014.76 | 1024.94 | 41.38 | 38.34 | 1095.18 | 10 |
| Total latency (ms) | 1116.78 | 1128.02 | 48.87 | 57.07 | 1204.30 | 10 |
| Peak board VRAM (MiB) | 761.00 | 761.00 | 0.00 | 0.00 | 761.00 | 10 |
| Peak process RAM (MiB) | 1146.94 | 1146.94 | 0.09 | 0.10 | 1147.05 | 10 |
| Board GPU utilization (%) | 68.09 | 65.86 | 6.79 | 3.69 | 71.88 | 10 |
| Engine CPU utilization (%) | 6.18 | 6.19 | 0.33 | 0.18 | 6.61 | 10 |
| Board-only tok/J | 1.62 | 1.61 | 0.06 | 0.11 | 1.68 | 10 |


## 512 prompt tokens

Exact measured output matches 10/10.

**llama.cpp** cold process-ready latency was 2920.48 ms (one observation). --device CUDA0 --n-gpu-layers 99 --ctx-size 2048 --parallel 1 --batch-size 1024 --ubatch-size 512 --flash-attn auto --cache-type-k f32 --cache-type-v f32 --log-verbosity 4.

| Metric | Median | Mean | Sample SD | IQR | P95 | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode (tok/s) | 123.96 | 122.91 | 4.17 | 2.70 | 126.35 | 10 |
| TTFT (ms) | 139.64 | 140.09 | 10.85 | 13.61 | 154.24 | 10 |
| Prefill (tok/s) | 3783.48 | 3772.49 | 304.98 | 366.00 | 4222.00 | 10 |
| Decode latency (ms) | 1024.50 | 1034.44 | 37.25 | 22.36 | 1097.53 | 10 |
| Total latency (ms) | 1163.19 | 1174.52 | 43.46 | 23.56 | 1249.56 | 10 |
| Peak board VRAM (MiB) | 761.00 | 761.00 | 0.00 | 0.00 | 761.00 | 10 |
| Peak process RAM (MiB) | 1148.95 | 1148.95 | 0.06 | 0.08 | 1149.02 | 10 |
| Board GPU utilization (%) | 72.95 | 72.39 | 3.12 | 3.76 | 75.65 | 10 |
| Engine CPU utilization (%) | 6.10 | 6.10 | 0.30 | 0.32 | 6.53 | 10 |
| Board-only tok/J | 1.48 | 1.47 | 0.04 | 0.06 | 1.52 | 10 |

**SPARK** cold process-ready latency was 3194.06 ms (one observation). Packed Q4_K_M CUDA, FP32 KV, graphs off.

| Metric | Median | Mean | Sample SD | IQR | P95 | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode (tok/s) | 50.29 | 50.14 | 0.59 | 0.82 | 50.79 | 10 |
| TTFT (ms) | 6969.26 | 7008.95 | 69.09 | 84.73 | 7118.98 | 10 |
| Prefill (tok/s) | 73.48 | 73.07 | 0.71 | 0.89 | 73.65 | 10 |
| Decode latency (ms) | 2525.42 | 2533.18 | 30.18 | 41.47 | 2578.59 | 10 |
| Total latency (ms) | 9504.01 | 9542.13 | 95.93 | 120.57 | 9692.61 | 10 |
| Peak board VRAM (MiB) | 683.00 | 683.00 | 0.00 | 0.00 | 683.00 | 10 |
| Peak process RAM (MiB) | 384.97 | 384.98 | 0.02 | 0.02 | 385.02 | 10 |
| Board GPU utilization (%) | 97.66 | 97.62 | 0.35 | 0.40 | 98.10 | 10 |
| Engine CPU utilization (%) | 4.15 | 4.18 | 0.47 | 0.31 | 4.90 | 10 |
| Board-only tok/J | 0.16 | 0.16 | 0.00 | 0.01 | 0.17 | 10 |


## 1024 prompt tokens

Exact measured output matches 10/10.

**SPARK** cold process-ready latency was 3957.16 ms (one observation). Packed Q4_K_M CUDA, FP32 KV, graphs off.

| Metric | Median | Mean | Sample SD | IQR | P95 | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode (tok/s) | 47.16 | 47.09 | 0.34 | 0.22 | 47.44 | 10 |
| TTFT (ms) | 14695.36 | 14701.45 | 33.76 | 39.95 | 14756.06 | 10 |
| Prefill (tok/s) | 69.69 | 69.66 | 0.16 | 0.19 | 69.82 | 10 |
| Decode latency (ms) | 2692.87 | 2697.30 | 19.37 | 12.64 | 2730.70 | 10 |
| Total latency (ms) | 17389.95 | 17398.75 | 41.38 | 46.04 | 17464.15 | 10 |
| Peak board VRAM (MiB) | 683.00 | 683.00 | 0.00 | 0.00 | 683.00 | 10 |
| Peak process RAM (MiB) | 384.98 | 384.98 | 0.00 | 0.00 | 384.98 | 10 |
| Board GPU utilization (%) | 98.45 | 98.48 | 0.17 | 0.23 | 98.73 | 10 |
| Engine CPU utilization (%) | 3.94 | 3.79 | 0.48 | 0.65 | 4.40 | 10 |
| Board-only tok/J | 0.10 | 0.10 | 0.00 | 0.00 | 0.10 | 10 |

**llama.cpp** cold process-ready latency was 1864.23 ms (one observation). --device CUDA0 --n-gpu-layers 99 --ctx-size 2048 --parallel 1 --batch-size 1024 --ubatch-size 512 --flash-attn auto --cache-type-k f32 --cache-type-v f32 --log-verbosity 4.

| Metric | Median | Mean | Sample SD | IQR | P95 | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode (tok/s) | 114.59 | 114.85 | 1.11 | 1.39 | 116.59 | 10 |
| TTFT (ms) | 255.04 | 254.21 | 3.60 | 5.59 | 258.21 | 10 |
| Prefill (tok/s) | 4066.21 | 4078.67 | 57.13 | 86.63 | 4160.81 | 10 |
| Decode latency (ms) | 1108.32 | 1105.86 | 10.66 | 13.43 | 1117.60 | 10 |
| Total latency (ms) | 1362.21 | 1360.07 | 10.29 | 14.04 | 1371.11 | 10 |
| Peak board VRAM (MiB) | 761.00 | 761.00 | 0.00 | 0.00 | 761.00 | 10 |
| Peak process RAM (MiB) | 1168.12 | 1168.12 | 0.09 | 0.14 | 1168.23 | 10 |
| Board GPU utilization (%) | 72.08 | 70.23 | 7.35 | 8.85 | 77.93 | 10 |
| Engine CPU utilization (%) | 6.10 | 6.18 | 0.18 | 0.27 | 6.39 | 10 |
| Board-only tok/J | 1.30 | 1.30 | 0.02 | 0.03 | 1.33 | 10 |

