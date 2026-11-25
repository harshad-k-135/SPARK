# Tuned CUDA comparison

Measured on NVIDIA GeForce RTX 3050 6GB Laptop GPU, 596.08, 6144 MiB.
Host 13th Gen Intel(R) Core(TM) i5-13450HX, Windows 10.0.26300, 15.69 GiB exposed RAM.
SPARK commit 77358ecb1a890ce0e3e482b8fb35d69a9775e97a. llama.cpp version: 10159 (f95de9776).
GGUF SHA-256 fb044e93939a70469c905781334f5de1e6c8b608ced6cbc8c9249bd4127d9526.

5 warm-ups, 10 measured runs per runtime and workload.
Each trial uses a single sequence, 128 forced greedy outputs and a 2048-token context. See [methodology](../docs/BENCHMARKING.md).

## 128 prompt tokens

Exact measured output matches 10/10.

**SPARK** cold process-ready latency was 3025.76 ms (one observation). Packed GGUF CUDA, FP32 KV, bucket graphs, 128-token projection batches, Q6_K block GEMV, Q4_K block GEMV.

| Metric | Median | Mean | Sample SD | IQR | P95 | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode (tok/s) | 117.77 | 117.72 | 0.42 | 0.22 | 118.15 | 10 |
| TTFT (ms) | 227.99 | 225.78 | 6.62 | 10.03 | 233.77 | 10 |
| Prefill (tok/s) | 566.17 | 571.92 | 16.92 | 25.94 | 595.42 | 10 |
| Decode latency (ms) | 1078.38 | 1078.84 | 3.84 | 2.05 | 1084.64 | 10 |
| Total latency (ms) | 1306.06 | 1304.62 | 4.99 | 3.82 | 1310.85 | 10 |
| Peak board VRAM (MiB) | 753.00 | 753.00 | 0.00 | 0.00 | 753.00 | 10 |
| Peak process RAM (MiB) | 427.41 | 427.34 | 0.13 | 0.20 | 427.41 | 10 |
| Board GPU utilization (%) | 76.33 | 76.31 | 5.21 | 9.75 | 82.58 | 10 |
| Engine CPU utilization (%) | 6.05 | 5.98 | 0.16 | 0.18 | 6.11 | 10 |
| Board-only tok/J | 1.29 | 1.28 | 0.01 | 0.00 | 1.29 | 10 |

**llama.cpp** cold process-ready latency was 1974.74 ms (one observation). --device CUDA0 --n-gpu-layers 99 --ctx-size 2048 --parallel 1 --batch-size 1024 --ubatch-size 512 --flash-attn auto --cache-type-k f32 --cache-type-v f32 --log-verbosity 4.

| Metric | Median | Mean | Sample SD | IQR | P95 | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode (tok/s) | 124.65 | 124.36 | 1.53 | 2.03 | 126.10 | 10 |
| TTFT (ms) | 98.02 | 98.70 | 11.32 | 12.09 | 113.03 | 10 |
| Prefill (tok/s) | 1349.10 | 1354.55 | 172.18 | 158.65 | 1623.08 | 10 |
| Decode latency (ms) | 1018.87 | 1021.35 | 12.68 | 16.65 | 1040.55 | 10 |
| Total latency (ms) | 1121.55 | 1120.06 | 19.09 | 12.52 | 1148.95 | 10 |
| Peak board VRAM (MiB) | 761.00 | 761.00 | 0.00 | 0.00 | 761.00 | 10 |
| Peak process RAM (MiB) | 1146.70 | 1146.71 | 0.07 | 0.12 | 1146.81 | 10 |
| Board GPU utilization (%) | 68.09 | 65.39 | 9.15 | 5.24 | 71.92 | 10 |
| Engine CPU utilization (%) | 6.18 | 6.18 | 0.19 | 0.21 | 6.44 | 10 |
| Board-only tok/J | 1.57 | 1.57 | 0.02 | 0.02 | 1.60 | 10 |


## 512 prompt tokens

Exact measured output matches 10/10.

**llama.cpp** cold process-ready latency was 2655.59 ms (one observation). --device CUDA0 --n-gpu-layers 99 --ctx-size 2048 --parallel 1 --batch-size 1024 --ubatch-size 512 --flash-attn auto --cache-type-k f32 --cache-type-v f32 --log-verbosity 4.

| Metric | Median | Mean | Sample SD | IQR | P95 | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode (tok/s) | 121.65 | 121.59 | 1.16 | 1.53 | 123.27 | 10 |
| TTFT (ms) | 144.40 | 142.56 | 6.37 | 6.44 | 148.69 | 10 |
| Prefill (tok/s) | 3618.39 | 3678.48 | 179.89 | 173.21 | 3967.22 | 10 |
| Decode latency (ms) | 1043.94 | 1044.57 | 9.96 | 13.18 | 1057.13 | 10 |
| Total latency (ms) | 1185.59 | 1187.13 | 15.45 | 20.54 | 1203.22 | 10 |
| Peak board VRAM (MiB) | 761.00 | 761.00 | 0.00 | 0.00 | 761.00 | 10 |
| Peak process RAM (MiB) | 1148.61 | 1148.61 | 0.05 | 0.08 | 1148.68 | 10 |
| Board GPU utilization (%) | 72.55 | 70.58 | 5.06 | 2.86 | 74.84 | 10 |
| Engine CPU utilization (%) | 6.25 | 6.34 | 0.37 | 0.45 | 6.89 | 10 |
| Board-only tok/J | 1.45 | 1.45 | 0.02 | 0.02 | 1.47 | 10 |

**SPARK** cold process-ready latency was 3108.29 ms (one observation). Packed GGUF CUDA, FP32 KV, bucket graphs, 128-token projection batches, Q6_K block GEMV, Q4_K block GEMV.

| Metric | Median | Mean | Sample SD | IQR | P95 | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode (tok/s) | 111.18 | 111.13 | 0.18 | 0.13 | 111.33 | 10 |
| TTFT (ms) | 838.72 | 838.07 | 16.29 | 19.84 | 860.24 | 10 |
| Prefill (tok/s) | 611.92 | 612.57 | 11.96 | 14.52 | 629.76 | 10 |
| Decode latency (ms) | 1142.30 | 1142.76 | 1.89 | 1.30 | 1145.76 | 10 |
| Total latency (ms) | 1981.90 | 1980.83 | 16.26 | 23.80 | 2001.16 | 10 |
| Peak board VRAM (MiB) | 753.00 | 753.00 | 0.00 | 0.00 | 753.00 | 10 |
| Peak process RAM (MiB) | 427.02 | 427.02 | 0.00 | 0.00 | 427.02 | 10 |
| Board GPU utilization (%) | 73.50 | 73.44 | 1.58 | 2.61 | 75.45 | 10 |
| Engine CPU utilization (%) | 5.85 | 5.90 | 0.15 | 0.26 | 6.11 | 10 |
| Board-only tok/J | 0.87 | 0.87 | 0.00 | 0.01 | 0.88 | 10 |


## 1024 prompt tokens

Exact measured output matches 10/10.

**SPARK** cold process-ready latency was 3422.57 ms (one observation). Packed GGUF CUDA, FP32 KV, bucket graphs, 128-token projection batches, Q6_K block GEMV, Q4_K block GEMV.

| Metric | Median | Mean | Sample SD | IQR | P95 | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode (tok/s) | 103.24 | 103.19 | 0.58 | 0.68 | 103.88 | 10 |
| TTFT (ms) | 1781.93 | 1777.69 | 42.54 | 41.22 | 1823.88 | 10 |
| Prefill (tok/s) | 575.33 | 576.91 | 14.23 | 13.33 | 599.52 | 10 |
| Decode latency (ms) | 1230.15 | 1230.78 | 6.95 | 8.09 | 1241.47 | 10 |
| Total latency (ms) | 3014.95 | 3008.47 | 45.69 | 50.00 | 3059.01 | 10 |
| Peak board VRAM (MiB) | 753.00 | 753.00 | 0.00 | 0.00 | 753.00 | 10 |
| Peak process RAM (MiB) | 427.25 | 427.25 | 0.00 | 0.00 | 427.25 | 10 |
| Board GPU utilization (%) | 82.42 | 82.79 | 1.59 | 1.00 | 85.66 | 10 |
| Engine CPU utilization (%) | 5.73 | 5.73 | 0.12 | 0.11 | 5.88 | 10 |
| Board-only tok/J | 0.57 | 0.57 | 0.00 | 0.00 | 0.58 | 10 |

**llama.cpp** cold process-ready latency was 1862.62 ms (one observation). --device CUDA0 --n-gpu-layers 99 --ctx-size 2048 --parallel 1 --batch-size 1024 --ubatch-size 512 --flash-attn auto --cache-type-k f32 --cache-type-v f32 --log-verbosity 4.

| Metric | Median | Mean | Sample SD | IQR | P95 | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode (tok/s) | 117.59 | 117.67 | 1.33 | 1.16 | 119.64 | 10 |
| TTFT (ms) | 247.59 | 248.79 | 4.13 | 6.22 | 254.85 | 10 |
| Prefill (tok/s) | 4187.84 | 4169.41 | 74.86 | 96.57 | 4258.99 | 10 |
| Decode latency (ms) | 1080.03 | 1079.46 | 12.21 | 10.62 | 1096.36 | 10 |
| Total latency (ms) | 1327.34 | 1328.25 | 13.01 | 14.81 | 1346.42 | 10 |
| Peak board VRAM (MiB) | 761.00 | 761.00 | 0.00 | 0.00 | 761.00 | 10 |
| Peak process RAM (MiB) | 1168.61 | 1168.58 | 0.09 | 0.11 | 1168.67 | 10 |
| Board GPU utilization (%) | 72.88 | 73.64 | 3.90 | 5.97 | 79.36 | 10 |
| Engine CPU utilization (%) | 6.21 | 6.19 | 0.18 | 0.18 | 6.42 | 10 |
| Board-only tok/J | 1.28 | 1.28 | 0.02 | 0.02 | 1.30 | 10 |

