# Results

| Gate | Status | Evidence |
| --- | --- | --- |
| CUDA runtime bootstrap | Passed | RTX 3050 detected. CUDA 12.8 allocated and freed 4,096 bytes. |
| Qwen GGUF import | Passed | Native C++ inspection validated the supplied file layout and classified all blocks. |
| Packed weight decoding | Passed | Fixed FP32, FP16, BF16, Q4_K, Q5_K, Q6_K, and Q8_0 cases passed. The native inspector read and decoded the first real embedding row. |
| 1 — Qwen CPU parity | Pending | Requires tokenizer, the text forward pass, and 20 × 64 reference continuations. |
| 2 — Qwen CUDA parity and baseline | Pending Gate 1 | |
| 3 — Optimized Qwen decode | Pending Gate 2 | |
| 4 — GPU comparison and control API | Pending Gate 3 | |

Target file `Qwen_Qwen3.5-0.8B-Q4_K_M.gguf`, SHA-256
`fb044e93939a70469c905781334f5de1e6c8b608ced6cbc8c9249bd4127d9526`.
The source revision of the supplied GGUF is unverified. The file is 579,615,840
bytes and contains 335 tensors in 25 blocks, including 18 recurrent blocks,
six attention blocks, and one auxiliary block. Hidden width is 1,024 and the
stored vocabulary has 248,320 entries.
The first Q6_K embedding row decoded to finite values from `-0.0759277` to
`0.0695343`. A deliberately truncated copy of the GGUF header was rejected.

The recorded [external Qwen API smoke test](../benchmark/results/qwen-local-smoke/02-qwen-api.png)
ran on CPU. GPU inference, live SPARK frontend captures, and matched performance
comparisons are pending.
