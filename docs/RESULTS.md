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
| 2 — Qwen CUDA parity and baseline | Parity passed | All 20 first tokens and 1,280 continuation IDs matched the FP32 reference. Maximum first-step logit-gap error was 0.001081. Baseline measurements and profiles pending. |
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
ran on CPU. Live SPARK frontend captures and matched performance comparisons
are pending.

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
