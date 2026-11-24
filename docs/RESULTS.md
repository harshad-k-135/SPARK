# Results

| Gate | Status | Evidence |
| --- | --- | --- |
| Bootstrap GPU allocation | Passed | CUDA 12.8 allocated and freed 4,096 bytes on the target GPU. See [environment](ENVIRONMENT.md). |
| 1 — CPU parity | Pending | Requires the pinned 1B model and 20 × 64 reference token IDs. |
| 2 — CUDA FP16 parity and baseline | Pending Gate 1 | |
| 3 — Optimized decode | Pending Gate 2 | |
| 4 — External comparison | Pending Gate 3 | |

No performance measurements yet.
