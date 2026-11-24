# Environment

Recorded 2026-11-24 on Windows under WDDM.

| Item | Observed |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| Compute capability | 8.6 |
| NVIDIA driver | 596.08 |
| Driver CUDA compatibility reported by `nvidia-smi` | 13.2 |
| Installed CUDA toolkit reported by `nvcc --version` | 12.8.61 |
| VRAM reported by `nvidia-smi` | 6,144 MiB total, 6,001 MiB free |
| Active Windows power scheme at bootstrap | Balanced |
| Phase 1 compiler available on PATH | MinGW GCC 14.2 |
| CUDA host compiler | Visual Studio 2022 Community detected. MSVC C++ tools and CMake were unavailable in the initial checks. |

The CUDA 12.8 runtime returned one device and successfully allocated and freed 4,096
bytes. `cudaMemGetInfo` returned 5,409,603,584 bytes free and 6,441,926,656
bytes total in the same process. Repeat with `powershell -File scripts/check_cuda.ps1`.
CUDA 12.8 was already installed. The older `CUDA_HOME` value refers to 11.8,
so the probe uses `CUDA_PATH` for the 12.8 DLL.
