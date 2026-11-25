# SPARK

SPARK (Specialized Public Accelerated Runtime for Kernels) targets batch-one
Qwen3.5-0.8B text generation on an NVIDIA GeForce RTX 3050 Laptop GPU. The model
target is the local Q4_K_M GGUF. SPARK reads the file in native C++20.

SPARK runs native text generation on CPU and CUDA from this GGUF. A local
FastAPI dashboard streams responses from the CUDA worker and compares saved
native benchmark runs.

## Build and inspect

```powershell
powershell -File scripts/build_cpu.ps1
& build/cpu/spark.exe inspect --model 'PATH\TO\Qwen_Qwen3.5-0.8B-Q4_K_M.gguf'
& build/cpu/spark.exe tokenize --model 'PATH\TO\Qwen_Qwen3.5-0.8B-Q4_K_M.gguf' --prompt 'What is 2 + 2?'
$env:OMP_NUM_THREADS = '6'
& build/cpu/spark.exe generate --model 'PATH\TO\Qwen_Qwen3.5-0.8B-Q4_K_M.gguf' --prompt 'What is 2 + 2? Answer with just the digit.' --chat --max-new-tokens 16
& build/cpu/spark.exe generate --model 'PATH\TO\Qwen_Qwen3.5-0.8B-Q4_K_M.gguf' --prompt 'What is 2 + 2? Answer with just the digit.' --chat --max-new-tokens 16 --device cuda --cuda-weights packed
```

Add `--tensors` to list each tensor's shape, format, byte offset, and size.
The inspector also decodes the first embedding row to verify weight access.
The tokenizer reads the vocabulary and merges directly from the GGUF.
With CMake, run `cmake --preset release` and `cmake --build --preset release`.
Use the executable under `build/release` or `build/release/Release`.
Run `ctest --preset release` or `build/cpu/spark_packed_weights_test.exe` to
check packed weight decoding.

Generation defaults to greedy sampling and a context of 2,048 tokens.
`--chat` selects the single-turn Qwen prompt format with thinking disabled.
`--temperature`, `--top-p`, and `--seed` configure sampling. The CPU
implementation expands approximately 3 GB of FP32 weights in host RAM.
The CUDA baseline expands the same weights into a device arena. The packed
path retains the GGUF encodings and unpacks weights inside CUDA projections. It uses
NVRTC and cuBLAS from the local CUDA toolkit. The Windows build uses the
toolkit pointed to by `CUDA_PATH`. Pass `--profile` on a CUDA request to
record GPU stage timings for the first decode step after prefill.

## Reproduce CPU correctness checks

Run the check against the same weights dequantized to FP32 by the external
reference. Use the b10159 Windows comparison binaries and Node.js. Omit
`-Full` for the three-prompt smoke test. The full test evaluates 20 prompts
and 64 greedy tokens per prompt. Both runtimes use FP32 key and value caches
for this CPU correctness comparison.

```powershell
powershell -File scripts\verify_qwen_cpu.ps1 -Model 'PATH\TO\Qwen.gguf' -ReferenceBin 'PATH\TO\llama-b10159-bin-win-cuda-13.3-x64' -Full
```

Results are written under `build/qwen-reference`. The CPU model and tests
use the original GGUF. The expanded comparison GGUF is a local build artifact.

## Run the dashboard locally

Build SPARK with CUDA enabled. Install the API requirements in a Python
environment and pass its interpreter with `-Python` if needed.

```powershell
python -m pip install -r api\requirements.txt
powershell -File scripts\run_dashboard.ps1 -Model 'PATH\TO\Qwen.gguf'
```

Open the loopback address printed by Uvicorn, normally
`http://127.0.0.1:8765/`. **Generate** starts the native CUDA worker on
demand with packed GGUF weights. **Compare** loads stored trial results and can queue another native
benchmark. The service serializes generation and benchmarks on the GPU.
An external GPU comparison requires a CUDA-capable llama.cpp build.

See [gate results](docs/RESULTS.md) and the [model layout](docs/DESIGN.md).
