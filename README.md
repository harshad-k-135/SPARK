# SPARK

SPARK (Specialized Public Accelerated Runtime for Kernels) targets batch-one
Qwen3.5-0.8B text generation on an NVIDIA GeForce RTX 3050 Laptop GPU. The model
target is the local Q4_K_M GGUF. SPARK reads the file in native C++20.

Current functionality is GGUF inspection. Native CPU and CUDA generation,
the control API, and the frontend still require implementation.

## Build and inspect

```powershell
powershell -File scripts/build_cpu.ps1
& build/cpu/spark.exe inspect --model 'PATH\TO\Qwen_Qwen3.5-0.8B-Q4_K_M.gguf'
```

Add `--tensors` to list each tensor's shape, format, byte offset, and size.
The inspector also decodes the first embedding row to verify weight access.
With CMake, run `cmake --preset release` and `cmake --build --preset release`.
Use the executable under `build/release` or `build/release/Release`.
Run `ctest --preset release` or `build/cpu/spark_packed_weights_test.exe` to
check packed weight decoding.

See [gate results](docs/RESULTS.md) and the [model layout](docs/DESIGN.md).
