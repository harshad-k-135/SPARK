# SPARK

SPARK (Specialized Public Accelerated Runtime for Kernels) targets batch-one
Qwen3.5-0.8B text generation on an NVIDIA GeForce RTX 3050 Laptop GPU. The model
target is the local Q4_K_M GGUF. SPARK reads the file in native C++20.

SPARK runs native text generation on CPU from this GGUF. CUDA generation,
the control API, and the frontend still require implementation.

## Build and inspect

```powershell
powershell -File scripts/build_cpu.ps1
& build/cpu/spark.exe inspect --model 'PATH\TO\Qwen_Qwen3.5-0.8B-Q4_K_M.gguf'
& build/cpu/spark.exe tokenize --model 'PATH\TO\Qwen_Qwen3.5-0.8B-Q4_K_M.gguf' --prompt 'What is 2 + 2?'
$env:OMP_NUM_THREADS = '6'
& build/cpu/spark.exe generate --model 'PATH\TO\Qwen_Qwen3.5-0.8B-Q4_K_M.gguf' --prompt 'What is 2 + 2? Answer with just the digit.' --chat --max-new-tokens 16
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

See [gate results](docs/RESULTS.md) and the [model layout](docs/DESIGN.md).
