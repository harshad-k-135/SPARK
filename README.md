# SPARK

SPARK (Specialized Public Accelerated Runtime for Kernels) is a specialized
local inference runtime for one user, one sequence, and a single NVIDIA GPU.
The current build runs a native C++20 FP32 CPU forward pass. The first CUDA
target is FP16 inference for Meta Llama 3.2 1B Instruct on the RTX 3050 6 GB
Laptop GPU. GPU implementation begins after the official CPU parity gate.

## Build

Run `cmake --preset release && cmake --build --preset release` with CMake 3.24
or newer and a C++20 compiler. On Windows, run
`powershell -File scripts/build_cpu.ps1`, which also supports MinGW GCC when
CMake is unavailable. The commands below use its GCC output under
`build/cpu`. With CMake, use the corresponding binaries in `build/release`
or `build/release/Release` for a multi-config generator.

## Export and generate

Download the official snapshot at
`9213176726f574b556790deb65791e0c5aa438b6` through your approved Hugging Face
account. Use a local snapshot containing the safetensors shards, config, and
tokenizer files. The exporter uses Python's standard library. [Model card and access](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

```powershell
$snapshot = 'models/hf-llama-3.2-1b-instruct'
python tools/export/export_hf_to_spark.py --source $snapshot --output models/spark-1b
python tools/export/export_tokenizer.py --source $snapshot --output models/spark-1b
& build/cpu/spark.exe generate --model models/spark-1b --prompt 'Describe how rain forms.' --max-new-tokens 64 --temperature 0
```

Generation uses a raw prompt with the model's BOS token. Supply UTF-8 text
with `--prompt-file` for multiline prompts. Positive temperatures support
`--top-p` and a repeatable `--seed`. The default context is 2,048 tokens and
the cap is 4,096 tokens.

## Run Gate 1

Use a Python environment compatible with PyTorch 2.6.0 and Transformers 4.51.3.

```powershell
python -m pip install -r tools/reference/requirements.txt
python tools/reference/generate_reference.py --source $snapshot
& build/cpu/spark_check.exe --gate1 models/spark-1b tests/golden/reference.json
```

The checker verifies Python token IDs, byte-exact prompt round-trips, first-step
logits, and all 64 greedy continuation tokens per prompt. The offline smoke
test can run with `ctest --preset release` or with
`python tools/reference/generate_smoke.py --check build/cpu/spark_check.exe --cli build/cpu/spark.exe`.

## Benchmark status

| Runtime | Status |
| --- | --- |
| spark-fp16-v0 | Awaits official CPU parity and CUDA baseline |
| spark-optimized | Awaits CUDA baseline |
| llama.cpp | Awaits matched GPU trials |

See [the gate ledger](docs/RESULTS.md) and the [benchmark protocol](docs/BENCHMARK_PROTOCOL.md).
