param([string]$OutputDirectory = 'build/cpu')

$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    if (Get-Command cmake -ErrorAction SilentlyContinue) {
        if ($OutputDirectory -ne 'build/cpu') {
            throw 'The CMake release preset controls the output directory.'
        }
        cmake --preset release
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        cmake --build --preset release
        exit $LASTEXITCODE
    }

    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
    $flags = @('-std=c++20', '-O3', '-Wall', '-Wextra', '-Wpedantic', '-Werror', '-I.')
    $extra = @()
    if ($env:CUDA_PATH -and (Test-Path "$env:CUDA_PATH\lib\x64\nvrtc.lib")) {
        $extra = @('-DSPARK_WITH_CUDA', '-isystem', "$env:CUDA_PATH\include",
            'cuda/qwen_gpu.cpp', 'cuda/memory/arena.cpp',
            "$env:CUDA_PATH\lib\x64\cuda.lib", "$env:CUDA_PATH\lib\x64\cudart.lib",
            "$env:CUDA_PATH\lib\x64\nvrtc.lib", "$env:CUDA_PATH\lib\x64\cublas.lib")
    }
    g++ @flags -fopenmp -municode engine/model_loader/gguf.cpp engine/model_loader/packed_weights.cpp engine/runtime/cli.cpp engine/tokenizer/tokenizer.cpp engine/sampling/sampler.cpp engine/transformer/cpu_model.cpp engine/transformer/delta_net.cpp @extra -o (Join-Path $OutputDirectory 'spark.exe')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    g++ @flags engine/model_loader/gguf.cpp engine/model_loader/packed_weights.cpp tests/unit/packed_weights.cpp -o (Join-Path $OutputDirectory 'spark_packed_weights_test.exe')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    g++ @flags engine/transformer/delta_net.cpp tests/unit/delta_net.cpp -o (Join-Path $OutputDirectory 'spark_delta_net_test.exe')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if ($extra.Count -gt 0) {
        g++ @flags -fopenmp engine/model_loader/gguf.cpp engine/model_loader/packed_weights.cpp engine/transformer/delta_net.cpp tests/unit/qwen_cuda.cpp @extra -o (Join-Path $OutputDirectory 'spark_cuda_test.exe')
    }
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
