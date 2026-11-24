$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    if (Get-Command cmake -ErrorAction SilentlyContinue) {
        cmake --preset release
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        cmake --build --preset release
        exit $LASTEXITCODE
    }

    New-Item -ItemType Directory -Path build/cpu -Force | Out-Null
    $flags = @('-std=c++20', '-O3', '-Wall', '-Wextra', '-Wpedantic', '-Werror', '-I.')
    g++ @flags -municode engine/model_loader/gguf.cpp engine/runtime/cli.cpp -o build/cpu/spark.exe
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
