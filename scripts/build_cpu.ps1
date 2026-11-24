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
    $sources = @(
        'engine/model_loader/json.cpp'
        'engine/model_loader/model.cpp'
        'engine/tokenizer/tokenizer.cpp'
        'engine/transformer/cpu_forward.cpp'
        'engine/runtime/session.cpp'
        'engine/sampling/sampler.cpp'
    )
    $flags = @('-std=c++20', '-O3', '-Wall', '-Wextra', '-Wpedantic', '-Werror', '-fopenmp', '-I.')
    $objects = @()
    foreach ($source in $sources) {
        $object = 'build/cpu/' + [IO.Path]::GetFileNameWithoutExtension($source) + '.o'
        g++ @flags -c $source -o $object
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        $objects += $object
    }
    g++ @flags -municode @objects engine/runtime/cli.cpp -o build/cpu/spark.exe
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    g++ @flags @objects tests/golden/check_reference.cpp -o build/cpu/spark_check.exe
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
