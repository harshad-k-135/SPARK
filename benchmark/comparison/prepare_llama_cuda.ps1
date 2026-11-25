param(
    [Parameter(Mandatory = $true)][string]$ReferenceBin,
    [string]$RuntimeZip,
    [string]$Model
)

$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $ReferenceBin).Path
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$stage = (New-Item -ItemType Directory -Force (Join-Path $root 'build/llama-b10159-cuda13')).FullName

foreach ($name in 'llama-cli.exe', 'ggml-cuda.dll') {
    if (-not (Test-Path -LiteralPath (Join-Path $source $name) -PathType Leaf)) {
        throw "Missing $name in the supplied b10159 binary directory."
    }
}
$version = & {
    $ErrorActionPreference = 'Continue'
    & (Join-Path $source 'llama-cli.exe') --version 2>&1
}
if ($LASTEXITCODE -ne 0 -or ($version -join ' ') -notmatch '10159 \(f95de9776\)') {
    throw 'Expected the pinned b10159 Windows x64 CUDA 13.3 build.'
}

Get-ChildItem -LiteralPath $source -File | Where-Object Extension -In '.dll','.exe' |
    Copy-Item -Destination $stage

$required = @('cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll')
if ($RuntimeZip) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead((Resolve-Path -LiteralPath $RuntimeZip).Path)
    try {
        $entries = foreach ($name in $required) {
            $matches = @($zip.Entries | Where-Object { $_.Name -ieq $name })
            if ($matches.Count -ne 1 -or $matches[0].Length -lt 1024) {
                throw "Expected exactly one $name in the official b10159 runtime archive."
            }
            $matches[0]
        }
        foreach ($entry in $entries) {
            [IO.Compression.ZipFileExtensions]::ExtractToFile(
                $entry, (Join-Path $stage $entry.Name), $true
            )
        }
    } finally {
        $zip.Dispose()
    }
}

$cli = Join-Path $stage 'llama-cli.exe'
Write-Output "Staged CLI at $cli"
$devices = & {
    $ErrorActionPreference = 'Continue'
    & $cli --list-devices 2>&1
}
$devices
$rtx = $devices | Select-String '^\s*(CUDA\d+)\s*:.*RTX 3050' | Select-Object -First 1
if ($LASTEXITCODE -ne 0 -or -not $rtx) {
    $missing = $required |
        Where-Object { -not (Test-Path -LiteralPath (Join-Path $stage $_) -PathType Leaf) }
    if ($missing) { throw "Missing runtime DLLs: $($missing -join ', '). Supply -RuntimeZip." }
    throw 'The RTX 3050 still did not enumerate with the staged runtime DLLs.'
}
if ($Model) {
    $modelPath = (Resolve-Path -LiteralPath $Model).Path
    $device = $rtx.Matches[0].Groups[1].Value
    $flags = @('--model', $modelPath, '--device', $device, '--n-gpu-layers', '99',
        '--single-turn', '--temp', '0', '--n-predict', '8',
        '--prompt', 'What is 2 + 2? Answer with just the digit.')
    & $cli @flags
    exit $LASTEXITCODE
}
