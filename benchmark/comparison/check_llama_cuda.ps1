param(
    [Parameter(Mandatory=$true)][string]$ReferenceBin
)

$ErrorActionPreference = 'Stop'
$directory = (Resolve-Path -LiteralPath $ReferenceBin).Path
$server = Join-Path $directory 'llama-server.exe'
$output = & $server --list-devices
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$output
if (-not ($output -match '^\s*CUDA\d+\s*:.*RTX 3050')) {
    Write-Error 'The external llama.cpp build did not enumerate the RTX 3050 as a CUDA device.'
    exit 1
}
