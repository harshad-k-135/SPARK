param(
    [Parameter(Mandatory=$true)][string]$Model,
    [string]$Spark = 'build/cpu/spark.exe',
    [string]$Python = 'python',
    [int]$Port = 8765
)
$ErrorActionPreference = 'Stop'
$env:SPARK_MODEL = (Resolve-Path -LiteralPath $Model).Path
$env:SPARK_EXE = (Resolve-Path -LiteralPath $Spark).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    & $Python -m uvicorn api.app:app --host 127.0.0.1 --port $Port
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
