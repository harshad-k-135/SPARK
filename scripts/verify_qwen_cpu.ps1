param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$ReferenceBin,
    [string]$Spark = 'build/cpu/spark.exe',
    [int]$Port = 18085,
    [switch]$Full
)

$ErrorActionPreference = 'Stop'
$Model = (Resolve-Path -LiteralPath $Model).Path
$ReferenceBin = (Resolve-Path -LiteralPath $ReferenceBin).Path
$Spark = (Resolve-Path -LiteralPath $Spark).Path
Push-Location (Join-Path $PSScriptRoot '..')
try {
    $dir = (New-Item -ItemType Directory -Force build/qwen-reference).FullName
    $sha = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Model)
    try {
        $hash = [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
    } finally {
        $stream.Dispose()
        $sha.Dispose()
    }
    $fp32 = Join-Path $dir "$hash-fp32.gguf"
    if (-not (Test-Path -LiteralPath $fp32)) {
        $partial = "$fp32.$PID.partial"
        $convert = @{
            FilePath = Join-Path $ReferenceBin 'llama-quantize.exe'
            WorkingDirectory = $ReferenceBin
            ArgumentList = '--allow-requantize --pure "{0}" "{1}" F32 6' -f $Model, $partial
            RedirectStandardOutput = Join-Path $dir 'convert-out.log'
            RedirectStandardError = Join-Path $dir 'convert-err.log'
            WindowStyle = 'Hidden'
            PassThru = $true
            Wait = $true
        }
        $conversion = Start-Process @convert
        if ($conversion.ExitCode -ne 0) { throw "FP32 reference conversion failed. See $dir." }
        Move-Item -LiteralPath $partial -Destination $fp32
    }
    $server = @{
        FilePath = Join-Path $ReferenceBin 'llama-server.exe'
        WorkingDirectory = $ReferenceBin
        ArgumentList = '-m "{0}" --host 127.0.0.1 --port {1} -ngl 0 -c 2048 -np 1 -t 6 -ctk f32 -ctv f32 -fa off' -f $fp32, $Port
        RedirectStandardOutput = Join-Path $dir 'server-out.log'
        RedirectStandardError = Join-Path $dir 'server-err.log'
        WindowStyle = 'Hidden'
        PassThru = $true
    }
    $process = Start-Process @server
    try {
        $url = "http://127.0.0.1:$Port"
        $ready = $false
        for ($attempt = 0; $attempt -lt 45; $attempt++) {
            if ($process.HasExited) { throw "Comparison server exited. See $dir." }
            $health = curl.exe -fs --max-time 2 "$url/health"
            if ($health -eq '{"status":"ok"}') {
                $ready = $true
                break
            }
            Start-Sleep -Milliseconds 500
        }
        if (-not $ready) { throw "Comparison server failed to become healthy. See $dir." }
        $env:TEMP = $dir
        $verify = @('benchmark/comparison/verify_qwen_generation.cjs', $Spark, $Model, $url)
        if ($Full) { $verify += '--full' }
        & node @verify | Tee-Object -FilePath (Join-Path $dir 'results.jsonl')
        $result = $LASTEXITCODE
    } finally {
        if (-not $process.HasExited) { Stop-Process -Id $process.Id }
    }
    exit $result
} finally {
    Pop-Location
}
