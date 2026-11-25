param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$ReferenceModel,
    [Parameter(Mandatory=$true)][string]$Spark,
    [string]$ReferenceServer = 'build/llama-b10159-cuda13/llama-server.exe',
    [string]$Output = 'build/qwen-cuda-golden.jsonl',
    [ValidateSet('off','bucket','single')][string]$Graph = 'off',
    [switch]$BatchPrefill,
    [switch]$Q6BlockGemv,
    [switch]$Fp16Prefill,
    [int]$Port = 18085
)

$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    $modelPath = (Resolve-Path -LiteralPath $Model).Path
    $reference = (Resolve-Path -LiteralPath $ReferenceModel).Path
    $executable = (Resolve-Path -LiteralPath $Spark).Path
    $llama = (Resolve-Path -LiteralPath $ReferenceServer).Path
    $work = (New-Item -ItemType Directory -Force "build/cuda-golden-$Port").FullName
    $env:SPARK_DEVICE = 'cuda'
    $env:SPARK_CUDA_WEIGHTS = 'packed'
    $env:SPARK_CUDA_GRAPH = $Graph
    $env:SPARK_FP16_PREFILL = [string][int]$Fp16Prefill.IsPresent
    $env:SPARK_Q6_BLOCK_GEMV = [string][int]$Q6BlockGemv.IsPresent
    $env:SPARK_BATCH_PREFILL = [string][int]$BatchPrefill.IsPresent
    $env:TEMP = $work
    $env:TMP = $work
    $arguments = '--model "{0}" --host 127.0.0.1 --port {1} -ngl 99 -c 2048 -np 1 -ctk f32 -ctv f32 -fa off --log-verbosity 4' -f $reference, $Port
    $errorLog = Join-Path $work 'reference-err.log'
    $server = Start-Process -FilePath $llama -ArgumentList $arguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $work 'reference-out.log') -RedirectStandardError $errorLog
    try {
        $url = "http://127.0.0.1:$Port"
        $ready = $false
        for ($attempt = 0; $attempt -lt 120; $attempt++) {
            if ($server.HasExited) { throw "Reference server exited. See $errorLog." }
            try {
                if ((Invoke-RestMethod "$url/health" -TimeoutSec 1).status -eq 'ok') {
                    $ready = $true
                    break
                }
            } catch {}
            Start-Sleep -Milliseconds 250
        }
        if (-not $ready -or -not (Select-String -LiteralPath $errorLog -Pattern 'offloaded (\d+)/\1 layers to GPU')) {
            throw "Full CUDA reference offload was not confirmed. See $errorLog."
        }
        if (Test-Path -LiteralPath $Output) { throw "Golden output already exists at $Output." }
        # JSON lines contain only ASCII keys and numeric values.
        node benchmark/comparison/verify_qwen_generation.cjs $executable $modelPath $url --full |
            ForEach-Object { $_ | Out-File -LiteralPath $Output -Append -Encoding ascii; $_ }
        exit $LASTEXITCODE
    } finally {
        if (-not $server.HasExited) { Stop-Process -Id $server.Id }
    }
} finally {
    Pop-Location
}
