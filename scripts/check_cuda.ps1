$ErrorActionPreference = 'Stop'
if (-not $env:CUDA_PATH) { throw 'Set CUDA_PATH to the CUDA 12 toolkit directory.' }
$library = Join-Path $env:CUDA_PATH 'bin\cudart64_12.dll'
if (-not (Test-Path -LiteralPath $library)) { throw "Missing CUDA runtime at $library" }

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class SparkCudaProbe {
    [DllImport(@"$library")] public static extern int cudaGetDeviceCount(out int count);
    [DllImport(@"$library")] public static extern int cudaMalloc(out IntPtr pointer, UIntPtr bytes);
    [DllImport(@"$library")] public static extern int cudaMemGetInfo(out UIntPtr free, out UIntPtr total);
    [DllImport(@"$library")] public static extern int cudaFree(IntPtr pointer);
    [DllImport(@"$library")] public static extern IntPtr cudaGetErrorString(int code);
}
"@

function Assert-Cuda([string]$operation, [int]$code) {
    if ($code -ne 0) {
        $message = [Runtime.InteropServices.Marshal]::PtrToStringAnsi([SparkCudaProbe]::cudaGetErrorString($code))
        throw "$operation failed ($code) at line $($MyInvocation.ScriptLineNumber). $message"
    }
}

$count = 0
Assert-Cuda cudaGetDeviceCount ([SparkCudaProbe]::cudaGetDeviceCount([ref]$count))
if ($count -lt 1) { throw 'CUDA reports zero devices.' }

$allocation = [IntPtr]::Zero
$bytes = [UIntPtr]::new([uint64]4096)
Assert-Cuda cudaMalloc ([SparkCudaProbe]::cudaMalloc([ref]$allocation, $bytes))
try {
    $free = [UIntPtr]::Zero
    $total = [UIntPtr]::Zero
    Assert-Cuda cudaMemGetInfo ([SparkCudaProbe]::cudaMemGetInfo([ref]$free, [ref]$total))
} finally {
    Assert-Cuda cudaFree ([SparkCudaProbe]::cudaFree($allocation))
}

[ordered]@{
    event = 'cuda_allocation_probe'
    devices = $count
    allocated_bytes = $bytes.ToUInt64()
    free_bytes = $free.ToUInt64()
    total_bytes = $total.ToUInt64()
} | ConvertTo-Json -Compress
