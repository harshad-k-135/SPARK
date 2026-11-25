param([Parameter(Mandatory=$true)][int]$ProcessId)

# Cumulative processor time (seconds) and current resident working set (bytes).
while ($true) {
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process) { break }
    '{0},{1}' -f $process.CPU.ToString([cultureinfo]::InvariantCulture),
        $process.WorkingSet64.ToString([cultureinfo]::InvariantCulture)
    Start-Sleep -Milliseconds 200
}
