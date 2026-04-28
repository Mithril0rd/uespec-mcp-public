$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$outputDir = Join-Path $PSScriptRoot "output"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$transcriptPath = Join-Path $outputDir "prerelease-demo-$timestamp.transcript.txt"

Push-Location $repoRoot
try {
    Start-Transcript -Path $transcriptPath -Force | Out-Null
    try {
        & (Join-Path $PSScriptRoot "prerelease-demo.ps1")
    }
    finally {
        Stop-Transcript | Out-Null
    }
}
finally {
    Pop-Location
}

Write-Host "Transcript written to $transcriptPath"
