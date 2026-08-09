param(
    [int]$RefreshSeconds = 5,
    [int]$LogLines = 30
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$statusPath = Join-Path $projectRoot "outputs\training-status.json"
$logPath = Join-Path $projectRoot "outputs\training.log"

while ($true) {
    Clear-Host
    Write-Host "TradingAI Training Monitor" -ForegroundColor Cyan
    Write-Host "Updated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "Press Ctrl+C to close this monitor."
    Write-Host ""
    Write-Host "STATUS" -ForegroundColor Yellow
    if (Test-Path -LiteralPath $statusPath) {
        Get-Content -LiteralPath $statusPath
    }
    else {
        Write-Host "No training status file exists yet."
    }

    Write-Host ""
    Write-Host "LATEST TRAINING LOG" -ForegroundColor Yellow
    if (Test-Path -LiteralPath $logPath) {
        Get-Content -LiteralPath $logPath -Tail $LogLines
    }
    else {
        Write-Host "No training log exists yet."
    }

    Start-Sleep -Seconds $RefreshSeconds
}
