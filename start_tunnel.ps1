$proc = Start-Process -FilePath "C:\Users\pragn\AppData\Local\Temp\cloudflared.exe" -ArgumentList "tunnel","--url","http://localhost:8000" -RedirectStandardOutput "$env:TEMP\cf_out.txt" -RedirectStandardError "$env:TEMP\cf_err.txt" -PassThru -NoNewWindow:$false
Start-Sleep -Seconds 15
$all = ""
if (Test-Path "$env:TEMP\cf_out.txt") { $all += Get-Content "$env:TEMP\cf_out.txt" -Raw }
if (Test-Path "$env:TEMP\cf_err.txt") { $all += Get-Content "$env:TEMP\cf_err.txt" -Raw }
if ($all -match "(https://[^\s]+trycloudflare\.com)") {
    Set-Content -Path "C:\Users\pragn\OneDrive\Documents\Default Project\voice-rag\TUNNEL_URL.txt" -Value $Matches[1]
    Write-Host "URL: $($Matches[1])"
} else {
    Write-Host "No URL found. Output:"
    Write-Host $all
}
