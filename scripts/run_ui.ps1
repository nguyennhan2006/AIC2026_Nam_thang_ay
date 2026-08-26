# Khoi dong UI THI DAU (React) o http://localhost:5173
#
# Chay o TERMINAL RIENG, song song voi .\scripts\run_competition.ps1 (backend).
# Hai tien trinh khac nhau, khong thay the nhau.
#
#     .\scripts\run_ui.ps1
#
# Vi sao can script rieng: mo http://localhost:8080 se ra MOT UI KHAC - API tu
# mount `online/ui` (HTML thuan) tai `/ui` va cho `/` chuyen huong vao do. Do la
# ban demo cu, khong co bang trong so nhanh, khong co tab chinh frame, khong co
# duong nop bai. Xem docs/36 muc 8.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath (Join-Path $repo "online/ui-react")

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Khong tim thay node. Cai Node.js roi chay lai."
}
if (-not (Test-Path "node_modules")) {
    Write-Host "node_modules chua co - dang chay npm install..."
    npm install
}

# `dist/` la ban build san va CU HON src (07/08 vs 13/08 luc viet dong nay), tuc
# thieu ca tab "Chinh frame" them ngay 10/08. Luon chay tu nguon bang `npm run
# dev` thay vi phuc vu dist, tru khi vua build lai.
$newestSrc  = (Get-ChildItem src -Recurse -File | Sort-Object LastWriteTime | Select-Object -Last 1).LastWriteTime
$newestDist = (Get-ChildItem dist -Recurse -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1).LastWriteTime
if ($newestDist -and $newestDist -lt $newestSrc) {
    Write-Host ("[luu y] dist/ ({0:yyyy-MM-dd}) cu hon src/ ({1:yyyy-MM-dd}) - dang chay tu nguon, dung mo dist." -f $newestDist, $newestSrc)
}

# Backend co the dang chay tren MAY KHAC (Vast.ai). Do o day de nguoi dung biet
# NGAY, thay vi mo UI, go truy van, roi doc mot loi fetch khong noi len dieu gi.
# /v1/health la duong duy nhat khong doi token (xem app.py api_key_guard).
$backendUp = $false
try {
    $null = Invoke-RestMethod -Uri "http://localhost:8080/v1/health" -TimeoutSec 2
    $backendUp = $true
} catch { }

Write-Host ""
Write-Host "UI:      http://localhost:5173"
if ($backendUp) {
    Write-Host "Backend: http://localhost:8080  (dang song)"
} else {
    Write-Host "Backend: KHONG thay gi o http://localhost:8080"
    Write-Host "  - Backend chay ngay tren may nay : mo terminal khac roi chay run_competition.ps1"
    Write-Host "  - Backend chay tren Vast.ai      : mo tunnel o terminal khac, giu nguyen API base:"
    Write-Host "        ssh -p <SSH_PORT> root@<HOST> -L 8080:localhost:8080 -N"
    Write-Host "    hoac dien API base = http://<IP_VAST>:<CONG_DA_MAP> trong QueryStudio"
    Write-Host "    (can uvicorn --host 0.0.0.0 va cong da duoc map luc tao may)."
}
Write-Host ""
Write-Host "Lan dau mo UI phai dien 2 o trong QueryStudio:"
Write-Host "  API base : http://localhost:8080"
Write-Host "  Token    : lay bang lenh duoi (luu vao localStorage, chi phai dien mot lan)"
Write-Host ""
$envFile = Join-Path $repo ".env.fpt.local"
if (Test-Path $envFile) {
    $m = Select-String -Path $envFile -Pattern '^AIC_ONLINE_API_KEY=(.+)$'
    if ($m) {
        Write-Host ("  -> Token: " + $m.Matches.Groups[1].Value)
    } else {
        # Khoa rong nghia la api_key_guard tat han (online/api/app.py), khong
        # phai loi. Noi ro de khoi ai di tim mot token khong ton tai.
        Write-Host "  -> AIC_ONLINE_API_KEY dang rong: de trong o Token la chay duoc."
    }
}
Write-Host ""

npm run dev
