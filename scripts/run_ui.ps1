# Khoi dong UI THI DAU (React) o http://localhost:5173
#
# Chay o TERMINAL RIENG, song song voi .\scripts\run_competition.ps1 (backend).
# Hai tien trinh khac nhau, khong thay the nhau.
#
#     .\scripts\run_ui.ps1
#
# Vi sao can script rieng: mo http://localhost:8000 se ra MOT UI KHAC — API tu
# mount `online/ui` (HTML thuan) tai `/ui` va cho `/` chuyen huong vao do. Do la
# ban demo cu, khong co bang trong so nhanh, khong co tab chinh frame, khong co
# duong nop bai. Xem docs/36 §8.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath (Join-Path $repo "online/ui-react")

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Khong tim thay node. Cai Node.js roi chay lai."
}
if (-not (Test-Path "node_modules")) {
    Write-Host "node_modules chua co — dang chay npm install..."
    npm install
}

# `dist/` la ban build san va CU HON src (07/08 vs 13/08 luc viet dong nay), tuc
# thieu ca tab "Chinh frame" them ngay 10/08. Luon chay tu nguon bang `npm run
# dev` thay vi phuc vu dist, tru khi vua build lai.
$newestSrc  = (Get-ChildItem src -Recurse -File | Sort-Object LastWriteTime | Select-Object -Last 1).LastWriteTime
$newestDist = (Get-ChildItem dist -Recurse -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1).LastWriteTime
if ($newestDist -and $newestDist -lt $newestSrc) {
    Write-Host ("[luu y] dist/ ({0:yyyy-MM-dd}) cu hon src/ ({1:yyyy-MM-dd}) — dang chay tu nguon, dung mo dist." -f $newestDist, $newestSrc)
}

Write-Host ""
Write-Host "UI:      http://localhost:5173"
Write-Host "Backend: http://localhost:8000  (chay .\scripts\run_competition.ps1 o terminal khac)"
Write-Host ""
Write-Host "Lan dau mo UI phai dien 2 o trong QueryStudio:"
Write-Host "  API base : http://localhost:8000"
Write-Host "  Token    : lay bang lenh duoi (luu vao localStorage, chi phai dien mot lan)"
Write-Host ""
$envFile = Join-Path $repo ".env.fpt.local"
if (Test-Path $envFile) {
    $m = Select-String -Path $envFile -Pattern '^AIC_ONLINE_API_KEY=(.+)$'
    if ($m) { Write-Host ("  -> Token: " + $m.Matches.Groups[1].Value) }
}
Write-Host ""

npm run dev
