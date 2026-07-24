$ErrorActionPreference = "Stop"

Write-Host "Starting AIC Local Video Search UI..."
python -m streamlit run "$PSScriptRoot\app.py" --server.port 8501
