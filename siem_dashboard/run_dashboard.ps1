param(
  [int]$Port = 8787
)

Set-Location -Path (Split-Path -Parent $PSScriptRoot)

python -m pip install -r requirements.txt
python -c "import uvicorn, fastapi, psutil; print('deps_ok')"

python -m uvicorn siem_dashboard.server:app --host 127.0.0.1 --port $Port

