# start_all.ps1
$env:PYTHONPATH = "C:/Users/Dell/mediLearn_agent"
Start-Process powershell -ArgumentList "uvicorn backend.hospital_A:app --port 8001"
Start-Process powershell -ArgumentList "uvicorn backend.hospital_B:app --port 8002"
Start-Process powershell -ArgumentList "uvicorn backend.hospital_C:app --port 8003"
Start-Process powershell -ArgumentList "uvicorn backend.server_main:app --port 8000"
