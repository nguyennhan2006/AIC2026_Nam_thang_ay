$env:AIC_METADATA_JSONL = "storage\exports_competition\scenes.jsonl"
python -m uvicorn online.api.app:app --host 127.0.0.1 --port 8001
