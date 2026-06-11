#!/bin/bash
set -e

echo "[PhishDetect] Starting FastAPI on :8000..."
python -m uvicorn api:app --host 0.0.0.0 --port 8000 &

echo "[PhishDetect] Starting Streamlit on :8501..."
python -m streamlit run app.py \
    --server.headless true \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.enableCORS false \
    --server.enableXsrfProtection false &

wait
