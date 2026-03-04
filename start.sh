#!/usr/bin/env bash
# libtcmalloc으로 메모리 관리 최적화
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"
echo "klein-tryon: LD_PRELOAD=${LD_PRELOAD}"

if [ "$SERVE_API_LOCALLY" == "true" ]; then
    echo "klein-tryon: LOCAL API 모드로 시작"
    python3 -u /app/runpod_app.py --rp_serve_api --rp_api_host=0.0.0.0
else
    echo "klein-tryon: SERVERLESS 모드로 시작"
    python3 -u /app/runpod_app.py
fi
