#!/bin/bash
# manga-image-translator + JSON proxy dual-process starter
# Part of the custom manga-image-translator build.
set -e

SHARED_PORT=${MT_SHARED_PORT:-5005}
PROXY_PORT=${MT_PROXY_PORT:-5003}

# -- Install extra packages if configured (e.g. paddleocr) --
if [ -n "$MT_EXTRA_PACKAGES" ]; then
    MT_PKG_NORMALIZED=$(echo "$MT_EXTRA_PACKAGES" | tr ',' ' ')
    echo "[mt_start] Installing extra packages: $MT_PKG_NORMALIZED"
    pip install -q $MT_PKG_NORMALIZED && echo "[mt_start] Extra packages installed OK" || echo "[mt_start] WARNING: some packages failed to install"

    # cv2 ABI guard: paddlepaddle pulls in old numpy 1.x opencv
    if ! python3 -c "import cv2" >/dev/null 2>&1; then
        echo "[mt_start] cv2 broken (numpy ABI mismatch), reinstalling opencv..."
        pip install -q --force-reinstall opencv-python-headless || true
    fi
    python3 -c "import cv2; print('[mt_start] cv2 OK:', cv2.__version__)" \
        || echo "[mt_start] WARNING: cv2 still broken after fix attempt"
fi

echo "[mt_start] Starting manga-translator shared API on port $SHARED_PORT..."
export MT_SHARED_URL="http://127.0.0.1:$SHARED_PORT"
python -m manga_translator shared --host 127.0.0.1 --port $SHARED_PORT --nonce None &
MT_PID=$!

echo "[mt_start] Waiting for shared API to be ready..."
for i in $(seq 1 90); do
    if python3 -c "import socket; s=socket.create_connection(('127.0.0.1',$SHARED_PORT),timeout=1); s.close(); print('ok')" 2>/dev/null | grep -q ok; then
        echo "[mt_start] Shared API ready (port open, ${i}s)"
        break
    fi
    sleep 2
done

echo "[mt_start] Starting JSON proxy on port $PROXY_PORT..."
python3 /app/json_proxy.py &
PROXY_PID=$!

echo "[mt_start] Both processes running. MT_PID=$MT_PID PROXY_PID=$PROXY_PID"
while kill -0 $MT_PID 2>/dev/null && kill -0 $PROXY_PID 2>/dev/null; do
    sleep 5
done
echo "[mt_start] A process exited, shutting down..."
kill $MT_PID $PROXY_PID 2>/dev/null || true
