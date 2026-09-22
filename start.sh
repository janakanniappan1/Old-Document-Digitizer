#!/bin/bash
set -e

echo "=== Starting Ollama daemon in background ==="
ollama serve > /tmp/ollama.log 2>&1 &

echo "=== Waiting for Ollama to become ready ==="
for i in {1..30}; do
    if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    sleep 1
done

TARGET_MODEL=${OLLAMA_MODEL:-"qwen2.5:1.5b"}
echo "=== Ensuring model $TARGET_MODEL is available ==="
if ! curl -s http://127.0.0.1:11434/api/tags | grep -q "$TARGET_MODEL"; then
    echo "Pulling $TARGET_MODEL in background..."
    ollama pull "$TARGET_MODEL" &
fi

PORT=${PORT:-7860}
echo "=== Starting Gunicorn on port $PORT ==="
cd /home/user/app/backend
exec gunicorn -w 1 -b 0.0.0.0:$PORT --timeout 300 app:app
