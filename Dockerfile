FROM python:3.10-slim

# System dependencies for OpenCV, PaddlePaddle, and Ollama
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    procps \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama binary
RUN curl -fsSL https://ollama.com/install.sh | sh

# Configure non-root user (UID 1000 required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    OLLAMA_HOST=http://127.0.0.1:11434 \
    OLLAMA_MODEL=qwen2.5:1.5b

WORKDIR /home/user/app

# Install Python dependencies first for Docker caching
COPY --chown=user:user backend/requirements.txt ./backend/
RUN pip install --no-cache-dir --user -r backend/requirements.txt

# Copy remaining repository files
COPY --chown=user:user . .

# Ensure start script is executable and runtime directories exist
RUN chmod +x start.sh && \
    mkdir -p backend/outputs backend/temp backend/model/feedback_data

EXPOSE 7860

CMD ["./start.sh"]
