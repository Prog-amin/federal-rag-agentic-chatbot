FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl git make wget cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Clone llama.cpp and build server
RUN git clone https://github.com/ggerganov/llama.cpp.git /llama.cpp && \
    cd /llama.cpp && \
    mkdir build && cd build && \
    cmake -DLLAMA_SERVER=ON .. && \
    make -j$(nproc)

# Download TinyLlama GGUF model
RUN mkdir -p /models && \
    wget -O /models/TinyLlama.gguf \
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/data/raw /tmp/data/processed templates static

# Set permissions
RUN chmod +x /app/app.py

# Expose the port
EXPOSE 7860

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting llama.cpp server..."\n\
/llama.cpp/build/bin/server -m /models/TinyLlama.gguf --host 0.0.0.0 --port 7861 --ctx-size 2048 --n-predict 512 &\n\
\n\
# Wait for llama.cpp server to start\n\
echo "Waiting for llama.cpp server to start..."\n\
sleep 15\n\
\n\
# Check if server is running\n\
if curl -s http://localhost:7861/health > /dev/null; then\n\
    echo "✅ llama.cpp server is running"\n\
else\n\
    echo "⚠️  llama.cpp server may not be ready yet"\n\
fi\n\
\n\
# Start the FastAPI application\n\
echo "Starting FastAPI application..."\n\
python app.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Use the startup script
CMD ["/app/start.sh"]