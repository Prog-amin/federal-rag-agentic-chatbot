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

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Clone llama.cpp and build server
RUN git clone https://github.com/ggerganov/llama.cpp.git /llama.cpp && \
    cd /llama.cpp && mkdir build && cd build && \
    cmake -DLLAMA_SERVER=ON .. && \
    make -j$(nproc)

# Download a working small model (Qwen 0.5B Chat GGUF)
RUN mkdir -p /models && \
    wget -O /models/qwen0.5b.gguf \
    https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/resolve/main/qwen1_5-0.5b-chat-q4_k_m.gguf

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/data/raw /tmp/data/processed templates static

# Ensure app.py is executable
RUN chmod +x /app/app.py

# Expose the main app port
EXPOSE 7860

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting llama.cpp server..."\n\
/llama.cpp/build/bin/server -m /models/qwen0.5b.gguf --host 0.0.0.0 --port 7861 --ctx-size 2048 --n-predict 512 &\n\
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

# Run everything using the script
CMD ["/app/start.sh"]
