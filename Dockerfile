FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create app user for better security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install system dependencies (including git for cloning)
RUN apt-get update && apt-get install -y \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Set up GitHub token for private repository access
ARG GITHUB_TOKEN
RUN git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"

# Clone your private repository
RUN git clone https://github.com/Prog-amin/-federal-rag-agentic-chatbot2.git /tmp/repo && \
    cp -r /tmp/repo/* /app/ && \
    rm -rf /tmp/repo && \
    git config --global --unset url."https://${GITHUB_TOKEN}@github.com/".insteadOf

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/raw /app/data/processed /app/templates /app/static && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Switch to non-root user
USER appuser

# Expose the port
EXPOSE 7860

# Start the FastAPI application directly
CMD ["python", "app.py"]