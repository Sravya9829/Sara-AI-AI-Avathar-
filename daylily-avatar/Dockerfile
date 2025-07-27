# Daylily AI Avatar Challenge - Docker with Audio Support
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio and AI
RUN apt-get update && apt-get install -y \
    ffmpeg \
    portaudio19-dev \
    python3-pyaudio \
    alsa-utils \
    pulseaudio \
    pulseaudio-utils \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for FFmpeg
RUN mkdir -p ffmpeg/bin

# Set audio environment variables
ENV PULSE_RUNTIME_PATH=/tmp/pulse-socket

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "main.py"]