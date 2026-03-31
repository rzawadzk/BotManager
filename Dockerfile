FROM python:3.12-slim AS base

WORKDIR /opt/bot-engine

# Install only runtime deps (no build tools needed)
COPY requirements.txt .
RUN pip install --no-cache-dir uvloop fastapi uvicorn python-multipart onnxruntime numpy

# Copy application code
COPY bot_engine.py pow_challenge.py realtime_server.py dashboard.py ./

# Create state directories
RUN mkdir -p /var/lib/bot-engine /run/bot-engine /var/www/bot-challenge

# Non-root user
RUN useradd -r -s /bin/false botengine && \
    chown -R botengine:botengine /var/lib/bot-engine /run/bot-engine /var/www/bot-challenge
USER botengine

EXPOSE 9999 8080

# Default: start the scoring server on TCP (for Docker networking)
CMD ["python3", "realtime_server.py", "--host", "0.0.0.0", "--port", "9999"]
