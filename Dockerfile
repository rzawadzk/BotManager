FROM python:3.12-slim AS base

WORKDIR /opt/bot-engine

# Install only runtime deps (no build tools needed). `redis` is bundled so
# STATE_BACKEND=redis works out of the box against the redis service in
# docker-compose.yml (C1.2 + C2.2). Everything else is optional and the
# engine degrades gracefully without it.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    uvloop \
    fastapi \
    uvicorn \
    python-multipart \
    onnxruntime \
    numpy \
    redis

# Copy application code
COPY bot_engine.py pow_challenge.py realtime_server.py dashboard.py redis_state.py ./

# Create state directories (HMAC secret file, SQLite DB, challenge page)
RUN mkdir -p /var/lib/bot-engine /run/bot-engine /var/www/bot-challenge

# Non-root user
RUN useradd -r -s /bin/false botengine && \
    chown -R botengine:botengine /var/lib/bot-engine /run/bot-engine /var/www/bot-challenge
USER botengine

# 9999 = scoring HTTP endpoint, 8080 = dashboard
EXPOSE 9999 8080

# Default: start the scoring server on TCP (for Docker bridge networking)
CMD ["python3", "realtime_server.py", "--host", "0.0.0.0", "--port", "9999"]
