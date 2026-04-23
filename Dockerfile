# syntax=docker/dockerfile:1.6
# ═══════════════════════════════════════════════════════════════════════════════
# Bot Engine v2.1 — Multi-stage Docker build
# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 (js-build) minifies client-side JS with terser → static/js/dist/*.
# Stage 2 (runtime)  copies Python code + the prebuilt dist/ and runs.
#
# Why multi-stage: terser needs node + npm at build time; the runtime image
# doesn't need them. Keeping them out trims ~200 MB off the final image and
# reduces the attack surface (fewer packages, no npm in prod).
#
# Why not rely on the committed dist/ files: a fresh-clone-and-build
# workflow (CI, git clone → docker build) still produces a correct image
# without requiring the operator to run `npm run build` by hand. If dist/
# is already present in the build context it's rebuilt anyway, which is
# cheap — the four sources together are under 25 KB.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: JS build ──
FROM node:20-alpine AS js-build

WORKDIR /build
# Install only terser (matches devDependencies in package.json). Bash is
# needed by tools/build_js.sh. --no-save because we don't want the
# node_modules carried into the runtime image.
RUN apk add --no-cache bash && \
    npm install --no-save terser@5

COPY tools/build_js.sh tools/
COPY static/js/src/ static/js/src/
RUN bash tools/build_js.sh


# ── Stage 2: runtime ──
FROM python:3.12-slim AS runtime

WORKDIR /opt/bot-engine

# Install Python deps from requirements.txt so the runtime matches what
# tests/CI install — single source of truth. --no-cache-dir keeps image
# size down; --no-compile skips .pyc generation for the system cache
# (the app writes its own to /opt/bot-engine/__pycache__ at startup).
COPY requirements.txt .
RUN pip install --no-cache-dir --no-compile \
    -r requirements.txt

# Application code. List explicitly — a wildcard COPY . can accidentally
# suck in __pycache__/, .git/, secrets, etc. The .dockerignore provides
# defence in depth.
COPY bot_engine.py \
     pow_challenge.py \
     realtime_server.py \
     dashboard.py \
     redis_state.py \
     db_worker.py \
     js_assets.py \
     train_bot_model.py \
     ./

# Client-side JS: source (fallback), built minified bundle (primary).
# Source is small enough that including it in the runtime image is
# harmless and js_assets.py's dist→src fallback keeps the image
# functional even if dist/ is absent in a degraded build.
COPY static/js/src/ static/js/src/
COPY --from=js-build /build/static/js/dist/ static/js/dist/

# Config directory (example profiles, bot_canary content). Read-only at
# runtime; operators override via env vars or a volume mount.
COPY config/ config/

# State directories. systemd creates these via StateDirectory= in the
# bare-metal deploy; in Docker we do it manually so the non-root user
# can write to them. /var/log/bot-engine is wired to the container's
# stdout by default (the CMD); mounting a volume there gets you files.
RUN mkdir -p /var/lib/bot-engine /run/bot-engine /var/log/bot-engine \
        /var/www/bot-challenge && \
    useradd -r -s /bin/false botengine && \
    chown -R botengine:botengine /var/lib/bot-engine /run/bot-engine \
        /var/log/bot-engine /var/www/bot-challenge /opt/bot-engine

USER botengine

# 9999 = scoring (bound to 0.0.0.0 for bridge networking)
# 8080 = dashboard
EXPOSE 9999 8080

# Container-level healthcheck — docker-compose also defines one, but
# this lets `docker run` users get the same signal without compose.
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9999/_bot_health')" || exit 1

# Default: scoring server on TCP for Docker bridge. Override via
# docker-compose command: for the dashboard container.
CMD ["python3", "realtime_server.py", "--host", "0.0.0.0", "--port", "9999"]
