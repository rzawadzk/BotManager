#!/usr/bin/env bash
# Bot Engine — post-deploy smoke test (C4)
# ============================================================================
# Boots the scoring server in a subshell, exercises the auth_request
# protocol end-to-end, and exits non-zero if anything looks off. Used
# for:
#   - local dev (`bash tools/smoke_test.sh`)
#   - Docker image verification (CI runs it after `docker compose up`)
#   - post-deploy sanity check on a new host
#
# Unlike `tests/` (pytest unit tests), this talks to a real server on a
# real port. Tests the seams tests can't see: process startup, env
# resolution, socket binding, shutdown drain.
#
# Env knobs:
#   SMOKE_PORT        TCP port to bind (default 19999 — out of the way)
#   SMOKE_TIMEOUT_S   boot timeout in seconds (default 10)
#   SMOKE_KEEP        "1" to leave the server running after the test
# ============================================================================

set -euo pipefail

PORT="${SMOKE_PORT:-19999}"
BOOT_TIMEOUT="${SMOKE_TIMEOUT_S:-10}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Isolated state — don't clobber a running deploy's DB / secret.
TMPDIR="$(mktemp -d -t bot-smoke-XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

export BOT_DB_PATH="$TMPDIR/smoke.db"
export BOT_HMAC_SECRET_FILE="$TMPDIR/hmac_secret"
export BOT_HMAC_SECRET="$(python3 -c 'import secrets;print(secrets.token_hex(32))')"
export BOT_ADMIN_ALLOW_IPS="127.0.0.1,::1"
export BOT_BLOCKLIST_OUTPUT="$TMPDIR/blocklist.conf"

# ── Boot the server in the background ──
echo "[smoke] starting scoring server on :$PORT (state=$TMPDIR)"
python3 "$ROOT/realtime_server.py" --host 127.0.0.1 --port "$PORT" \
    > "$TMPDIR/server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    if [[ "${SMOKE_KEEP:-0}" == "1" ]]; then
      echo "[smoke] SMOKE_KEEP=1 — leaving server pid=$SERVER_PID running"
      return
    fi
    # SIGTERM → graceful drain → process exits on its own.
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    # Give it 5 s to drain before escalating; this exercises the
    # shutdown path too.
    for _ in $(seq 1 50); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

# ── Wait for the health endpoint ──
# The scoring server is ready when /_bot_health returns 200. Poll for
# up to BOOT_TIMEOUT seconds; fail noisily if it doesn't come up.
echo -n "[smoke] waiting for boot"
for i in $(seq 1 "$BOOT_TIMEOUT"); do
  if curl -sf "http://127.0.0.1:$PORT/" \
       -H "X-Real-IP: 127.0.0.1" \
       -H "X-Original-URI: /_bot_health" >/dev/null 2>&1; then
    echo " — ready after ${i}s"
    break
  fi
  echo -n "."
  sleep 1
  if [[ "$i" == "$BOOT_TIMEOUT" ]]; then
    echo ""
    echo "[smoke] ERROR: server failed to boot within ${BOOT_TIMEOUT}s"
    echo "--- server log ---"
    cat "$TMPDIR/server.log"
    exit 1
  fi
done

# Helper — the scoring server routes on X-Original-URI (nginx
# auth_request semantics), not the HTTP request path. Every probe
# needs to spoof the nginx header set.
probe() {
  local uri="$1"
  curl -sf "http://127.0.0.1:$PORT/" \
    -H "X-Real-IP: 127.0.0.1" \
    -H "X-Original-URI: $uri" \
    -H "X-Original-Method: GET"
}

# ── 1. Health endpoint returns JSON with status=ok ──
health="$(probe "/_bot_health")"
if ! echo "$health" | python3 -c 'import json,sys;d=json.load(sys.stdin);assert d["status"]=="ok",d' 2>/dev/null; then
  echo "[smoke] FAIL: /_bot_health did not return status=ok: $health"
  exit 2
fi
echo "[smoke] OK: /_bot_health"

# ── 2. Admin endpoints: localhost is in default allowlist ──
stats="$(probe "/_bot_stats")"
if ! echo "$stats" | python3 -c 'import json,sys;json.load(sys.stdin)' 2>/dev/null; then
  echo "[smoke] FAIL: /_bot_stats did not return JSON: $stats"
  exit 3
fi
echo "[smoke] OK: /_bot_stats"

# ── 3. Prometheus metrics: text exposition format ──
# Any response containing a `# TYPE` directive is a sanity pass.
metrics="$(probe "/_metrics")"
if ! echo "$metrics" | grep -q '^# TYPE '; then
  echo "[smoke] FAIL: /_metrics missing Prometheus TYPE directives"
  exit 4
fi
echo "[smoke] OK: /_metrics"

# ── 4. auth_request path: a benign curl should get an allow verdict ──
# For a "normal" request (not an admin path) we expect an X-Bot-Action
# header back. The server returns an HTTP status + headers that Nginx
# uses to decide whether to forward the original request.
response="$(curl -sfi "http://127.0.0.1:$PORT/" \
    -H "X-Real-IP: 127.0.0.1" \
    -H "X-Original-URI: /" \
    -H "X-Original-Method: GET" \
    -H "User-Agent: smoke-test/1.0")"
action="$(echo "$response" | awk -F': ' '/^X-Bot-Action:/{print $2}' | tr -d '\r')"
if [[ -z "$action" ]]; then
  echo "[smoke] FAIL: no X-Bot-Action header from /_bot_auth"
  echo "$response"
  exit 5
fi
echo "[smoke] OK: /_bot_auth → X-Bot-Action=$action"

# ── 5. Graceful shutdown on SIGTERM ──
# The cleanup trap sends SIGTERM and waits. We manually signal here so
# the exit path reports the result.
kill -TERM "$SERVER_PID"
for _ in $(seq 1 50); do
  kill -0 "$SERVER_PID" 2>/dev/null || break
  sleep 0.1
done
if kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "[smoke] FAIL: server did not exit within 5s of SIGTERM"
  exit 6
fi
echo "[smoke] OK: graceful shutdown"

# Prevent the trap from re-killing the (already-exited) process and
# from printing spurious error.
SERVER_PID=""
trap 'rm -rf "$TMPDIR"' EXIT

echo ""
echo "[smoke] ✅ all checks passed"
