# Bot Engine v2.1

Real-time bot-detection and request-scoring service designed to sit behind
OpenResty / Nginx via `auth_request`. Scores every inbound request inline,
returns an allow / challenge / block verdict within microseconds, and
writes a dynamic blocklist back to Nginx when abusive IPs cross the
threshold.

## Architecture

```
                         ┌────────────────────┐
                         │  Client (browser/  │
                         │  bot/API consumer) │
                         └──────────┬─────────┘
                                    │ HTTPS
                                    ▼
                         ┌────────────────────┐
                         │  OpenResty / Nginx │
                         │  (TLS, WAF, Lua    │
                         │   signal capture)  │
                         └──────────┬─────────┘
                                    │ auth_request
                                    │ (Unix socket)
                                    ▼
                         ┌────────────────────┐
                         │  realtime_server   │
                         │  asyncio + uvloop  │───▶ Redis (optional)
                         │  ─ scoring engine  │     cross-node state
                         │  ─ PoW challenge   │
                         │  ─ ML inference    │───▶ SQLite
                         │  ─ drip / canary   │     persistence
                         └────────────────────┘
```

| Component | Role |
| --- | --- |
| `realtime_server.py` | Async scoring server. Receives signals from Nginx, returns allow / challenge / block. |
| `bot_engine.py` | Scoring engine: session tracking, drDNS, ML, agentic-AI detection. |
| `pow_challenge.py` | Multi-batch PoW + biometric captcha + HMAC-signed API protection. |
| `db_worker.py` | Dedicated SQLite writer thread with bounded queue and graceful drain. |
| `redis_state.py` | Optional Redis-backed session / rate / challenge stores for multi-node deploys. |
| `dashboard.py` | FastAPI management UI (read-only, IP-allowlisted, basic auth / proxy auth). |
| `nginx/bot_engine.conf` | Reference Nginx integration config. |
| `static/js/` | Client-side JS (PoW worker, captcha, canary), built with terser. |

## Quick start (Docker)

```bash
git clone <your-fork>
cd BotManager
cp .env.example .env
# Generate secrets
python3 -c "import secrets; print('BOT_HMAC_SECRET=' + secrets.token_hex(32))" >> .env
python3 -c "import secrets; print('DASHBOARD_PASS=' + secrets.token_urlsafe(24))" >> .env
# If you plan to use Redis (multi-node):
python3 -c "import secrets; print('REDIS_PASSWORD=' + secrets.token_urlsafe(32))" >> .env
# Review .env, then:
docker compose up -d                       # scoring + dashboard
docker compose --profile redis up -d       # add shared Redis state
docker compose logs -f scoring
```

Scoring listens on `127.0.0.1:9999`, dashboard on `127.0.0.1:8080`.
Put Nginx in front for TLS termination and public exposure.

## Quick start (bare-metal, systemd)

```bash
# As root
install -d -o bot-engine -g bot-engine /opt/bot-engine /var/lib/bot-engine \
    /var/log/bot-engine /var/backups/bot-engine
rsync -a --exclude tests/ --exclude .git/ ./ /opt/bot-engine/
pip3 install -r /opt/bot-engine/requirements.txt

# Secrets (mode 0600)
install -m 0600 /dev/null /etc/bot-engine/env
cat > /etc/bot-engine/env <<EOF
BOT_HMAC_SECRET=$(python3 -c 'import secrets;print(secrets.token_hex(32))')
DASHBOARD_PASS=$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')
EOF

# Units
install -m 0644 systemd/*.service systemd/*.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now bot-engine.service bot-dashboard.service \
    bot-engine-backup.timer

# Nginx
cp nginx/bot_engine.conf /etc/nginx/conf.d/
cp nginx/openresty_signals.lua /etc/nginx/lua/
nginx -t && systemctl reload nginx
```

## Required configuration

| Variable | Required | Default | Notes |
| --- | --- | --- | --- |
| `BOT_HMAC_SECRET` | **yes** | (none) | 32+ chars, no known placeholders. Fails startup in strict mode. |
| `DASHBOARD_PASS` | **yes** | (none) | 12+ chars, not in weak list. Fails dashboard startup. |
| `REDIS_PASSWORD` | yes if `--profile redis` | (none) | Passed via `--requirepass` and `BOT_REDIS_PASSWORD` — never embedded in `REDIS_URL`. |
| `BOT_DB_PATH` | no | `/var/lib/bot-engine/bot_scores.db` | SQLite persistence path. |
| `BOT_ADMIN_ALLOW_IPS` | no | localhost + RFC1918 | CIDRs allowed to hit `/_bot_stats` and `/_metrics`. |
| `BOT_LOG_LEVEL` | no | `INFO` | `DEBUG` for diagnostics. |
| `DASHBOARD_ALLOW_IPS` | no | localhost + RFC1918 | Dashboard network allowlist. |
| `DASHBOARD_TRUST_PROXY_AUTH` | no | `false` | Admit proxy-header auth (oauth2-proxy / authentik / Tailscale). See `.env.example`. |

Full list: see `.env.example` and `config/bot_engine.example.yaml`.

## Observability

- **Health:** `GET /_bot_health` — 200 if all subsystems nominal, 503 otherwise.
- **Stats:** `GET /_bot_stats` — JSON traffic snapshot. **Admin-IP-only**.
- **Prometheus:** `GET /_metrics` — text exposition format. **Admin-IP-only**.
- **Logs:** stdout (or journald under systemd). Set `BOT_LOG_LEVEL=DEBUG`
  to increase verbosity without redeploying.

## Backups & restore

Daily backups via `systemd/bot-engine-backup.{service,timer}` or manually:

```bash
bash tools/backup_db.sh                           # default paths
bash tools/backup_db.sh /custom.db /dest/dir
BOT_BACKUP_RETENTION_DAYS=7 bash tools/backup_db.sh
```

The script uses SQLite's online `.backup` command — safe to run while the
engine is taking writes. Retention defaults to 14 days.

Restore:

```bash
systemctl stop bot-engine
cp /var/backups/bot-engine/bot_scores-<stamp>.db /var/lib/bot-engine/bot_scores.db
chown bot-engine:bot-engine /var/lib/bot-engine/bot_scores.db
systemctl start bot-engine
```

## Smoke-test a deploy

```bash
bash tools/smoke_test.sh             # spawns a server on :19999
SMOKE_PORT=20000 bash tools/smoke_test.sh
```

Exercises health, admin endpoints, the auth_request path, and SIGTERM
drain. Non-zero exit = something's wrong.

## Running the tests

```bash
pip install -r requirements.txt
BOT_STRICT_HMAC=false BOT_HMAC_SECRET_FILE=/tmp/test_hmac python3 -m pytest tests/
```

## Building the client-side JS

JS sources live in `static/js/src/`; minified bundles in `static/js/dist/`.
The minified output is committed so a fresh clone works without npm, but
to rebuild:

```bash
bash tools/build_js.sh         # needs terser (npm / npx)
bash tools/build_js.sh --check # CI: non-zero if dist/ is stale
```

The Docker build does this in the `js-build` stage automatically.

## Security notes

- `BOT_HMAC_SECRET` and the `.hmac_secret` fallback file must be 0600. The
  strict-mode validator rejects world/group-readable files at boot.
- Dashboard password validation is strict by default: unset, placeholder,
  or <12 chars fails startup. Disable only for local dev with
  `DASHBOARD_STRICT_AUTH=false`.
- `/_bot_stats` and `/_metrics` are admin-IP gated, not authenticated. Do
  **not** widen `BOT_ADMIN_ALLOW_IPS` to `0.0.0.0/0`.
- Redis (when enabled) requires a password. The password travels out-of-band
  (`BOT_REDIS_PASSWORD`) so it doesn't appear in `ps` output.
- Graceful shutdown: SIGTERM drains the DB write queue via `db_worker.py`.
  Systemd `TimeoutStopSec=15` is enough for a 10k-item queue on typical disks.

## Version history

| Rev | Highlights |
| --- | --- |
| **C1** | nginx auth cache, Redis state backend, real-traffic ML ingestion. |
| **C2** | Strict HMAC secret, Docker bridge networking, IPv6 /64 grouping, OAuth2-proxy trust. |
| **C3** | Async SQLite write queue, terser build pipeline for client JS. |
| **C4** | Operator-facing hardening: dashboard strict password, admin IP gate, HMAC file perm check, BOT_LOG_LEVEL, Redis auth, Dockerfile multi-stage build, backup tool, smoke test, this README. |

## Layout

```
.
├── bot_engine.py           # Scoring engine
├── pow_challenge.py        # PoW + captcha + API HMAC
├── realtime_server.py      # Async scoring server
├── dashboard.py            # Management UI
├── redis_state.py          # Redis-backed state stores
├── db_worker.py            # SQLite writer thread (C3 #8)
├── js_assets.py            # Loads minified JS (C3 #9)
├── config/                 # Example profiles (incl. bot_canary URL)
├── nginx/                  # Reference Nginx + Lua integration
├── static/js/
│   ├── src/                # Source JS
│   └── dist/               # Minified (committed)
├── systemd/                # Unit files (engine, dashboard, backup timer)
├── tools/
│   ├── build_js.sh         # terser minification
│   ├── backup_db.sh        # SQLite online backup (C4)
│   └── smoke_test.sh       # End-to-end post-deploy check (C4)
└── tests/                  # pytest unit suite
```
