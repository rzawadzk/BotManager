#!/usr/bin/env bash
# Bot Engine — SQLite backup (C4)
# ============================================================================
# Takes a consistent online backup of the scoring DB using SQLite's
# `.backup` command. Unlike `cp`, `.backup` plays nicely with concurrent
# writers — it locks cooperatively and produces a clean WAL-checkpointed
# file even while db_worker is actively committing.
#
# Usage:
#   bash tools/backup_db.sh                             # default paths
#   bash tools/backup_db.sh /custom/src.db /dest/dir    # explicit paths
#   BOT_BACKUP_RETENTION_DAYS=7 bash tools/backup_db.sh # keep 7d, prune older
#
# Cron example (daily at 03:15):
#   15 3 * * * /opt/bot-engine/tools/backup_db.sh >> /var/log/bot-engine/backup.log 2>&1
#
# systemd timer example: see systemd/bot-engine-backup.{service,timer}.
#
# Restore (stop the engine first — the backup is a plain SQLite file):
#   systemctl stop bot-engine
#   cp /var/backups/bot-engine/bot_scores-2026-04-22.db /var/lib/bot-engine/bot_scores.db
#   chown bot-engine:bot-engine /var/lib/bot-engine/bot_scores.db
#   systemctl start bot-engine
# ============================================================================

set -euo pipefail

SRC="${1:-${BOT_DB_PATH:-/var/lib/bot-engine/bot_scores.db}}"
DEST_DIR="${2:-${BOT_BACKUP_DIR:-/var/backups/bot-engine}}"
RETENTION_DAYS="${BOT_BACKUP_RETENTION_DAYS:-14}"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: source DB not found: $SRC" >&2
  exit 1
fi

# Timestamped filename so successive runs don't clobber. Date-level
# granularity is enough — more than one backup per day is cheap
# re-assurance, not extra recoverability.
STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
BASE="$(basename "$SRC" .db)"
DEST="$DEST_DIR/${BASE}-${STAMP}.db"

mkdir -p "$DEST_DIR"

# SQLite's `.backup` is the supported online-backup API. It handles WAL
# checkpointing and produces a self-contained file. Timeout 10 s so a
# pathological lock situation doesn't hang the cron job forever.
#
# We use the .backup dot-command rather than the C API because the
# `sqlite3` CLI is always present where the engine runs (the runtime
# image installs it transitively) and doesn't need any bindings.
sqlite3 "$SRC" ".timeout 10000" ".backup '$DEST'"

# Verify the file is a sane SQLite DB. `PRAGMA integrity_check` does
# deep validation; quick_check is lighter and adequate for a post-
# backup sanity touch.
if ! sqlite3 "$DEST" "PRAGMA quick_check;" | grep -qi "^ok$"; then
  echo "ERROR: backup $DEST failed quick_check" >&2
  exit 2
fi

# File permissions: the DB holds scoring data which includes client IP
# addresses. 0640 (owner rw, group r) matches the engine's StateDirectoryMode.
chmod 0640 "$DEST"
echo "backup: $DEST ($(du -h "$DEST" | cut -f1))"

# Prune old backups. Uses `find -mtime` so time-based pruning tracks
# file mtime, not the parsed name — works even if the STAMP scheme
# changes later.
if [[ "$RETENTION_DAYS" -gt 0 ]]; then
  deleted=$(find "$DEST_DIR" -maxdepth 1 -name "${BASE}-*.db" -type f \
              -mtime +"$RETENTION_DAYS" -print -delete | wc -l | tr -d ' ')
  if [[ "$deleted" -gt 0 ]]; then
    echo "pruned: $deleted backup(s) older than ${RETENTION_DAYS}d"
  fi
fi
