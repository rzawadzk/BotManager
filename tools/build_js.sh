#!/usr/bin/env bash
# Bot Engine — client-side JS build (C3 #9)
# ============================================================================
# Minifies the four source files under static/js/src/ into static/js/dist/
# using terser. Python side (js_assets.py) loads dist/*.min.js at import time,
# with a fallback to the un-minified sources if the dist directory is empty
# (so a fresh checkout can still run — pay the size cost but stay functional).
#
# Usage:
#   bash tools/build_js.sh          # build all
#   bash tools/build_js.sh --check  # exit non-zero if dist is stale
#   npm run build                   # same as bare invocation
#
# Requires terser. Install via `npm install --no-save terser` or, if you
# don't want a node_modules at all, `npx terser ...` which will fetch it
# on-demand. The script detects either.
# ============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/static/js/src"
DST="$ROOT/static/js/dist"

# Sources to minify. Order doesn't matter — each is independent.
FILES=(pow_worker.js pow_challenge.js captcha.js bot_canary.js)

# ── Mode ──
CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
fi

# ── Locate terser ──
# Prefer a local install (faster, hermetic). Fall back to npx so devs
# without node_modules can still `bash tools/build_js.sh` and it just
# works — npx fetches terser transparently.
TERSER=""
if [[ -x "$ROOT/node_modules/.bin/terser" ]]; then
  TERSER="$ROOT/node_modules/.bin/terser"
elif command -v terser >/dev/null 2>&1; then
  TERSER="terser"
elif command -v npx >/dev/null 2>&1; then
  TERSER="npx --yes terser"
else
  echo "ERROR: terser not found. Install with:" >&2
  echo "  cd $ROOT && npm install --no-save terser" >&2
  echo "or make sure 'terser' / 'npx' are on PATH." >&2
  exit 2
fi

echo "Using terser: $TERSER"

mkdir -p "$DST"

# ── Build / check loop ──
# Terser flags:
#   --compress  enable shrinking passes (dead-code, constant-fold, inline)
#   --mangle    shorten identifiers — safe because our code is an IIFE
#               (no external consumers)
#   --ecma 2015 we use const/let/arrow-fns in pow_worker; don't downlevel
#   --source-map  emit .map files alongside; useful for debugging but
#                 NOT loaded at runtime by js_assets.py.
# We deliberately do NOT minify bot_canary.js's URL strings away —
# terser's default won't, but if someone bumps the --unsafe flag they
# could, which would defeat the purpose of the canary. Keep flags
# conservative.
stale=0
for f in "${FILES[@]}"; do
  src="$SRC/$f"
  out="${f%.js}.min.js"
  dst="$DST/$out"

  if [[ ! -f "$src" ]]; then
    echo "ERROR: source missing: $src" >&2
    exit 3
  fi

  if [[ "$CHECK_ONLY" == "1" ]]; then
    # Check mode: dist must exist and be newer than source.
    if [[ ! -f "$dst" || "$src" -nt "$dst" ]]; then
      echo "STALE: $dst"
      stale=1
    fi
    continue
  fi

  before=$(wc -c < "$src" | tr -d ' ')
  $TERSER "$src" \
    --compress \
    --mangle \
    --ecma 2015 \
    --source-map "url='$out.map'" \
    --output "$dst"
  after=$(wc -c < "$dst" | tr -d ' ')

  pct=$(( after * 100 / before ))
  printf "  %-22s %7s → %7s bytes (%d%%)\n" "$f" "$before" "$after" "$pct"
done

if [[ "$CHECK_ONLY" == "1" ]]; then
  if [[ "$stale" == "1" ]]; then
    echo ""
    echo "Some dist files are stale. Run: bash tools/build_js.sh"
    exit 1
  fi
  echo "All dist files up to date."
  exit 0
fi

echo "Build complete → $DST"
