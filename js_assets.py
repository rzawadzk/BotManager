"""Load client-side JS assets for embedding into HTML templates (C3 #9).

Before C3 the PoW worker, the challenge orchestrator, and the captcha
script all lived as ~550 lines of f-string-templated JS inside
pow_challenge.py. That made minification impossible (terser can't see
the JS), made unit-testing the JS awkward, and meant every edit to the
JS forced a Python-level reload.

Now the JS lives under ``static/js/src/`` as regular files and gets
minified into ``static/js/dist/*.min.js`` by ``tools/build_js.sh``. This
module loads the minified bundle at import time with a fallback to the
un-minified sources if the build step hasn't been run (dev checkout,
test environments, fresh clone). In either mode the loaded bytes become
module-level constants that pow_challenge.py can splice into the HTML
template.

Why not ``<script src="...">``? Two reasons:

  1. The scoring server doesn't serve static files — nginx does, and we
     don't want to couple the challenge page to an extra nginx route or
     a CORS dance for the worker blob.
  2. Inlining means one round-trip for the whole challenge, which
     matters when the user is already being kept waiting by a PoW.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("bot-engine.js-assets")


# Repo-root-relative layout. The loader walks up from this file, not from
# the CWD, so tests and scripts can import it from any working directory.
_HERE = Path(__file__).resolve().parent
_SRC_DIR = _HERE / "static" / "js" / "src"
_DIST_DIR = _HERE / "static" / "js" / "dist"

# Source → minified filename mapping. The dict is the authoritative list
# of assets; adding a new file means one line here plus one line in
# tools/build_js.sh. pow_worker and pow_challenge are the hot path; the
# rest are nice-to-have.
_ASSETS = {
    "pow_worker":    ("pow_worker.js",    "pow_worker.min.js"),
    "pow_challenge": ("pow_challenge.js", "pow_challenge.min.js"),
    "captcha":       ("captcha.js",       "captcha.min.js"),
    "bot_canary":    ("bot_canary.js",    "bot_canary.min.js"),
}


def _read_asset(key: str) -> str:
    """Return the JS text for ``key``, preferring dist over src.

    Fails loudly (``FileNotFoundError``) only if NEITHER file exists —
    that means the repo is broken, not just unbuilt.
    """
    src_name, dist_name = _ASSETS[key]
    dist_path = _DIST_DIR / dist_name
    src_path = _SRC_DIR / src_name

    # Prefer minified — it's what we want in prod. If the dist file is
    # empty (e.g. an aborted build left a zero-byte file) we treat it as
    # missing and fall back to source.
    if dist_path.is_file() and dist_path.stat().st_size > 0:
        logger.debug("js-assets: loaded %s from dist (%d bytes)",
                     key, dist_path.stat().st_size)
        return dist_path.read_text(encoding="utf-8")

    if src_path.is_file():
        logger.info(
            "js-assets: %s dist missing — loading unminified source "
            "from %s. Run `npm run build` for production.",
            key, src_path,
        )
        return src_path.read_text(encoding="utf-8")

    raise FileNotFoundError(
        f"js-assets: neither dist nor src found for {key!r}. "
        f"Looked at: {dist_path}, {src_path}"
    )


# ── Public surface: module-level constants, loaded once ──
# Pattern: eager-load at import so startup cost is paid once per process,
# not once per challenge rendered. A typical total size (minified) is
# under 15 KB, so memory cost is negligible.

POW_WORKER_JS: str = _read_asset("pow_worker")
POW_CHALLENGE_JS: str = _read_asset("pow_challenge")
CAPTCHA_JS: str = _read_asset("captcha")
BOT_CANARY_JS: str = _read_asset("bot_canary")


def reload() -> None:
    """Re-read all assets from disk. For tests / dev hot-reload only.

    Production code should never call this — the module-level constants
    are the intended API. Tests use this to switch between dist and src
    without re-importing the whole module.
    """
    global POW_WORKER_JS, POW_CHALLENGE_JS, CAPTCHA_JS, BOT_CANARY_JS
    POW_WORKER_JS = _read_asset("pow_worker")
    POW_CHALLENGE_JS = _read_asset("pow_challenge")
    CAPTCHA_JS = _read_asset("captcha")
    BOT_CANARY_JS = _read_asset("bot_canary")


def manifest() -> dict:
    """Return a {key: {"source": "dist"|"src", "size": int}} map.

    Useful for startup log lines and ops dashboards so operators can see
    which bundle is live. Called infrequently (startup + debug), so
    re-stats the disk each time rather than caching.
    """
    out = {}
    for key, (src_name, dist_name) in _ASSETS.items():
        dist_path = _DIST_DIR / dist_name
        src_path = _SRC_DIR / src_name
        if dist_path.is_file() and dist_path.stat().st_size > 0:
            out[key] = {"source": "dist", "size": dist_path.stat().st_size}
        elif src_path.is_file():
            out[key] = {"source": "src", "size": src_path.stat().st_size}
        else:
            out[key] = {"source": "missing", "size": 0}
    return out
