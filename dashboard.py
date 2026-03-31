#!/usr/bin/env python3
"""
Bot Engine Dashboard - FastAPI-based management dashboard for the bot management system.

Usage:
    python3 dashboard.py [--host HOST] [--port PORT] [--db DB_PATH]

Dependencies:
    pip install fastapi uvicorn

Default database path: /var/lib/bot-engine/bot_scores.db
Default listen address: 127.0.0.1:8080
"""

import argparse
import csv
import hmac
import io
import os
import secrets
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

app = FastAPI(title="Bot Engine Dashboard", version="2.1")
security = HTTPBasic(auto_error=False)

DB_PATH = "/var/lib/bot-engine/bot_scores.db"

# ── Basic auth ──
# Set via environment or defaults. Change in production!
DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "admin")
DASHBOARD_PASS = os.environ.get("DASHBOARD_PASS", "changeme")

# ── IP allowlist ──
# Comma-separated list of IPs/CIDRs allowed to access the dashboard.
# Default: localhost only. Set DASHBOARD_ALLOW_IPS="0.0.0.0/0" to allow all.
_raw_allow = os.environ.get("DASHBOARD_ALLOW_IPS", "127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16")
import ipaddress as _ipaddress
DASHBOARD_ALLOW_NETS: list[_ipaddress.IPv4Network | _ipaddress.IPv6Network] = []
for _cidr in _raw_allow.split(","):
    _cidr = _cidr.strip()
    if _cidr:
        try:
            DASHBOARD_ALLOW_NETS.append(_ipaddress.ip_network(_cidr, strict=False))
        except ValueError:
            pass


def _ip_allowed(client_ip: str) -> bool:
    """Check if a client IP is in the allowlist."""
    try:
        addr = _ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    return any(addr in net for net in DASHBOARD_ALLOW_NETS)


def _verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Validate basic auth credentials with constant-time comparison."""
    user_ok = hmac.compare_digest(credentials.username.encode(), DASHBOARD_USER.encode())
    pass_ok = hmac.compare_digest(credentials.password.encode(), DASHBOARD_PASS.encode())
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# ---------------------------------------------------------------------------
# CSS Theme
# ---------------------------------------------------------------------------

CSS = """
:root {
    --bg: #1a1a2e;
    --card: #16213e;
    --card-border: #0f3460;
    --text: #e0e0e0;
    --accent: #00d4ff;
    --danger: #ff4757;
    --warning: #ffa502;
    --success: #2ed573;
    --muted: #888;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
nav {
    background: var(--card);
    border-bottom: 1px solid var(--card-border);
    padding: 12px 20px;
    display: flex;
    gap: 24px;
    align-items: center;
}
nav .brand { font-size: 1.3em; font-weight: 700; color: var(--accent); }
nav a { color: var(--text); font-size: 0.95em; }
nav a:hover { color: var(--accent); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin: 20px 0; }
.card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 20px;
}
.card h3 { color: var(--accent); margin-bottom: 12px; font-size: 1em; text-transform: uppercase; letter-spacing: 1px; }
.stat-big { font-size: 3em; font-weight: 700; font-family: 'Courier New', monospace; }
.mono { font-family: 'Courier New', monospace; }
table { width: 100%; border-collapse: collapse; margin-top: 10px; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--card-border); }
th { color: var(--accent); font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; }
tr:hover { background: rgba(0,212,255,0.05); }
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 0.8em; font-weight: 600;
}
.badge-bad { background: var(--danger); color: #fff; }
.badge-suspect { background: var(--warning); color: #000; }
.badge-good { background: var(--success); color: #000; }
.badge-unknown { background: var(--muted); color: #fff; }
.gauge-wrap { text-align: center; padding: 20px 0; }
.gauge {
    width: 160px; height: 80px; margin: 0 auto;
    border-radius: 160px 160px 0 0;
    background: conic-gradient(
        var(--success) 0deg, var(--success) 90deg,
        var(--warning) 90deg, var(--warning) 180deg,
        var(--danger) 180deg, var(--danger) 270deg,
        var(--danger) 270deg
    );
    position: relative;
    overflow: hidden;
}
.gauge::after {
    content: '';
    position: absolute;
    bottom: 0; left: 20px; right: 20px;
    height: 60px;
    background: var(--card);
    border-radius: 100px 100px 0 0;
}
.donut {
    width: 180px; height: 180px; border-radius: 50%;
    margin: 0 auto;
}
.donut-hole {
    width: 100px; height: 100px; border-radius: 50%;
    background: var(--card);
    position: relative; top: -140px;
    margin: 0 auto;
}
.bar-chart { display: flex; align-items: flex-end; gap: 4px; height: 150px; margin-top: 12px; }
.bar-chart .bar {
    flex: 1;
    background: var(--accent);
    border-radius: 3px 3px 0 0;
    min-width: 8px;
    position: relative;
}
.bar-chart .bar:hover { background: #33dfff; }
.bar-chart .bar .bar-label {
    position: absolute; bottom: -20px; left: 50%; transform: translateX(-50%);
    font-size: 0.6em; color: var(--muted); white-space: nowrap;
}
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 24px; justify-content: center; }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.85em; }
.legend-dot { width: 12px; height: 12px; border-radius: 50%; }
input, select, textarea {
    background: var(--bg);
    border: 1px solid var(--card-border);
    color: var(--text);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.95em;
}
input:focus, select:focus, textarea:focus { outline: none; border-color: var(--accent); }
button, .btn {
    background: var(--accent);
    color: #000;
    border: none;
    padding: 8px 20px;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    font-size: 0.95em;
}
button:hover, .btn:hover { opacity: 0.85; }
.form-row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin: 12px 0; }
.pagination { display: flex; gap: 8px; margin: 20px 0; align-items: center; }
.pagination a, .pagination span {
    padding: 6px 14px; border-radius: 6px;
    background: var(--card); border: 1px solid var(--card-border);
}
.pagination .active { background: var(--accent); color: #000; font-weight: 600; }
.flash { padding: 12px; border-radius: 6px; margin: 12px 0; background: rgba(46,213,115,0.15); border: 1px solid var(--success); }
.wide-card { grid-column: 1 / -1; }
"""

# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------


class DashboardDB:
    """Manages all database interactions for the dashboard."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def ensure_tables(self):
        """Create dashboard-specific tables if they do not exist."""
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ip TEXT NOT NULL,
                    label TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ip TEXT NOT NULL,
                    action TEXT,
                    score REAL,
                    classification TEXT,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.commit()
        finally:
            conn.close()

    # -- helpers --

    @staticmethod
    def _rows_to_dicts(rows) -> list[dict]:
        return [dict(r) for r in rows]

    def _safe_query(self, sql: str, params: tuple = (), fetchone: bool = False):
        """Run a read query, returning dicts. Returns empty on any error."""
        try:
            conn = self._connect()
            cur = conn.execute(sql, params)
            if fetchone:
                row = cur.fetchone()
                result = dict(row) if row else {}
            else:
                result = self._rows_to_dicts(cur.fetchall())
            conn.close()
            return result
        except Exception:
            return {} if fetchone else []

    # -- public API --

    def get_global_stats(self) -> dict:
        """Return total IPs tracked, blocked, challenged, and average score."""
        defaults = {"total_ips": 0, "blocked": 0, "challenged": 0, "avg_score": 0.0}
        try:
            conn = self._connect()
            cur = conn.execute(
                "SELECT COUNT(*) AS total_ips, "
                "COALESCE(SUM(CASE WHEN classification='bad' THEN 1 ELSE 0 END),0) AS blocked, "
                "COALESCE(SUM(CASE WHEN classification='suspect' THEN 1 ELSE 0 END),0) AS challenged, "
                "COALESCE(AVG(score),0) AS avg_score "
                "FROM ip_scores"
            )
            row = cur.fetchone()
            conn.close()
            return dict(row) if row else defaults
        except Exception:
            return defaults

    def get_top_threats(self, limit: int = 50) -> list[dict]:
        return self._safe_query(
            "SELECT ip, score, classification, request_count, last_seen "
            "FROM ip_scores ORDER BY score DESC LIMIT ?",
            (limit,),
        )

    def get_recent_events(self, limit: int = 100) -> list[dict]:
        return self._safe_query(
            "SELECT ip, action, score, classification, timestamp "
            "FROM events ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )

    def get_ip_detail(self, ip: str) -> dict:
        detail = self._safe_query(
            "SELECT * FROM ip_scores WHERE ip = ?", (ip,), fetchone=True
        )
        events = self._safe_query(
            "SELECT * FROM events WHERE ip = ? ORDER BY timestamp DESC LIMIT 50",
            (ip,),
        )
        feedback = self._safe_query(
            "SELECT * FROM feedback WHERE ip = ? ORDER BY created_at DESC", (ip,)
        )
        detail["events"] = events
        detail["feedback"] = feedback
        return detail

    def get_classification_breakdown(self) -> dict:
        defaults = {"good": 0, "suspect": 0, "bad": 0, "unknown": 0}
        try:
            conn = self._connect()
            cur = conn.execute(
                "SELECT classification, COUNT(*) AS cnt FROM ip_scores GROUP BY classification"
            )
            for row in cur.fetchall():
                key = row["classification"] or "unknown"
                defaults[key] = row["cnt"]
            conn.close()
        except Exception:
            pass
        return defaults

    def get_hourly_traffic(self, hours: int = 24) -> list[dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        return self._safe_query(
            "SELECT strftime('%%Y-%%m-%%d %%H:00', timestamp) AS hour, COUNT(*) AS count "
            "FROM events WHERE timestamp >= ? GROUP BY hour ORDER BY hour",
            (cutoff,),
        )

    def save_feedback(self, ip: str, label: str, notes: str):
        try:
            conn = self._connect()
            conn.execute(
                "INSERT INTO feedback (ip, label, notes) VALUES (?, ?, ?)",
                (ip, label, notes),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_feedback(self, limit: int = 500) -> list[dict]:
        return self._safe_query(
            "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,)
        )

    def export_training_data(self) -> list[dict]:
        return self._safe_query(
            "SELECT f.ip, f.label, f.notes, f.created_at, "
            "s.score, s.classification, s.request_count "
            "FROM feedback f LEFT JOIN ip_scores s ON f.ip = s.ip "
            "ORDER BY f.created_at"
        )

    def get_explorer_page(
        self,
        classification: Optional[str] = None,
        search: Optional[str] = None,
        page: int = 1,
        per_page: int = 50,
    ) -> tuple[list[dict], int]:
        """Return a page of IPs and total count, with optional filters."""
        conditions = []
        params: list = []
        if classification:
            conditions.append("classification = ?")
            params.append(classification)
        if search:
            conditions.append("ip LIKE ?")
            params.append(f"%{search}%")
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        try:
            conn = self._connect()
            count_row = conn.execute(
                f"SELECT COUNT(*) AS cnt FROM ip_scores{where}", params
            ).fetchone()
            total = count_row["cnt"] if count_row else 0
            offset = (page - 1) * per_page
            rows = self._rows_to_dicts(
                conn.execute(
                    f"SELECT ip, score, classification, request_count, last_seen "
                    f"FROM ip_scores{where} ORDER BY score DESC LIMIT ? OFFSET ?",
                    params + [per_page, offset],
                ).fetchall()
            )
            conn.close()
            return rows, total
        except Exception:
            return [], 0


def get_db() -> DashboardDB:
    return DashboardDB(DB_PATH)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
def startup():
    get_db().ensure_tables()


# ---------------------------------------------------------------------------
# Auth middleware — protects all routes except /api/health
# ---------------------------------------------------------------------------

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path == "/api/health":
        return await call_next(request)

    # IP allowlist check (before auth, to reject early)
    client_ip = request.client.host if request.client else ""
    if not _ip_allowed(client_ip):
        return JSONResponse(status_code=403, content={"detail": "Forbidden"})

    import base64
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Basic "):
        return JSONResponse(
            status_code=401, content={"detail": "Authentication required"},
            headers={"WWW-Authenticate": "Basic realm=\"Bot Engine Dashboard\""},
        )
    try:
        decoded = base64.b64decode(auth_header[6:]).decode()
        username, password = decoded.split(":", 1)
    except Exception:
        return JSONResponse(
            status_code=401, content={"detail": "Invalid credentials"},
            headers={"WWW-Authenticate": "Basic"},
        )
    user_ok = hmac.compare_digest(username.encode(), DASHBOARD_USER.encode())
    pass_ok = hmac.compare_digest(password.encode(), DASHBOARD_PASS.encode())
    if not (user_ok and pass_ok):
        return JSONResponse(
            status_code=401, content={"detail": "Invalid credentials"},
            headers={"WWW-Authenticate": "Basic"},
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def badge(cls: str) -> str:
    cls = cls or "unknown"
    css = {
        "bad": "badge-bad",
        "suspect": "badge-suspect",
        "good": "badge-good",
    }.get(cls, "badge-unknown")
    return f'<span class="badge {css}">{cls}</span>'


def page_html(title: str, body: str, refresh: int = 0) -> HTMLResponse:
    meta_refresh = (
        f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{meta_refresh}
<title>{title} - Bot Engine Dashboard</title>
<style>{CSS}</style>
</head>
<body>
<nav>
    <span class="brand">Bot Engine Dashboard</span>
    <a href="/">Dashboard</a>
    <a href="/explorer">Traffic Explorer</a>
    <a href="/feedback">ML Feedback</a>
    <a href="/api/health">API Health</a>
</nav>
<div class="container">
{body}
</div>
</body>
</html>"""
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# Dashboard Pages
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def main_dashboard():
    db = get_db()
    stats = db.get_global_stats()
    breakdown = db.get_classification_breakdown()
    threats = db.get_top_threats(limit=10)
    hourly = db.get_hourly_traffic(hours=24)

    # Gauge value (avg score mapped 0-100)
    avg = stats.get("avg_score", 0) or 0
    avg_display = f"{avg:.1f}"

    # Donut chart
    total_bd = sum(breakdown.values()) or 1
    good_pct = breakdown.get("good", 0) / total_bd * 100
    suspect_pct = breakdown.get("suspect", 0) / total_bd * 100
    bad_pct = breakdown.get("bad", 0) / total_bd * 100
    unknown_pct = breakdown.get("unknown", 0) / total_bd * 100

    cum1 = good_pct
    cum2 = cum1 + suspect_pct
    cum3 = cum2 + bad_pct
    donut_grad = (
        f"conic-gradient("
        f"var(--success) 0% {cum1:.1f}%, "
        f"var(--warning) {cum1:.1f}% {cum2:.1f}%, "
        f"var(--danger) {cum2:.1f}% {cum3:.1f}%, "
        f"var(--muted) {cum3:.1f}% 100%"
        f")"
    )

    # Bar chart
    max_count = max((h.get("count", 0) for h in hourly), default=1) or 1
    bars_html = ""
    for h in hourly:
        pct = h.get("count", 0) / max_count * 100
        label = h.get("hour", "")[-5:-3] if h.get("hour") else ""
        bars_html += (
            f'<div class="bar" style="height:{pct:.0f}%" title="{h.get("hour","")}: {h.get("count",0)}">'
            f'<span class="bar-label">{label}h</span></div>'
        )

    # Threats table
    threats_rows = ""
    for t in threats:
        threats_rows += (
            f'<tr>'
            f'<td><a href="/ip/{t.get("ip","")}" class="mono">{t.get("ip","")}</a></td>'
            f'<td class="mono">{t.get("score",0):.1f}</td>'
            f'<td>{badge(t.get("classification",""))}</td>'
            f'<td>{t.get("request_count",0)}</td>'
            f'<td>{t.get("last_seen","")}</td>'
            f'</tr>'
        )

    body = f"""
<h1 style="margin:20px 0 4px">Dashboard</h1>
<p style="color:var(--muted);margin-bottom:20px">Real-time bot threat overview</p>

<div class="grid">
    <div class="card" style="text-align:center">
        <h3>Global Threat Score</h3>
        <div class="gauge-wrap">
            <div class="gauge"></div>
            <div class="stat-big" style="margin-top:-20px;position:relative;z-index:1;
                 color:{'var(--danger)' if avg > 60 else 'var(--warning)' if avg > 30 else 'var(--success)'}">
                {avg_display}
            </div>
            <div style="color:var(--muted);font-size:0.85em">avg score</div>
        </div>
    </div>
    <div class="card" style="text-align:center">
        <h3>Traffic Breakdown</h3>
        <div class="donut" style="background:{donut_grad}"></div>
        <div class="donut-hole" style="display:flex;align-items:center;justify-content:center">
            <span class="stat-big" style="font-size:1.4em">{total_bd}</span>
        </div>
        <div class="legend">
            <span class="legend-item"><span class="legend-dot" style="background:var(--success)"></span> Good ({breakdown.get("good",0)})</span>
            <span class="legend-item"><span class="legend-dot" style="background:var(--warning)"></span> Suspect ({breakdown.get("suspect",0)})</span>
            <span class="legend-item"><span class="legend-dot" style="background:var(--danger)"></span> Bad ({breakdown.get("bad",0)})</span>
            <span class="legend-item"><span class="legend-dot" style="background:var(--muted)"></span> Unknown ({breakdown.get("unknown",0)})</span>
        </div>
    </div>
    <div class="card">
        <h3>Stats</h3>
        <p style="margin:10px 0"><span style="color:var(--accent);font-size:2em;font-weight:700" class="mono">{stats.get("total_ips",0)}</span><br>Total IPs Tracked</p>
        <p style="margin:10px 0"><span style="color:var(--danger);font-size:2em;font-weight:700" class="mono">{stats.get("blocked",0)}</span><br>Blocked</p>
        <p style="margin:10px 0"><span style="color:var(--warning);font-size:2em;font-weight:700" class="mono">{stats.get("challenged",0)}</span><br>Challenged</p>
    </div>
</div>

<div class="grid">
    <div class="card wide-card">
        <h3>Hourly Traffic (last 24h)</h3>
        <div class="bar-chart">
            {bars_html if bars_html else '<div style="color:var(--muted)">No traffic data yet.</div>'}
        </div>
    </div>
</div>

<div class="grid">
    <div class="card wide-card">
        <h3>Top 10 Threats</h3>
        <table>
            <thead><tr><th>IP</th><th>Score</th><th>Classification</th><th>Requests</th><th>Last Seen</th></tr></thead>
            <tbody>
                {threats_rows if threats_rows else '<tr><td colspan="5" style="color:var(--muted)">No data yet.</td></tr>'}
            </tbody>
        </table>
    </div>
</div>
"""
    return page_html("Dashboard", body, refresh=30)


@app.get("/explorer", response_class=HTMLResponse)
def traffic_explorer(
    search: str = Query(default="", alias="search"),
    classification: str = Query(default="", alias="classification"),
    page: int = Query(default=1, alias="page"),
):
    db = get_db()
    per_page = 50
    rows, total = db.get_explorer_page(
        classification=classification or None,
        search=search or None,
        page=page,
        per_page=per_page,
    )
    total_pages = max(1, (total + per_page - 1) // per_page)

    table_rows = ""
    for r in rows:
        table_rows += (
            f'<tr>'
            f'<td><a href="/ip/{r.get("ip","")}" class="mono">{r.get("ip","")}</a></td>'
            f'<td class="mono">{r.get("score",0):.1f}</td>'
            f'<td>{badge(r.get("classification",""))}</td>'
            f'<td>{r.get("request_count",0)}</td>'
            f'<td>{r.get("last_seen","")}</td>'
            f'</tr>'
        )

    # Pagination
    pagination = '<div class="pagination">'
    if page > 1:
        pagination += f'<a href="/explorer?search={search}&classification={classification}&page={page-1}">Prev</a>'
    for p in range(1, total_pages + 1):
        if abs(p - page) < 4 or p == 1 or p == total_pages:
            active = ' class="active"' if p == page else ""
            pagination += f'<a href="/explorer?search={search}&classification={classification}&page={p}"{active}>{p}</a>'
        elif abs(p - page) == 4:
            pagination += '<span>...</span>'
    if page < total_pages:
        pagination += f'<a href="/explorer?search={search}&classification={classification}&page={page+1}">Next</a>'
    pagination += "</div>"

    body = f"""
<h1 style="margin:20px 0 4px">Traffic Explorer</h1>
<p style="color:var(--muted);margin-bottom:20px">Search and filter tracked IPs</p>

<form method="get" action="/explorer" class="form-row">
    <input type="text" name="search" value="{search}" placeholder="Search by IP..." style="min-width:220px">
    <select name="classification">
        <option value="">All Classifications</option>
        <option value="good" {"selected" if classification == "good" else ""}>Good</option>
        <option value="suspect" {"selected" if classification == "suspect" else ""}>Suspect</option>
        <option value="bad" {"selected" if classification == "bad" else ""}>Bad</option>
        <option value="unknown" {"selected" if classification == "unknown" else ""}>Unknown</option>
    </select>
    <button type="submit">Filter</button>
</form>

<div class="card">
    <p style="color:var(--muted);margin-bottom:8px">Showing {len(rows)} of {total} IPs (page {page}/{total_pages})</p>
    <table>
        <thead><tr><th>IP</th><th>Score</th><th>Classification</th><th>Requests</th><th>Last Seen</th></tr></thead>
        <tbody>
            {table_rows if table_rows else '<tr><td colspan="5" style="color:var(--muted)">No results.</td></tr>'}
        </tbody>
    </table>
    {pagination}
</div>
"""
    return page_html("Traffic Explorer", body)


@app.get("/ip/{ip_address}", response_class=HTMLResponse)
def ip_detail_page(ip_address: str):
    db = get_db()
    detail = db.get_ip_detail(ip_address)
    events = detail.pop("events", [])
    feedback_items = detail.pop("feedback", [])

    # Build detail fields
    detail_rows = ""
    for k, v in detail.items():
        detail_rows += f"<tr><td><strong>{k}</strong></td><td class='mono'>{v}</td></tr>"

    # Events timeline
    events_rows = ""
    for e in events:
        events_rows += (
            f'<tr><td>{e.get("timestamp","")}</td>'
            f'<td>{e.get("action","")}</td>'
            f'<td class="mono">{e.get("score","")}</td>'
            f'<td>{badge(e.get("classification",""))}</td></tr>'
        )

    # Feedback history
    fb_rows = ""
    for f in feedback_items:
        fb_rows += (
            f'<tr><td>{f.get("created_at","")}</td>'
            f'<td>{f.get("label","")}</td>'
            f'<td>{f.get("notes","")}</td></tr>'
        )

    body = f"""
<h1 style="margin:20px 0 4px" class="mono">{ip_address}</h1>
<p style="color:var(--muted);margin-bottom:20px">IP detail and scoring breakdown</p>

<div class="grid">
    <div class="card">
        <h3>Scoring Breakdown</h3>
        <table>
            <tbody>
                {detail_rows if detail_rows else '<tr><td colspan="2" style="color:var(--muted)">No data for this IP.</td></tr>'}
            </tbody>
        </table>
    </div>
    <div class="card">
        <h3>Human-in-the-Loop Feedback</h3>
        <form method="post" action="/api/feedback">
            <input type="hidden" name="ip" value="{ip_address}">
            <div class="form-row">
                <select name="label" required>
                    <option value="">Select label...</option>
                    <option value="human">Human</option>
                    <option value="good_bot">Good Bot</option>
                    <option value="bad_bot">Bad Bot</option>
                </select>
            </div>
            <div class="form-row">
                <textarea name="notes" rows="3" placeholder="Optional notes..." style="width:100%"></textarea>
            </div>
            <div class="form-row">
                <button type="submit">Submit Feedback</button>
            </div>
        </form>
        {('<h3 style="margin-top:20px">Previous Feedback</h3>'
          '<table><thead><tr><th>Date</th><th>Label</th><th>Notes</th></tr></thead><tbody>'
          + fb_rows + '</tbody></table>') if fb_rows else ''}
    </div>
</div>

<div class="grid">
    <div class="card wide-card">
        <h3>Recent Events</h3>
        <table>
            <thead><tr><th>Timestamp</th><th>Action</th><th>Score</th><th>Classification</th></tr></thead>
            <tbody>
                {events_rows if events_rows else '<tr><td colspan="4" style="color:var(--muted)">No events recorded.</td></tr>'}
            </tbody>
        </table>
    </div>
</div>
"""
    return page_html(f"IP {ip_address}", body)


@app.get("/feedback", response_class=HTMLResponse)
def feedback_page():
    db = get_db()
    items = db.get_feedback(limit=500)

    # Label stats
    label_counts: dict[str, int] = {}
    for item in items:
        lbl = item.get("label", "unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    stats_html = ""
    for lbl, cnt in sorted(label_counts.items()):
        stats_html += f'<span class="badge badge-unknown" style="margin-right:8px">{lbl}: {cnt}</span>'

    rows = ""
    for item in items:
        rows += (
            f'<tr>'
            f'<td><a href="/ip/{item.get("ip","")}" class="mono">{item.get("ip","")}</a></td>'
            f'<td>{item.get("label","")}</td>'
            f'<td>{item.get("notes","")}</td>'
            f'<td>{item.get("created_at","")}</td>'
            f'</tr>'
        )

    body = f"""
<h1 style="margin:20px 0 4px">ML Feedback</h1>
<p style="color:var(--muted);margin-bottom:20px">Human-in-the-loop labels for model retraining</p>

<div class="grid">
    <div class="card">
        <h3>Label Distribution</h3>
        <div style="margin:12px 0">{stats_html if stats_html else '<span style="color:var(--muted)">No feedback yet.</span>'}</div>
        <p style="margin-top:12px">Total entries: <strong>{len(items)}</strong></p>
    </div>
    <div class="card">
        <h3>Export</h3>
        <p style="margin:12px 0">Download labeled data as CSV for ML retraining.</p>
        <a href="/api/export-training-data" class="btn" style="display:inline-block;text-decoration:none">Export Training Data (CSV)</a>
    </div>
</div>

<div class="card" style="margin-top:20px">
    <h3>All Feedback Entries</h3>
    <table>
        <thead><tr><th>IP</th><th>Label</th><th>Notes</th><th>Date</th></tr></thead>
        <tbody>
            {rows if rows else '<tr><td colspan="4" style="color:var(--muted)">No feedback entries yet.</td></tr>'}
        </tbody>
    </table>
</div>
"""
    return page_html("ML Feedback", body)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def api_health():
    return {"status": "ok"}


@app.get("/api/stats")
def api_stats():
    return get_db().get_global_stats()


@app.get("/api/threats")
def api_threats(limit: int = Query(default=50)):
    return get_db().get_top_threats(limit=limit)


@app.get("/api/ip/{ip}")
def api_ip_detail(ip: str):
    return get_db().get_ip_detail(ip)


@app.post("/api/feedback")
def api_feedback(
    ip: str = Form(...),
    label: str = Form(...),
    notes: str = Form(default=""),
):
    get_db().save_feedback(ip, label, notes)
    # Redirect back to the IP detail page after submission
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url=f"/ip/{ip}", status_code=303)


@app.get("/api/export-training-data")
def api_export_training_data():
    data = get_db().export_training_data()
    output = io.StringIO()
    if data:
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    else:
        output.write("ip,label,notes,created_at,score,classification,request_count\n")
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=training_data.csv"},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Bot Engine Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--db", default="/var/lib/bot-engine/bot_scores.db")
    args = parser.parse_args()
    DB_PATH = args.db
    uvicorn.run(app, host=args.host, port=args.port)
