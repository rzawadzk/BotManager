#!/usr/bin/env python3
"""
Bot Engine v2.1 — Real-Time Scoring Server
============================================
Inline request evaluation via Unix socket, designed for OpenResty/Nginx
auth_request. Every request is scored before it reaches the backend.

Architecture:
  ┌────────────┐   auth_request   ┌──────────────────────────────┐
  │  OpenResty │ ───────────────▶ │  This Server (Unix socket)   │
  │  + Lua     │ ◀─────────────── │  asyncio + uvloop            │
  │  signals   │   200/202/403/   │                              │
  │  capture   │   429 + headers  │  BotScoringEngine v2.1       │
  └────────────┘                  │  ├─ drDNS Verification       │
                                  │  ├─ ONNX ML Inference        │
                                  │  ├─ Session Tracking (IAJ)   │
                                  │  ├─ Agentic AI Detection     │
                                  │  ├─ Deception Engine         │
                                  │  └─ Multi-Batch PoW + HMAC   │
                                  └──────────────────────────────┘

Signals received via headers (set by OpenResty Lua):
  X-Real-IP, X-Forwarded-For, X-Original-URI, X-Original-Method,
  User-Agent, X-JA4-Hash, X-H2-Fingerprint, X-H3-Params,
  X-Header-Order, X-Conn-Timing, X-API-Key, X-Timestamp, X-Signature

Responses:
  200 → Allow          403 → Block
  202 → Challenge      429 → Rate limit
  401 → API auth fail

Response headers (consumed by Nginx):
  X-Bot-Score, X-Bot-Action, X-Bot-Classification
  X-Bot-Drip-Delay (invisible temporal challenge, ms)
  X-Challenge-Redirect, X-Honeypot-Links

Usage:
  python3 realtime_server.py --socket /run/bot-engine/scoring.sock
  python3 realtime_server.py --host 127.0.0.1 --port 9999

Dependencies: Python 3.10+ stdlib.
  Optional: uvloop (pip install uvloop) for ~30% faster event loop.

Author: Rafal — VPS Bot Management
Version: 2.1 — March 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional
from urllib.parse import unquote
import ipaddress
import re

_SAFE_HEADER_RE = re.compile(r"[\x00-\x1f\x7f\r\n]")


def _sanitize_ip(raw: str) -> str:
    """Validate and normalize an IP address. Returns 'invalid' on bad input."""
    raw = raw.strip()
    # Strip port suffix if present (e.g. "1.2.3.4:8080", "[::1]:443")
    if raw.startswith("["):
        bracket_end = raw.find("]")
        if bracket_end != -1:
            raw = raw[1:bracket_end]
    elif "." in raw and ":" in raw:
        raw = raw.rsplit(":", 1)[0]
    try:
        return str(ipaddress.ip_address(raw))
    except ValueError:
        return "invalid"


def _sanitize_header(value: str, max_len: int = 2048) -> str:
    """Strip control characters and truncate."""
    return _SAFE_HEADER_RE.sub("", value)[:max_len]

# ── Import the scoring engine ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from bot_engine import (
        BotScoringEngine, RequestSignals, ThreatScore, ScoreDatabase,
        BlocklistWriter, CONFIG, GOOD_BOT_UAS,
    )
except ImportError:
    print("ERROR: bot_engine.py must be in the same directory or on PYTHONPATH")
    sys.exit(1)

# ── Import challenge / API protection modules ──
try:
    from pow_challenge import (
        ProofOfWorkEngine, ChallengeVerificationHandler,
        BiometricCaptcha, APIProtector,
        generate_challenge_html,
    )
    POW_AVAILABLE = True
except ImportError:
    POW_AVAILABLE = False

# Try to use uvloop for better performance
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# SERVER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SERVER_CONFIG = {
    # ── Socket ──
    "SOCKET_PATH": os.environ.get("BOT_SOCKET", "/run/bot-engine/scoring.sock"),
    "TCP_HOST": "127.0.0.1",
    "TCP_PORT": 9999,

    # ── Performance ──
    "MAX_CONCURRENT": 500,
    "REQUEST_TIMEOUT_MS": 10,
    "KEEPALIVE_TIMEOUT": 60,

    # ── Challenge routing ──
    "CHALLENGE_THRESHOLD": 35,
    "CHALLENGE_PATH": "/_bot_challenge",
    "CAPTCHA_PATH": "/_bot_captcha",
    "CHALLENGE_COOKIE": "bc_pow",
    "CHALLENGE_COOKIE_TTL": 3600,

    # ── Whitelist (IPs that skip scoring entirely) ──
    "WHITELIST_IPS": set(os.environ.get("BOT_WHITELIST_IPS", "").split(",")) - {""},

    # ── Stats ──
    "STATS_LOG_INTERVAL": 60,

    # ── Persistence ──
    "SNAPSHOT_INTERVAL": 300,
    "DB_PATH": CONFIG.get("DB_PATH", "/var/lib/bot-engine/bot_scores.db"),

    # ── API Protection ──
    "API_PREFIX": "/api/",
}


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class Metrics:
    """Performance counters for the scoring server."""

    def __init__(self):
        self.requests_total = 0
        self.requests_allowed = 0
        self.requests_blocked = 0
        self.requests_rate_limited = 0
        self.requests_challenged = 0
        self.requests_whitelisted = 0
        self.requests_api_rejected = 0
        self.requests_honeypot = 0
        self.requests_errors = 0
        self.latency_sum_us = 0
        self.latency_max_us = 0
        self.latency_count = 0
        self._start_time = time.monotonic()

    def record_request(self, action: str, latency_us: int):
        self.requests_total += 1
        self.latency_sum_us += latency_us
        self.latency_count += 1
        if latency_us > self.latency_max_us:
            self.latency_max_us = latency_us

        counter_map = {
            "allow": "requests_allowed",
            "block": "requests_blocked",
            "rate_limit": "requests_rate_limited",
            "challenge": "requests_challenged",
            "api_reject": "requests_api_rejected",
            "honeypot": "requests_honeypot",
        }
        attr = counter_map.get(action)
        if attr:
            setattr(self, attr, getattr(self, attr) + 1)

    def record_error(self):
        self.requests_errors += 1

    def snapshot(self) -> dict:
        elapsed = time.monotonic() - self._start_time
        avg_latency = (
            self.latency_sum_us / self.latency_count
            if self.latency_count > 0 else 0
        )
        rps = self.requests_total / elapsed if elapsed > 0 else 0
        return {
            "uptime_seconds": int(elapsed),
            "total_requests": self.requests_total,
            "rps": round(rps, 1),
            "allowed": self.requests_allowed,
            "blocked": self.requests_blocked,
            "rate_limited": self.requests_rate_limited,
            "challenged": self.requests_challenged,
            "api_rejected": self.requests_api_rejected,
            "honeypot_hits": self.requests_honeypot,
            "whitelisted": self.requests_whitelisted,
            "errors": self.requests_errors,
            "avg_latency_us": round(avg_latency, 1),
            "max_latency_us": self.latency_max_us,
            "engine_tracked_ips": 0,
            "uvloop": UVLOOP_AVAILABLE,
            "pow_available": POW_AVAILABLE,
        }

    def reset_latency(self):
        self.latency_sum_us = 0
        self.latency_max_us = 0
        self.latency_count = 0


# ═══════════════════════════════════════════════════════════════════════════════
# IP RATE TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class RateTracker:
    """Sliding window rate tracker using per-second bucket counts."""

    def __init__(self, window_seconds: int = 10, max_ips: int = 100_000):
        self.window = window_seconds
        self.max_ips = max_ips
        self._data: dict[str, dict[int, int]] = {}

    def record(self, ip: str, now: float) -> int:
        bucket = int(now)
        cutoff = bucket - self.window

        if ip not in self._data:
            if len(self._data) >= self.max_ips:
                evict_ip = next(iter(self._data))
                del self._data[evict_ip]
            self._data[ip] = {}

        buckets = self._data[ip]
        buckets[bucket] = buckets.get(bucket, 0) + 1

        count = 0
        expired = []
        for b, c in buckets.items():
            if b > cutoff:
                count += c
            else:
                expired.append(b)
        for b in expired:
            del buckets[b]

        return count


# ═══════════════════════════════════════════════════════════════════════════════
# CHALLENGE TOKEN STORE
# ═══════════════════════════════════════════════════════════════════════════════

class ChallengeStore:
    """Track which IPs have been challenged and whether they passed."""

    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self.verified: dict[str, float] = {}
        self.pending: dict[str, tuple[str, float]] = {}
        self.failures: dict[str, int] = defaultdict(int)

    def is_verified(self, ip: str) -> bool:
        if ip in self.verified:
            if time.time() < self.verified[ip]:
                return True
            del self.verified[ip]
        return False

    def mark_verified(self, ip: str):
        self.verified[ip] = time.time() + self.ttl

    def issue_challenge(self, ip: str, token: str):
        self.pending[ip] = (token, time.time())

    def verify(self, ip: str, token: str) -> bool:
        if ip not in self.pending:
            return False
        expected_token, issued_at = self.pending[ip]
        if token == expected_token and (time.time() - issued_at) < 60:
            self.verified[ip] = time.time() + self.ttl
            del self.pending[ip]
            return True
        self.failures[ip] += 1
        return False

    def needs_challenge(self, ip: str) -> bool:
        if self.is_verified(ip):
            return False
        if ip in self.pending:
            _, issued_at = self.pending[ip]
            if time.time() - issued_at > 60:
                del self.pending[ip]
                return True
            return False
        return True

    def cleanup(self):
        now = time.time()
        self.verified = {ip: exp for ip, exp in self.verified.items() if exp > now}
        self.pending = {ip: (t, ts) for ip, (t, ts) in self.pending.items() if now - ts < 60}


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP PROTOCOL PARSER
# ═══════════════════════════════════════════════════════════════════════════════

class HTTPParser:
    """Minimal HTTP/1.1 request parser for auth_request subrequests."""

    @staticmethod
    def parse_request(data: bytes) -> Optional[dict]:
        try:
            header_end = data.find(b"\r\n\r\n")
            if header_end == -1:
                return None

            header_block = data[:header_end].decode("utf-8", errors="replace")
            lines = header_block.split("\r\n")

            if not lines:
                return None

            parts = lines[0].split(" ", 2)
            if len(parts) < 2:
                return None

            result = {
                "method": parts[0],
                "path": parts[1],
                "headers": {},
                "header_order": [],
                "body": b"",
            }

            for line in lines[1:]:
                if ": " in line:
                    key, value = line.split(": ", 1)
                    key_lower = key.lower()
                    result["headers"][key_lower] = value
                    result["header_order"].append(key_lower)

            # Extract body if present
            if header_end + 4 < len(data):
                result["body"] = data[header_end + 4:]

            return result

        except Exception:
            return None

    @staticmethod
    def build_response(status_code: int, headers: dict | None = None,
                       body: str = "") -> bytes:
        status_messages = {
            200: "OK", 202: "Accepted", 401: "Unauthorized",
            403: "Forbidden", 429: "Too Many Requests",
            500: "Internal Server Error",
        }
        status_msg = status_messages.get(status_code, "Unknown")

        resp_headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body.encode())),
            "Connection": "close",
        }
        if headers:
            resp_headers.update(headers)

        header_lines = "\r\n".join(f"{k}: {v}" for k, v in resp_headers.items())
        response = f"HTTP/1.1 {status_code} {status_msg}\r\n{header_lines}\r\n\r\n{body}"
        return response.encode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# REAL-TIME SCORING SERVER v2.1
# ═══════════════════════════════════════════════════════════════════════════════

class RealtimeScoringServer:
    """
    Async HTTP server that evaluates every request inline.

    v2.1 integrates:
    - BotScoringEngine with drDNS, ML, session tracking, agentic AI, deception
    - Multi-batch PoW challenges
    - Biometric captcha
    - API HMAC-SHA256 protection
    - Invisible drip temporal challenges
    - Dynamic honeypot link injection
    """

    def __init__(self, config: dict | None = None):
        self.config = config or SERVER_CONFIG
        self.engine = BotScoringEngine()
        self._load_persisted_scores()
        self.metrics = Metrics()
        self.rate_tracker = RateTracker(window_seconds=10)
        self.challenges = ChallengeStore(
            ttl=self.config["CHALLENGE_COOKIE_TTL"]
        )

        # ── PoW engine (multi-batch) ──
        self.pow_engine: Optional[ProofOfWorkEngine] = None
        self.challenge_handler: Optional[ChallengeVerificationHandler] = None
        self.biometric_captcha: Optional[BiometricCaptcha] = None
        self.api_protector: Optional[APIProtector] = None

        if POW_AVAILABLE:
            self.pow_engine = ProofOfWorkEngine()
            self.challenge_handler = ChallengeVerificationHandler(self.pow_engine)
            self.biometric_captcha = BiometricCaptcha()
            self.api_protector = APIProtector()

        # ── Whitelist ──
        self.whitelist = self.config.get("WHITELIST_IPS", set())

        # ── Logging ──
        self.logger = logging.getLogger("bot-engine-rt")
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s"
            ))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # ── Semaphore for concurrency limiting ──
        self.semaphore = asyncio.Semaphore(self.config["MAX_CONCURRENT"])

        self._running = False
        self._server = None

    # ──────────────────────────────────────────────────────────────────────────
    # State recovery
    # ──────────────────────────────────────────────────────────────────────────

    def _load_persisted_scores(self):
        """Load previously saved scores from SQLite on startup."""
        try:
            db = ScoreDatabase(self.config["DB_PATH"])
            scores = db.load_scores()
            db.close()
            if scores:
                self.engine.scores.update(scores)
                logging.getLogger("bot-engine-rt").info(
                    f"[STARTUP] Restored {len(scores)} IP scores from DB"
                )
        except Exception as e:
            logging.getLogger("bot-engine-rt").warning(
                f"[STARTUP] Could not load persisted scores: {e}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Connection handler
    # ──────────────────────────────────────────────────────────────────────────

    async def handle_connection(self, reader: asyncio.StreamReader,
                                writer: asyncio.StreamWriter):
        """Handle a single auth_request connection from Nginx."""
        start_us = time.monotonic_ns() // 1000

        try:
            async with self.semaphore:
                # Read HTTP request with timeout
                try:
                    data = await asyncio.wait_for(
                        reader.read(16384),
                        timeout=self.config["REQUEST_TIMEOUT_MS"] / 1000
                    )
                except asyncio.TimeoutError:
                    self.metrics.record_error()
                    writer.write(HTTPParser.build_response(200))  # Fail open
                    await writer.drain()
                    return

                if not data:
                    return

                parsed = HTTPParser.parse_request(data)
                if not parsed:
                    self.metrics.record_error()
                    writer.write(HTTPParser.build_response(200))  # Fail open
                    await writer.drain()
                    return

                # ── Extract request metadata from headers ──
                headers = parsed["headers"]
                raw_ip = (
                    headers.get("x-real-ip")
                    or headers.get("x-forwarded-for", "").split(",")[0].strip()
                    or "unknown"
                )
                ip = _sanitize_ip(raw_ip) if raw_ip != "unknown" else "unknown"
                original_uri = _sanitize_header(unquote(headers.get("x-original-uri", "/")))
                original_method = _sanitize_header(headers.get("x-original-method", "GET"), 10)
                user_agent = _sanitize_header(headers.get("user-agent", ""), 512)
                ja4_hash = headers.get("x-ja4-hash") or None
                h2_fp = headers.get("x-h2-fingerprint") or None
                h3_params = headers.get("x-h3-params") or None
                cookie = headers.get("cookie", "")
                api_key = headers.get("x-api-key") or None
                api_timestamp = headers.get("x-timestamp") or None
                api_signature = headers.get("x-signature") or None

                header_order_raw = headers.get("x-header-order", "")
                header_order = (
                    header_order_raw.split("|") if header_order_raw else []
                )

                # Check for honeypot hit marker from Nginx
                is_honeypot = headers.get("x-honeypot-hit") == "true"

                # ── Fast path: whitelisted IPs ──
                if ip in self.whitelist:
                    self.metrics.requests_whitelisted += 1
                    response = self._build_decision_response(
                        200, "allow", 0, "whitelisted"
                    )
                    writer.write(response)
                    await writer.drain()
                    return

                # ── Fast path: already verified by challenge ──
                if self.challenges.is_verified(ip):
                    response = self._build_decision_response(
                        200, "allow", 0, "challenge_verified"
                    )
                    writer.write(response)
                    await writer.drain()
                    latency_us = (time.monotonic_ns() // 1000) - start_us
                    self.metrics.record_request("allow", latency_us)
                    return

                # ── Health check endpoint ──
                if original_uri == "/_bot_health":
                    response = HTTPParser.build_response(200, body='{"status":"ok"}')
                    writer.write(response)
                    await writer.drain()
                    return

                # ── Stats endpoint ──
                if original_uri == "/_bot_stats":
                    stats = self.metrics.snapshot()
                    stats["engine_tracked_ips"] = len(self.engine.scores)
                    if self.pow_engine:
                        stats["pow_stats"] = self.pow_engine.get_stats()
                    response = HTTPParser.build_response(
                        200, body=json.dumps(stats)
                    )
                    writer.write(response)
                    await writer.drain()
                    return

                # ── Challenge verification (POST from JS challenge page) ──
                if original_uri == self.config["CHALLENGE_PATH"]:
                    response = self._handle_challenge_verification(ip, headers, parsed)
                    writer.write(response)
                    await writer.drain()
                    return

                # ── Biometric captcha verification ──
                if original_uri == self.config.get("CAPTCHA_PATH", "/_bot_captcha"):
                    response = self._handle_captcha_verification(ip, parsed)
                    writer.write(response)
                    await writer.drain()
                    return

                # ── API HMAC verification for /api/ paths ──
                if (original_uri.startswith(self.config["API_PREFIX"])
                        and self.api_protector and api_key):
                    body_hash = hashlib.sha256(
                        parsed.get("body", b"")
                    ).hexdigest()
                    api_ok, api_reason = self.api_protector.verify_api_request(
                        method=original_method,
                        path=original_uri,
                        headers={
                            "x-api-key": api_key or "",
                            "x-timestamp": api_timestamp or "",
                            "x-signature": api_signature or "",
                        },
                        body_hash=body_hash,
                    )
                    if not api_ok:
                        self.logger.info(
                            f"[API_REJECT] {ip} key={api_key} reason={api_reason}"
                        )
                        response = HTTPParser.build_response(
                            401,
                            headers={"X-Bot-Action": "api_reject"},
                            body=json.dumps({"error": "unauthorized", "reason": api_reason}),
                        )
                        writer.write(response)
                        await writer.drain()
                        latency_us = (time.monotonic_ns() // 1000) - start_us
                        self.metrics.record_request("api_reject", latency_us)
                        return

                # ── Pre-resolve drDNS asynchronously for claimed good bots ──
                drdns_result = None
                if user_agent and GOOD_BOT_UAS.search(user_agent):
                    try:
                        drdns_result = await self.engine.drdns.async_verify(ip)
                    except Exception:
                        drdns_result = (False, "")

                # ── Score the request ──
                req = RequestSignals(
                    ip=ip,
                    timestamp=time.time(),
                    method=original_method,
                    path=original_uri,
                    user_agent=user_agent,
                    ja4_hash=ja4_hash,
                    h2_fingerprint=h2_fp,
                    h3_params=h3_params,
                    header_order=header_order,
                    headers=dict(headers),
                    cookie=cookie,
                    drdns_result=drdns_result,
                )

                # If honeypot hit marker is set, force the path so deception detects it
                if is_honeypot:
                    req.path = original_uri  # Already set, but ensure deception sees it

                threat = self.engine.evaluate(req)

                # ── Track request rate ──
                rate = self.rate_tracker.record(ip, time.time())

                # ── Decision logic ──
                status_code, action = self._decide(threat, rate, ip)

                # ── Build response with extra v2.1 headers ──
                extra_headers = {}

                # Drip challenge delay
                if threat.drip_challenge_pending:
                    drip_ms = CONFIG.get("DRIP_CHALLENGE_DELAY_MS", 150)
                    extra_headers["X-Bot-Drip-Delay"] = str(drip_ms)

                # Honeypot links for injection
                if action == "allow" and hasattr(self.engine, "deception"):
                    links = self.engine.deception.get_honeypot_links()
                    if links:
                        # Send as comma-separated for Nginx sub_filter
                        extra_headers["X-Honeypot-Links"] = ",".join(links[:3])

                response = self._build_decision_response(
                    status_code, action, threat.total_score,
                    threat.classification, extra_headers
                )

                writer.write(response)
                await writer.drain()

                latency_us = (time.monotonic_ns() // 1000) - start_us
                self.metrics.record_request(action, latency_us)

                # Log noteworthy decisions
                if action in ("block", "challenge", "honeypot"):
                    self.logger.info(
                        f"[{action.upper()}] {ip} score={threat.total_score:.1f} "
                        f"reasons={','.join(threat.reasons[:5])} "
                        f"uri={original_uri} ua={user_agent[:60]}"
                    )

        except Exception as e:
            self.metrics.record_error()
            self.logger.error(f"Handler error: {e}", exc_info=True)
            try:
                writer.write(HTTPParser.build_response(200))  # Fail open
                await writer.drain()
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────────
    # Decision engine
    # ──────────────────────────────────────────────────────────────────────────

    def _decide(self, threat: ThreatScore, rate: int, ip: str) -> tuple[int, str]:
        """
        Make the allow/block/challenge/rate_limit/honeypot decision.

        Priority:
          1. Honeypot hit → 403 block (logged as honeypot)
          2. Score >= BLOCK_THRESHOLD → 403 block
          3. Agentic AI detected + high score → 403 block
          4. Score >= CHALLENGE_THRESHOLD and not yet challenged → 202 challenge
          5. Rate exceeds burst limit → 429 rate_limit
          6. Score >= SUSPECT_THRESHOLD → 429 rate_limit
          7. Otherwise → 200 allow
        """
        score = threat.total_score

        # Honeypot hit is an instant block
        if threat.honeypot_hit:
            return 403, "honeypot"

        # Hard block
        if score >= CONFIG["BLOCK_THRESHOLD"]:
            return 403, "block"

        # Agentic AI with high confidence
        if threat.is_agentic_ai and score >= 50:
            return 403, "block"

        # Challenge for medium-score traffic
        if (score >= self.config["CHALLENGE_THRESHOLD"]
                and self.challenges.needs_challenge(ip)):
            if self.pow_engine:
                challenge = self.pow_engine.generate_challenge(ip, score)
                self.challenges.issue_challenge(ip, challenge.challenge_id)
            else:
                token = hashlib.sha256(
                    f"{ip}{time.time()}{os.urandom(8).hex()}".encode()
                ).hexdigest()[:32]
                self.challenges.issue_challenge(ip, token)
            return 202, "challenge"

        # Rate limit if bursting
        if rate > CONFIG.get("BURST_MAX_REQUESTS", 15):
            return 429, "rate_limit"

        # Soft rate limit for suspect
        if score >= CONFIG["SUSPECT_THRESHOLD"]:
            return 429, "rate_limit"

        return 200, "allow"

    # ──────────────────────────────────────────────────────────────────────────
    # Response builders
    # ──────────────────────────────────────────────────────────────────────────

    def _build_decision_response(self, status_code: int, action: str,
                                 score: float, classification: str,
                                 extra_headers: dict | None = None) -> bytes:
        headers = {
            "X-Bot-Score": f"{score:.1f}",
            "X-Bot-Action": action,
            "X-Bot-Classification": classification,
        }

        if action == "challenge":
            headers["X-Challenge-Redirect"] = self.config["CHALLENGE_PATH"]

        if extra_headers:
            headers.update(extra_headers)

        body = json.dumps({
            "action": action,
            "score": round(score, 1),
            "classification": classification,
        })

        return HTTPParser.build_response(status_code, headers, body)

    # ──────────────────────────────────────────────────────────────────────────
    # Challenge & captcha verification handlers
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_challenge_verification(self, ip: str, headers: dict,
                                       parsed: dict) -> bytes:
        """Handle POST from the multi-batch PoW challenge page."""
        if not self.challenge_handler:
            return self._build_decision_response(200, "allow", 0, "pow_unavailable")

        try:
            body = json.loads(parsed.get("body", b"{}"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = {}

        result = self.challenge_handler.handle_verification(ip, body)

        if result.get("verified"):
            self.challenges.mark_verified(ip)
            self.logger.info(f"[CHALLENGE_PASS] {ip}")
            response_headers = {}
            cookie = result.get("cookie")
            if cookie:
                response_headers["Set-Cookie"] = (
                    f"{self.config['CHALLENGE_COOKIE']}={cookie}; "
                    f"Path=/; HttpOnly; Secure; SameSite=Strict; "
                    f"Max-Age={self.config['CHALLENGE_COOKIE_TTL']}"
                )
            return HTTPParser.build_response(
                200,
                headers={
                    "X-Bot-Action": "allow",
                    "X-Bot-Classification": "verified",
                    **response_headers,
                },
                body=json.dumps(result),
            )
        else:
            self.logger.warning(
                f"[CHALLENGE_FAIL] {ip} reasons={result.get('reasons', [])}"
            )
            return self._build_decision_response(403, "block", 100, "challenge_failed")

    def _handle_captcha_verification(self, ip: str, parsed: dict) -> bytes:
        """Handle POST from the biometric captcha page."""
        if not self.biometric_captcha:
            return self._build_decision_response(200, "allow", 0, "captcha_unavailable")

        try:
            body = json.loads(parsed.get("body", b"{}"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = {}

        trace_data = body.get("trace", [])
        passed, score, reasons = self.biometric_captcha.verify_captcha(ip, trace_data)

        if passed:
            self.challenges.mark_verified(ip)
            self.logger.info(f"[CAPTCHA_PASS] {ip} score={score:.1f}")
            cookie = None
            if self.pow_engine:
                cookie = self.pow_engine.get_cookie_value(ip)
            result = {"verified": True, "cookie": cookie}
            resp_headers: dict[str, str] = {
                "X-Bot-Action": "allow",
                "X-Bot-Classification": "verified",
            }
            if cookie:
                resp_headers["Set-Cookie"] = (
                    f"{self.config['CHALLENGE_COOKIE']}={cookie}; "
                    f"Path=/; HttpOnly; Secure; SameSite=Strict; "
                    f"Max-Age={self.config['CHALLENGE_COOKIE_TTL']}"
                )
            return HTTPParser.build_response(
                200, headers=resp_headers,
                body=json.dumps(result),
            )
        else:
            self.logger.warning(f"[CAPTCHA_FAIL] {ip} score={score:.1f} {reasons}")
            return self._build_decision_response(403, "block", score, "captcha_failed")

    # ──────────────────────────────────────────────────────────────────────────
    # Background tasks
    # ──────────────────────────────────────────────────────────────────────────

    async def _stats_logger(self):
        while self._running:
            await asyncio.sleep(self.config["STATS_LOG_INTERVAL"])
            stats = self.metrics.snapshot()
            stats["engine_tracked_ips"] = len(self.engine.scores)
            self.logger.info(f"[STATS] {json.dumps(stats)}")
            self.metrics.reset_latency()

    async def _snapshot_saver(self):
        db: Optional[ScoreDatabase] = None
        while self._running:
            await asyncio.sleep(self.config["SNAPSHOT_INTERVAL"])
            try:
                # Lazy-connect / reconnect on failure
                if db is None:
                    db = ScoreDatabase(self.config["DB_PATH"])

                # Batch save all non-zero scores in one transaction
                to_save = [
                    t for t in self.engine.scores.values()
                    if t.total_score > 0
                ]
                if to_save:
                    saved = db.save_scores_batch(to_save)
                    self.logger.info(f"[SNAPSHOT] Saved {saved} scores to DB")

                # Update Nginx blocklist
                blocklist = self.engine.get_aggregated_blocklist()
                if blocklist:
                    BlocklistWriter.write_nginx_deny(
                        blocklist, CONFIG.get(
                            "BLOCKLIST_OUTPUT",
                            "/etc/nginx/conf.d/dynamic_blocklist.conf"
                        )
                    )
            except Exception as e:
                self.logger.error(f"Snapshot error: {e}", exc_info=True)
                # Force reconnect on next cycle
                if db:
                    try:
                        db.close()
                    except Exception:
                        pass
                    db = None

        # Cleanup on shutdown
        if db:
            db.close()

    async def _challenge_cleanup(self):
        while self._running:
            await asyncio.sleep(60)
            self.challenges.cleanup()
            if self.pow_engine:
                self.pow_engine._evict_expired()

    async def _session_cleanup(self):
        """Evict expired sessions from the tracking subsystem."""
        while self._running:
            await asyncio.sleep(120)
            if hasattr(self.engine, "session_tracker"):
                self.engine.session_tracker.evict_expired()

    # ──────────────────────────────────────────────────────────────────────────
    # Server lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def _start_background_tasks(self):
        asyncio.create_task(self._stats_logger())
        asyncio.create_task(self._snapshot_saver())
        asyncio.create_task(self._challenge_cleanup())
        asyncio.create_task(self._session_cleanup())

    async def start_unix(self, socket_path: str):
        Path(socket_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass

        self._running = True
        self._server = await asyncio.start_unix_server(
            self.handle_connection, path=socket_path,
        )
        os.chmod(socket_path, 0o666)

        self.logger.info(
            f"Bot Engine v2.1 scoring server started on unix:{socket_path} "
            f"(uvloop={'yes' if UVLOOP_AVAILABLE else 'no'}, "
            f"pow={'yes' if POW_AVAILABLE else 'no'}, "
            f"ml={'yes' if self.engine.ml_scorer.is_available else 'no'})"
        )

        self._start_background_tasks()

        async with self._server:
            await self._server.serve_forever()

    async def start_tcp(self, host: str, port: int):
        self._running = True
        self._server = await asyncio.start_server(
            self.handle_connection, host=host, port=port,
        )

        self.logger.info(
            f"Bot Engine v2.1 scoring server started on tcp:{host}:{port} "
            f"(uvloop={'yes' if UVLOOP_AVAILABLE else 'no'}, "
            f"pow={'yes' if POW_AVAILABLE else 'no'}, "
            f"ml={'yes' if self.engine.ml_scorer.is_available else 'no'})"
        )

        self._start_background_tasks()

        async with self._server:
            await self._server.serve_forever()

    def stop(self):
        self._running = False
        if self._server:
            self._server.close()
        self.logger.info("Server stopping...")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Bot Engine v2.1 — Real-Time Scoring Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Production (Unix socket)
  python3 realtime_server.py --socket /run/bot-engine/scoring.sock

  # Debug (TCP)
  python3 realtime_server.py --host 127.0.0.1 --port 9999

  # With IP whitelist
  python3 realtime_server.py --whitelist 10.0.0.1,10.0.0.2
        """
    )

    parser.add_argument("--socket", type=str,
                        default=SERVER_CONFIG["SOCKET_PATH"],
                        help="Unix socket path")
    parser.add_argument("--host", type=str, default=None,
                        help="TCP host (overrides socket)")
    parser.add_argument("--port", type=int, default=SERVER_CONFIG["TCP_PORT"],
                        help="TCP port (used with --host)")
    parser.add_argument("--whitelist", type=str, default="",
                        help="Comma-separated IPs to whitelist")

    args = parser.parse_args()

    if args.whitelist:
        SERVER_CONFIG["WHITELIST_IPS"].update(
            ip.strip() for ip in args.whitelist.split(",") if ip.strip()
        )

    server = RealtimeScoringServer(SERVER_CONFIG)

    loop = asyncio.new_event_loop()

    def signal_handler():
        server.stop()
        loop.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        if args.host:
            loop.run_until_complete(server.start_tcp(args.host, args.port))
        else:
            loop.run_until_complete(server.start_unix(args.socket))
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        loop.close()


if __name__ == "__main__":
    main()
