#!/usr/bin/env python3
"""
Bot Engine — Core Scoring Engine v2.1
=======================================
Provides the BotScoringEngine, RequestSignals, ThreatScore, and supporting
subsystems for advanced bot detection including:

- Double-Reverse DNS verification (drDNS) for crawler identity
- ONNX ML model inference (graceful degradation)
- Session tracking with temporal jitter and identity drift
- Agentic AI countermeasures (micro-jitter biometrics, drip challenges)
- Deception engine (honeypot paths, semantic probing traps)

All operations are designed for the scoring hot path (<1ms rule-based,
<5ms with ML inference). DNS lookups are cached. ONNX and numpy are
optional dependencies — the engine degrades gracefully without them.

Author: Rafal — VPS Bot Management
Version: 2.1 — March 2026
"""

from __future__ import annotations

import hashlib
import ipaddress
import math
import os
import re
import socket
import sqlite3
import statistics
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Optional ML dependencies ────────────────────────────────────────────────
try:
    import numpy as np
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    ort = None  # type: ignore[assignment]
    _ONNX_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Thresholds ──
    "BLOCK_THRESHOLD": 70,
    "SUSPECT_THRESHOLD": 40,
    "BURST_MAX_REQUESTS": 15,

    # ── Persistence ──
    "DB_PATH": os.environ.get(
        "BOT_DB_PATH", "/var/lib/bot-engine/bot_scores.db"
    ),
    "BLOCKLIST_OUTPUT": os.environ.get(
        "BOT_BLOCKLIST_OUTPUT", "/etc/nginx/conf.d/dynamic_blocklist.conf"
    ),

    # ── ML / ONNX ──
    "ONNX_MODEL_PATH": os.environ.get(
        "BOT_ONNX_MODEL_PATH", "/var/lib/bot-engine/bot_model.onnx"
    ),

    # ── drDNS ──
    "DRDNS_CACHE_TTL": 3600,

    # ── Session Tracking ──
    "SESSION_WINDOW_SECONDS": 300,

    # ── Honeypot / Deception ──
    "HONEYPOT_PATHS": [
        "/api/v2/internal/users",
        "/admin/export.csv",
        "/backup/db.sql",
        "/.env",
        "/wp-admin/setup-config.php",
        "/debug/vars",
        "/server-status",
        "/api/v1/admin/keys",
        "/config/database.yml",
        "/.git/config",
    ],

    # ── Drip Challenge ──
    "DRIP_CHALLENGE_DELAY_MS": 150,

    # ── Agentic AI ──
    "MICRO_JITTER_THRESHOLD": 0.02,
}


def load_config_file(path: str) -> dict:
    """Load a YAML or TOML config file and merge into CONFIG.

    Returns the merged config dict. Supports .yaml/.yml and .toml extensions.
    Unknown keys are silently accepted to allow forward-compatible config files.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml  # type: ignore[import-untyped]
        with open(p) as f:
            user_config = yaml.safe_load(f) or {}
    elif suffix == ".toml":
        import tomllib  # Python 3.11+
        with open(p, "rb") as f:
            user_config = tomllib.load(f)
    else:
        raise ValueError(f"Unsupported config format '{suffix}'. Use .yaml, .yml, or .toml")

    if not isinstance(user_config, dict):
        raise ValueError("Config file must contain a top-level mapping")

    CONFIG.update(user_config)
    return CONFIG


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RequestSignals:
    """All signals extracted from a single HTTP request."""

    ip: str
    timestamp: float
    method: str = "GET"
    path: str = "/"
    user_agent: str = ""
    ja4_hash: Optional[str] = None
    h2_fingerprint: Optional[str] = None
    h3_params: Optional[str] = None
    header_order: list = field(default_factory=list)
    headers: dict = field(default_factory=dict)
    cookie: str = ""
    body_hash: Optional[str] = None
    api_signature: Optional[str] = None
    telemetry: Optional[dict] = None
    # Pre-resolved drDNS result (set by async server to avoid blocking)
    # None = not checked yet, tuple = (verified, identity)
    drdns_result: Optional[tuple] = None


@dataclass
class ThreatScore:
    """Aggregated threat score for an IP / session."""

    ip: str
    total_score: float = 0.0
    classification: str = "unknown"  # good | suspect | bad | unknown
    reasons: list = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    request_count: int = 0
    ml_score: float = 0.0
    identity_verified: bool = False
    session_id: Optional[str] = None
    is_agentic_ai: bool = False
    honeypot_hit: bool = False
    drip_challenge_pending: bool = False

    def classify(self, config: dict = CONFIG) -> None:
        """Classify the threat based on total_score and thresholds."""
        if self.total_score >= config["BLOCK_THRESHOLD"]:
            self.classification = "bad"
        elif self.total_score >= config["SUSPECT_THRESHOLD"]:
            self.classification = "suspect"
        elif self.total_score > 0:
            self.classification = "unknown"
        else:
            self.classification = "good"


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWN BOT PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

GOOD_BOT_UAS = re.compile(
    r"(Googlebot|Bingbot|Slurp|DuckDuckBot|Baiduspider|YandexBot"
    r"|facebookexternalhit|Twitterbot|LinkedInBot|Applebot"
    r"|UptimeRobot|Pingdom|Site24x7)",
    re.IGNORECASE,
)

BAD_BOT_UAS = re.compile(
    r"(python-requests|python-urllib|curl/|wget/|scrapy|httpclient"
    r"|java/|libwww-perl|Go-http-client|node-fetch|axios"
    r"|PhantomJS|HeadlessChrome|Selenium|Puppeteer|Playwright)",
    re.IGNORECASE,
)

PROBE_PATHS = re.compile(
    r"(\.env|\.git|wp-admin|wp-login|phpinfo|\.php$|/admin"
    r"|/config|/backup|\.sql|\.bak|/debug|/actuator|/\.)",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# DOUBLE-REVERSE DNS VERIFICATION (drDNS)
# ═══════════════════════════════════════════════════════════════════════════════

class DrDNSVerifier:
    """
    Verify that IPs claiming to be known crawlers actually belong to those
    organisations using double-reverse DNS: IP -> PTR -> forward A -> match IP.

    Also maintains a CIDR allowlist for fast-path verification of known
    crawler IP ranges.
    """

    # Known crawler CIDR ranges
    KNOWN_CIDRS: dict[str, list[ipaddress.IPv4Network | ipaddress.IPv6Network]] = {
        "google": [
            ipaddress.ip_network("66.249.64.0/19"),
            ipaddress.ip_network("64.233.160.0/19"),
        ],
        "bing": [
            ipaddress.ip_network("40.77.167.0/24"),
            ipaddress.ip_network("157.55.39.0/24"),
            ipaddress.ip_network("207.46.13.0/24"),
        ],
        "facebook": [
            ipaddress.ip_network("69.63.176.0/20"),
            ipaddress.ip_network("66.220.144.0/20"),
        ],
        "apple": [
            ipaddress.ip_network("17.0.0.0/8"),
        ],
    }

    # PTR domain suffixes that map to known crawlers
    DOMAIN_MAP: dict[str, str] = {
        ".googlebot.com": "google",
        ".google.com": "google",
        ".search.msn.com": "bing",
        ".crawl.yahoo.net": "yahoo",
        ".yandex.com": "yandex",
        ".yandex.ru": "yandex",
        ".yandex.net": "yandex",
        ".facebook.com": "facebook",
        ".fbsv.net": "facebook",
        ".apple.com": "apple",
        ".applebot.apple.com": "apple",
    }

    def __init__(self, cache_ttl: int = CONFIG["DRDNS_CACHE_TTL"]) -> None:
        self._cache: dict[str, tuple[float, bool, str]] = {}
        self._cache_ttl = cache_ttl
        self._lock = threading.Lock()

    def _check_cidr(self, ip: str) -> tuple[bool, str]:
        """Fast-path: check if IP falls within a known crawler CIDR range."""
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False, ""
        for identity, networks in self.KNOWN_CIDRS.items():
            for net in networks:
                if addr in net:
                    return True, identity
        return False, ""

    def _ptr_lookup(self, ip: str) -> Optional[str]:
        """Perform reverse DNS lookup: IP -> hostname."""
        try:
            hostname, _aliases, _addrs = socket.gethostbyaddr(ip)
            return hostname
        except (socket.herror, socket.gaierror, OSError):
            return None

    def _forward_lookup(self, hostname: str) -> list[str]:
        """Perform forward DNS lookup: hostname -> IPs."""
        try:
            _name, _aliases, addrs = socket.gethostbyname_ex(hostname)
            return addrs
        except (socket.herror, socket.gaierror, OSError):
            return []

    def _identify_by_ptr(self, hostname: str) -> str:
        """Match a PTR hostname to a known crawler identity."""
        hostname_lower = hostname.lower()
        for suffix, identity in self.DOMAIN_MAP.items():
            if hostname_lower.endswith(suffix):
                return identity
        return ""

    def verify(self, ip: str) -> tuple[bool, str]:
        """
        Verify an IP address via double-reverse DNS.

        Returns:
            (is_verified, identity) where identity is e.g. 'google', 'bing', ''
        """
        # Check cache first
        with self._lock:
            if ip in self._cache:
                ts, verified, identity = self._cache[ip]
                if time.time() - ts < self._cache_ttl:
                    return verified, identity

        # Fast-path: CIDR check
        cidr_match, cidr_identity = self._check_cidr(ip)
        if cidr_match:
            with self._lock:
                self._cache[ip] = (time.time(), True, cidr_identity)
            return True, cidr_identity

        # Full drDNS: IP -> PTR -> forward -> match
        try:
            hostname = self._ptr_lookup(ip)
            if not hostname:
                with self._lock:
                    self._cache[ip] = (time.time(), False, "")
                return False, ""

            identity = self._identify_by_ptr(hostname)
            if not identity:
                with self._lock:
                    self._cache[ip] = (time.time(), False, "")
                return False, ""

            # Forward lookup to confirm IP matches
            resolved_ips = self._forward_lookup(hostname)
            verified = ip in resolved_ips

            with self._lock:
                self._cache[ip] = (time.time(), verified, identity if verified else "")
            return verified, identity if verified else ""

        except Exception:
            # Never crash the scoring hot path
            with self._lock:
                self._cache[ip] = (time.time(), False, "")
            return False, ""

    async def async_verify(self, ip: str) -> tuple[bool, str]:
        """Async wrapper — runs blocking DNS lookups in a thread pool."""
        import asyncio
        return await asyncio.to_thread(self.verify, ip)

    def evict_expired(self) -> int:
        """Remove expired cache entries. Returns number evicted."""
        now = time.time()
        with self._lock:
            expired = [
                ip for ip, (ts, _, _) in self._cache.items()
                if now - ts >= self._cache_ttl
            ]
            for ip in expired:
                del self._cache[ip]
            return len(expired)


# ═══════════════════════════════════════════════════════════════════════════════
# ONNX ML INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

class MLScorer:
    """
    Score requests using an ONNX ML model.

    Gracefully degrades when onnxruntime or numpy are not installed, or
    when the model file does not exist. In degraded mode, predict()
    always returns 0.0.
    """

    N_FEATURES = 16

    # ── Staleness thresholds ──
    MAX_MODEL_AGE_DAYS = 30       # Flag if model file older than this
    DRIFT_WINDOW = 1000           # Rolling window size for prediction stats
    DRIFT_THRESHOLD = 0.15        # Flag if mean prediction shifts by this much

    def __init__(self, model_path: str = CONFIG["ONNX_MODEL_PATH"]) -> None:
        self._session: object | None = None
        self._input_name: str = ""
        self._available = False
        self._model_path = model_path
        self._model_loaded_at: float = 0.0
        self._model_file_mtime: float = 0.0

        # Staleness tracking: rolling prediction distribution
        self._predictions: list[float] = []
        self._baseline_mean: float | None = None
        self._feedback_matches = 0
        self._feedback_total = 0

        if not _ONNX_AVAILABLE:
            return

        if not os.path.isfile(model_path):
            return

        try:
            self._session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            self._available = True
            self._model_loaded_at = time.time()
            self._model_file_mtime = os.path.getmtime(model_path)
        except Exception:
            self._session = None

    @property
    def is_available(self) -> bool:
        """Whether the ML model is loaded and ready for inference."""
        return self._available

    def staleness_report(self) -> dict:
        """
        Return a staleness diagnostic for the current ML model.

        Checks:
          - model_age_days: how old the model file is
          - prediction_drift: whether the rolling mean prediction has shifted
            significantly from the baseline (first DRIFT_WINDOW predictions)
          - feedback_accuracy: if human feedback has been provided, what fraction
            of predictions agreed with the human label
          - needs_retrain: True if any staleness indicator is triggered
        """
        report: dict = {
            "model_loaded": self._available,
            "model_age_days": 0.0,
            "prediction_drift": 0.0,
            "feedback_accuracy": None,
            "needs_retrain": False,
            "reasons": [],
        }

        if not self._available:
            report["reasons"].append("no_model_loaded")
            return report

        # Model age
        age_days = (time.time() - self._model_file_mtime) / 86400
        report["model_age_days"] = round(age_days, 1)
        if age_days > self.MAX_MODEL_AGE_DAYS:
            report["needs_retrain"] = True
            report["reasons"].append(f"model_age_{age_days:.0f}d")

        # Prediction drift
        if len(self._predictions) >= self.DRIFT_WINDOW:
            if self._baseline_mean is not None:
                current_mean = sum(self._predictions[-self.DRIFT_WINDOW:]) / self.DRIFT_WINDOW
                drift = abs(current_mean - self._baseline_mean)
                report["prediction_drift"] = round(drift, 4)
                if drift > self.DRIFT_THRESHOLD:
                    report["needs_retrain"] = True
                    report["reasons"].append(f"prediction_drift_{drift:.3f}")

        # Feedback accuracy
        if self._feedback_total >= 20:
            accuracy = self._feedback_matches / self._feedback_total
            report["feedback_accuracy"] = round(accuracy, 3)
            if accuracy < 0.7:
                report["needs_retrain"] = True
                report["reasons"].append(f"low_feedback_accuracy_{accuracy:.2f}")

        return report

    def record_prediction(self, score: float) -> None:
        """Track a prediction for drift monitoring."""
        self._predictions.append(score)
        # Set baseline from first full window
        if self._baseline_mean is None and len(self._predictions) == self.DRIFT_WINDOW:
            self._baseline_mean = sum(self._predictions) / self.DRIFT_WINDOW
        # Cap memory usage
        if len(self._predictions) > self.DRIFT_WINDOW * 3:
            self._predictions = self._predictions[-self.DRIFT_WINDOW * 2:]

    def record_feedback(self, predicted_bad: bool, actual_bad: bool) -> None:
        """Record human feedback to track prediction accuracy."""
        self._feedback_total += 1
        if predicted_bad == actual_bad:
            self._feedback_matches += 1

    @staticmethod
    def _hash_to_int(h: Optional[str]) -> int:
        """Convert a hex-like hash string to a bounded integer feature."""
        if not h:
            return 0
        try:
            return int(hashlib.md5(h.encode("utf-8", errors="replace")).hexdigest()[:8], 16) % 1_000_000
        except Exception:
            return 0

    @staticmethod
    def _encode_method(method: str) -> int:
        """Encode HTTP method as integer."""
        mapping = {"GET": 1, "POST": 2, "PUT": 3, "DELETE": 4, "PATCH": 5,
                   "HEAD": 6, "OPTIONS": 7}
        return mapping.get(method.upper(), 0)

    def extract_features(self, req: RequestSignals) -> list[float]:
        """
        Extract a fixed-length feature vector from RequestSignals.

        Features (16 total):
         0: JA4 hash encoded as int
         1: H2 fingerprint encoded as int
         2: H3 params present (bool)
         3: User-Agent length
         4: Header count
         5: Path depth (number of / segments)
         6: HTTP method encoded
         7: Has Accept-Language header (bool)
         8: Has cookie (bool)
         9: Has body hash (bool)
        10: Has API signature (bool)
        11: Header order length
        12: User-Agent has version-like substring (bool)
        13: Path contains query-like pattern (bool)
        14: Has Accept header (bool)
        15: Has telemetry (bool)
        """
        ua = req.user_agent or ""
        headers = req.headers or {}
        return [
            float(self._hash_to_int(req.ja4_hash)),
            float(self._hash_to_int(req.h2_fingerprint)),
            1.0 if req.h3_params else 0.0,
            float(len(ua)),
            float(len(headers)),
            float(req.path.count("/")),
            float(self._encode_method(req.method)),
            1.0 if "accept-language" in headers else 0.0,
            1.0 if req.cookie else 0.0,
            1.0 if req.body_hash else 0.0,
            1.0 if req.api_signature else 0.0,
            float(len(req.header_order)),
            1.0 if re.search(r"\d+\.\d+", ua) else 0.0,
            1.0 if "?" in req.path or "=" in req.path else 0.0,
            1.0 if "accept" in headers else 0.0,
            1.0 if req.telemetry else 0.0,
        ]

    def predict(self, features: dict | RequestSignals) -> float:
        """
        Run ML inference and return a score between 0 and 100.

        Accepts either a RequestSignals object or a pre-extracted feature dict.
        Returns 0.0 if the model is not available.
        """
        if not self._available or self._session is None or np is None:
            return 0.0

        try:
            if isinstance(features, RequestSignals):
                feat_list = self.extract_features(features)
            elif isinstance(features, dict):
                # Allow passing a dict with a 'features' key or direct feature values
                feat_list = list(features.values()) if features else [0.0] * self.N_FEATURES
            else:
                return 0.0

            # Pad or truncate to N_FEATURES
            feat_list = feat_list[:self.N_FEATURES]
            while len(feat_list) < self.N_FEATURES:
                feat_list.append(0.0)

            input_array = np.array([feat_list], dtype=np.float32)
            outputs = self._session.run(None, {self._input_name: input_array})

            # Model output: probability (0..1) -> scale to 0..100
            raw = float(outputs[0].flat[0])
            return max(0.0, min(100.0, raw * 100.0))

        except Exception:
            return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _SessionRecord:
    """Internal record for a single IP's session data."""
    timestamps: list[float] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    user_agents: set[str] = field(default_factory=set)
    ja4_hashes: set[str] = field(default_factory=set)
    cookies: set[str] = field(default_factory=set)


class SessionTracker:
    """
    Track per-IP sessions within a rolling time window.

    Computes temporal jitter (Inter-Arrival Jitter) and identity drift
    to detect metronomic bots and identity-rotating scrapers.
    """

    def __init__(self, window_seconds: int = CONFIG["SESSION_WINDOW_SECONDS"]) -> None:
        self._window = window_seconds
        self._sessions: dict[str, _SessionRecord] = {}
        self._lock = threading.Lock()

    def _evict(self, record: _SessionRecord, now: float) -> None:
        """Remove entries older than the rolling window."""
        cutoff = now - self._window
        # Find the index of the first non-expired timestamp
        idx = 0
        for i, ts in enumerate(record.timestamps):
            if ts >= cutoff:
                idx = i
                break
        else:
            # All expired
            idx = len(record.timestamps)

        if idx > 0:
            record.timestamps = record.timestamps[idx:]
            record.paths = record.paths[idx:]

    def update(self, req: RequestSignals) -> str:
        """
        Record a request in the session tracker.

        Returns a session_id (hash of IP + window start).
        """
        now = req.timestamp
        ip = req.ip

        with self._lock:
            if ip not in self._sessions:
                self._sessions[ip] = _SessionRecord()

            rec = self._sessions[ip]
            self._evict(rec, now)

            rec.timestamps.append(now)
            rec.paths.append(req.path)
            if req.user_agent:
                rec.user_agents.add(req.user_agent)
            if req.ja4_hash:
                rec.ja4_hashes.add(req.ja4_hash)
            if req.cookie:
                rec.cookies.add(req.cookie)

        window_start = int(now // self._window) * self._window
        session_id = hashlib.sha256(
            f"{ip}:{window_start}".encode()
        ).hexdigest()[:16]
        return session_id

    def temporal_jitter(self, ip: str) -> float:
        """
        Compute Inter-Arrival Jitter (IAJ): the standard deviation of
        inter-request time intervals.

        Low jitter (<0.05s) indicates metronomic bot behaviour.
        High jitter (>1s) suggests human browsing.
        Returns -1.0 if insufficient data.
        """
        with self._lock:
            rec = self._sessions.get(ip)
            if not rec or len(rec.timestamps) < 3:
                return -1.0
            timestamps = list(rec.timestamps)

        intervals = [
            timestamps[i + 1] - timestamps[i]
            for i in range(len(timestamps) - 1)
        ]
        if len(intervals) < 2:
            return -1.0

        try:
            return statistics.stdev(intervals)
        except statistics.StatisticsError:
            return -1.0

    def identity_drift(self, ip: str) -> float:
        """
        Measure identity drift: how many distinct UA / JA4 / cookie
        combinations have been seen from this IP in the current window.

        Higher drift suggests a bot rotating fingerprints.
        Returns 0.0 for a single consistent identity.
        """
        with self._lock:
            rec = self._sessions.get(ip)
            if not rec:
                return 0.0
            n_ua = len(rec.user_agents)
            n_ja4 = len(rec.ja4_hashes)
            n_cookie = len(rec.cookies)

        # Drift = total distinct identities minus the baseline (1 each)
        drift = max(0, n_ua - 1) + max(0, n_ja4 - 1) + max(0, n_cookie - 1)
        return float(drift)

    def request_count(self, ip: str) -> int:
        """Return number of requests in the current window for an IP."""
        with self._lock:
            rec = self._sessions.get(ip)
            return len(rec.timestamps) if rec else 0

    def evict_expired(self) -> int:
        """Remove all fully-expired sessions. Returns count removed."""
        now = time.time()
        cutoff = now - self._window
        removed = 0
        with self._lock:
            expired_ips = [
                ip for ip, rec in self._sessions.items()
                if not rec.timestamps or rec.timestamps[-1] < cutoff
            ]
            for ip in expired_ips:
                del self._sessions[ip]
                removed += 1
        return removed


# ═══════════════════════════════════════════════════════════════════════════════
# AGENTIC AI COUNTERMEASURES
# ═══════════════════════════════════════════════════════════════════════════════

class AgenticAIDetector:
    """
    Detect agentic AI clients through micro-jitter biometric analysis
    and invisible drip challenges.

    Micro-jitter analysis looks for the absence of biological tremors
    (3-12 Hz oscillation) in mouse/touch telemetry — real humans exhibit
    these involuntary movements, automated agents do not.

    Drip challenges inject tiny timing delays (100-200ms) into responses;
    bots that immediately retry or consistently beat the expected delay
    are flagged.
    """

    def __init__(
        self,
        jitter_threshold: float = CONFIG["MICRO_JITTER_THRESHOLD"],
        drip_delay_ms: int = CONFIG["DRIP_CHALLENGE_DELAY_MS"],
    ) -> None:
        self._jitter_threshold = jitter_threshold
        self._drip_delay_ms = drip_delay_ms
        # IP -> list of (expected_ms, actual_ms)
        self._drip_records: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._drip_lock = threading.Lock()

    def analyze_biometrics(self, telemetry: Optional[dict]) -> tuple[float, list[str]]:
        """
        Analyse mouse/touch telemetry for biological micro-tremors.

        Expects telemetry dict with:
            - 'mouse_moves': list of (timestamp_ms, x, y) tuples
            - Optionally 'touch_events': similar format

        Returns:
            (score, reasons) where score is 0-100 (higher = more bot-like)
        """
        if not telemetry:
            return 0.0, []

        mouse_moves = telemetry.get("mouse_moves", [])
        if len(mouse_moves) < 10:
            return 0.0, ["insufficient_telemetry"]

        reasons: list[str] = []
        score = 0.0

        try:
            # Extract timestamps and positions
            timestamps = [float(m[0]) for m in mouse_moves]
            x_positions = [float(m[1]) for m in mouse_moves]
            y_positions = [float(m[2]) for m in mouse_moves]

            # Compute velocity series
            dt_series = []
            dx_series = []
            dy_series = []
            for i in range(1, len(timestamps)):
                dt = (timestamps[i] - timestamps[i - 1]) / 1000.0  # ms -> s
                if dt <= 0:
                    continue
                dt_series.append(dt)
                dx_series.append((x_positions[i] - x_positions[i - 1]) / dt)
                dy_series.append((y_positions[i] - y_positions[i - 1]) / dt)

            if len(dt_series) < 5:
                return 0.0, ["insufficient_movement_data"]

            # Compute average sampling rate
            avg_dt = sum(dt_series) / len(dt_series)
            if avg_dt <= 0:
                return 0.0, ["zero_sample_rate"]
            sample_rate = 1.0 / avg_dt

            # Spectral analysis for biological tremor band (3-12 Hz)
            # Using a simple approach: compute power in tremor band via
            # discrete differences (approximation of spectral energy)
            tremor_energy = self._compute_tremor_energy(
                dx_series, dy_series, dt_series, sample_rate
            )

            # Total energy
            total_energy = sum(v ** 2 for v in dx_series) + sum(v ** 2 for v in dy_series)
            if total_energy < 1e-10:
                score += 60.0
                reasons.append("zero_movement_energy")
                return min(100.0, score), reasons

            tremor_ratio = tremor_energy / total_energy

            if tremor_ratio < self._jitter_threshold:
                score += 70.0
                reasons.append("no_biological_tremor")
            elif tremor_ratio < self._jitter_threshold * 3:
                score += 30.0
                reasons.append("low_biological_tremor")

            # Check for perfectly linear movements (bots often move in
            # straight lines with no micro-corrections)
            linearity = self._compute_linearity(x_positions, y_positions)
            if linearity > 0.98:
                score += 20.0
                reasons.append("linear_movement")

        except Exception:
            return 0.0, ["telemetry_analysis_error"]

        return min(100.0, score), reasons

    @staticmethod
    def _compute_tremor_energy(
        dx: list[float],
        dy: list[float],
        dt: list[float],
        sample_rate: float,
    ) -> float:
        """
        Estimate spectral energy in the 3-12 Hz biological tremor band.

        Uses a simple bandpass approach: compute second differences
        (acceleration) and measure energy at tremor frequencies.
        """
        if len(dx) < 4 or sample_rate < 24.0:
            # Need at least 2x Nyquist for 12 Hz
            return 0.0

        # Compute acceleration (second derivative of position)
        ax = [dx[i + 1] - dx[i] for i in range(len(dx) - 1)]
        ay = [dy[i + 1] - dy[i] for i in range(len(dy) - 1)]

        # Simple energy estimate: high-frequency components correspond
        # to tremor. We approximate by looking at sign changes and
        # magnitude of acceleration (oscillatory behaviour).
        tremor_energy = 0.0
        for i in range(1, len(ax)):
            # Sign change in acceleration = oscillation
            if (ax[i] > 0) != (ax[i - 1] > 0):
                tremor_energy += ax[i] ** 2
            if i < len(ay) and (ay[i] > 0) != (ay[i - 1] > 0):
                tremor_energy += ay[i] ** 2

        return tremor_energy

    @staticmethod
    def _compute_linearity(x: list[float], y: list[float]) -> float:
        """
        Compute how linear a series of points is (R-squared of linear fit).

        Returns value 0..1 where 1.0 = perfectly straight line.
        """
        n = len(x)
        if n < 3:
            return 0.0

        try:
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            ss_xx = sum((xi - mean_x) ** 2 for xi in x)
            ss_yy = sum((yi - mean_y) ** 2 for yi in y)
            ss_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))

            if ss_xx < 1e-10 or ss_yy < 1e-10:
                return 0.0

            r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)
            return max(0.0, min(1.0, r_squared))
        except Exception:
            return 0.0

    def should_drip_challenge(self, ip: str, score: float) -> bool:
        """
        Determine whether to issue a drip challenge to this IP.

        Drip challenges are issued when the score is in the suspect range
        and we need more signal to confirm bot/human.
        """
        return CONFIG["SUSPECT_THRESHOLD"] <= score < CONFIG["BLOCK_THRESHOLD"]

    def record_drip_response(
        self, ip: str, expected_delay_ms: int, actual_delay_ms: int
    ) -> None:
        """Record a drip challenge response for analysis."""
        with self._drip_lock:
            records = self._drip_records[ip]
            records.append((expected_delay_ms, actual_delay_ms))
            # Keep only last 20 records per IP
            if len(records) > 20:
                self._drip_records[ip] = records[-20:]

    def get_drip_score(self, ip: str) -> float:
        """
        Analyse drip challenge responses for an IP.

        Returns 0-100 score: high = bot (consistently beats expected delay).
        """
        with self._drip_lock:
            records = self._drip_records.get(ip, [])

        if len(records) < 3:
            return 0.0

        # Count how many times actual << expected
        beat_count = 0
        for expected, actual in records:
            if expected > 0 and actual < expected * 0.5:
                beat_count += 1

        ratio = beat_count / len(records)
        return min(100.0, ratio * 100.0)


# ═══════════════════════════════════════════════════════════════════════════════
# DECEPTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class DeceptionEngine:
    """
    Deception-based bot detection using dynamic honeypot injection and
    semantic probing traps.

    Honeypot paths are URLs that look attractive to scrapers but are never
    linked from legitimate pages. Any access to these paths is a strong
    bot signal. Additionally, invisible links can be injected into HTML
    responses — only bots that parse and follow all links will hit them.
    """

    def __init__(self, honeypot_paths: list[str] | None = None) -> None:
        self._honeypot_paths: set[str] = set(
            honeypot_paths or CONFIG["HONEYPOT_PATHS"]
        )
        # IP -> list of (timestamp, path) for trap accesses
        self._trap_accesses: dict[str, list[tuple[float, str]]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_honeypot_hit(self, path: str) -> bool:
        """Check if the requested path is a honeypot."""
        # Normalise path
        clean = path.split("?")[0].rstrip("/")
        for hp in self._honeypot_paths:
            hp_clean = hp.rstrip("/")
            if clean == hp_clean:
                return True
        return False

    def get_honeypot_links(self) -> list[str]:
        """
        Return HTML link tags for honeypot paths, styled to be invisible.

        These should be injected into HTML responses. Only bots that parse
        all links (ignoring CSS visibility) will follow them.
        """
        links: list[str] = []
        for path in sorted(self._honeypot_paths):
            links.append(
                f'<a href="{path}" style="position:absolute;left:-9999px;'
                f'opacity:0;height:0;width:0;overflow:hidden" '
                f'tabindex="-1" aria-hidden="true"></a>'
            )
        return links

    def record_trap_access(self, ip: str, path: str) -> None:
        """Record that an IP accessed a trap/honeypot path."""
        with self._lock:
            self._trap_accesses[ip].append((time.time(), path))
            # Keep last 50 per IP
            if len(self._trap_accesses[ip]) > 50:
                self._trap_accesses[ip] = self._trap_accesses[ip][-50:]

    def get_trap_score(self, ip: str) -> float:
        """
        Compute a trap score for an IP based on honeypot/trap accesses.

        Returns 0-100. Any honeypot access is a very strong signal.
        """
        with self._lock:
            accesses = self._trap_accesses.get(ip, [])

        if not accesses:
            return 0.0

        # Each distinct honeypot hit adds significant score
        distinct_paths = {path for _, path in accesses}
        # 1 path = 80, 2+ = 100
        if len(distinct_paths) >= 2:
            return 100.0
        return 80.0


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class BotScoringEngine:
    """
    Main scoring engine orchestrating all bot detection subsystems.

    Evaluates request signals through:
    1. drDNS verification (for claimed good bots)
    2. Session tracking (jitter + drift analysis)
    3. ONNX ML inference (if model loaded)
    4. Rule-based scoring (UA, path, header heuristics)
    5. Agentic AI detection (biometrics + drip challenges)
    6. Deception checks (honeypot hit detection)

    Final score = weighted combination of all sub-scores, with overrides
    for honeypot hits (immediate 100) and verified good bots (immediate 0).
    """

    # Score combination weights
    WEIGHT_RULES = 0.35
    WEIGHT_ML = 0.30
    WEIGHT_SESSION = 0.20
    WEIGHT_AGENTIC = 0.15

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or CONFIG
        self.scores: dict[str, ThreatScore] = {}

        # Subsystems
        self.drdns = DrDNSVerifier(cache_ttl=self.config["DRDNS_CACHE_TTL"])
        self.ml_scorer = MLScorer(model_path=self.config["ONNX_MODEL_PATH"])
        self.session_tracker = SessionTracker(
            window_seconds=self.config["SESSION_WINDOW_SECONDS"]
        )
        self.agentic_detector = AgenticAIDetector(
            jitter_threshold=self.config["MICRO_JITTER_THRESHOLD"],
            drip_delay_ms=self.config["DRIP_CHALLENGE_DELAY_MS"],
        )
        self.deception = DeceptionEngine(
            honeypot_paths=self.config.get("HONEYPOT_PATHS")
        )

    def evaluate(self, req: RequestSignals) -> ThreatScore:
        """
        Score a single request. Returns the cumulative ThreatScore for the IP.

        Orchestrates all detection subsystems and combines their outputs.
        """
        if req.ip not in self.scores:
            self.scores[req.ip] = ThreatScore(ip=req.ip)

        threat = self.scores[req.ip]
        threat.last_seen = req.timestamp
        threat.request_count += 1

        reasons: list[str] = []

        # ── 1. drDNS verification for claimed good bots ──
        is_claimed_good = bool(GOOD_BOT_UAS.search(req.user_agent)) if req.user_agent else False
        if is_claimed_good:
            try:
                # Use pre-resolved result from async server if available
                if req.drdns_result is not None:
                    verified, identity = req.drdns_result
                else:
                    verified, identity = self.drdns.verify(req.ip)
            except Exception:
                verified, identity = False, ""

            if verified:
                threat.identity_verified = True
                threat.total_score = 0.0
                threat.classification = "good"
                threat.reasons = [f"drdns_verified:{identity}"]
                threat.ml_score = 0.0
                threat.is_agentic_ai = False
                threat.honeypot_hit = False
                return threat
            else:
                reasons.append("drdns_failed_for_claimed_bot")

        # ── 2. Session tracking ──
        session_id = self.session_tracker.update(req)
        threat.session_id = session_id

        jitter = self.session_tracker.temporal_jitter(req.ip)
        drift = self.session_tracker.identity_drift(req.ip)
        req_count = self.session_tracker.request_count(req.ip)

        session_score = 0.0
        if jitter >= 0.0 and jitter < 0.05 and req_count >= 5:
            session_score += 50.0
            reasons.append("metronomic_jitter")
        elif jitter >= 0.0 and jitter < 0.15 and req_count >= 5:
            session_score += 20.0
            reasons.append("low_jitter")

        if drift >= 3.0:
            session_score += 40.0
            reasons.append("high_identity_drift")
        elif drift >= 1.0:
            session_score += 15.0
            reasons.append("moderate_identity_drift")

        if req_count > self.config["BURST_MAX_REQUESTS"]:
            session_score += 30.0
            reasons.append("burst_rate")

        session_score = min(100.0, session_score)

        # ── 3. ML inference ──
        ml_score = 0.0
        if self.ml_scorer.is_available:
            try:
                ml_score = self.ml_scorer.predict(req)
                self.ml_scorer.record_prediction(ml_score)
                threat.ml_score = ml_score
                if ml_score > 70:
                    reasons.append("ml_high_score")
                elif ml_score > 40:
                    reasons.append("ml_moderate_score")
            except Exception:
                ml_score = 0.0
        threat.ml_score = ml_score

        # ── 4. Rule-based scoring ──
        rule_score = 0.0

        # User-Agent scoring
        ua = req.user_agent
        if not ua:
            rule_score += 20
            reasons.append("empty_ua")
        elif BAD_BOT_UAS.search(ua):
            rule_score += 35
            reasons.append("bad_bot_ua")
        elif is_claimed_good:
            # Already failed drDNS above — penalise spoofing
            rule_score += 40
            reasons.append("spoofed_good_bot_ua")

        # Path scoring
        if PROBE_PATHS.search(req.path):
            rule_score += 25
            reasons.append("probe_path")

        # Header analysis
        if not req.header_order:
            rule_score += 5
            reasons.append("no_header_order")
        if "accept-language" not in req.headers:
            rule_score += 10
            reasons.append("no_accept_language")
        if "accept" not in req.headers:
            rule_score += 8
            reasons.append("no_accept_header")

        # TLS fingerprint
        if req.ja4_hash:
            # Placeholder for known headless browser JA4 hashes
            pass

        # Method scoring
        if req.method not in ("GET", "HEAD", "POST", "OPTIONS"):
            rule_score += 10
            reasons.append(f"unusual_method:{req.method}")

        rule_score = min(100.0, max(0.0, rule_score))

        # ── 5. Agentic AI detection ──
        agentic_score = 0.0
        if req.telemetry:
            try:
                bio_score, bio_reasons = self.agentic_detector.analyze_biometrics(
                    req.telemetry
                )
                agentic_score += bio_score
                reasons.extend(bio_reasons)
            except Exception:
                pass

        drip_score = self.agentic_detector.get_drip_score(req.ip)
        if drip_score > 0:
            agentic_score = max(agentic_score, drip_score)
            if drip_score > 50:
                reasons.append("drip_challenge_failed")

        agentic_score = min(100.0, agentic_score)
        threat.is_agentic_ai = agentic_score > 50.0

        # Check if a new drip challenge should be issued
        # (computed after initial scoring estimate)
        preliminary_score = (
            rule_score * self.WEIGHT_RULES
            + ml_score * self.WEIGHT_ML
            + session_score * self.WEIGHT_SESSION
            + agentic_score * self.WEIGHT_AGENTIC
        )
        threat.drip_challenge_pending = self.agentic_detector.should_drip_challenge(
            req.ip, preliminary_score
        )

        # ── 6. Deception checks ──
        honeypot_hit = self.deception.is_honeypot_hit(req.path)
        if honeypot_hit:
            self.deception.record_trap_access(req.ip, req.path)
            threat.honeypot_hit = True
            threat.total_score = 100.0
            reasons.append("honeypot_hit")
            threat.reasons = reasons
            threat.classification = "bad"
            return threat

        # Also check historical trap score
        trap_score = self.deception.get_trap_score(req.ip)
        if trap_score > 0:
            threat.honeypot_hit = True
            threat.total_score = 100.0
            reasons.append("previous_honeypot_hit")
            threat.reasons = reasons
            threat.classification = "bad"
            return threat

        # ── 7. Combine sub-scores ──
        combined = (
            rule_score * self.WEIGHT_RULES
            + ml_score * self.WEIGHT_ML
            + session_score * self.WEIGHT_SESSION
            + agentic_score * self.WEIGHT_AGENTIC
        )

        threat.total_score = max(0.0, min(100.0, combined))
        threat.reasons = reasons
        threat.classify(self.config)

        return threat

    def get_aggregated_blocklist(self) -> list[str]:
        """Return list of IPs that should be blocked."""
        threshold = self.config["BLOCK_THRESHOLD"]
        return [
            ip for ip, threat in self.scores.items()
            if threat.total_score >= threshold
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# SCORE DATABASE (SQLite persistence)
# ═══════════════════════════════════════════════════════════════════════════════

class ScoreDatabase:
    """Persist ThreatScores to SQLite for analysis and recovery."""

    def __init__(self, db_path: str | None = None) -> None:
        db_path = db_path or CONFIG["DB_PATH"]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create or migrate the scores table to the v2.1 schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                ip TEXT PRIMARY KEY,
                total_score REAL,
                classification TEXT,
                reasons TEXT,
                first_seen REAL,
                last_seen REAL,
                request_count INTEGER,
                updated_at REAL,
                ml_score REAL DEFAULT 0.0,
                identity_verified INTEGER DEFAULT 0,
                session_id TEXT,
                is_agentic_ai INTEGER DEFAULT 0,
                honeypot_hit INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

        # Migrate existing tables: add new columns if they don't exist
        existing_columns = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(scores)").fetchall()
        }
        migrations = [
            ("ml_score", "REAL DEFAULT 0.0"),
            ("identity_verified", "INTEGER DEFAULT 0"),
            ("session_id", "TEXT"),
            ("is_agentic_ai", "INTEGER DEFAULT 0"),
            ("honeypot_hit", "INTEGER DEFAULT 0"),
        ]
        for col_name, col_type in migrations:
            if col_name not in existing_columns:
                try:
                    self.conn.execute(
                        f"ALTER TABLE scores ADD COLUMN {col_name} {col_type}"
                    )
                except sqlite3.OperationalError:
                    pass  # Column already exists
        self.conn.commit()

    def save_score(self, threat: ThreatScore) -> None:
        """Insert or update a ThreatScore record."""
        self.conn.execute("""
            INSERT OR REPLACE INTO scores
            (ip, total_score, classification, reasons, first_seen,
             last_seen, request_count, updated_at, ml_score,
             identity_verified, session_id, is_agentic_ai, honeypot_hit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            threat.ip,
            threat.total_score,
            threat.classification,
            ",".join(threat.reasons),
            threat.first_seen,
            threat.last_seen,
            threat.request_count,
            time.time(),
            threat.ml_score,
            1 if threat.identity_verified else 0,
            threat.session_id,
            1 if threat.is_agentic_ai else 0,
            1 if threat.honeypot_hit else 0,
        ))
        self.conn.commit()

    def save_scores_batch(self, threats: list[ThreatScore]) -> int:
        """Batch insert/update scores in a single transaction. Returns count saved."""
        now = time.time()
        rows = [
            (
                t.ip, t.total_score, t.classification,
                ",".join(t.reasons), t.first_seen, t.last_seen,
                t.request_count, now, t.ml_score,
                1 if t.identity_verified else 0, t.session_id,
                1 if t.is_agentic_ai else 0, 1 if t.honeypot_hit else 0,
            )
            for t in threats
        ]
        self.conn.executemany("""
            INSERT OR REPLACE INTO scores
            (ip, total_score, classification, reasons, first_seen,
             last_seen, request_count, updated_at, ml_score,
             identity_verified, session_id, is_agentic_ai, honeypot_hit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        self.conn.commit()
        return len(rows)

    def load_scores(self) -> dict[str, ThreatScore]:
        """Load all scores from the database."""
        cursor = self.conn.execute("""
            SELECT ip, total_score, classification, reasons, first_seen,
                   last_seen, request_count, updated_at, ml_score,
                   identity_verified, session_id, is_agentic_ai, honeypot_hit
            FROM scores
        """)
        scores: dict[str, ThreatScore] = {}
        for row in cursor:
            (ip, total, cls, reasons_str, first, last, count, _updated,
             ml_sc, id_ver, sess_id, is_agent, hp_hit) = row
            scores[ip] = ThreatScore(
                ip=ip,
                total_score=total,
                classification=cls,
                reasons=reasons_str.split(",") if reasons_str else [],
                first_seen=first,
                last_seen=last,
                request_count=count,
                ml_score=ml_sc or 0.0,
                identity_verified=bool(id_ver),
                session_id=sess_id,
                is_agentic_ai=bool(is_agent),
                honeypot_hit=bool(hp_hit),
            )
        return scores

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCKLIST WRITER
# ═══════════════════════════════════════════════════════════════════════════════

class BlocklistWriter:
    """Write Nginx deny directives for blocked IPs."""

    @staticmethod
    def write_nginx_deny(ips: list[str], output_path: str,
                         reload_nginx: bool = True) -> None:
        """Write an Nginx-compatible blocklist file and optionally reload Nginx."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        lines = [f"deny {ip};" for ip in sorted(ips)]
        content = (
            "# Auto-generated by bot-engine — do not edit\n"
            f"# Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# Blocked IPs: {len(ips)}\n\n"
            + "\n".join(lines)
            + "\n"
        )
        # Atomic write
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(content)
        os.replace(tmp_path, output_path)

        if reload_nginx:
            BlocklistWriter.reload_nginx()

    @staticmethod
    def reload_nginx() -> bool:
        """Send reload signal to Nginx. Returns True on success."""
        import subprocess
        # Try nginx -s reload first (works without knowing the PID)
        for cmd in (
            ["nginx", "-s", "reload"],
            ["openresty", "-s", "reload"],
            ["systemctl", "reload", "nginx"],
        ):
            try:
                result = subprocess.run(
                    cmd, capture_output=True, timeout=5,
                )
                if result.returncode == 0:
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return False
