#!/usr/bin/env python3
"""
Bot Engine — Redis State Backend (C1.2)
=========================================
Optional Redis-backed implementations of SessionTracker, RateTracker, and
ChallengeStore. Enable by setting in config.yaml:

    STATE_BACKEND: redis
    REDIS_URL: redis://localhost:6379/0

When enabled, these classes are drop-in replacements for the in-memory
versions in bot_engine.py / realtime_server.py — they expose the SAME
public methods with the SAME signatures. Horizontal scaling: multiple
scoring engine nodes can share a single Redis and all see the same
session/rate/challenge state, so a request to node A counts against an
IP's budget even if their previous request hit node B.

Graceful degradation: if the redis package is not installed or the Redis
instance is unreachable at startup, the factory in `state_factory.py`
falls back to in-memory with a warning. The engine NEVER fails closed
on missing Redis.

Data model:
  bot:sess:{ip}:ts        sorted set, score=timestamp, member=ts:path
  bot:sess:{ip}:ua        set of user-agents
  bot:sess:{ip}:ja4       set of JA4 hashes
  bot:sess:{ip}:cookies   set of cookie values
  bot:sess:{ip}:meta      hash with first_seen, last_access
  bot:rate:{ip}           sorted set, score=timestamp, member=timestamp
  bot:chal:verified:{ip}  string (expiry ts), EXPIRE applied
  bot:chal:pending:{ip}   hash {token, issued_at}, EXPIRE applied
  bot:chal:failures:{ip}  integer counter

All keys set with EXPIRE so that abandoned sessions self-clean without
explicit GC.
"""

from __future__ import annotations

import ipaddress
import statistics
import time
from typing import Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False


def _ip_bucket(ip: str, ipv6_prefix: int) -> str:
    """Bucket an IP for state tracking (C2.3).

    Mirrors bot_engine.ip_bucket so redis_state has no import cycle with
    bot_engine. IPv6 → /ipv6_prefix network; IPv4 unchanged; invalid
    inputs pass through.
    """
    if not ip:
        return ip
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    if isinstance(addr, ipaddress.IPv4Address):
        return ip
    if ipv6_prefix >= 128:
        return ip
    return str(ipaddress.ip_network(f"{ip}/{ipv6_prefix}", strict=False))


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION TRACKER (Redis)
# ═══════════════════════════════════════════════════════════════════════════════

class RedisSessionTracker:
    """
    Session tracker using Redis for shared state across scoring engine nodes.

    Public interface matches bot_engine.SessionTracker exactly:
        update(req) -> session_id
        temporal_jitter(ip) -> float
        identity_drift(ip) -> float
        request_count(ip) -> int
        has_been_seen(ip) -> bool
        time_since_first_seen(ip, now=None) -> float | None
        evict_expired() -> int   (no-op — Redis EXPIRE handles it)
    """

    KEY_PREFIX = "bot:sess:"

    def __init__(
        self,
        redis_client,
        window_seconds: int = 300,
        ipv6_prefix: int = 64,
    ) -> None:
        if not REDIS_AVAILABLE:
            raise RuntimeError("redis package not installed")
        self._r = redis_client
        self._window = window_seconds
        self._ipv6_prefix = ipv6_prefix
        # Buffer beyond the logical window so keys don't vanish mid-eviction
        self._key_ttl = window_seconds + 60

    def _bucket(self, ip: str) -> str:
        """Collapse IPv6 to /ipv6_prefix so a bucket shares state (C2.3)."""
        return _ip_bucket(ip, self._ipv6_prefix)

    def _k(self, ip: str, suffix: str) -> str:
        # Embed the bucket, not the raw IP, so /64 addresses share state
        return f"{self.KEY_PREFIX}{self._bucket(ip)}:{suffix}"

    def update(self, req) -> str:
        """Record a request. Returns a session_id (bucket + window start).

        IPv6 addresses are collapsed to /ipv6_prefix so a scraper cycling
        source addresses inside one allocation can't reset per-IP state.
        """
        import hashlib
        now = req.timestamp
        bucket = self._bucket(req.ip)
        cutoff = now - self._window

        ts_key = self._k(req.ip, "ts")
        ua_key = self._k(req.ip, "ua")
        ja4_key = self._k(req.ip, "ja4")
        cook_key = self._k(req.ip, "cookies")
        meta_key = self._k(req.ip, "meta")

        pipe = self._r.pipeline()
        # Evict expired timestamps first (keeps sorted-set small)
        pipe.zremrangebyscore(ts_key, "-inf", f"({cutoff}")
        # Store timestamp with path as member (ts encoded for uniqueness)
        member = f"{now:.6f}:{req.path}"
        pipe.zadd(ts_key, {member: now})
        pipe.expire(ts_key, self._key_ttl)

        if req.user_agent:
            pipe.sadd(ua_key, req.user_agent)
            pipe.expire(ua_key, self._key_ttl)
        if req.ja4_hash:
            pipe.sadd(ja4_key, req.ja4_hash)
            pipe.expire(ja4_key, self._key_ttl)
        if req.cookie:
            pipe.sadd(cook_key, req.cookie)
            pipe.expire(cook_key, self._key_ttl)

        # first_seen is set only on the first request (HSETNX — NX = only if absent)
        pipe.hsetnx(meta_key, "first_seen", f"{now:.6f}")
        pipe.hset(meta_key, "last_access", f"{now:.6f}")
        pipe.expire(meta_key, self._key_ttl)
        pipe.execute()

        window_start = int(now // self._window) * self._window
        return hashlib.sha256(f"{bucket}:{window_start}".encode()).hexdigest()[:16]

    def _timestamps(self, ip: str) -> list[float]:
        cutoff = time.time() - self._window
        raw = self._r.zrangebyscore(self._k(ip, "ts"), cutoff, "+inf", withscores=True)
        return [score for _member, score in raw]

    def temporal_jitter(self, ip: str) -> float:
        timestamps = self._timestamps(ip)
        if len(timestamps) < 3:
            return -1.0
        intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        if len(intervals) < 2:
            return -1.0
        try:
            return statistics.stdev(intervals)
        except statistics.StatisticsError:
            return -1.0

    def identity_drift(self, ip: str) -> float:
        pipe = self._r.pipeline()
        pipe.scard(self._k(ip, "ua"))
        pipe.scard(self._k(ip, "ja4"))
        pipe.scard(self._k(ip, "cookies"))
        n_ua, n_ja4, n_cookie = pipe.execute()
        return float(max(0, n_ua - 1) + max(0, n_ja4 - 1) + max(0, n_cookie - 1))

    def request_count(self, ip: str) -> int:
        cutoff = time.time() - self._window
        return self._r.zcount(self._k(ip, "ts"), cutoff, "+inf")

    def has_been_seen(self, ip: str) -> bool:
        return bool(self._r.exists(self._k(ip, "meta")))

    def time_since_first_seen(self, ip: str, now: Optional[float] = None) -> Optional[float]:
        if now is None:
            now = time.time()
        raw = self._r.hget(self._k(ip, "meta"), "first_seen")
        if raw is None:
            return None
        try:
            return now - float(raw)
        except (TypeError, ValueError):
            return None

    def evict_expired(self) -> int:
        """No-op — Redis handles expiration via EXPIRE. Returned for interface parity."""
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# RATE TRACKER (Redis)
# ═══════════════════════════════════════════════════════════════════════════════

class RedisRateTracker:
    """
    Sliding-window rate tracker. Public interface matches
    realtime_server.RateTracker:
        record(ip, now) -> int
    """

    KEY_PREFIX = "bot:rate:"

    def __init__(
        self,
        redis_client,
        window_seconds: int = 10,
        ipv6_prefix: int = 64,
    ) -> None:
        if not REDIS_AVAILABLE:
            raise RuntimeError("redis package not installed")
        self._r = redis_client
        self._window = window_seconds
        self._ipv6_prefix = ipv6_prefix
        self._key_ttl = window_seconds + 60

    def _k(self, ip: str) -> str:
        # Rate-limit per /64 so an IPv6 scraper can't dodge limits by
        # cycling source addresses inside one allocation (C2.3).
        bucket = _ip_bucket(ip, self._ipv6_prefix)
        return f"{self.KEY_PREFIX}{bucket}"

    def record(self, ip: str, now: float) -> int:
        key = self._k(ip)
        cutoff = now - self._window
        # Unique member to avoid dedup (ZADD replaces same member). Use
        # a monotonic suffix with nanosecond precision.
        member = f"{now:.9f}"
        pipe = self._r.pipeline()
        pipe.zremrangebyscore(key, "-inf", f"({cutoff}")
        pipe.zadd(key, {member: now})
        pipe.expire(key, self._key_ttl)
        pipe.zcount(key, cutoff, "+inf")
        results = pipe.execute()
        return int(results[-1])


# ═══════════════════════════════════════════════════════════════════════════════
# CHALLENGE STORE (Redis)
# ═══════════════════════════════════════════════════════════════════════════════

class RedisChallengeStore:
    """
    Challenge token store. Public interface matches
    realtime_server.ChallengeStore:
        is_verified(ip) -> bool
        mark_verified(ip)
        issue_challenge(ip, token)
        verify(ip, token) -> bool
        needs_challenge(ip) -> bool
        cleanup()   (no-op — Redis EXPIRE handles it)
    """

    KEY_PREFIX = "bot:chal:"
    PENDING_TTL = 60   # Challenges expire 60s after issue
    FAILURE_TTL = 3600 # Keep failure counts for 1h

    def __init__(
        self,
        redis_client,
        ttl: int = 3600,
        ipv6_prefix: int = 64,
    ) -> None:
        if not REDIS_AVAILABLE:
            raise RuntimeError("redis package not installed")
        self._r = redis_client
        self._ttl = ttl
        self._ipv6_prefix = ipv6_prefix

    def _b(self, ip: str) -> str:
        return _ip_bucket(ip, self._ipv6_prefix)

    def _vk(self, ip: str) -> str:
        return f"{self.KEY_PREFIX}verified:{self._b(ip)}"

    def _pk(self, ip: str) -> str:
        return f"{self.KEY_PREFIX}pending:{self._b(ip)}"

    def _fk(self, ip: str) -> str:
        return f"{self.KEY_PREFIX}failures:{self._b(ip)}"

    def is_verified(self, ip: str) -> bool:
        return bool(self._r.exists(self._vk(ip)))

    def mark_verified(self, ip: str) -> None:
        self._r.setex(self._vk(ip), self._ttl, "1")

    def issue_challenge(self, ip: str, token: str) -> None:
        key = self._pk(ip)
        pipe = self._r.pipeline()
        pipe.hset(key, mapping={"token": token, "issued_at": f"{time.time():.6f}"})
        pipe.expire(key, self.PENDING_TTL)
        pipe.execute()

    def verify(self, ip: str, token: str) -> bool:
        key = self._pk(ip)
        data = self._r.hgetall(key)
        if not data:
            return False
        # Handle bytes (default) or str decoding
        expected_token = data.get("token") or data.get(b"token", b"")
        if isinstance(expected_token, bytes):
            expected_token = expected_token.decode("utf-8", errors="replace")
        issued_raw = data.get("issued_at") or data.get(b"issued_at", b"0")
        if isinstance(issued_raw, bytes):
            issued_raw = issued_raw.decode("utf-8", errors="replace")
        try:
            issued_at = float(issued_raw)
        except (TypeError, ValueError):
            issued_at = 0.0

        if token == expected_token and (time.time() - issued_at) < self.PENDING_TTL:
            self.mark_verified(ip)
            self._r.delete(key)
            return True

        # Record failure
        fkey = self._fk(ip)
        pipe = self._r.pipeline()
        pipe.incr(fkey)
        pipe.expire(fkey, self.FAILURE_TTL)
        pipe.execute()
        return False

    def needs_challenge(self, ip: str) -> bool:
        if self.is_verified(ip):
            return False
        key = self._pk(ip)
        if self._r.exists(key):
            issued_raw = self._r.hget(key, "issued_at")
            if isinstance(issued_raw, bytes):
                issued_raw = issued_raw.decode("utf-8", errors="replace")
            try:
                issued_at = float(issued_raw or 0)
            except (TypeError, ValueError):
                issued_at = 0.0
            if time.time() - issued_at > self.PENDING_TTL:
                self._r.delete(key)
                return True
            return False
        return True

    def cleanup(self) -> None:
        """No-op — Redis handles expiration via EXPIRE."""
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def build_redis_client(url: str, *, decode_responses: bool = True):
    """Build a Redis client from a URL, or raise a clear error."""
    if not REDIS_AVAILABLE:
        raise RuntimeError(
            "redis package not installed. `pip install redis` to enable "
            "the Redis state backend."
        )
    client = redis.from_url(url, decode_responses=decode_responses)
    # Ping to fail fast if unreachable
    client.ping()
    return client


def try_build_redis_state(config: dict, logger=None):
    """
    Attempt to construct (session_tracker, rate_tracker, challenge_store)
    backed by Redis using ``config["REDIS_URL"]``. Returns the tuple on
    success, or None if Redis is unavailable / unreachable. Callers should
    fall back to the in-memory implementations on None.
    """
    if config.get("STATE_BACKEND") != "redis":
        return None
    url = config.get("REDIS_URL")
    if not url:
        if logger:
            logger.warning("STATE_BACKEND=redis but REDIS_URL is unset; falling back to in-memory")
        return None
    try:
        client = build_redis_client(url)
    except Exception as exc:
        if logger:
            logger.warning(f"Redis backend unavailable ({exc}); falling back to in-memory")
        return None

    ipv6_prefix = int(config.get("IPV6_BUCKET_PREFIX", 64))
    session = RedisSessionTracker(
        client,
        window_seconds=config.get("SESSION_WINDOW_SECONDS", 300),
        ipv6_prefix=ipv6_prefix,
    )
    rate = RedisRateTracker(client, window_seconds=10, ipv6_prefix=ipv6_prefix)
    challenge = RedisChallengeStore(
        client,
        ttl=config.get("CHALLENGE_COOKIE_TTL", 3600),
        ipv6_prefix=ipv6_prefix,
    )
    return (session, rate, challenge)
