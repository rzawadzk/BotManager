"""Tests for C1 — perf/scale improvements.

Covers:
  - Nginx cache response headers (Cache-Control wiring in realtime_server)
  - Redis state backends (SessionTracker, RateTracker, ChallengeStore) via fakeredis
  - Fallback behaviour when Redis is unavailable
  - train_bot_model real-traffic ingestion: parsing, auto-labelling, ip stats
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import time
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub heavy ML dependencies before importing train_bot_model so the test
# file can run without onnxruntime/xgboost/skl2onnx installed.
for _m in (
    "onnxruntime",
    "xgboost",
    "sklearn",
    "sklearn.metrics",
    "sklearn.model_selection",
    "skl2onnx",
    "skl2onnx.common",
    "skl2onnx.common.data_types",
):
    sys.modules.setdefault(_m, types.ModuleType(_m))

import sklearn.metrics as _sm
for _fn in (
    "accuracy_score", "auc", "classification_report", "confusion_matrix",
    "f1_score", "precision_score", "recall_score", "roc_auc_score",
):
    if not hasattr(_sm, _fn):
        setattr(_sm, _fn, lambda *a, **k: 0.0)

import sklearn.model_selection as _sms
if not hasattr(_sms, "train_test_split"):
    _sms.train_test_split = lambda *a, **k: (None,) * 4

import skl2onnx as _skl2onnx
if not hasattr(_skl2onnx, "convert_sklearn"):
    _skl2onnx.convert_sklearn = lambda *a, **k: None

import skl2onnx.common.data_types as _dt
if not hasattr(_dt, "FloatTensorType"):
    _dt.FloatTensorType = object


# ═══════════════════════════════════════════════════════════════════════════════
# Redis state backends
# ═══════════════════════════════════════════════════════════════════════════════

fakeredis = pytest.importorskip("fakeredis")

from redis_state import (  # noqa: E402
    RedisSessionTracker,
    RedisRateTracker,
    RedisChallengeStore,
    try_build_redis_state,
)


@pytest.fixture
def redis_client():
    """Fresh fakeredis per test — decode_responses=True to match production."""
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    yield client
    client.flushall()


def _make_req(ip="10.0.0.1", ua="curl/7.88", path="/", ja4="t13d1516h2", cookie=""):
    """Build a minimal RequestSignals-like object (SessionTracker only reads attrs)."""
    return types.SimpleNamespace(
        ip=ip,
        timestamp=time.time(),
        user_agent=ua,
        path=path,
        ja4_hash=ja4,
        cookie=cookie,
    )


class TestRedisSessionTracker:
    def test_first_request_records_first_seen(self, redis_client):
        tracker = RedisSessionTracker(redis_client, window_seconds=300)
        req = _make_req()
        sid = tracker.update(req)
        assert isinstance(sid, str) and len(sid) == 16
        assert tracker.has_been_seen(req.ip) is True
        assert tracker.request_count(req.ip) == 1

    def test_second_node_sees_first_node_state(self, redis_client):
        """Simulates two scoring engine nodes sharing one Redis."""
        node_a = RedisSessionTracker(redis_client, window_seconds=300)
        node_b = RedisSessionTracker(redis_client, window_seconds=300)

        node_a.update(_make_req(ip="1.2.3.4", ua="ua-a"))
        node_b.update(_make_req(ip="1.2.3.4", ua="ua-b"))

        # Both nodes see the combined state
        assert node_a.request_count("1.2.3.4") == 2
        assert node_b.request_count("1.2.3.4") == 2
        assert node_a.identity_drift("1.2.3.4") == 1.0  # 2 UAs → drift 1

    def test_identity_drift_counts_unique_signals(self, redis_client):
        tracker = RedisSessionTracker(redis_client)
        ip = "1.1.1.1"
        tracker.update(_make_req(ip=ip, ua="ua1", ja4="ja4a", cookie="c1"))
        tracker.update(_make_req(ip=ip, ua="ua2", ja4="ja4a", cookie="c1"))
        tracker.update(_make_req(ip=ip, ua="ua3", ja4="ja4b", cookie="c2"))
        # UAs: 3 distinct → 2 drift; JA4s: 2 → 1; cookies: 2 → 1 = 4
        assert tracker.identity_drift(ip) == 4.0

    def test_temporal_jitter_needs_three_samples(self, redis_client):
        tracker = RedisSessionTracker(redis_client)
        ip = "1.1.1.1"
        # Only 2 samples → jitter is -1 (insufficient data sentinel)
        tracker.update(_make_req(ip=ip))
        tracker.update(_make_req(ip=ip))
        assert tracker.temporal_jitter(ip) == -1.0

    def test_time_since_first_seen(self, redis_client):
        tracker = RedisSessionTracker(redis_client)
        ip = "1.1.1.1"
        t0 = time.time()
        req = types.SimpleNamespace(
            ip=ip, timestamp=t0, user_agent="ua", path="/",
            ja4_hash="j", cookie="",
        )
        tracker.update(req)
        delta = tracker.time_since_first_seen(ip, now=t0 + 5.0)
        assert delta is not None
        assert 4.9 < delta < 5.1

    def test_unknown_ip_has_been_seen_false(self, redis_client):
        tracker = RedisSessionTracker(redis_client)
        assert tracker.has_been_seen("203.0.113.99") is False
        assert tracker.time_since_first_seen("203.0.113.99") is None


class TestRedisRateTracker:
    def test_record_returns_running_count(self, redis_client):
        tracker = RedisRateTracker(redis_client, window_seconds=10)
        t = time.time()
        assert tracker.record("1.1.1.1", t) == 1
        assert tracker.record("1.1.1.1", t + 0.1) == 2
        assert tracker.record("1.1.1.1", t + 0.2) == 3

    def test_sliding_window_evicts_old(self, redis_client):
        tracker = RedisRateTracker(redis_client, window_seconds=2)
        t = time.time()
        tracker.record("1.1.1.1", t)
        tracker.record("1.1.1.1", t + 0.1)
        # Fast-forward 3s — first two should be evicted
        assert tracker.record("1.1.1.1", t + 3.0) == 1

    def test_per_ip_isolation(self, redis_client):
        tracker = RedisRateTracker(redis_client)
        t = time.time()
        tracker.record("1.1.1.1", t)
        tracker.record("2.2.2.2", t)
        tracker.record("2.2.2.2", t + 0.01)
        assert tracker.record("1.1.1.1", t + 0.02) == 2
        assert tracker.record("2.2.2.2", t + 0.03) == 3


class TestRedisChallengeStore:
    def test_issue_and_verify_success(self, redis_client):
        store = RedisChallengeStore(redis_client, ttl=3600)
        store.issue_challenge("1.1.1.1", "tok-abc")
        assert store.verify("1.1.1.1", "tok-abc") is True
        assert store.is_verified("1.1.1.1") is True

    def test_verify_wrong_token_fails(self, redis_client):
        store = RedisChallengeStore(redis_client)
        store.issue_challenge("1.1.1.1", "tok-abc")
        assert store.verify("1.1.1.1", "wrong") is False
        assert store.is_verified("1.1.1.1") is False

    def test_verify_no_pending(self, redis_client):
        store = RedisChallengeStore(redis_client)
        assert store.verify("1.1.1.1", "anything") is False

    def test_needs_challenge_transitions(self, redis_client):
        store = RedisChallengeStore(redis_client)
        # Fresh IP → needs challenge
        assert store.needs_challenge("1.1.1.1") is True
        store.issue_challenge("1.1.1.1", "tok")
        # Pending but not expired → doesn't re-challenge yet
        assert store.needs_challenge("1.1.1.1") is False
        store.verify("1.1.1.1", "tok")
        # Verified → doesn't challenge
        assert store.needs_challenge("1.1.1.1") is False

    def test_verified_ttl_applied(self, redis_client):
        store = RedisChallengeStore(redis_client, ttl=7)
        store.mark_verified("1.1.1.1")
        ttl = redis_client.ttl("bot:chal:verified:1.1.1.1")
        assert 0 < ttl <= 7


class TestRedisStateFactory:
    def test_not_enabled_returns_none(self):
        assert try_build_redis_state({"STATE_BACKEND": "memory"}) is None

    def test_missing_url_returns_none(self):
        result = try_build_redis_state({"STATE_BACKEND": "redis"})
        assert result is None

    def test_unreachable_returns_none(self):
        # redis:// with a nonexistent socket → connection fails
        result = try_build_redis_state({
            "STATE_BACKEND": "redis",
            "REDIS_URL": "redis://127.0.0.1:1/0",  # port 1 unused
        })
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# realtime_server — Cache-Control header wiring (C1.1)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_http_response(raw: bytes) -> tuple[int, dict, str]:
    """Mini parser for the HTTP/1.1 response bytes produced by HTTPParser.build_response."""
    text = raw.decode("utf-8")
    head, _, body = text.partition("\r\n\r\n")
    lines = head.split("\r\n")
    status_line = lines[0]
    status_code = int(status_line.split()[1])
    headers = {}
    for line in lines[1:]:
        k, _, v = line.partition(":")
        headers[k.strip()] = v.strip()
    return status_code, headers, body


class TestCacheControlHeaders:
    """_build_decision_response must set Cache-Control so nginx can cache allow
    verdicts and skip caching of block/challenge/429."""

    @pytest.fixture
    def server(self):
        import realtime_server
        # Build a minimal engine-free server: we only need _build_decision_response.
        # Use __new__ to skip __init__ side effects (sockets, state backends).
        s = realtime_server.RealtimeScoringServer.__new__(
            realtime_server.RealtimeScoringServer
        )
        # Attach just enough state for the method to run
        s.config = {"CHALLENGE_COOKIE_TTL": 3600, "CHALLENGE_PATH": "/_challenge"}
        return s

    def test_allow_is_cacheable(self, server):
        raw = server._build_decision_response(
            status_code=200, action="allow", score=5.0, classification="unknown",
        )
        status, headers, _ = _parse_http_response(raw)
        assert status == 200
        cc = headers.get("Cache-Control", "")
        assert "private" in cc
        assert "max-age=60" in cc
        assert headers.get("X-Bot-Action") == "allow"

    def test_block_is_not_cached(self, server):
        raw = server._build_decision_response(
            status_code=403, action="block", score=95.0, classification="bad",
        )
        status, headers, _ = _parse_http_response(raw)
        assert status == 403
        assert "no-store" in headers.get("Cache-Control", "")

    def test_challenge_is_not_cached(self, server):
        raw = server._build_decision_response(
            status_code=202, action="challenge", score=55.0, classification="suspect",
        )
        status, headers, _ = _parse_http_response(raw)
        assert status == 202
        assert "no-store" in headers.get("Cache-Control", "")

    def test_rate_limited_is_not_cached(self, server):
        raw = server._build_decision_response(
            status_code=429, action="rate_limit", score=40.0, classification="suspect",
        )
        status, headers, _ = _parse_http_response(raw)
        assert status == 429
        assert "no-store" in headers.get("Cache-Control", "")

    def test_allow_with_non_200_status_not_cached(self, server):
        # Defensive: if somehow action=allow pairs with a non-200 status,
        # we should still refuse to cache.
        raw = server._build_decision_response(
            status_code=202, action="allow", score=5.0, classification="unknown",
        )
        _, headers, _ = _parse_http_response(raw)
        assert "no-store" in headers.get("Cache-Control", "")


# ═══════════════════════════════════════════════════════════════════════════════
# train_bot_model — real-traffic ingestion (C1.3)
# ═══════════════════════════════════════════════════════════════════════════════

import train_bot_model as tbm  # noqa: E402


LOG_SAMPLE = textwrap.dedent("""\
    192.0.2.1 [15/Apr/2026:12:34:56 +0000] "GET /index.html HTTP/1.1" 200 score=5 action=allow class=human ua="Mozilla/5.0 (X11; Linux)" ja4=t13d1516h2_x1_y1 rt=0.021
    192.0.2.1 [15/Apr/2026:12:34:57 +0000] "GET /about.html HTTP/1.1" 200 score=6 action=allow class=human ua="Mozilla/5.0 (X11; Linux)" ja4=t13d1516h2_x1_y1 rt=0.018
    192.0.2.1 [15/Apr/2026:12:34:58 +0000] "GET /contact HTTP/1.1" 200 score=7 action=allow class=human ua="Mozilla/5.0 (X11; Linux)" ja4=t13d1516h2_x1_y1 rt=0.019
    203.0.113.5 [15/Apr/2026:12:35:10 +0000] "GET /api/v2/internal/secrets HTTP/1.1" 200 score=95 action=block class=honeypot ua="curl/7.88" ja4=- rt=0.002
    198.51.100.2 [15/Apr/2026:12:35:15 +0000] "GET /sitemap.xml HTTP/1.1" 200 score=10 action=allow class=verified_good_bot ua="Googlebot/2.1" ja4=t13d-bot rt=0.008
    203.0.113.99 [15/Apr/2026:12:35:20 +0000] "POST /login HTTP/1.1" 403 score=88 action=block class=bad ua="" ja4=- rt=0.003
    203.0.113.42 [15/Apr/2026:12:35:30 +0000] "GET /only-once HTTP/1.1" 200 score=30 action=challenge class=suspect ua="Mozilla" ja4=- rt=0.005
""")


@pytest.fixture
def sample_log_path():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(LOG_SAMPLE)
        path = f.name
    yield path
    os.unlink(path)


class TestLogParsing:
    def test_parses_all_valid_rows(self, sample_log_path):
        rows = list(tbm.parse_bot_access_log(sample_log_path))
        assert len(rows) == 7
        r0 = rows[0]
        assert r0["ip"] == "192.0.2.1"
        assert r0["method"] == "GET"
        assert r0["path"] == "/index.html"
        assert r0["status"] == 200
        assert r0["score"] == 5.0
        assert r0["action"] == "allow"
        assert r0["cls"] == "human"
        assert "Mozilla" in r0["ua"]
        assert r0["ts"] > 0

    def test_skips_malformed_lines(self, tmp_path):
        p = tmp_path / "mixed.log"
        p.write_text(
            "malformed garbage without structure\n"
            '10.0.0.1 [15/Apr/2026:12:34:56 +0000] "GET / HTTP/1.1" 200 '
            'score=5 action=allow class=human ua="x" ja4=j rt=0.01\n'
            "another garbage line\n"
        )
        rows = list(tbm.parse_bot_access_log(str(p)))
        assert len(rows) == 1
        assert rows[0]["ip"] == "10.0.0.1"

    def test_handles_dash_numeric_fields(self, tmp_path):
        # upstream headers missing → '-' instead of numeric
        p = tmp_path / "dash.log"
        p.write_text(
            '10.0.0.1 [15/Apr/2026:12:34:56 +0000] "GET / HTTP/1.1" 200 '
            'score=- action=allow class=- ua="x" ja4=- rt=-\n'
        )
        rows = list(tbm.parse_bot_access_log(str(p)))
        assert len(rows) == 1
        assert rows[0]["score"] == 0.0
        assert rows[0]["rt"] == 0.0


class TestAutoLabelling:
    def test_honeypot_path_is_bad(self):
        row = {"path": "/api/v2/internal/foo", "cls": "suspect",
               "action": "allow", "score": 0, "status": 200, "ip": "x"}
        assert tbm.autolabel_row(row, {}) == 1

    def test_honeypot_class_is_bad(self):
        row = {"path": "/", "cls": "honeypot", "action": "block",
               "score": 50, "status": 200, "ip": "x"}
        assert tbm.autolabel_row(row, {}) == 1

    def test_verified_good_bot_is_good(self):
        row = {"path": "/sitemap.xml", "cls": "verified_good_bot",
               "action": "allow", "score": 10, "status": 200, "ip": "x"}
        assert tbm.autolabel_row(row, {}) == 0

    def test_high_score_block_is_bad(self):
        row = {"path": "/login", "cls": "bad", "action": "block",
               "score": 88, "status": 403, "ip": "x"}
        assert tbm.autolabel_row(row, {}) == 1

    def test_low_score_block_not_labelled(self):
        # action=block but score below threshold → skip (ambiguous)
        row = {"path": "/login", "cls": "suspect", "action": "block",
               "score": 60, "status": 403, "ip": "x"}
        assert tbm.autolabel_row(row, {}) is None

    def test_burst_rate_labelled_bad(self):
        row = {"path": "/", "cls": "suspect", "action": "allow",
               "score": 55, "status": 200, "ip": "x"}
        stats = {"x": {"request_rate": 200.0, "count": 300}}
        assert tbm.autolabel_row(row, stats) == 1

    def test_human_single_request_skipped(self):
        # Need ≥3 requests to label as human-good
        row = {"path": "/", "cls": "human", "action": "allow",
               "score": 5, "status": 200, "ip": "x"}
        stats = {"x": {"count": 1}}
        assert tbm.autolabel_row(row, stats) is None

    def test_human_with_history_good(self):
        row = {"path": "/", "cls": "human", "action": "allow",
               "score": 5, "status": 200, "ip": "x"}
        stats = {"x": {"count": 5, "request_rate": 2.0}}
        assert tbm.autolabel_row(row, stats) == 0

    def test_ambiguous_challenge_skipped(self):
        row = {"path": "/", "cls": "suspect", "action": "challenge",
               "score": 40, "status": 200, "ip": "x"}
        assert tbm.autolabel_row(row, {}) is None


class TestIPStats:
    def test_single_ip_counts(self, sample_log_path):
        rows = list(tbm.parse_bot_access_log(sample_log_path))
        stats = tbm._compute_ip_stats(rows)
        assert stats["192.0.2.1"]["count"] == 3
        assert stats["192.0.2.1"]["identity_drift"] == 1  # one UA

    def test_jitter_needs_three_timestamps(self, sample_log_path):
        rows = list(tbm.parse_bot_access_log(sample_log_path))
        stats = tbm._compute_ip_stats(rows)
        # 192.0.2.1 has 3 requests → jitter is computable (possibly 0)
        assert "jitter" in stats["192.0.2.1"]
        assert isinstance(stats["192.0.2.1"]["jitter"], float)
        # Singleton IPs → jitter defaults to 0
        assert stats["203.0.113.5"]["jitter"] == 0.0


class TestIngestRealTraffic:
    def test_end_to_end(self, sample_log_path):
        X, y = tbm.ingest_real_traffic(sample_log_path)
        # 3 humans + 1 honeypot + 1 good-bot + 1 high-block = 6 labelled;
        # 1 ambiguous challenge skipped
        assert X.shape == (6, tbm.NUM_FEATURES)
        assert y.shape == (6,)
        assert X.dtype.name == "float32"
        assert y.dtype.name == "int32"
        assert set(y.tolist()) == {0, 1}
        # 4 good (3 humans + 1 verified_good_bot), 2 bad (honeypot + high-block)
        assert int((y == 0).sum()) == 4
        assert int((y == 1).sum()) == 2

    def test_empty_log_returns_empty_arrays(self, tmp_path):
        p = tmp_path / "empty.log"
        p.write_text("")
        X, y = tbm.ingest_real_traffic(str(p))
        assert X.shape == (0, tbm.NUM_FEATURES)
        assert y.shape == (0,)

    def test_max_rows_caps_most_recent(self, sample_log_path):
        X, y = tbm.ingest_real_traffic(sample_log_path, max_rows=3)
        # Last 3 rows: verified_good_bot, bad-block, suspect-challenge →
        # 1 good + 1 bad + 1 skipped = 2 labelled
        assert X.shape == (2, tbm.NUM_FEATURES)


class TestUAEntropy:
    def test_empty_is_zero(self):
        assert tbm._ua_entropy("") == 0.0

    def test_single_char_repeated_is_zero(self):
        assert tbm._ua_entropy("aaaaa") == 0.0

    def test_diverse_ua_has_high_entropy(self):
        # Shannon bound for 16 chars: log2(16) = 4, so a rich UA should exceed
        # 2.5 bits. Mozilla UA strings are diverse.
        assert tbm._ua_entropy("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36") > 3.5
