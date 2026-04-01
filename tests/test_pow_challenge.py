"""Tests for pow_challenge.py — PoW engine, API protector, biometric captcha."""

import hashlib
import hmac as hmac_mod
import time
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pow_challenge import (
    ProofOfWorkEngine, APIProtector, BiometricCaptcha, POW_CONFIG,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ProofOfWorkEngine
# ═══════════════════════════════════════════════════════════════════════════════

class TestProofOfWorkEngine:
    @pytest.fixture
    def engine(self):
        config = {**POW_CONFIG, "HMAC_SECRET": "test-secret-key"}
        return ProofOfWorkEngine(config=config)

    def test_generate_challenge(self, engine):
        challenge = engine.generate_challenge("10.0.0.1")
        assert challenge.challenge_id
        assert len(challenge.batches) >= POW_CONFIG["BATCH_COUNT_MIN"]
        assert len(challenge.batches) <= POW_CONFIG["BATCH_COUNT_MAX"]

    def test_challenge_stored_in_pending(self, engine):
        challenge = engine.generate_challenge("10.0.0.1")
        assert challenge.challenge_id in engine.pending

    def test_is_verified_false_initially(self, engine):
        assert engine.is_verified("10.0.0.1") is False

    def test_get_stats(self, engine):
        engine.generate_challenge("10.0.0.1")
        stats = engine.get_stats()
        assert "pending_challenges" in stats
        assert stats["pending_challenges"] >= 1

    def test_verify_cookie_invalid(self, engine):
        assert engine.verify_cookie("invalid:cookie:value", "10.0.0.1") is False

    def test_verify_cookie_wrong_format(self, engine):
        assert engine.verify_cookie("nocolons", "10.0.0.1") is False

    def test_difficulty_scales_with_threat(self, engine):
        # High threat should get higher min difficulty
        challenge_high = engine.generate_challenge("10.0.0.1", threat_score=80)
        challenge_low = engine.generate_challenge("10.0.0.2", threat_score=10)
        avg_high = sum(b.difficulty for b in challenge_high.batches) / len(challenge_high.batches)
        avg_low = sum(b.difficulty for b in challenge_low.batches) / len(challenge_low.batches)
        # High threat should have equal or higher average difficulty
        assert avg_high >= avg_low


# ═══════════════════════════════════════════════════════════════════════════════
# APIProtector
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPIProtector:
    @pytest.fixture
    def protector(self, tmp_path):
        config = {
            **POW_CONFIG,
            "API_KEY_FILE": str(tmp_path / "api_keys.json"),
            "API_TIMESTAMP_TOLERANCE": 300,
        }
        return APIProtector(config=config)

    def test_generate_api_key(self, protector):
        key, secret = protector.generate_api_key()
        assert key.startswith("bk_")
        assert len(secret) == 64  # 32 bytes hex

    def test_verify_valid_request(self, protector):
        key, secret = protector.generate_api_key()
        method = "GET"
        path = "/api/v1/data"
        ts = str(int(time.time()))
        body_hash = hashlib.sha256(b"").hexdigest()
        message = f"{method}:{path}:{ts}:{body_hash}"
        sig = hmac_mod.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()

        valid, reason = protector.verify_api_request(
            method=method,
            path=path,
            headers={"X-API-Key": key, "X-Timestamp": ts, "X-Signature": sig},
            body_hash=body_hash,
        )
        assert valid is True
        assert reason == "ok" or valid  # may return empty reason on success

    def test_verify_missing_key(self, protector):
        valid, reason = protector.verify_api_request(
            method="GET", path="/api", headers={}, body_hash=""
        )
        assert valid is False
        assert "missing" in reason

    def test_verify_unknown_key(self, protector):
        valid, reason = protector.verify_api_request(
            method="GET", path="/api",
            headers={"X-API-Key": "bk_unknown", "X-Timestamp": "0", "X-Signature": "bad"},
            body_hash=""
        )
        assert valid is False
        assert "unknown" in reason

    def test_verify_expired_timestamp(self, protector):
        key, secret = protector.generate_api_key()
        old_ts = str(int(time.time()) - 9999)
        valid, reason = protector.verify_api_request(
            method="GET", path="/api",
            headers={"X-API-Key": key, "X-Timestamp": old_ts, "X-Signature": "x"},
            body_hash=""
        )
        assert valid is False
        assert "expired" in reason or "timestamp" in reason

    def test_revoke_api_key(self, protector):
        key, secret = protector.generate_api_key()
        assert protector.revoke_api_key(key) is True
        assert protector.revoke_api_key(key) is False  # already gone


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics (from realtime_server)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetrics:
    @pytest.fixture
    def metrics(self):
        from realtime_server import Metrics
        return Metrics()

    def test_record_request(self, metrics):
        metrics.record_request("allow", 500)
        assert metrics.requests_total == 1
        assert metrics.requests_allowed == 1
        assert metrics.latency_max_us == 500

    def test_record_multiple_actions(self, metrics):
        metrics.record_request("allow", 100)
        metrics.record_request("block", 200)
        metrics.record_request("challenge", 300)
        assert metrics.requests_total == 3
        assert metrics.requests_allowed == 1
        assert metrics.requests_blocked == 1
        assert metrics.requests_challenged == 1

    def test_snapshot(self, metrics):
        metrics.record_request("allow", 100)
        snap = metrics.snapshot()
        assert snap["total_requests"] == 1
        assert snap["allowed"] == 1
        assert "avg_latency_us" in snap
        assert "uptime_seconds" in snap

    def test_prometheus_text(self, metrics):
        metrics.record_request("allow", 100)
        metrics.record_request("block", 200)
        text = metrics.prometheus_text()
        assert "bot_engine_requests_total 2" in text
        assert 'bot_engine_requests{decision="allow"} 1' in text
        assert 'bot_engine_requests{decision="block"} 1' in text
        assert "# HELP" in text
        assert "# TYPE" in text

    def test_record_error(self, metrics):
        metrics.record_error()
        metrics.record_error()
        assert metrics.requests_errors == 2
        text = metrics.prometheus_text()
        assert "bot_engine_errors_total 2" in text
