"""Tests for C2 — deploy/security improvements.

Covers:
  - C2.1 Strict HMAC secret: weak-placeholder rejection, min-length enforcement,
        unwritable persist path, valid-secret acceptance, non-strict back-compat.
  - C2.3 IPv6 /64 grouping: ip_bucket helper, SessionTracker, RateTracker,
        ChallengeStore, BlocklistWriter CIDR collapse.
  - C2.4 OAuth2 proxy-header trust: admit-with-header when flag on + IP
        allowlisted; fall through to basic auth when flag off; reject when
        header missing even with flag on; reject when IP outside allowlist.

Tests for the Redis path are covered separately in test_c1.py — here we
focus on the in-memory trackers and the pure helpers.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════════
# C2.1 Strict HMAC secret
# ═══════════════════════════════════════════════════════════════════════════════

from pow_challenge import ProofOfWorkEngine  # noqa: E402


def _resolve(secret: str | None = None, *, persist_path: str, strict: bool = True,
             min_len: int = 32) -> bytes:
    """Call _resolve_secret directly so tests don't need a full POW_CONFIG."""
    cfg = {
        "STRICT_HMAC_SECRET": strict,
        "HMAC_SECRET_MIN_LEN": min_len,
        "HMAC_SECRET_FILE": persist_path,
    }
    if secret is not None:
        cfg["HMAC_SECRET"] = secret
    return ProofOfWorkEngine._resolve_secret(cfg)


class TestStrictHMACSecret:
    def test_rejects_known_weak_placeholder(self, tmp_path):
        # A placeholder like "change_me_in_production" is rejected even though
        # it's long enough to satisfy the min-length gate — strict mode cares
        # about "did they actually set a secret" not just length.
        with pytest.raises(RuntimeError, match="placeholder"):
            _resolve("change_me_in_production", persist_path=str(tmp_path / "h"))

    def test_rejects_short_secret(self, tmp_path):
        # 9 chars < 32 min
        with pytest.raises(RuntimeError, match="short|too short"):
            _resolve("too-short", persist_path=str(tmp_path / "h"))

    def test_accepts_strong_secret(self, tmp_path):
        # 64 chars, not in placeholder list
        secret = "a" * 64
        got = _resolve(secret, persist_path=str(tmp_path / "h"))
        assert got == secret.encode()

    def test_persists_generated_secret_when_none_configured(self, tmp_path):
        persist = tmp_path / "hmac_secret"
        # First call: no HMAC_SECRET set, file doesn't exist → generate + persist
        first = _resolve(None, persist_path=str(persist))
        assert persist.exists()
        assert len(first) >= 32
        # File permissions: should be 0600 (not world-readable)
        mode = persist.stat().st_mode & 0o777
        assert mode == 0o600, f"expected 0600, got {oct(mode)}"
        # Second call: must read the SAME bytes back — deterministic
        second = _resolve(None, persist_path=str(persist))
        assert first == second

    def test_unwritable_persist_path_fails_hard(self, tmp_path):
        # Block the persist path by making its parent a regular file — any
        # attempt to create the dir tree will fail with FileExistsError/OSError.
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("I am a file, not a directory")
        bad_path = blocker / "hmac_secret"  # parent is a file
        with pytest.raises(RuntimeError):
            _resolve(None, persist_path=str(bad_path))

    def test_non_strict_mode_accepts_weak_secret(self, tmp_path):
        # Back-compat: STRICT_HMAC_SECRET=False lets short / weak strings
        # through so local dev and existing tests keep working.
        got = _resolve("short", persist_path=str(tmp_path / "h"), strict=False)
        assert got == b"short"

    def test_weak_placeholders_set_includes_common_leaks(self):
        # If someone adds a new placeholder, at minimum these classics must
        # stay rejected — they represent real production leak patterns.
        wp = ProofOfWorkEngine.WEAK_PLACEHOLDERS
        for s in ("change_me_in_production", "changeme", "secret", "test", ""):
            assert s in wp, f"expected {s!r} in WEAK_PLACEHOLDERS"


# ═══════════════════════════════════════════════════════════════════════════════
# C2.3 IPv6 /64 grouping — pure helper
# ═══════════════════════════════════════════════════════════════════════════════

from bot_engine import (  # noqa: E402
    BlocklistWriter,
    RequestSignals,
    SessionTracker,
    ip_bucket,
)


class TestIPBucket:
    def test_ipv4_unchanged(self):
        assert ip_bucket("1.2.3.4", 64) == "1.2.3.4"
        assert ip_bucket("10.0.0.1", 64) == "10.0.0.1"

    def test_ipv6_groups_to_64(self):
        b1 = ip_bucket("2001:db8::1", 64)
        b2 = ip_bucket("2001:db8::ffff", 64)
        b3 = ip_bucket("2001:db8:0:0:dead:beef::", 64)
        # All three share the same /64 and should collapse to same bucket
        assert b1 == b2 == b3
        # Bucket is a network string, not a plain IP
        assert "/" in b1

    def test_ipv6_prefix_128_preserves_exact_address(self):
        assert ip_bucket("2001:db8::1", 128) == "2001:db8::1"

    def test_different_64s_are_distinct(self):
        a = ip_bucket("2001:db8:a::1", 64)
        b = ip_bucket("2001:db8:b::1", 64)
        assert a != b

    def test_invalid_input_returned_as_is(self):
        # Unparseable strings pass through so callers don't need to pre-validate
        assert ip_bucket("not-an-ip", 64) == "not-an-ip"
        assert ip_bucket("", 64) == ""


# ═══════════════════════════════════════════════════════════════════════════════
# C2.3 IPv6 /64 grouping — SessionTracker
# ═══════════════════════════════════════════════════════════════════════════════

def _req(ip: str, ua: str = "ua-x", path: str = "/", ts: float | None = None) -> RequestSignals:
    return RequestSignals(
        ip=ip,
        timestamp=ts if ts is not None else time.time(),
        method="GET",
        path=path,
        user_agent=ua,
    )


class TestSessionTrackerIPv6:
    def test_groups_ipv6_by_64(self):
        st = SessionTracker(window_seconds=60, ipv6_prefix=64)
        st.update(_req("2001:db8::1"))
        st.update(_req("2001:db8::2"))
        # Two different addresses, same /64 — bucket sees 2 requests
        assert st.request_count("2001:db8::1") == 2
        assert st.request_count("2001:db8::2") == 2  # same bucket
        # Any address in the /64 should be "seen"
        assert st.has_been_seen("2001:db8::9999")

    def test_ipv4_unaffected(self):
        st = SessionTracker(window_seconds=60, ipv6_prefix=64)
        st.update(_req("1.2.3.4"))
        st.update(_req("1.2.3.5"))
        assert st.request_count("1.2.3.4") == 1
        assert st.request_count("1.2.3.5") == 1
        assert not st.has_been_seen("1.2.3.6")

    def test_prefix_128_keeps_per_address_tracking(self):
        st = SessionTracker(window_seconds=60, ipv6_prefix=128)
        st.update(_req("2001:db8::1"))
        st.update(_req("2001:db8::2"))
        # Legacy /128 mode — each address is its own bucket
        assert st.request_count("2001:db8::1") == 1
        assert st.request_count("2001:db8::2") == 1


# ═══════════════════════════════════════════════════════════════════════════════
# C2.3 IPv6 /64 grouping — in-memory realtime trackers
# ═══════════════════════════════════════════════════════════════════════════════

class TestRealtimeInMemoryTrackersIPv6:
    def test_rate_tracker_collapses_ipv6(self):
        from realtime_server import RateTracker

        rt = RateTracker(window_seconds=10, ipv6_prefix=64)
        t = time.time()
        n1 = rt.record("2001:db8::1", t)
        n2 = rt.record("2001:db8::ffff", t)
        assert n1 == 1
        assert n2 == 2

    def test_rate_tracker_ipv4_isolated(self):
        from realtime_server import RateTracker

        rt = RateTracker(window_seconds=10, ipv6_prefix=64)
        t = time.time()
        assert rt.record("1.2.3.4", t) == 1
        assert rt.record("1.2.3.5", t) == 1

    def test_challenge_store_shares_verification_across_64(self):
        from realtime_server import ChallengeStore

        cs = ChallengeStore(ttl=3600, ipv6_prefix=64)
        cs.mark_verified("2001:db8::1")
        assert cs.is_verified("2001:db8::abcd")  # same /64
        assert not cs.is_verified("2001:db9::1")  # different /64

    def test_challenge_store_ipv4_isolated(self):
        from realtime_server import ChallengeStore

        cs = ChallengeStore(ttl=3600, ipv6_prefix=64)
        cs.mark_verified("1.2.3.4")
        assert not cs.is_verified("1.2.3.5")

    def test_challenge_store_verify_round_trip_ipv6(self):
        from realtime_server import ChallengeStore

        cs = ChallengeStore(ttl=3600, ipv6_prefix=64)
        cs.issue_challenge("2001:db8::1", "tok-abc")
        # A different address in the same /64 can complete the challenge —
        # because the verification bucket is the /64, not the individual IP.
        assert cs.verify("2001:db8::dead", "tok-abc") is True
        assert cs.is_verified("2001:db8::beef") is True


# ═══════════════════════════════════════════════════════════════════════════════
# C2.3 IPv6 /64 grouping — BlocklistWriter
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlocklistWriterIPv6:
    def test_collapses_ipv6_to_64(self, tmp_path):
        out = tmp_path / "blocklist.conf"
        ips = [
            "1.2.3.4",
            "2001:db8::1",
            "2001:db8::2",     # same /64 as above — dedupe
            "2001:db8::abcd",  # same /64 — dedupe
            "2001:db9::1",     # different /64
        ]
        BlocklistWriter.write_nginx_deny(
            ips, str(out), reload_nginx=False, ipv6_prefix=64
        )
        content = out.read_text()
        # IPv4 preserved as bare IP (Nginx treats `deny 1.2.3.4;` as /32)
        assert "deny 1.2.3.4;" in content
        # /64 networks present
        assert "2001:db8::/64" in content
        assert "2001:db9::/64" in content
        # Exactly one entry per /64 — no duplicates
        db8_lines = [l for l in content.splitlines() if "2001:db8:" in l]
        assert len(db8_lines) == 1

    def test_prefix_128_preserves_individual_ipv6(self, tmp_path):
        out = tmp_path / "blocklist.conf"
        BlocklistWriter.write_nginx_deny(
            ["2001:db8::1", "2001:db8::2"], str(out),
            reload_nginx=False, ipv6_prefix=128,
        )
        content = out.read_text()
        assert "deny 2001:db8::1;" in content
        assert "deny 2001:db8::2;" in content


# ═══════════════════════════════════════════════════════════════════════════════
# C2.4 OAuth2 proxy-header trust
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from fastapi.testclient import TestClient  # noqa: F401
    _HAS_TESTCLIENT = True
except Exception:
    _HAS_TESTCLIENT = False


def _fresh_dashboard(monkeypatch, **env):
    """Import dashboard.py under a custom env so module-level globals re-read.

    DASHBOARD_TRUST_PROXY_AUTH and DASHBOARD_ALLOW_IPS are evaluated at
    module import, so per-test configuration requires a fresh module load.
    """
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    if "dashboard" in sys.modules:
        del sys.modules["dashboard"]
    import dashboard as _d
    return _d


@pytest.mark.skipif(not _HAS_TESTCLIENT, reason="fastapi.testclient not available")
class TestDashboardProxyAuth:
    """Exercise the auth_middleware end-to-end via TestClient.

    TestClient sets ``request.client.host`` to the literal string
    ``"testclient"`` which can't be parsed as an IP — so we monkeypatch
    ``_ip_allowed`` per-test to simulate the allowlist decision. That
    keeps the tests focused on the proxy-header logic rather than on
    TestClient's transport peculiarities.
    """

    def test_trust_flag_off_requires_basic_auth(self, monkeypatch):
        d = _fresh_dashboard(monkeypatch, DASHBOARD_TRUST_PROXY_AUTH="false")
        monkeypatch.setattr(d, "_ip_allowed", lambda ip: True)
        client = TestClient(d.app)
        # Proxy header present but trust flag off → still requires Basic
        r = client.get("/api/stats", headers={"X-Forwarded-User": "alice"})
        assert r.status_code == 401

    def test_trust_flag_on_with_header_admits(self, monkeypatch):
        d = _fresh_dashboard(
            monkeypatch,
            DASHBOARD_TRUST_PROXY_AUTH="true",
            DASHBOARD_PROXY_USER_HEADER="X-Forwarded-User",
        )
        monkeypatch.setattr(d, "_ip_allowed", lambda ip: True)
        client = TestClient(d.app)
        r = client.get("/api/stats", headers={"X-Forwarded-User": "alice"})
        # Must not be challenged for basic auth; internal DB errors are
        # acceptable — what we care about is the auth layer admitting us.
        assert r.status_code not in (401, 403)

    def test_trust_flag_on_without_header_falls_through(self, monkeypatch):
        d = _fresh_dashboard(monkeypatch, DASHBOARD_TRUST_PROXY_AUTH="true")
        monkeypatch.setattr(d, "_ip_allowed", lambda ip: True)
        client = TestClient(d.app)
        # No proxy header → falls through to Basic Auth path → 401
        r = client.get("/api/stats")
        assert r.status_code == 401
        assert "WWW-Authenticate" in r.headers

    def test_ip_not_in_allowlist_rejected_even_with_header(self, monkeypatch):
        # IP allowlist is the trust boundary — forging the proxy header
        # from outside it must not work.
        d = _fresh_dashboard(monkeypatch, DASHBOARD_TRUST_PROXY_AUTH="true")
        monkeypatch.setattr(d, "_ip_allowed", lambda ip: False)
        client = TestClient(d.app)
        r = client.get("/api/stats", headers={"X-Forwarded-User": "alice"})
        assert r.status_code == 403

    def test_custom_header_name_respected(self, monkeypatch):
        d = _fresh_dashboard(
            monkeypatch,
            DASHBOARD_TRUST_PROXY_AUTH="true",
            DASHBOARD_PROXY_USER_HEADER="X-Authentik-User",
        )
        monkeypatch.setattr(d, "_ip_allowed", lambda ip: True)
        client = TestClient(d.app)
        # The default X-Forwarded-User must NOT admit — we only trust the
        # configured header name.
        r = client.get("/api/stats", headers={"X-Forwarded-User": "alice"})
        assert r.status_code == 401
        # The configured header admits.
        r = client.get("/api/stats", headers={"X-Authentik-User": "alice"})
        assert r.status_code not in (401, 403)

    def test_empty_header_value_falls_through(self, monkeypatch):
        # An empty / whitespace-only header value must NOT admit — otherwise
        # a misbehaving proxy that strips the value but keeps the key would
        # silently bypass auth.
        d = _fresh_dashboard(monkeypatch, DASHBOARD_TRUST_PROXY_AUTH="true")
        monkeypatch.setattr(d, "_ip_allowed", lambda ip: True)
        client = TestClient(d.app)
        r = client.get("/api/stats", headers={"X-Forwarded-User": "   "})
        assert r.status_code == 401
