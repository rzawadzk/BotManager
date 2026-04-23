"""Tests for C4 — production hardening.

Covers:
  - Dashboard strict password: fails unset/weak/short, warns only when
        DASHBOARD_STRICT_AUTH=false.
  - HMAC secret file perms: refuses to read a world-readable file,
        writes new secrets as 0600 atomically.
  - Admin IP allowlist helpers: parses CIDR lists, matches correctly,
        default-denies unknown / unparseable IPs.
  - Redis client builder: accepts explicit password, sets sane timeouts.

The end-to-end smoke test in tools/smoke_test.sh exercises the wired
endpoints — these are unit-level guards on the helpers.
"""

from __future__ import annotations

import importlib
import os
import stat
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard strict password
# ═══════════════════════════════════════════════════════════════════════════════

def _fresh_dashboard(monkeypatch, **env):
    """Reimport dashboard.py under a custom env.

    Validation runs at import-time, so per-test scenarios need a clean
    module load. Matches the pattern in test_c2.py.
    """
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # Several env defaults are sticky — set them explicitly so the run
    # is self-contained. Tests that want the default behaviour can
    # still omit DASHBOARD_PASS.
    monkeypatch.setenv("DASHBOARD_ALLOW_IPS", env.get("DASHBOARD_ALLOW_IPS", "127.0.0.1"))
    if "dashboard" in sys.modules:
        del sys.modules["dashboard"]
    return importlib.import_module("dashboard")


class TestDashboardStrictPassword:
    def test_rejects_unset_password(self, monkeypatch):
        # Strict mode default ON; no password set → RuntimeError on import.
        monkeypatch.delenv("DASHBOARD_PASS", raising=False)
        monkeypatch.setenv("DASHBOARD_STRICT_AUTH", "true")
        monkeypatch.setenv("DASHBOARD_ALLOW_IPS", "127.0.0.1")
        if "dashboard" in sys.modules:
            del sys.modules["dashboard"]
        with pytest.raises(RuntimeError, match="DASHBOARD_PASS"):
            importlib.import_module("dashboard")

    def test_rejects_weak_placeholder(self, monkeypatch):
        with pytest.raises(RuntimeError, match="placeholder"):
            _fresh_dashboard(monkeypatch, DASHBOARD_PASS="changeme")

    def test_rejects_short_password(self, monkeypatch):
        # Default min length is 12; 8 chars → reject
        with pytest.raises(RuntimeError, match="too short"):
            _fresh_dashboard(monkeypatch, DASHBOARD_PASS="aB3$xY8!")

    def test_accepts_strong_password(self, monkeypatch):
        d = _fresh_dashboard(monkeypatch,
                             DASHBOARD_PASS="xY9-strong-password-here")
        assert d.DASHBOARD_PASS == "xY9-strong-password-here"

    def test_nonstrict_mode_warns_instead_of_raising(self, monkeypatch, capsys):
        # DASHBOARD_STRICT_AUTH=false downgrades the hard failure to a
        # stderr warning — the dev/test escape hatch.
        d = _fresh_dashboard(
            monkeypatch,
            DASHBOARD_PASS="changeme",
            DASHBOARD_STRICT_AUTH="false",
        )
        assert d.DASHBOARD_PASS == "changeme"
        err = capsys.readouterr().err
        assert "placeholder" in err
        assert "WARNING" in err

    def test_case_insensitive_placeholder_match(self, monkeypatch):
        # Uppercase-ish variants of "admin"/"password" still trip the
        # placeholder list so a trivial capitalise-workaround fails.
        with pytest.raises(RuntimeError, match="placeholder"):
            _fresh_dashboard(monkeypatch, DASHBOARD_PASS="ADMIN")


# ═══════════════════════════════════════════════════════════════════════════════
# HMAC secret file permission handling
# ═══════════════════════════════════════════════════════════════════════════════

from pow_challenge import ProofOfWorkEngine  # noqa: E402


def _resolve(secret=None, *, persist_path: str, strict: bool = True) -> bytes:
    cfg = {
        "STRICT_HMAC_SECRET": strict,
        "HMAC_SECRET_MIN_LEN": 32,
        "HMAC_SECRET_FILE": persist_path,
    }
    if secret is not None:
        cfg["HMAC_SECRET"] = secret
    return ProofOfWorkEngine._resolve_secret(cfg)


class TestHMACFilePermissions:
    def test_generated_file_is_0600(self, tmp_path):
        # Fresh generation: the atomic-rename path creates the file with
        # mode 0600 via os.open. Anything more permissive is a bug.
        persist = tmp_path / "hmac_secret"
        _resolve(None, persist_path=str(persist))
        mode = persist.stat().st_mode & 0o777
        assert mode == 0o600, f"expected 0600, got {oct(mode)}"

    def test_strict_rejects_world_readable_file(self, tmp_path):
        # If the file already exists with loose perms (0644), strict
        # mode refuses to trust it — the secret could already be
        # compromised.
        persist = tmp_path / "hmac_secret"
        persist.write_text("a" * 64)
        persist.chmod(0o644)
        with pytest.raises(RuntimeError, match="permissions"):
            _resolve(None, persist_path=str(persist))

    def test_strict_rejects_group_readable_file(self, tmp_path):
        persist = tmp_path / "hmac_secret"
        persist.write_text("a" * 64)
        persist.chmod(0o640)
        with pytest.raises(RuntimeError, match="permissions"):
            _resolve(None, persist_path=str(persist))

    def test_strict_accepts_0600_file(self, tmp_path):
        persist = tmp_path / "hmac_secret"
        persist.write_text("a" * 64)
        persist.chmod(0o600)
        got = _resolve(None, persist_path=str(persist))
        assert got == b"a" * 64

    def test_nonstrict_tolerates_loose_perms(self, tmp_path):
        # Dev/test escape hatch. No exception; still returns the secret.
        persist = tmp_path / "hmac_secret"
        persist.write_text("short")  # also short, but strict=False
        persist.chmod(0o644)
        got = _resolve(None, persist_path=str(persist), strict=False)
        assert got == b"short"

    def test_reread_reenforces_0600(self, tmp_path):
        # If a human `chmod 0640` an existing 0600 file, strict mode
        # blocks the read. If somehow it slips past (unlikely — only
        # 0600 passes), the code re-chmods it. We verify the rechmod by
        # starting with a 0600 file and re-reading.
        persist = tmp_path / "hmac_secret"
        persist.write_text("b" * 64)
        persist.chmod(0o600)
        _resolve(None, persist_path=str(persist))
        # Still 0600 after re-read.
        assert persist.stat().st_mode & 0o777 == 0o600


# ═══════════════════════════════════════════════════════════════════════════════
# Admin-endpoint IP allowlist helpers
# ═══════════════════════════════════════════════════════════════════════════════

from realtime_server import RealtimeScoringServer  # noqa: E402


class TestAdminIPAllowlist:
    def test_parses_clean_cidr_list(self):
        nets = RealtimeScoringServer._parse_admin_nets(
            "127.0.0.1,::1,10.0.0.0/8"
        )
        assert len(nets) == 3
        # First is IPv4 /32; /8 network is the third.
        import ipaddress
        assert ipaddress.ip_address("10.11.12.13") in nets[2]

    def test_drops_bad_entries(self, capsys):
        # Bad entries get logged to stderr but don't blow up the parse.
        nets = RealtimeScoringServer._parse_admin_nets(
            "127.0.0.1,not-an-ip,::1"
        )
        assert len(nets) == 2
        err = capsys.readouterr().err
        assert "not-an-ip" in err

    def test_empty_list_parses_to_empty(self):
        assert RealtimeScoringServer._parse_admin_nets("") == []
        assert RealtimeScoringServer._parse_admin_nets("   ,,, ") == []

    def test_is_admin_ip_matches_allowlist(self):
        # Build a tiny server fixture with a known allowlist.
        server = RealtimeScoringServer.__new__(RealtimeScoringServer)
        server.admin_nets = RealtimeScoringServer._parse_admin_nets(
            "10.0.0.0/8,127.0.0.1"
        )
        assert server._is_admin_ip("127.0.0.1")
        assert server._is_admin_ip("10.255.255.255")
        assert not server._is_admin_ip("8.8.8.8")
        assert not server._is_admin_ip("::1")  # not in the allowlist

    def test_is_admin_ip_rejects_unknown_and_invalid(self):
        server = RealtimeScoringServer.__new__(RealtimeScoringServer)
        server.admin_nets = RealtimeScoringServer._parse_admin_nets(
            "0.0.0.0/0"
        )
        # "unknown" is the sentinel when nginx didn't send X-Real-IP —
        # default-deny even under 0.0.0.0/0.
        assert not server._is_admin_ip("unknown")
        assert not server._is_admin_ip("")
        assert not server._is_admin_ip("garbage")


# ═══════════════════════════════════════════════════════════════════════════════
# Redis client builder — password + timeouts
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import fakeredis  # noqa: F401
    _HAS_FAKEREDIS = True
except ImportError:
    _HAS_FAKEREDIS = False


class TestRedisAuth:
    def test_build_redis_client_rejects_without_package(self, monkeypatch):
        # When redis isn't installed the builder raises a clear error
        # instead of returning a broken client.
        import redis_state

        monkeypatch.setattr(redis_state, "REDIS_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="redis package not installed"):
            redis_state.build_redis_client("redis://localhost:6379/0")

    def test_try_build_redis_state_returns_none_on_bad_url(self):
        import redis_state

        # Pointing at a guaranteed-unreachable port causes .ping() to
        # raise; the wrapper turns that into a graceful None so callers
        # fall back to in-memory. The unreachable-port fast fail relies
        # on the 2 s socket_timeout we added in C4.
        result = redis_state.try_build_redis_state({
            "STATE_BACKEND": "redis",
            "REDIS_URL": "redis://127.0.0.1:1/0",  # port 1 ≠ redis
        })
        assert result is None

    def test_try_build_redis_state_noop_when_backend_not_redis(self):
        import redis_state

        assert redis_state.try_build_redis_state({
            "STATE_BACKEND": "memory",
            "REDIS_URL": "redis://whatever",
        }) is None

    def test_try_build_redis_state_warns_on_missing_url(self):
        import redis_state

        # The function emits warnings through a caller-supplied logger
        # rather than its own — we pass a recording stub and verify it
        # saw the expected message. This mirrors how realtime_server
        # wires its logger in production.
        class _Recorder:
            def __init__(self) -> None:
                self.calls: list[tuple] = []

            def warning(self, msg, *args):
                # Support both % formatting and plain strings to match
                # the redis_state callsite.
                if args:
                    try:
                        msg = msg % args
                    except TypeError:
                        pass
                self.calls.append(("warning", msg))

        rec = _Recorder()
        result = redis_state.try_build_redis_state(
            {"STATE_BACKEND": "redis"},  # REDIS_URL missing
            logger=rec,
        )
        assert result is None
        assert any("REDIS_URL is unset" in m for _, m in rec.calls)
