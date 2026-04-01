"""Tests for dashboard.py — auth rate limiting."""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import (
    _check_auth_rate_limit, _record_auth_failure,
    _auth_failures, _AUTH_FAIL_MAX,
)


class TestAuthRateLimiter:
    def setup_method(self):
        """Clear state between tests."""
        _auth_failures.clear()

    def test_no_failures_not_limited(self):
        assert _check_auth_rate_limit("192.168.1.1") is False

    def test_below_threshold_not_limited(self):
        for _ in range(_AUTH_FAIL_MAX - 1):
            _record_auth_failure("192.168.1.1")
        assert _check_auth_rate_limit("192.168.1.1") is False

    def test_at_threshold_limited(self):
        for _ in range(_AUTH_FAIL_MAX):
            _record_auth_failure("192.168.1.1")
        assert _check_auth_rate_limit("192.168.1.1") is True

    def test_different_ips_independent(self):
        for _ in range(_AUTH_FAIL_MAX):
            _record_auth_failure("192.168.1.1")
        assert _check_auth_rate_limit("192.168.1.1") is True
        assert _check_auth_rate_limit("192.168.1.2") is False

    def test_old_failures_expire(self):
        # Manually insert old timestamps
        _auth_failures["192.168.1.1"] = [time.time() - 600] * _AUTH_FAIL_MAX
        # Old entries should be pruned — not rate-limited
        assert _check_auth_rate_limit("192.168.1.1") is False
