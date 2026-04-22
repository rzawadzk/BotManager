#!/usr/bin/env python3
"""
Bot Engine — Proof-of-Work Challenge System v2.1
==================================================
Multi-batch cryptographic proof-of-work with biometric captcha and
API HMAC protection.  Forces browsers to spend real CPU time before
accessing protected resources.  Combined with telemetry collection
and an interactive tracing captcha to detect headless automation.

WHY THIS WORKS AGAINST PLAYWRIGHT:
  1. CPU cost: Each challenge requires ~2-4 seconds of browser computation
     across 10-12 sequential sub-puzzles.  For a real user this happens
     once per session (then cookie-bypassed).  For a bot farm scraping
     100k pages, that is 100k x 3s = ~83 CPU-hours.

  2. Economic pressure: Running Playwright at scale requires servers.
     PoW makes each request cost real compute, destroying the cost
     advantage of scraping over paying for data access.

  3. Sequential batches: Each batch depends on the previous solution,
     preventing parallelisation across cores or GPU offload.

  4. Telemetry during PoW: While the browser solves the puzzle we
     collect mouse/canvas/environment signals. The bot cannot skip the
     telemetry phase because it needs to solve the PoW.

  5. Biometric captcha: An interactive tracing challenge that measures
     biological motor noise, path coverage, and timing.

  6. Server verification is O(1): The server generates the challenge in
     microseconds and verifies the solution in microseconds.

  7. API HMAC protection: Non-browser /api/ clients must sign every
     request with HMAC-SHA256 for replay and tamper protection.

PROTOCOL (v2.1 multi-batch):
  1. Server generates MultiBatchChallenge with 10-12 sub-puzzles
  2. Client receives challenge via HTML page
  3. Client solves batches sequentially; batch N uses hash of batch N-1
     solution as additional salt
  4. Client POSTs all nonces + telemetry to beacon URL
  5. Server verifies every batch in order
  6. Server sets verified cookie

DIFFICULTY CALIBRATION:
  Per-batch difficulty varies 14-18 bits.
  Total expected solve time: ~2-4 seconds on modern hardware.

Author: Rafal — VPS Bot Management
Version: 2.1 — March 2026
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import random
import secrets
import struct
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

POW_CONFIG = {
    # ── Difficulty ──
    # Number of leading zero BITS required in SHA-256(prefix + nonce).
    # Higher = harder. Scale dynamically based on threat score.
    "DIFFICULTY_DEFAULT": 18,       # ~1 second on modern hardware
    "DIFFICULTY_HIGH": 20,          # ~3.5 seconds (for high-score suspects)
    "DIFFICULTY_LOW": 15,           # ~0.2 seconds (for low-score suspects)

    # ── Multi-batch settings ──
    "BATCH_COUNT_MIN": 10,          # Minimum sub-puzzles
    "BATCH_COUNT_MAX": 12,          # Maximum sub-puzzles
    "BATCH_DIFFICULTY_MIN": 14,     # Easiest batch difficulty
    "BATCH_DIFFICULTY_MAX": 18,     # Hardest batch difficulty

    # ── Challenge lifecycle ──
    "CHALLENGE_TTL_SECONDS": 30,    # Challenge must be solved within this window
    "CHALLENGE_MAX_PENDING": 50000, # Max pending challenges in memory
    "VERIFIED_TTL_SECONDS": 3600,   # Verified status lasts 1 hour
    "MAX_ATTEMPTS_PER_IP": 5,       # Max solve attempts before hard-block

    # ── HMAC secret for challenge signing ──
    # MUST be set via BOT_HMAC_SECRET env var in production.
    # Generate: python3 -c "import secrets; print(secrets.token_hex(32))"
    # Fallback auto-generates and persists to disk so it survives restarts.
    "HMAC_SECRET": os.environ.get("BOT_HMAC_SECRET", ""),

    # ── Strict HMAC secret validation (C2.1) ──
    # When true, the engine refuses to start if the secret is empty / weak /
    # a known placeholder AND cannot be auto-generated to a persistent path.
    # Defaults to on for production safety; flip to "false" only for tests
    # or ephemeral dev containers where a transient secret is acceptable.
    "STRICT_HMAC_SECRET": os.environ.get("BOT_STRICT_HMAC", "true").lower() != "false",

    # Minimum secret length in characters when STRICT_HMAC_SECRET is enabled.
    "HMAC_SECRET_MIN_LEN": int(os.environ.get("BOT_HMAC_SECRET_MIN_LEN", "32")),

    # Persisted-secret path — created on first run so restarts don't rotate
    # the HMAC key. Must be writable by the scoring engine user and 0600.
    "HMAC_SECRET_FILE": os.environ.get(
        "BOT_HMAC_SECRET_FILE", "/var/lib/bot-engine/.hmac_secret"
    ),

    # ── Telemetry collection time ──
    "TELEMETRY_COLLECT_MS": 3000,   # Collect mouse/env data for 3 seconds

    # ── Cookie ──
    "COOKIE_NAME": "bc_pow",
    "COOKIE_DOMAIN": "",            # Set to your domain
    "COOKIE_SECURE": True,
    "COOKIE_HTTPONLY": True,
    "COOKIE_SAMESITE": "Strict",

    # ── API protection ──
    "API_KEY_FILE": "/var/lib/bot-engine/api_keys.json",
    "API_TIMESTAMP_TOLERANCE": 300,  # seconds

    # ── Biometric captcha ──
    "BIOMETRIC_CAPTCHA_MIN_TRACE_TIME_S": 1.0,
    "BIOMETRIC_CAPTCHA_MAX_TRACE_TIME_S": 5.0,
    "BIOMETRIC_CAPTCHA_MIN_COVERAGE": 0.80,
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Challenge:
    """A single proof-of-work challenge (v1 compat)."""
    challenge_id: str
    prefix: str
    difficulty: int
    ip: str
    issued_at: float
    hmac_sig: str = ""

    def is_expired(self) -> bool:
        return time.time() - self.issued_at > POW_CONFIG["CHALLENGE_TTL_SECONDS"]


@dataclass
class ChallengeBatch:
    """One sub-puzzle in a multi-batch challenge."""
    batch_index: int
    prefix: str
    difficulty: int
    salt: str           # Empty for first batch; hash of previous solution for rest
    hmac_sig: str = ""


@dataclass
class MultiBatchChallenge:
    """A complete multi-batch PoW challenge."""
    challenge_id: str
    batches: list[ChallengeBatch]
    ip: str
    issued_at: float

    def is_expired(self) -> bool:
        return time.time() - self.issued_at > POW_CONFIG["CHALLENGE_TTL_SECONDS"]


@dataclass
class VerifiedSession:
    """A session that has passed the PoW challenge."""
    ip: str
    verified_at: float
    score_at_verification: float
    difficulty_solved: int
    solve_time_ms: float
    telemetry_score: float = 0.0

    def is_expired(self) -> bool:
        return time.time() - self.verified_at > POW_CONFIG["VERIFIED_TTL_SECONDS"]


# ═══════════════════════════════════════════════════════════════════════════════
# SERVER-SIDE: CHALLENGE GENERATION & VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class ProofOfWorkEngine:
    """
    Server-side PoW challenge generation and verification.

    Thread-safe for use in the async scoring server.
    All operations are O(1) — no heavy computation server-side.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or POW_CONFIG
        self.secret = self._resolve_secret(self.config)

        # ── State ──
        self.pending: dict[str, MultiBatchChallenge] = {}
        self.verified: dict[str, VerifiedSession] = {}
        self.attempts: dict[str, int] = {}

    # Secrets that look plausible but are well-known placeholders. Any of
    # these lands the engine in a refuse-to-start state under strict mode
    # so a "forgot to set BOT_HMAC_SECRET" deploy fails loudly instead of
    # silently using a guessable key.
    WEAK_PLACEHOLDERS = frozenset({
        "change_me_in_production",
        "CHANGE_ME_IN_PRODUCTION",
        "changeme",
        "change_me",
        "changeme123",
        "secret",
        "test",
        "test-secret",
        "hmac_secret",
        "placeholder",
        "",
    })

    @classmethod
    def _resolve_secret(cls, config: dict | str) -> bytes:
        """Resolve HMAC secret with strict-mode validation.

        Resolution order:
          1. ``config["HMAC_SECRET"]`` if present and non-empty
          2. persisted secret file (``HMAC_SECRET_FILE``), if readable
          3. freshly-generated secret persisted to that file

        In strict mode (``STRICT_HMAC_SECRET`` true, the default), the
        engine refuses to start if:
          - the configured secret is a known weak placeholder
          - the configured secret is shorter than ``HMAC_SECRET_MIN_LEN``
          - no secret is configured AND the persist path cannot be read
            or written

        A single-arg str is accepted for back-compat with tests that call
        ``_resolve_secret("my-test-secret")`` directly — strict validation
        is bypassed in that mode.
        """
        # Back-compat: tests may pass a bare string.
        if isinstance(config, (str, bytes)):
            value = config.decode() if isinstance(config, bytes) else config
            return (value or secrets.token_hex(32)).encode()

        configured = config.get("HMAC_SECRET", "") or ""
        strict = bool(config.get("STRICT_HMAC_SECRET", True))
        min_len = int(config.get("HMAC_SECRET_MIN_LEN", 32))
        secret_path = Path(
            config.get("HMAC_SECRET_FILE", "/var/lib/bot-engine/.hmac_secret")
        )

        if configured:
            if strict:
                if configured in cls.WEAK_PLACEHOLDERS:
                    raise RuntimeError(
                        "BOT_HMAC_SECRET is a known weak placeholder "
                        f"({configured!r}). Set a strong secret with: "
                        "python3 -c 'import secrets; print(secrets.token_hex(32))' "
                        "or disable strict mode with BOT_STRICT_HMAC=false."
                    )
                if len(configured) < min_len:
                    raise RuntimeError(
                        f"BOT_HMAC_SECRET is too short ({len(configured)} chars; "
                        f"need ≥{min_len}). Generate one with: "
                        "python3 -c 'import secrets; print(secrets.token_hex(32))'"
                    )
            return configured.encode()

        # No configured secret — try the persisted file.
        try:
            if secret_path.is_file():
                persisted = secret_path.read_text().strip()
                if persisted:
                    if strict and len(persisted) < min_len:
                        raise RuntimeError(
                            f"Persisted HMAC secret at {secret_path} is too short "
                            f"({len(persisted)} chars; need ≥{min_len}). "
                            "Delete the file to regenerate."
                        )
                    return persisted.encode()
        except OSError as exc:
            if strict:
                raise RuntimeError(
                    f"BOT_HMAC_SECRET is unset and persisted secret at "
                    f"{secret_path} is unreadable: {exc}. Set BOT_HMAC_SECRET "
                    "or disable strict mode with BOT_STRICT_HMAC=false."
                ) from exc

        # Generate and persist.
        new_secret = secrets.token_hex(32)
        try:
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            secret_path.write_text(new_secret)
            secret_path.chmod(0o600)
        except OSError as exc:
            if strict:
                raise RuntimeError(
                    f"BOT_HMAC_SECRET is unset and cannot persist a generated "
                    f"secret to {secret_path}: {exc}. The engine will not start "
                    "without a stable HMAC key — set BOT_HMAC_SECRET or make "
                    "the path writable."
                ) from exc
            # Non-strict: continue with ephemeral secret (tests, dev)
        return new_secret.encode()

    # ── helpers ──

    def _sign_batch(self, challenge_id: str, batch: ChallengeBatch, ip: str) -> str:
        # Salt is excluded from signature — it's derived at solve time
        # from the previous batch's solution and is not known at generation.
        sig_data = (
            f"{challenge_id}:{batch.batch_index}:{batch.prefix}"
            f":{batch.difficulty}:{ip}"
        ).encode()
        return hmac.new(self.secret, sig_data, hashlib.sha256).hexdigest()

    # ── generate ──

    def generate_challenge(self, ip: str, threat_score: float = 0) -> MultiBatchChallenge:
        """
        Generate a new multi-batch PoW challenge for an IP.

        Returns a MultiBatchChallenge with 10-12 sub-puzzles of varying
        difficulty (14-18 bits).  Each batch is HMAC-signed independently.
        """
        batch_count = random.randint(
            self.config["BATCH_COUNT_MIN"],
            self.config["BATCH_COUNT_MAX"],
        )
        challenge_id = secrets.token_hex(16)

        # Distribute difficulties: harder for higher threat scores
        diff_min = self.config["BATCH_DIFFICULTY_MIN"]
        diff_max = self.config["BATCH_DIFFICULTY_MAX"]

        # Bias difficulties upward for higher threat scores
        if threat_score > 60:
            diff_min = max(diff_min, 16)
        elif threat_score < 30:
            diff_max = min(diff_max, 16)

        batches: list[ChallengeBatch] = []
        for i in range(batch_count):
            prefix = secrets.token_hex(16)
            difficulty = random.randint(diff_min, diff_max)
            salt = ""  # first batch has empty salt; rest filled at verify-time

            batch = ChallengeBatch(
                batch_index=i,
                prefix=prefix,
                difficulty=difficulty,
                salt=salt,
            )
            batch.hmac_sig = self._sign_batch(challenge_id, batch, ip)
            batches.append(batch)

        challenge = MultiBatchChallenge(
            challenge_id=challenge_id,
            batches=batches,
            ip=ip,
            issued_at=time.time(),
        )

        if len(self.pending) >= self.config["CHALLENGE_MAX_PENDING"]:
            self._evict_expired()

        self.pending[challenge_id] = challenge
        return challenge

    # ── verify ──

    def verify_solution(
        self,
        challenge_id: str,
        nonces: list[str],
        ip: str,
    ) -> tuple[bool, str]:
        """
        Verify a multi-batch PoW solution.

        Args:
            challenge_id: The challenge identifier.
            nonces: List of nonce strings, one per batch.
            ip: Client IP address.

        Returns:
            (success, reason) tuple.

        Verification steps per batch:
          1. HMAC signature is valid (re-derived with correct salt)
          2. SHA-256(prefix + salt + nonce) has required leading zero bits
          3. Salt for batch N = SHA-256(prefix_{N-1} + salt_{N-1} + nonce_{N-1})
        """
        # ── Rate limit ──
        self.attempts[ip] = self.attempts.get(ip, 0) + 1
        if self.attempts[ip] > self.config["MAX_ATTEMPTS_PER_IP"]:
            return False, "too_many_attempts"

        # ── Find challenge ──
        challenge = self.pending.get(challenge_id)
        if not challenge:
            return False, "unknown_challenge"

        if challenge.is_expired():
            del self.pending[challenge_id]
            return False, "expired"

        if challenge.ip != ip:
            return False, "ip_mismatch"

        if len(nonces) != len(challenge.batches):
            return False, "nonce_count_mismatch"

        # ── Verify each batch sequentially ──
        running_salt = ""
        for i, batch in enumerate(challenge.batches):
            # Recompute expected salt and re-sign
            expected_batch = ChallengeBatch(
                batch_index=batch.batch_index,
                prefix=batch.prefix,
                difficulty=batch.difficulty,
                salt=running_salt,
            )
            expected_sig = self._sign_batch(challenge_id, expected_batch, ip)
            if not hmac.compare_digest(batch.hmac_sig, expected_sig):
                return False, f"invalid_signature_batch_{i}"

            # Verify proof of work for this batch
            hash_input = f"{batch.prefix}{running_salt}{nonces[i]}".encode()
            hash_result = hashlib.sha256(hash_input).digest()

            if not self._check_leading_zeros(hash_result, batch.difficulty):
                return False, f"insufficient_work_batch_{i}"

            # Compute salt for next batch = hex digest of this solution's hash
            running_salt = hashlib.sha256(hash_input).hexdigest()

        # ── Success ──
        del self.pending[challenge_id]
        self.attempts.pop(ip, None)

        total_difficulty = sum(b.difficulty for b in challenge.batches)
        self.verified[ip] = VerifiedSession(
            ip=ip,
            verified_at=time.time(),
            score_at_verification=0,
            difficulty_solved=total_difficulty,
            solve_time_ms=0,
        )

        return True, "verified"

    # ── session / cookie management ──

    def is_verified(self, ip: str) -> bool:
        """Check if an IP has a valid verified session."""
        session = self.verified.get(ip)
        if session and not session.is_expired():
            return True
        if session:
            del self.verified[ip]
        return False

    def get_cookie_value(self, ip: str) -> Optional[str]:
        """Generate a signed cookie value for a verified IP."""
        session = self.verified.get(ip)
        if not session:
            return None

        ts = str(int(session.verified_at))
        ip_hash = hashlib.sha256(ip.encode()).hexdigest()[:16]
        payload = f"{ip_hash}:{ts}"
        sig = hmac.new(self.secret, payload.encode(), hashlib.sha256).hexdigest()[:32]
        return f"{payload}:{sig}"

    def verify_cookie(self, cookie_value: str, ip: str) -> bool:
        """Verify a PoW cookie value."""
        try:
            parts = cookie_value.split(":")
            if len(parts) != 3:
                return False

            ip_hash, ts_str, sig = parts
            expected_ip_hash = hashlib.sha256(ip.encode()).hexdigest()[:16]
            if ip_hash != expected_ip_hash:
                return False

            ts = int(ts_str)
            if time.time() - ts > self.config["VERIFIED_TTL_SECONDS"]:
                return False

            payload = f"{ip_hash}:{ts_str}"
            expected_sig = hmac.new(
                self.secret, payload.encode(), hashlib.sha256
            ).hexdigest()[:32]
            if not hmac.compare_digest(sig, expected_sig):
                return False

            return True
        except (ValueError, IndexError):
            return False

    # ── utility ──

    @staticmethod
    def _check_leading_zeros(hash_bytes: bytes, required_bits: int) -> bool:
        """Check if a hash has the required number of leading zero bits."""
        full_bytes = required_bits // 8
        remaining_bits = required_bits % 8

        for i in range(full_bytes):
            if hash_bytes[i] != 0:
                return False

        if remaining_bits > 0:
            mask = 0xFF << (8 - remaining_bits)
            if hash_bytes[full_bytes] & mask != 0:
                return False

        return True

    def _evict_expired(self):
        """Remove expired challenges and sessions."""
        now = time.time()
        expired_challenges = [
            cid for cid, c in self.pending.items() if c.is_expired()
        ]
        for cid in expired_challenges:
            del self.pending[cid]

        expired_verified = [
            ip for ip, s in self.verified.items() if s.is_expired()
        ]
        for ip in expired_verified:
            del self.verified[ip]

        active_ips = {c.ip for c in self.pending.values()} | set(self.verified.keys())
        self.attempts = {
            ip: count for ip, count in self.attempts.items()
            if ip in active_ips
        }

    def get_stats(self) -> dict:
        """Return current state statistics."""
        return {
            "pending_challenges": len(self.pending),
            "verified_sessions": len(self.verified),
            "tracked_attempts": len(self.attempts),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CLIENT-SIDE: MULTI-BATCH JS CHALLENGE PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_challenge_html(
    challenge: MultiBatchChallenge,
    redirect_url: str = "/",
    telemetry_collect_ms: int = 3000,
) -> str:
    """
    Generate the complete HTML challenge page for a multi-batch PoW.

    The page:
      1. Shows a progress bar with batch-by-batch status
      2. Runs the PoW computation in a Web Worker (non-blocking)
      3. Worker solves batches sequentially; each solution salts the next
      4. Simultaneously collects telemetry
      5. POSTs all nonces + telemetry to the verification endpoint
      6. On success, sets cookie and redirects

    The JavaScript lives in ``static/js/src/pow_worker.js`` and
    ``static/js/src/pow_challenge.js`` and is loaded via ``js_assets``;
    the values that change per challenge (challenge_id, batches,
    redirect, telemetry window) travel through a single JSON script
    block so we never have to str.format the JS itself. That means
    terser's output lands here byte-for-byte unmodified (C3 #9).
    """
    from js_assets import POW_WORKER_JS, POW_CHALLENGE_JS

    # Per-challenge config → JSON document in a <script type="application/json">
    # block. Using JSON (not f-string interpolation into JS) sidesteps
    # every escape hazard: the challenge_id can contain any character
    # json.dumps knows how to escape, and it still parses as data rather
    # than code. The one edge we handle explicitly is ``</script`` — if
    # a field value contains it, the browser would end the <script> tag
    # early. Replace ``</`` with ``<\/`` so the HTML parser sees nothing
    # script-like.
    config_payload = {
        "challenge_id": challenge.challenge_id,
        "batches": [
            {
                "batch_index": b.batch_index,
                "prefix": b.prefix,
                "difficulty": b.difficulty,
            }
            for b in challenge.batches
        ],
        "redirect": redirect_url,
        "collect_ms": telemetry_collect_ms,
    }
    config_json = json.dumps(config_payload).replace("</", "<\\/")

    # Inline CSS kept here (not f-string-interpolated) so we never have
    # to double brace `{ }` in the style rules. The CSS is small enough
    # that inlining beats a separate round-trip for the challenge page.
    style = (
        "*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }"
        "body {"
        "  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;"
        "  display: flex; justify-content: center; align-items: center;"
        "  min-height: 100vh; background: #f8f9fa; color: #333;"
        "}"
        ".card {"
        "  background: white; border-radius: 12px; padding: 48px;"
        "  box-shadow: 0 4px 24px rgba(0,0,0,0.10); text-align: center;"
        "  max-width: 460px; width: 92%;"
        "}"
        ".spinner {"
        "  width: 48px; height: 48px; margin: 0 auto 24px;"
        "  border: 4px solid #e9ecef; border-top-color: #495057;"
        "  border-radius: 50%; animation: spin 0.8s linear infinite;"
        "}"
        "@keyframes spin { to { transform: rotate(360deg); } }"
        "h2 { font-size: 18px; font-weight: 600; margin-bottom: 8px; }"
        "p.desc { font-size: 14px; color: #6c757d; line-height: 1.5; }"
        ".progress-outer {"
        "  margin-top: 24px; height: 6px; background: #e9ecef;"
        "  border-radius: 3px; overflow: hidden;"
        "}"
        ".progress-bar {"
        "  height: 100%; width: 0%;"
        "  background: linear-gradient(90deg, #495057, #6c757d);"
        "  transition: width 0.3s ease;"
        "}"
        ".batch-label { margin-top: 10px; font-size: 13px; color: #868e96; font-weight: 500; }"
        ".status { margin-top: 8px; font-size: 12px; color: #adb5bd; }"
        ".error { color: #dc3545; }"
        ".success { color: #28a745; }"
        "noscript p { color: #dc3545; font-weight: 500; }"
    )

    body = (
        '<div class="card">'
        '  <div class="spinner" id="spinner"></div>'
        "  <h2>Verifying your browser</h2>"
        '  <p class="desc">This is a one-time security check. It should complete in a few seconds.</p>'
        '  <div class="progress-outer"><div class="progress-bar" id="progress"></div></div>'
        '  <div class="batch-label" id="batchLabel">Preparing puzzles...</div>'
        '  <div class="status" id="status">Initializing...</div>'
        "  <noscript><p>JavaScript is required to verify your browser.</p></noscript>"
        "</div>"
    )

    # Assembly: HTML shell + config JSON + worker body (as a <script
    # type="text/js-worker"> so the browser won't execute it inline;
    # pow_challenge.js reads its textContent and spawns a Blob worker)
    # + the orchestrator itself.
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Verifying your browser</title>"
        f"<style>{style}</style>"
        "</head><body>"
        f"{body}"
        '<script id="challenge-config" type="application/json">'
        f"{config_json}"
        "</script>"
        '<script id="worker-src" type="text/js-worker">\n'
        f"{POW_WORKER_JS}"
        "\n</script>"
        "<script>\n"
        f"{POW_CHALLENGE_JS}"
        "\n</script>"
        "</body></html>"
    )




# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION ENDPOINT HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class ChallengeVerificationHandler:
    """
    Handles the POST from the JS challenge page.

    Verifies the multi-batch PoW solution, analyzes the telemetry,
    and returns a signed cookie if everything checks out.
    """

    def __init__(self, pow_engine: ProofOfWorkEngine):
        self.pow = pow_engine

    def handle_verification(self, ip: str, body: dict) -> dict:
        """
        Process a challenge solution submission.

        Args:
            ip: Client IP address
            body: JSON body from the challenge page POST

        Returns:
            dict with {verified, cookie, telemetry_score, reasons}
        """
        challenge_id = body.get("challenge_id", "")
        nonces = body.get("nonces", [])
        solve_time_ms = body.get("solve_time_ms", 0)
        telemetry_data = body.get("telemetry", {})

        result: dict = {
            "verified": False,
            "cookie": None,
            "telemetry_score": 0,
            "reasons": [],
        }

        # ── Step 1: Verify PoW ──
        success, reason = self.pow.verify_solution(challenge_id, nonces, ip)
        if not success:
            result["reasons"].append(f"PoW failed: {reason}")
            return result

        # ── Step 2: Validate solve time ──
        if solve_time_ms < 50:
            result["reasons"].append(
                f"Suspiciously fast solve: {solve_time_ms}ms"
            )
            result["telemetry_score"] += 20

        # ── Step 3: Analyze telemetry ──
        tel_score, tel_reasons = self._analyze_telemetry(telemetry_data)
        result["telemetry_score"] += tel_score
        result["reasons"].extend(tel_reasons)

        # ── Step 4: Decision ──
        if result["telemetry_score"] > 50:
            result["reasons"].append(
                f"Telemetry score too high: {result['telemetry_score']}"
            )
            return result

        # ── Step 5: Issue cookie ──
        session = self.pow.verified.get(ip)
        if session:
            session.solve_time_ms = solve_time_ms
            session.telemetry_score = result["telemetry_score"]

        result["verified"] = True
        result["cookie"] = self.pow.get_cookie_value(ip)

        return result

    def _analyze_telemetry(self, data: dict) -> tuple[float, list[str]]:
        """Quick telemetry analysis (subset of full v2 analyzers)."""
        score = 0.0
        reasons: list[str] = []

        if data.get("webdriver") is True:
            score += 25
            reasons.append("webdriver=true")

        if data.get("plugins", 0) == 0:
            score += 8
            reasons.append("No browser plugins")

        if data.get("outerW", 1) == 0 or data.get("outerH", 1) == 0:
            score += 10
            reasons.append("No window chrome (outerW/H=0)")

        renderer = data.get("webglRenderer", "")
        if "swiftshader" in renderer.lower() or "llvmpipe" in renderer.lower():
            score += 12
            reasons.append(f"Headless WebGL: {renderer}")

        raf_avg = data.get("rafAvg", 16.7)
        if 0 < raf_avg < 10:
            score += 10
            reasons.append(f"RAF too fast: {raf_avg:.1f}ms")

        moves = data.get("mouseMoves", [])
        if len(moves) < 3:
            score += 8
            reasons.append(f"Minimal mouse movement: {len(moves)} events")

        sw = data.get("screenW", 0)
        sh = data.get("screenH", 0)
        if (sw, sh) == (800, 600) or (sw, sh) == (1024, 768):
            score += 5
            reasons.append(f"Default headless screen: {sw}x{sh}")

        return score, reasons


# ═══════════════════════════════════════════════════════════════════════════════
# BIOMETRIC CAPTCHA
# ═══════════════════════════════════════════════════════════════════════════════

class BiometricCaptcha:
    """
    Interactive tracing challenge: user must trace a curved Bezier path
    on a canvas.  Server validates path coverage, biological motor noise,
    timing, and optional pressure variation.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or POW_CONFIG
        self.secret = ProofOfWorkEngine._resolve_secret(self.config)
        # captcha_id -> curve params
        self._pending: dict[str, dict] = {}

    def _generate_bezier_points(self) -> list[dict]:
        """Generate control points for a random cubic Bezier curve."""
        # Canvas is 400x300
        cw, ch = 400, 300
        margin = 40
        points = []
        # Start point (left side)
        points.append({
            "x": random.randint(margin, cw // 4),
            "y": random.randint(margin, ch - margin),
        })
        # Control point 1
        points.append({
            "x": random.randint(cw // 4, cw // 2),
            "y": random.randint(margin, ch - margin),
        })
        # Control point 2
        points.append({
            "x": random.randint(cw // 2, 3 * cw // 4),
            "y": random.randint(margin, ch - margin),
        })
        # End point (right side)
        points.append({
            "x": random.randint(3 * cw // 4, cw - margin),
            "y": random.randint(margin, ch - margin),
        })
        return points

    @staticmethod
    def _bezier_at(t: float, points: list[dict]) -> tuple[float, float]:
        """Evaluate a cubic Bezier curve at parameter t in [0,1]."""
        p0, p1, p2, p3 = points
        u = 1.0 - t
        x = (u**3 * p0["x"] + 3 * u**2 * t * p1["x"]
             + 3 * u * t**2 * p2["x"] + t**3 * p3["x"])
        y = (u**3 * p0["y"] + 3 * u**2 * t * p1["y"]
             + 3 * u * t**2 * p2["y"] + t**3 * p3["y"])
        return x, y

    def _sample_curve(self, points: list[dict], n: int = 100) -> list[tuple[float, float]]:
        """Sample n points along the Bezier curve."""
        return [self._bezier_at(i / (n - 1), points) for i in range(n)]

    def generate_captcha(self, ip: str) -> dict:
        """
        Generate a biometric tracing captcha.

        Returns dict with:
          - captcha_id: unique identifier
          - curve_points: the 4 Bezier control points
          - html: complete HTML page for the captcha
        """
        captcha_id = secrets.token_hex(16)
        curve_points = self._generate_bezier_points()

        # Sign for tamper-proofing
        sig_data = f"{captcha_id}:{ip}:{json.dumps(curve_points)}".encode()
        sig = hmac.new(self.secret, sig_data, hashlib.sha256).hexdigest()

        self._pending[captcha_id] = {
            "ip": ip,
            "curve_points": curve_points,
            "sig": sig,
            "issued_at": time.time(),
        }

        points_json = json.dumps(curve_points)

        html = self._render_captcha_html(captcha_id, points_json)

        return {
            "captcha_id": captcha_id,
            "curve_points": curve_points,
            "html": html,
        }

    def verify_captcha(
        self, ip: str, captcha_id: str, trace_data: list[dict],
    ) -> tuple[bool, float, list[str]]:
        """
        Verify a biometric captcha submission.

        Args:
            ip: Client IP
            captcha_id: The captcha identifier
            trace_data: list of dicts with keys: x, y, t, pressure (optional)

        Returns:
            (passed, score, reasons)
            score 0.0 = definitely human, 1.0 = definitely bot
        """
        reasons: list[str] = []
        score = 0.0

        pending = self._pending.pop(captcha_id, None)
        if not pending:
            return False, 1.0, ["Unknown or expired captcha"]

        if pending["ip"] != ip:
            return False, 1.0, ["IP mismatch"]

        if time.time() - pending["issued_at"] > 60:
            return False, 1.0, ["Captcha expired"]

        curve_points = pending["curve_points"]

        if len(trace_data) < 5:
            return False, 1.0, ["Too few trace points"]

        # ── Timing ──
        timestamps = [p.get("t", 0) for p in trace_data]
        if timestamps:
            trace_duration_s = (max(timestamps) - min(timestamps)) / 1000.0
        else:
            trace_duration_s = 0

        min_time = self.config["BIOMETRIC_CAPTCHA_MIN_TRACE_TIME_S"]
        max_time = self.config["BIOMETRIC_CAPTCHA_MAX_TRACE_TIME_S"]

        if trace_duration_s < min_time:
            score += 0.35
            reasons.append(f"Trace too fast: {trace_duration_s:.2f}s (min {min_time}s)")
        elif trace_duration_s > max_time:
            score += 0.15
            reasons.append(f"Trace too slow: {trace_duration_s:.2f}s (max {max_time}s)")

        # ── Path coverage ──
        curve_samples = self._sample_curve(curve_points, 100)
        coverage_threshold = 20.0  # pixels
        covered = set()
        for sample_idx, (sx, sy) in enumerate(curve_samples):
            for tp in trace_data:
                dx = tp.get("x", 0) - sx
                dy = tp.get("y", 0) - sy
                if math.sqrt(dx * dx + dy * dy) < coverage_threshold:
                    covered.add(sample_idx)
                    break

        coverage_ratio = len(covered) / len(curve_samples)
        min_coverage = self.config["BIOMETRIC_CAPTCHA_MIN_COVERAGE"]
        if coverage_ratio < min_coverage:
            score += 0.3
            reasons.append(
                f"Insufficient path coverage: {coverage_ratio:.0%} "
                f"(need {min_coverage:.0%})"
            )

        # ── Biological motor noise ──
        # Real humans have micro-deviations: compute standard deviation of
        # distance from the nearest curve point.
        distances: list[float] = []
        for tp in trace_data:
            min_dist = float("inf")
            for sx, sy in curve_samples:
                dx = tp.get("x", 0) - sx
                dy = tp.get("y", 0) - sy
                d = math.sqrt(dx * dx + dy * dy)
                if d < min_dist:
                    min_dist = d
            distances.append(min_dist)

        if distances:
            mean_dist = sum(distances) / len(distances)
            variance = sum((d - mean_dist) ** 2 for d in distances) / len(distances)
            std_dist = math.sqrt(variance)
        else:
            mean_dist = 0
            std_dist = 0

        # Too perfect (std < 1 pixel) => likely programmatic
        if std_dist < 1.0:
            score += 0.25
            reasons.append(
                f"Trace too perfect: stddev={std_dist:.2f}px (suspiciously low)"
            )
        # Way too noisy (std > 50) => random garbage
        if std_dist > 50.0:
            score += 0.2
            reasons.append(
                f"Trace too noisy: stddev={std_dist:.2f}px (suspiciously high)"
            )

        # ── Pressure variation (if available) ──
        pressures = [
            p.get("pressure", -1) for p in trace_data
            if p.get("pressure", -1) >= 0
        ]
        if len(pressures) > 10:
            p_mean = sum(pressures) / len(pressures)
            p_var = sum((p - p_mean) ** 2 for p in pressures) / len(pressures)
            p_std = math.sqrt(p_var)
            # Zero variation = fake
            if p_std < 0.001:
                score += 0.1
                reasons.append("No pressure variation (likely non-touch)")

        passed = score < 0.5 and coverage_ratio >= min_coverage
        return passed, score, reasons

    @staticmethod
    def _render_captcha_html(captcha_id: str, points_json: str) -> str:
        """Render the full HTML for the biometric captcha page.

        The JavaScript lives in ``static/js/src/captcha.js`` and is loaded
        via ``js_assets``; the per-captcha config (captcha_id, curve
        points) travels through a JSON script block so we never have to
        str.format JS source (C3 #9).
        """
        from js_assets import CAPTCHA_JS

        # points_json arrives as a serialised JSON string already (the
        # caller built it with json.dumps(curve_points)). Parse and
        # re-serialise alongside captcha_id so the config block is a
        # single well-formed JSON document.
        try:
            points = json.loads(points_json)
        except (TypeError, ValueError):
            points = []
        config_json = json.dumps({
            "captcha_id": captcha_id,
            "points": points,
        }).replace("</", "<\\/")

        style = (
            "*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }"
            "body {"
            "  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;"
            "  display: flex; justify-content: center; align-items: center;"
            "  min-height: 100vh; background: #f8f9fa; color: #333;"
            "}"
            ".card {"
            "  background: white; border-radius: 12px; padding: 36px;"
            "  box-shadow: 0 4px 24px rgba(0,0,0,0.10); text-align: center;"
            "  max-width: 500px; width: 94%;"
            "}"
            "h2 { font-size: 18px; font-weight: 600; margin-bottom: 8px; }"
            "p.desc { font-size: 14px; color: #6c757d; line-height: 1.5; margin-bottom: 16px; }"
            "canvas {"
            "  border: 2px solid #dee2e6; border-radius: 8px;"
            "  cursor: crosshair; display: block; margin: 0 auto;"
            "  touch-action: none;"
            "}"
            ".btn {"
            "  margin-top: 16px; padding: 10px 32px; border: none; border-radius: 6px;"
            "  background: #495057; color: white; font-size: 14px; font-weight: 500;"
            "  cursor: pointer; transition: background 0.2s;"
            "}"
            ".btn:hover { background: #343a40; }"
            ".btn:disabled { background: #adb5bd; cursor: not-allowed; }"
            ".status { margin-top: 10px; font-size: 12px; color: #adb5bd; min-height: 18px; }"
            ".error { color: #dc3545; }"
            ".success { color: #28a745; }"
        )

        body = (
            '<div class="card">'
            "<h2>Trace the path below</h2>"
            '<p class="desc">Use your mouse or finger to trace along the highlighted curve.</p>'
            '<canvas id="captchaCanvas" width="400" height="300"></canvas>'
            '<button class="btn" id="submitBtn" disabled>Submit</button>'
            '<div class="status" id="status"></div>'
            "</div>"
        )

        return (
            "<!DOCTYPE html>"
            '<html lang="en"><head>'
            '<meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>Security Verification</title>"
            f"<style>{style}</style>"
            "</head><body>"
            f"{body}"
            '<script id="captcha-config" type="application/json">'
            f"{config_json}"
            "</script>"
            "<script>\n"
            f"{CAPTCHA_JS}"
            "\n</script>"
            "</body></html>"
        )




# ═══════════════════════════════════════════════════════════════════════════════
# API PROTECTION — HMAC-SHA256 SIGNATURE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class APIProtector:
    """
    HMAC-SHA256 signed request verification for non-browser /api/ clients.

    Expected headers:
      X-API-Key:    The client's API key
      X-Timestamp:  Unix timestamp (seconds)
      X-Signature:  HMAC-SHA256(api_secret, "{method}:{path}:{timestamp}:{body_hash}")

    Provides replay protection via timestamp window and per-key secrets.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or POW_CONFIG
        # api_key -> api_secret
        self._keys: dict[str, str] = {}
        self._load_keys()

    def _key_file_path(self) -> Path:
        return Path(self.config["API_KEY_FILE"])

    def _load_keys(self) -> None:
        """Load API keys from the JSON config file if it exists."""
        path = self._key_file_path()
        if path.is_file():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._keys = data
            except (json.JSONDecodeError, OSError):
                pass

    def _save_keys(self) -> None:
        """Persist current API keys to the JSON config file."""
        path = self._key_file_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(self._keys, f, indent=2)
        except OSError:
            pass  # Best-effort; in-memory keys still work

    def generate_api_key(self) -> tuple[str, str]:
        """
        Generate a new API key pair.

        Returns:
            (api_key, api_secret) — the client needs both.
        """
        api_key = "bk_" + secrets.token_hex(16)
        api_secret = secrets.token_hex(32)
        self._keys[api_key] = api_secret
        self._save_keys()
        return api_key, api_secret

    def add_api_key(self, api_key: str, api_secret: str) -> None:
        """Add an externally-generated API key."""
        self._keys[api_key] = api_secret
        self._save_keys()

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key. Returns True if it existed."""
        if api_key in self._keys:
            del self._keys[api_key]
            self._save_keys()
            return True
        return False

    def verify_api_request(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body_hash: str,
    ) -> tuple[bool, str]:
        """
        Verify an HMAC-signed API request.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path (e.g. /api/v1/data)
            headers: Request headers dict
            body_hash: SHA-256 hex digest of request body
                       (empty string hash for GET requests)

        Returns:
            (valid, reason) tuple
        """
        api_key = headers.get("X-API-Key", headers.get("x-api-key", ""))
        timestamp_str = headers.get("X-Timestamp", headers.get("x-timestamp", ""))
        signature = headers.get("X-Signature", headers.get("x-signature", ""))

        if not api_key:
            return False, "missing_api_key"
        if not timestamp_str:
            return False, "missing_timestamp"
        if not signature:
            return False, "missing_signature"

        # ── Check API key exists ──
        api_secret = self._keys.get(api_key)
        if api_secret is None:
            return False, "unknown_api_key"

        # ── Check timestamp freshness (replay protection) ──
        try:
            timestamp = int(timestamp_str)
        except ValueError:
            return False, "invalid_timestamp"

        now = int(time.time())
        tolerance = self.config["API_TIMESTAMP_TOLERANCE"]
        if abs(now - timestamp) > tolerance:
            return False, "timestamp_expired"

        # ── Verify HMAC signature ──
        message = f"{method}:{path}:{timestamp_str}:{body_hash}"
        expected_sig = hmac.new(
            api_secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_sig):
            return False, "invalid_signature"

        return True, "valid"

    @staticmethod
    def compute_request_signature(
        api_secret: str,
        method: str,
        path: str,
        timestamp: str,
        body_hash: str,
    ) -> str:
        """
        Compute the HMAC-SHA256 signature for a request.
        Utility for clients constructing signed requests.
        """
        message = f"{method}:{path}:{timestamp}:{body_hash}"
        return hmac.new(
            api_secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Run a complete test suite for all components."""
    import time as _time

    print("=" * 64)
    print("  Proof-of-Work Challenge System v2.1 — Self Test")
    print("=" * 64)

    errors = 0

    # ──────────────────────────────────────────────────────────────
    # 1. Multi-batch challenge: generate -> solve -> verify
    # ──────────────────────────────────────────────────────────────
    print("\n[1] Multi-batch PoW challenge")
    engine = ProofOfWorkEngine()

    challenge = engine.generate_challenge("192.168.1.100", threat_score=45)
    print(f"  Challenge ID:  {challenge.challenge_id}")
    print(f"  Batches:       {len(challenge.batches)}")
    for b in challenge.batches:
        print(f"    Batch {b.batch_index}: difficulty={b.difficulty}, prefix={b.prefix[:12]}...")

    print(f"\n  Solving {len(challenge.batches)} batches sequentially...")
    start = _time.time()
    nonces: list[str] = []
    running_salt = ""

    for batch in challenge.batches:
        nonce = 0
        while True:
            candidate = f"{batch.prefix}{running_salt}{nonce:x}".encode()
            hash_result = hashlib.sha256(candidate).digest()
            if ProofOfWorkEngine._check_leading_zeros(hash_result, batch.difficulty):
                break
            nonce += 1
        nonce_hex = f"{nonce:x}"
        nonces.append(nonce_hex)
        running_salt = hashlib.sha256(
            f"{batch.prefix}{running_salt}{nonce_hex}".encode()
        ).hexdigest()

    elapsed = _time.time() - start
    print(f"  Solved in {elapsed:.2f}s")
    print(f"  Nonces: {nonces}")

    success, reason = engine.verify_solution(
        challenge.challenge_id, nonces, "192.168.1.100"
    )
    print(f"  Verification: {'PASS' if success else 'FAIL'} ({reason})")
    if not success:
        errors += 1

    # Cookie
    cookie = engine.get_cookie_value("192.168.1.100")
    print(f"  Cookie: {cookie}")
    is_valid = engine.verify_cookie(cookie, "192.168.1.100")
    print(f"  Cookie valid: {is_valid}")
    if not is_valid:
        errors += 1

    is_valid_wrong = engine.verify_cookie(cookie, "10.0.0.1")
    print(f"  Cookie valid (wrong IP): {is_valid_wrong}")
    if is_valid_wrong:
        errors += 1

    # Wrong nonce count
    challenge2 = engine.generate_challenge("192.168.1.200")
    ok2, reason2 = engine.verify_solution(challenge2.challenge_id, ["bad"], "192.168.1.200")
    print(f"  Wrong nonce count: {'PASS' if not ok2 else 'FAIL'} ({reason2})")
    if ok2:
        errors += 1

    # HTML generation
    challenge3 = engine.generate_challenge("192.168.1.101", threat_score=50)
    html = generate_challenge_html(challenge3, "/original-page")
    print(f"\n  Challenge HTML: {len(html):,} bytes")
    print(f"  Contains Worker: {'worker-src' in html}")
    print(f"  Contains batch progress: {'batchLabel' in html}")
    print(f"  Contains telemetry: {'mouseMoves' in html}")

    print(f"\n  Stats: {engine.get_stats()}")

    # ──────────────────────────────────────────────────────────────
    # 2. API HMAC signature verification
    # ──────────────────────────────────────────────────────────────
    print("\n[2] API HMAC signature verification")
    api = APIProtector()

    api_key, api_secret = api.generate_api_key()
    print(f"  Generated key:    {api_key}")
    print(f"  Generated secret: {api_secret[:16]}...")

    ts = str(int(_time.time()))
    body = '{"query": "test"}'
    body_hash = hashlib.sha256(body.encode()).hexdigest()
    sig = APIProtector.compute_request_signature(
        api_secret, "POST", "/api/v1/data", ts, body_hash
    )

    headers = {
        "X-API-Key": api_key,
        "X-Timestamp": ts,
        "X-Signature": sig,
    }

    valid, vreason = api.verify_api_request("POST", "/api/v1/data", headers, body_hash)
    print(f"  Valid signature:   {'PASS' if valid else 'FAIL'} ({vreason})")
    if not valid:
        errors += 1

    # Tampered signature
    bad_headers = dict(headers)
    bad_headers["X-Signature"] = "deadbeef" * 8
    valid2, vreason2 = api.verify_api_request("POST", "/api/v1/data", bad_headers, body_hash)
    print(f"  Tampered sig:      {'PASS' if not valid2 else 'FAIL'} ({vreason2})")
    if valid2:
        errors += 1

    # Expired timestamp
    old_headers = dict(headers)
    old_headers["X-Timestamp"] = str(int(_time.time()) - 600)
    old_sig = APIProtector.compute_request_signature(
        api_secret, "POST", "/api/v1/data", old_headers["X-Timestamp"], body_hash
    )
    old_headers["X-Signature"] = old_sig
    valid3, vreason3 = api.verify_api_request("POST", "/api/v1/data", old_headers, body_hash)
    print(f"  Expired timestamp: {'PASS' if not valid3 else 'FAIL'} ({vreason3})")
    if valid3:
        errors += 1

    # Unknown key
    unk_headers = dict(headers)
    unk_headers["X-API-Key"] = "bk_unknown"
    valid4, vreason4 = api.verify_api_request("POST", "/api/v1/data", unk_headers, body_hash)
    print(f"  Unknown key:       {'PASS' if not valid4 else 'FAIL'} ({vreason4})")
    if valid4:
        errors += 1

    # ──────────────────────────────────────────────────────────────
    # 3. Biometric captcha generation
    # ──────────────────────────────────────────────────────────────
    print("\n[3] Biometric captcha")
    captcha = BiometricCaptcha()

    result = captcha.generate_captcha("192.168.1.100")
    print(f"  Captcha ID:        {result['captcha_id']}")
    print(f"  Curve points:      {len(result['curve_points'])} control points")
    print(f"  HTML size:         {len(result['html']):,} bytes")
    print(f"  Contains canvas:   {'captchaCanvas' in result['html']}")
    print(f"  Contains bezier:   {'bezierCurveTo' in result['html']}")

    # Quick verify with synthetic perfect trace (should pass with some noise)
    curve_pts = result["curve_points"]
    fake_trace: list[dict] = []
    base_t = 1000
    for i in range(80):
        t_param = i / 79.0
        bx, by = BiometricCaptcha._bezier_at(t_param, curve_pts)
        fake_trace.append({
            "x": bx + random.uniform(-5, 5),
            "y": by + random.uniform(-5, 5),
            "t": base_t + i * 30,  # ~2.4 seconds total
            "pressure": 0.5 + random.uniform(-0.1, 0.1),
        })
    passed, score, reasons = captcha.verify_captcha(
        "192.168.1.100", result["captcha_id"], fake_trace
    )
    print(f"  Synthetic trace:   passed={passed}, score={score:.2f}")
    if reasons:
        for r in reasons:
            print(f"    - {r}")
    if not passed:
        errors += 1
        print("  WARNING: synthetic trace did not pass (may vary due to randomness)")

    # ──────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    if errors == 0:
        print("  All tests passed!")
    else:
        print(f"  {errors} test(s) FAILED")
    print("=" * 64)

    return errors == 0


if __name__ == "__main__":
    self_test()
