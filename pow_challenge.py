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
    "HMAC_SECRET": os.environ.get(
        "BOT_HMAC_SECRET",
        "CHANGE_ME_IN_PRODUCTION_" + secrets.token_hex(16)
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
        self.secret = self.config["HMAC_SECRET"].encode()

        # ── State ──
        self.pending: dict[str, MultiBatchChallenge] = {}
        self.verified: dict[str, VerifiedSession] = {}
        self.attempts: dict[str, int] = {}

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
    """
    # Serialise batches to JSON for embedding in the page
    batches_json = json.dumps([
        {
            "batch_index": b.batch_index,
            "prefix": b.prefix,
            "difficulty": b.difficulty,
        }
        for b in challenge.batches
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Verifying your browser</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh; background: #f8f9fa; color: #333;
  }}
  .card {{
    background: white; border-radius: 12px; padding: 48px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.10); text-align: center;
    max-width: 460px; width: 92%;
  }}
  .spinner {{
    width: 48px; height: 48px; margin: 0 auto 24px;
    border: 4px solid #e9ecef; border-top-color: #495057;
    border-radius: 50%; animation: spin 0.8s linear infinite;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  h2 {{ font-size: 18px; font-weight: 600; margin-bottom: 8px; }}
  p.desc {{ font-size: 14px; color: #6c757d; line-height: 1.5; }}
  .progress-outer {{
    margin-top: 24px; height: 6px; background: #e9ecef;
    border-radius: 3px; overflow: hidden;
  }}
  .progress-bar {{
    height: 100%; width: 0%; background: linear-gradient(90deg, #495057, #6c757d);
    transition: width 0.3s ease;
  }}
  .batch-label {{
    margin-top: 10px; font-size: 13px; color: #868e96; font-weight: 500;
  }}
  .status {{ margin-top: 8px; font-size: 12px; color: #adb5bd; }}
  .error {{ color: #dc3545; }}
  .success {{ color: #28a745; }}
  noscript p {{ color: #dc3545; font-weight: 500; }}
</style>
</head>
<body>
<div class="card">
  <div class="spinner" id="spinner"></div>
  <h2>Verifying your browser</h2>
  <p class="desc">This is a one-time security check. It should complete in a few seconds.</p>
  <div class="progress-outer"><div class="progress-bar" id="progress"></div></div>
  <div class="batch-label" id="batchLabel">Preparing puzzles...</div>
  <div class="status" id="status">Initializing...</div>
  <noscript><p>JavaScript is required to verify your browser.</p></noscript>
</div>

<!-- Web Worker for sequential multi-batch PoW -->
<script id="worker-src" type="text/js-worker">
  const K = new Uint32Array([
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
  ]);

  function sha256(msg) {{
    const msgLen = msg.length;
    const bitLen = msgLen * 8;
    const padLen = ((msgLen + 9 + 63) & ~63);
    const buf = new Uint8Array(padLen);
    for (let i = 0; i < msgLen; i++) buf[i] = msg.charCodeAt(i);
    buf[msgLen] = 0x80;
    const view = new DataView(buf.buffer);
    view.setUint32(padLen - 4, bitLen, false);

    let h0=0x6a09e667, h1=0xbb67ae85, h2=0x3c6ef372, h3=0xa54ff53a;
    let h4=0x510e527f, h5=0x9b05688c, h6=0x1f83d9ab, h7=0x5be0cd19;
    const w = new Uint32Array(64);

    for (let off = 0; off < padLen; off += 64) {{
      for (let i = 0; i < 16; i++) w[i] = view.getUint32(off + i*4, false);
      for (let i = 16; i < 64; i++) {{
        const s0 = (w[i-15]>>>7 | w[i-15]<<25) ^ (w[i-15]>>>18 | w[i-15]<<14) ^ (w[i-15]>>>3);
        const s1 = (w[i-2]>>>17 | w[i-2]<<15) ^ (w[i-2]>>>19 | w[i-2]<<13) ^ (w[i-2]>>>10);
        w[i] = (w[i-16] + s0 + w[i-7] + s1) | 0;
      }}
      let a=h0,b=h1,c=h2,d=h3,e=h4,f=h5,g=h6,h=h7;
      for (let i = 0; i < 64; i++) {{
        const S1 = (e>>>6|e<<26)^(e>>>11|e<<21)^(e>>>25|e<<7);
        const ch = (e&f)^(~e&g);
        const t1 = (h+S1+ch+K[i]+w[i])|0;
        const S0 = (a>>>2|a<<30)^(a>>>13|a<<19)^(a>>>22|a<<10);
        const maj = (a&b)^(a&c)^(b&c);
        const t2 = (S0+maj)|0;
        h=g; g=f; f=e; e=(d+t1)|0; d=c; c=b; b=a; a=(t1+t2)|0;
      }}
      h0=(h0+a)|0; h1=(h1+b)|0; h2=(h2+c)|0; h3=(h3+d)|0;
      h4=(h4+e)|0; h5=(h5+f)|0; h6=(h6+g)|0; h7=(h7+h)|0;
    }}
    return [h0,h1,h2,h3,h4,h5,h6,h7];
  }}

  function sha256hex(msg) {{
    const h = sha256(msg);
    let out = "";
    for (let i = 0; i < 8; i++) {{
      out += ("00000000" + (h[i]>>>0).toString(16)).slice(-8);
    }}
    return out;
  }}

  function checkLeadingZeros(hash, bits) {{
    const fullBytes = bits >> 3;
    const remBits = bits & 7;
    for (let i = 0; i < fullBytes; i++) {{
      const byte = (hash[i>>2] >>> (24 - (i&3)*8)) & 0xFF;
      if (byte !== 0) return false;
    }}
    if (remBits > 0) {{
      const byte = (hash[fullBytes>>2] >>> (24 - (fullBytes&3)*8)) & 0xFF;
      if ((byte & (0xFF << (8 - remBits))) !== 0) return false;
    }}
    return true;
  }}

  self.onmessage = function(e) {{
    const batches = e.data.batches;
    const totalBatches = batches.length;
    const startTime = Date.now();
    const nonces = [];
    let runningSalt = "";

    function solveBatch(batchIdx) {{
      if (batchIdx >= totalBatches) {{
        self.postMessage({{
          done: true,
          nonces: nonces,
          elapsed: Date.now() - startTime,
        }});
        return;
      }}

      const batch = batches[batchIdx];
      const prefix = batch.prefix;
      const difficulty = batch.difficulty;
      let nonce = 0;
      const batchSize = 5000;

      function solveChunk() {{
        for (let i = 0; i < batchSize; i++) {{
          const candidate = prefix + runningSalt + nonce.toString(16);
          const hash = sha256(candidate);
          if (checkLeadingZeros(hash, difficulty)) {{
            const foundNonce = nonce.toString(16);
            nonces.push(foundNonce);
            // Compute running salt for next batch
            runningSalt = sha256hex(candidate);
            self.postMessage({{
              done: false,
              batchDone: true,
              batchIndex: batchIdx,
              totalBatches: totalBatches,
              nonce: foundNonce,
              elapsed: Date.now() - startTime,
            }});
            setTimeout(function() {{ solveBatch(batchIdx + 1); }}, 0);
            return;
          }}
          nonce++;
        }}
        self.postMessage({{
          done: false,
          batchDone: false,
          batchIndex: batchIdx,
          totalBatches: totalBatches,
          hashes: nonce,
          elapsed: Date.now() - startTime,
        }});
        setTimeout(solveChunk, 0);
      }}
      solveChunk();
    }}
    solveBatch(0);
  }};
</script>

<!-- Main script: orchestrates multi-batch PoW + telemetry -->
<script>
(function() {{
  "use strict";

  var CHALLENGE_ID = "{challenge.challenge_id}";
  var BATCHES = {batches_json};
  var REDIRECT = "{redirect_url}";
  var COLLECT_MS = {telemetry_collect_ms};
  var BEACON_URL = "/_bot_challenge";

  var progress = document.getElementById("progress");
  var status = document.getElementById("status");
  var batchLabel = document.getElementById("batchLabel");
  var spinner = document.getElementById("spinner");

  // ── Telemetry collection (runs in parallel with PoW) ──
  var telemetry = {{
    ts: Date.now(),
    webdriver: !!navigator.webdriver,
    plugins: navigator.plugins ? navigator.plugins.length : 0,
    languages: navigator.languages ? Array.from(navigator.languages) : [],
    platform: navigator.platform || "",
    hwConcurrency: navigator.hardwareConcurrency || 0,
    deviceMemory: navigator.deviceMemory || null,
    maxTouchPoints: navigator.maxTouchPoints || 0,
    screenW: screen.width, screenH: screen.height,
    colorDepth: screen.colorDepth,
    outerW: window.outerWidth, outerH: window.outerHeight,
    dpr: window.devicePixelRatio || 1,
    perfRes: 0, rafAvg: 0, rafStd: 0,
    canvasHash: "", webglRenderer: "", webglVendor: "",
    mouseMoves: [], mouseClicks: [], scrollEvents: [], keyCount: 0,
    apis: [],
    notifPerm: "",
  }};

  var t1 = performance.now(), t2 = performance.now();
  telemetry.perfRes = t2 - t1;

  var rafTs = []; var rafN = 0;
  function mRAF(ts) {{ rafTs.push(ts); if (++rafN < 30) requestAnimationFrame(mRAF); }}
  requestAnimationFrame(mRAF);

  try {{
    var c = document.createElement("canvas"); c.width=256; c.height=64;
    var x = c.getContext("2d");
    x.textBaseline="top"; x.font="14px Arial";
    x.fillStyle="#f60"; x.fillRect(125,1,62,20);
    x.fillStyle="#069"; x.fillText("BotCk,.+@#$",2,15);
    x.fillStyle="rgba(102,204,0,0.7)"; x.fillText("BotCk,.+@#$",4,17);
    x.globalCompositeOperation="multiply";
    x.fillStyle="rgb(255,0,255)"; x.beginPath(); x.arc(50,50,50,0,Math.PI*2,true); x.fill();
    var hv=0; var d=c.toDataURL();
    for(var i=0;i<d.length;i++) {{ hv=((hv<<5)-hv)+d.charCodeAt(i); hv|=0; }}
    telemetry.canvasHash = hv.toString(16);
  }} catch(e) {{}}

  try {{
    var c2 = document.createElement("canvas");
    var gl = c2.getContext("webgl")||c2.getContext("experimental-webgl");
    if (gl) {{
      var dbg = gl.getExtension("WEBGL_debug_renderer_info");
      if (dbg) {{
        telemetry.webglRenderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)||"";
        telemetry.webglVendor = gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL)||"";
      }}
    }}
  }} catch(e) {{}}

  document.addEventListener("mousemove", function(e) {{
    if (telemetry.mouseMoves.length < 200)
      telemetry.mouseMoves.push([e.clientX, e.clientY, e.timeStamp|0]);
  }});
  document.addEventListener("click", function(e) {{
    telemetry.mouseClicks.push([e.clientX, e.clientY, e.timeStamp|0]);
  }});
  document.addEventListener("scroll", function() {{
    telemetry.scrollEvents.push([window.scrollY, Date.now()]);
  }});
  document.addEventListener("keydown", function() {{ telemetry.keyCount++; }});

  if (navigator.permissions) {{
    navigator.permissions.query({{name:"notifications"}}).then(function(r) {{
      telemetry.notifPerm = r.state;
    }}).catch(function(){{}});
  }}

  ["Bluetooth","BatteryManager","Gamepad","MediaDevices","Credential",
   "PaymentRequest","Presentation","WakeLock","USB","Serial","HID","XRSystem"
  ].forEach(function(a) {{ if (a in window || a in navigator) telemetry.apis.push(a); }});

  // ── Start Web Worker ──
  status.textContent = "Solving puzzles...";
  batchLabel.textContent = "Solving puzzle 1 of " + BATCHES.length + "...";

  var workerSrc = document.getElementById("worker-src").textContent;
  var blob = new Blob([workerSrc], {{type: "application/javascript"}});
  var worker = new Worker(URL.createObjectURL(blob));

  var solved = false;
  var solvedNonces = [];
  var solveElapsed = 0;

  worker.onmessage = function(e) {{
    var msg = e.data;

    if (msg.done) {{
      solved = true;
      solvedNonces = msg.nonces;
      solveElapsed = msg.elapsed;
      progress.style.width = "100%";
      batchLabel.textContent = "All puzzles solved!";
      status.textContent = "Verified! Redirecting...";
      status.className = "status success";
      spinner.style.borderTopColor = "#28a745";
      worker.terminate();
      submitResult();
    }} else if (msg.batchDone) {{
      var pct = ((msg.batchIndex + 1) / msg.totalBatches) * 100;
      progress.style.width = pct + "%";
      batchLabel.textContent = "Solved puzzle " + (msg.batchIndex + 1) + " of " + msg.totalBatches;
      if (msg.batchIndex + 1 < msg.totalBatches) {{
        status.textContent = "Starting puzzle " + (msg.batchIndex + 2) + "...";
      }}
    }} else {{
      var basePct = (msg.batchIndex / msg.totalBatches) * 100;
      var expectedHashes = Math.pow(2, BATCHES[msg.batchIndex].difficulty);
      var inBatchPct = Math.min(0.95, msg.hashes / expectedHashes);
      var totalPct = basePct + inBatchPct * (100 / msg.totalBatches);
      progress.style.width = Math.min(95, totalPct) + "%";
      batchLabel.textContent = "Solving puzzle " + (msg.batchIndex + 1) + " of " + msg.totalBatches + "...";
      var rate = msg.hashes / (msg.elapsed / 1000);
      status.textContent = "Working... " + (rate/1000|0) + "k hashes/s";
    }}
  }};

  worker.postMessage({{ batches: BATCHES }});

  // ── Submit all nonces + telemetry ──
  function submitResult() {{
    if (rafTs.length > 2) {{
      var intervals = [];
      for (var i=1; i<rafTs.length; i++) intervals.push(rafTs[i]-rafTs[i-1]);
      var sum = intervals.reduce(function(a,b){{return a+b;}},0);
      telemetry.rafAvg = sum/intervals.length;
      var sq = intervals.reduce(function(a,b){{return a+Math.pow(b-telemetry.rafAvg,2);}},0);
      telemetry.rafStd = Math.sqrt(sq/intervals.length);
    }}

    var payload = {{
      challenge_id: CHALLENGE_ID,
      nonces: solvedNonces,
      solve_time_ms: solveElapsed,
      telemetry: telemetry,
    }};

    fetch(BEACON_URL, {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify(payload),
      credentials: "same-origin",
    }})
    .then(function(resp) {{
      if (resp.ok) return resp.json();
      throw new Error("Verification failed");
    }})
    .then(function(data) {{
      if (data.verified) {{
        setTimeout(function() {{ window.location.href = REDIRECT; }}, 300);
      }} else {{
        status.textContent = "Verification failed. Retrying...";
        status.className = "status error";
        setTimeout(function() {{ window.location.reload(); }}, 2000);
      }}
    }})
    .catch(function(err) {{
      status.textContent = "Error: " + err.message;
      status.className = "status error";
      setTimeout(function() {{ window.location.reload(); }}, 3000);
    }});
  }}

  // ── Timeout fallback ──
  setTimeout(function() {{
    if (!solved) {{
      worker.terminate();
      status.textContent = "Challenge timeout. Reloading...";
      status.className = "status error";
      setTimeout(function() {{ window.location.reload(); }}, 2000);
    }}
  }}, 30000);

}})();
</script>
</body>
</html>"""


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
        self.secret = self.config["HMAC_SECRET"].encode()
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
        """Render the full HTML for the biometric captcha page."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Security Verification</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh; background: #f8f9fa; color: #333;
  }}
  .card {{
    background: white; border-radius: 12px; padding: 36px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.10); text-align: center;
    max-width: 500px; width: 94%;
  }}
  h2 {{ font-size: 18px; font-weight: 600; margin-bottom: 8px; }}
  p.desc {{ font-size: 14px; color: #6c757d; line-height: 1.5; margin-bottom: 16px; }}
  canvas {{
    border: 2px solid #dee2e6; border-radius: 8px;
    cursor: crosshair; display: block; margin: 0 auto;
    touch-action: none;
  }}
  .btn {{
    margin-top: 16px; padding: 10px 32px; border: none; border-radius: 6px;
    background: #495057; color: white; font-size: 14px; font-weight: 500;
    cursor: pointer; transition: background 0.2s;
  }}
  .btn:hover {{ background: #343a40; }}
  .btn:disabled {{ background: #adb5bd; cursor: not-allowed; }}
  .status {{ margin-top: 10px; font-size: 12px; color: #adb5bd; min-height: 18px; }}
  .error {{ color: #dc3545; }}
  .success {{ color: #28a745; }}
</style>
</head>
<body>
<div class="card">
  <h2>Trace the path below</h2>
  <p class="desc">Use your mouse or finger to trace along the highlighted curve.</p>
  <canvas id="captchaCanvas" width="400" height="300"></canvas>
  <button class="btn" id="submitBtn" disabled>Submit</button>
  <div class="status" id="status"></div>
</div>

<script>
(function() {{
  "use strict";

  var CAPTCHA_ID = "{captcha_id}";
  var POINTS = {points_json};
  var BEACON_URL = "/_bot_captcha";

  var canvas = document.getElementById("captchaCanvas");
  var ctx = canvas.getContext("2d");
  var submitBtn = document.getElementById("submitBtn");
  var statusEl = document.getElementById("status");
  var traceData = [];
  var isTracing = false;
  var hasStarted = false;

  // ── Draw the Bezier curve ──
  function drawCurve() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Background
    ctx.fillStyle = "#f8f9fa";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Animated dashed guide
    ctx.beginPath();
    ctx.moveTo(POINTS[0].x, POINTS[0].y);
    ctx.bezierCurveTo(
      POINTS[1].x, POINTS[1].y,
      POINTS[2].x, POINTS[2].y,
      POINTS[3].x, POINTS[3].y
    );
    ctx.strokeStyle = "#adb5bd";
    ctx.lineWidth = 3;
    ctx.setLineDash([8, 6]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Solid thicker guide
    ctx.beginPath();
    ctx.moveTo(POINTS[0].x, POINTS[0].y);
    ctx.bezierCurveTo(
      POINTS[1].x, POINTS[1].y,
      POINTS[2].x, POINTS[2].y,
      POINTS[3].x, POINTS[3].y
    );
    ctx.strokeStyle = "rgba(73, 80, 87, 0.25)";
    ctx.lineWidth = 24;
    ctx.lineCap = "round";
    ctx.stroke();

    // Start/end markers
    ctx.beginPath();
    ctx.arc(POINTS[0].x, POINTS[0].y, 8, 0, Math.PI*2);
    ctx.fillStyle = "#28a745";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(POINTS[3].x, POINTS[3].y, 8, 0, Math.PI*2);
    ctx.fillStyle = "#dc3545";
    ctx.fill();
  }}

  // ── Draw user trace ──
  function drawTrace() {{
    if (traceData.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(traceData[0].x, traceData[0].y);
    for (var i = 1; i < traceData.length; i++) {{
      ctx.lineTo(traceData[i].x, traceData[i].y);
    }}
    ctx.strokeStyle = "#495057";
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.stroke();
  }}

  function getPos(e) {{
    var rect = canvas.getBoundingClientRect();
    var clientX, clientY, pressure;
    if (e.touches && e.touches.length > 0) {{
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
      pressure = e.touches[0].force || 0;
    }} else {{
      clientX = e.clientX;
      clientY = e.clientY;
      pressure = e.pressure || 0;
    }}
    return {{
      x: clientX - rect.left,
      y: clientY - rect.top,
      t: Date.now(),
      pressure: pressure,
    }};
  }}

  function onStart(e) {{
    e.preventDefault();
    isTracing = true;
    if (!hasStarted) {{
      hasStarted = true;
      traceData = [];
    }}
    var pos = getPos(e);
    traceData.push(pos);
  }}

  function onMove(e) {{
    e.preventDefault();
    if (!isTracing) return;
    var pos = getPos(e);
    traceData.push(pos);
    drawCurve();
    drawTrace();
  }}

  function onEnd(e) {{
    e.preventDefault();
    isTracing = false;
    if (traceData.length > 5) {{
      submitBtn.disabled = false;
    }}
  }}

  canvas.addEventListener("mousedown", onStart);
  canvas.addEventListener("mousemove", onMove);
  canvas.addEventListener("mouseup", onEnd);
  canvas.addEventListener("mouseleave", onEnd);
  canvas.addEventListener("touchstart", onStart, {{passive: false}});
  canvas.addEventListener("touchmove", onMove, {{passive: false}});
  canvas.addEventListener("touchend", onEnd, {{passive: false}});
  canvas.addEventListener("touchcancel", onEnd, {{passive: false}});

  submitBtn.addEventListener("click", function() {{
    submitBtn.disabled = true;
    statusEl.textContent = "Verifying...";

    fetch(BEACON_URL, {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify({{
        captcha_id: CAPTCHA_ID,
        trace_data: traceData,
      }}),
      credentials: "same-origin",
    }})
    .then(function(resp) {{
      if (resp.ok) return resp.json();
      throw new Error("Verification failed");
    }})
    .then(function(data) {{
      if (data.passed) {{
        statusEl.textContent = "Verified!";
        statusEl.className = "status success";
      }} else {{
        statusEl.textContent = "Please try again.";
        statusEl.className = "status error";
        traceData = [];
        hasStarted = false;
        drawCurve();
        setTimeout(function() {{
          submitBtn.disabled = false;
          statusEl.textContent = "";
          statusEl.className = "status";
        }}, 1500);
      }}
    }})
    .catch(function(err) {{
      statusEl.textContent = "Error: " + err.message;
      statusEl.className = "status error";
    }});
  }});

  drawCurve();
}})();
</script>
</body>
</html>"""


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
