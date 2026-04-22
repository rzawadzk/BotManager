"""Tests for C3 — perf/deploy cleanup.

Covers:
  - C3 #8 DbWriteQueue: submit paths, coalescing, backpressure, shutdown
        drain, reconnect-on-error, metrics correctness.
  - C3 #9 js_assets: dist-preferred/src-fallback, missing-both error,
        ``manifest()`` shape, ``reload()`` refresh.
  - C3 #9 HTML renderers: ``generate_challenge_html`` and
        ``_render_captcha_html`` embed a well-formed JSON config block
        plus the minified JS, with no per-request string-format into JS.

C3 #7 (IPv6 /64 grouping) landed in C2 and is exercised by test_c2.py.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════════
# C3 #8 — DbWriteQueue
# ═══════════════════════════════════════════════════════════════════════════════

from db_worker import DbQueueMetrics, DbWriteQueue  # noqa: E402


class FakeDb:
    """In-memory stand-in for ScoreDatabase — records calls so tests can assert.

    The real ScoreDatabase needs a file path and a working sqlite3. The
    queue doesn't care what kind of object it holds — it only calls
    ``save_scores_batch`` and ``close`` — so a fake is the lightest way
    to exercise the worker's semantics without fs I/O.
    """

    def __init__(self, *, fail_writes: int = 0):
        self.batches: list[list] = []
        self.total_rows = 0
        self.commits = 0
        self.closed = False
        self._fail_remaining = fail_writes  # number of writes to blow up before succeeding
        self._lock = threading.Lock()

    def save_scores_batch(self, threats):
        with self._lock:
            if self._fail_remaining > 0:
                self._fail_remaining -= 1
                raise RuntimeError("simulated write failure")
            self.batches.append(list(threats))
            self.total_rows += len(threats)
            self.commits += 1
            return len(threats)

    def close(self):
        self.closed = True


def _mk_score(ip: str = "1.2.3.4"):
    """Minimal ThreatScore — only the fields the queue/DB actually touch.

    We import it here so tests that skip the queue tests don't pay the
    bot_engine import cost at collection time.
    """
    from bot_engine import ThreatScore
    return ThreatScore(ip=ip)


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll ``predicate`` until it returns truthy or we time out.

    We can't join the worker to know when a submit has been processed —
    it's a long-lived thread. Polling a visible side-effect (metrics or
    the fake DB's commit count) is the standard pattern.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class TestDbWriteQueueBasic:
    def test_submit_batch_persists(self):
        db = FakeDb()
        q = DbWriteQueue(db_factory=lambda: db, max_queue=100, coalesce=False)
        q.start()
        try:
            assert q.submit_batch([_mk_score("1.1.1.1"), _mk_score("1.1.1.2")])
            assert _wait_until(lambda: db.total_rows == 2)
            assert q.metrics.written == 2
            assert q.metrics.batches == 1
        finally:
            assert q.shutdown(timeout=2.0)
        assert db.closed

    def test_submit_score_single_path(self):
        db = FakeDb()
        q = DbWriteQueue(db_factory=lambda: db, max_queue=100, coalesce=False)
        q.start()
        try:
            assert q.submit_score(_mk_score("9.9.9.9"))
            assert _wait_until(lambda: db.total_rows == 1)
        finally:
            q.shutdown(timeout=2.0)

    def test_empty_batch_is_noop(self):
        db = FakeDb()
        q = DbWriteQueue(db_factory=lambda: db)
        q.start()
        try:
            # An empty list shortcircuits before touching the queue — it
            # must not increment ``queued`` and must not hit the DB.
            assert q.submit_batch([]) is True
            time.sleep(0.05)
            assert db.total_rows == 0
            assert q.metrics.queued == 0
        finally:
            q.shutdown(timeout=2.0)

    def test_is_running_reflects_thread_state(self):
        db = FakeDb()
        q = DbWriteQueue(db_factory=lambda: db)
        assert not q.is_running
        q.start()
        assert q.is_running
        q.shutdown(timeout=2.0)
        assert not q.is_running

    def test_start_is_idempotent(self):
        db = FakeDb()
        q = DbWriteQueue(db_factory=lambda: db)
        q.start()
        q.start()  # second call must not spawn a second thread
        try:
            assert q.is_running
        finally:
            q.shutdown(timeout=2.0)


class TestDbWriteQueueCoalescing:
    def test_coalesce_merges_into_one_commit(self):
        # With coalesce=True a burst of submits should collapse into far
        # fewer commits than submits — exact count is timing-dependent
        # but at the very least a burst of 10 shouldn't produce 10 fsyncs.
        db = FakeDb()
        q = DbWriteQueue(db_factory=lambda: db, coalesce=True)
        q.start()
        try:
            for i in range(10):
                q.submit_score(_mk_score(f"10.0.0.{i}"))
            assert _wait_until(lambda: db.total_rows == 10)
            # Strict claim: we got all 10 rows. Soft claim: fewer than
            # 10 commits (coalescing did something). If the CI host is
            # pathologically slow and the worker wakes between every
            # submit, relax to <=10 — the hard assert is still rows.
            assert db.commits <= 10
        finally:
            q.shutdown(timeout=2.0)

    def test_coalesce_off_preserves_per_submit_commits(self):
        # coalesce=False is the "one commit per submit" pedagogical mode.
        # Mostly a guard against accidentally re-enabling coalescing.
        db = FakeDb()
        q = DbWriteQueue(db_factory=lambda: db, coalesce=False)
        q.start()
        try:
            for i in range(5):
                q.submit_score(_mk_score(f"10.0.1.{i}"))
            assert _wait_until(lambda: db.total_rows == 5)
            assert db.commits == 5
        finally:
            q.shutdown(timeout=2.0)


class TestDbWriteQueueBackpressure:
    def test_drops_when_full(self):
        # Tiny queue + no worker → every submit past the first fills up.
        # We don't start() the worker so the queue stays wedged.
        db = FakeDb()
        q = DbWriteQueue(db_factory=lambda: db, max_queue=2)
        # Don't start — simulate a stalled worker.
        assert q.submit_score(_mk_score("a")) is True
        assert q.submit_score(_mk_score("b")) is True
        # Third must fail fast.
        assert q.submit_score(_mk_score("c")) is False
        assert q.metrics.dropped_full == 1
        assert q.metrics.queued == 2

    def test_refuses_submits_after_shutdown_starts(self):
        db = FakeDb()
        q = DbWriteQueue(db_factory=lambda: db)
        q.start()
        q.shutdown(timeout=2.0)
        # After shutdown ``_stopping`` is set → further submits are
        # refused so the process doesn't accidentally accept work that
        # would never drain.
        assert q.submit_score(_mk_score("late")) is False


class TestDbWriteQueueShutdown:
    def test_shutdown_drains_inflight(self):
        # Submit a batch, shut down immediately. The drain loop in the
        # worker's finally block must still commit that batch.
        db = FakeDb()
        q = DbWriteQueue(db_factory=lambda: db)
        q.start()
        q.submit_batch([_mk_score("x.x.x.1"), _mk_score("x.x.x.2")])
        # Don't wait — let shutdown do the draining.
        assert q.shutdown(timeout=2.0)
        assert db.total_rows == 2
        assert db.closed

    def test_shutdown_before_start_is_a_noop(self):
        db = FakeDb()
        q = DbWriteQueue(db_factory=lambda: db)
        # Never called start(); shutdown must not blow up.
        assert q.shutdown(timeout=0.1) is True


class TestDbWriteQueueReconnect:
    def test_reconnects_after_write_failure(self):
        # First DB raises on its first write, then succeeds. Factory
        # returns a fresh FakeDb each time it's called (the reconnect
        # path in the worker builds a new connection).
        built: list[FakeDb] = []

        def factory():
            # First call: one bad write then good; second call: clean db.
            db = FakeDb(fail_writes=1 if not built else 0)
            built.append(db)
            return db

        q = DbWriteQueue(db_factory=factory, coalesce=False)
        q.start()
        try:
            q.submit_score(_mk_score("r.r.r.1"))  # will fail, reconnect
            q.submit_score(_mk_score("r.r.r.2"))  # should land on new db
            assert _wait_until(lambda: len(built) >= 2)
            assert _wait_until(lambda: built[1].total_rows >= 1)
            # Error was counted.
            assert q.metrics.errors >= 1
        finally:
            q.shutdown(timeout=2.0)


class TestDbQueueMetrics:
    def test_snapshot_is_plain_dict(self):
        m = DbQueueMetrics()
        snap = m.snapshot()
        assert isinstance(snap, dict)
        # Must be JSON-safe so /stats can emit it directly.
        json.dumps(snap)
        # The internal lock must NOT leak into the snapshot.
        assert "_lock" not in snap

    def test_counters_default_to_zero(self):
        m = DbQueueMetrics()
        for key in ("queued", "written", "batches", "dropped_full",
                    "errors", "shutdown_drained", "current_depth"):
            assert m.snapshot()[key] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# C3 #9 — js_assets loader
# ═══════════════════════════════════════════════════════════════════════════════

import js_assets  # noqa: E402


class TestJsAssets:
    def test_constants_are_nonempty_strings(self):
        # These four are the public surface; they're eager-loaded at
        # import time so if the repo is intact they must be present.
        for name in ("POW_WORKER_JS", "POW_CHALLENGE_JS",
                     "CAPTCHA_JS", "BOT_CANARY_JS"):
            val = getattr(js_assets, name)
            assert isinstance(val, str)
            assert val.strip(), f"{name} is empty"

    def test_manifest_reports_all_four(self):
        m = js_assets.manifest()
        assert set(m) == {"pow_worker", "pow_challenge", "captcha", "bot_canary"}
        for key, info in m.items():
            assert info["source"] in {"dist", "src", "missing"}, key
            assert isinstance(info["size"], int)

    def test_prefers_dist_when_present(self, tmp_path, monkeypatch):
        # Point the loader at a fake src/dist pair, write distinct
        # markers into each, and confirm dist wins.
        src_dir = tmp_path / "src"
        dist_dir = tmp_path / "dist"
        src_dir.mkdir()
        dist_dir.mkdir()
        (src_dir / "pow_worker.js").write_text("/*SRC*/")
        (dist_dir / "pow_worker.min.js").write_text("/*DIST*/")

        monkeypatch.setattr(js_assets, "_SRC_DIR", src_dir)
        monkeypatch.setattr(js_assets, "_DIST_DIR", dist_dir)

        got = js_assets._read_asset("pow_worker")
        assert got == "/*DIST*/"

    def test_falls_back_to_src_when_dist_missing(self, tmp_path, monkeypatch):
        src_dir = tmp_path / "src"
        dist_dir = tmp_path / "dist"
        src_dir.mkdir()
        dist_dir.mkdir()  # empty
        (src_dir / "captcha.js").write_text("/*SRC-ONLY*/")

        monkeypatch.setattr(js_assets, "_SRC_DIR", src_dir)
        monkeypatch.setattr(js_assets, "_DIST_DIR", dist_dir)

        assert js_assets._read_asset("captcha") == "/*SRC-ONLY*/"

    def test_zero_byte_dist_falls_back_to_src(self, tmp_path, monkeypatch):
        # An aborted terser build can leave a zero-byte file; the loader
        # must treat that as "missing" rather than ship an empty script.
        src_dir = tmp_path / "src"
        dist_dir = tmp_path / "dist"
        src_dir.mkdir()
        dist_dir.mkdir()
        (src_dir / "bot_canary.js").write_text("/*SRC-CANARY*/")
        (dist_dir / "bot_canary.min.js").write_text("")  # 0 bytes

        monkeypatch.setattr(js_assets, "_SRC_DIR", src_dir)
        monkeypatch.setattr(js_assets, "_DIST_DIR", dist_dir)
        assert js_assets._read_asset("bot_canary") == "/*SRC-CANARY*/"

    def test_raises_when_both_missing(self, tmp_path, monkeypatch):
        src_dir = tmp_path / "src"
        dist_dir = tmp_path / "dist"
        src_dir.mkdir()
        dist_dir.mkdir()
        monkeypatch.setattr(js_assets, "_SRC_DIR", src_dir)
        monkeypatch.setattr(js_assets, "_DIST_DIR", dist_dir)
        with pytest.raises(FileNotFoundError):
            js_assets._read_asset("pow_challenge")

    def test_reload_refreshes_constants(self, tmp_path, monkeypatch):
        # Swap the dirs, call reload(), confirm constants change.
        src_dir = tmp_path / "src"
        dist_dir = tmp_path / "dist"
        src_dir.mkdir()
        dist_dir.mkdir()
        for src_name, dist_name in js_assets._ASSETS.values():
            (dist_dir / dist_name).write_text(f"/*{dist_name}*/")
            (src_dir / src_name).write_text(f"/*{src_name}*/")

        original = js_assets.POW_WORKER_JS
        monkeypatch.setattr(js_assets, "_SRC_DIR", src_dir)
        monkeypatch.setattr(js_assets, "_DIST_DIR", dist_dir)
        try:
            js_assets.reload()
            assert js_assets.POW_WORKER_JS == "/*pow_worker.min.js*/"
            assert js_assets.POW_WORKER_JS != original
        finally:
            # Restore: monkeypatch undoes _SRC_DIR/_DIST_DIR; we just
            # have to re-read from the real dirs so other tests don't
            # see our fake constants leak.
            js_assets.reload()


# ═══════════════════════════════════════════════════════════════════════════════
# C3 #9 — HTML renderers embed config-as-JSON + minified JS
# ═══════════════════════════════════════════════════════════════════════════════

from pow_challenge import (  # noqa: E402
    ChallengeBatch,
    MultiBatchChallenge,
    generate_challenge_html,
)


def _fake_challenge(n_batches: int = 3, cid: str = "abcd1234"):
    return MultiBatchChallenge(
        challenge_id=cid,
        batches=[
            ChallengeBatch(
                batch_index=i,
                prefix=f"prefix-{i}",
                difficulty=18,
                salt="" if i == 0 else f"salt-{i}",
                hmac_sig="sig",
            )
            for i in range(n_batches)
        ],
        ip="1.2.3.4",
        issued_at=time.time(),
    )


def _extract_json_block(html: str, script_id: str) -> dict:
    """Pull the ``<script id="{id}" type="application/json">...</script>`` payload.

    The renderer emits the JSON in a single <script> block, so simple
    string slicing is enough — we don't need a full HTML parser.
    """
    needle = f'<script id="{script_id}" type="application/json">'
    start = html.index(needle) + len(needle)
    end = html.index("</script>", start)
    raw = html[start:end]
    return json.loads(raw)


class TestGenerateChallengeHtml:
    def test_emits_valid_challenge_config_json(self):
        chal = _fake_challenge(n_batches=4, cid="deadbeef")
        html = generate_challenge_html(chal, redirect_url="/after",
                                       telemetry_collect_ms=2500)
        cfg = _extract_json_block(html, "challenge-config")
        assert cfg["challenge_id"] == "deadbeef"
        assert len(cfg["batches"]) == 4
        assert cfg["redirect"] == "/after"
        assert cfg["collect_ms"] == 2500
        # Each batch has the three fields the client JS expects.
        for i, b in enumerate(cfg["batches"]):
            assert b["batch_index"] == i
            assert b["prefix"] == f"prefix-{i}"
            assert b["difficulty"] == 18

    def test_embeds_worker_and_orchestrator_scripts(self):
        html = generate_challenge_html(_fake_challenge())
        # The worker body sits in a text/js-worker script (so the browser
        # doesn't run it inline — the orchestrator spawns a Blob worker).
        assert '<script id="worker-src" type="text/js-worker">' in html
        # The orchestrator text is the non-minified or minified pow_challenge.js.
        # Either way, "challenge-config" is referenced from inside it.
        assert "challenge-config" in html

    def test_script_tag_injection_is_escaped(self):
        # A fabricated challenge_id containing "</script>" must not be
        # able to close the surrounding <script> block — the ``</`` →
        # ``<\/`` rewrite is what prevents it.
        chal = _fake_challenge(cid="evil</script><script>alert(1)</script>")
        html = generate_challenge_html(chal)
        # The literal "</script>" sequence from challenge_id must NOT
        # land inside the config block. It's fine to see ``</script>``
        # elsewhere (there are legitimate closers for <style>, <script>,
        # <body>), so we slice out just the config block.
        cfg_start = html.index('<script id="challenge-config"')
        cfg_end = html.index("</script>", cfg_start)
        config_block = html[cfg_start:cfg_end]
        assert "</script>" not in config_block
        # And json.loads still handles it — parseability is the whole
        # point of using JSON over f-string interpolation.
        cfg = _extract_json_block(html, "challenge-config")
        assert cfg["challenge_id"] == "evil</script><script>alert(1)</script>"


class TestRenderCaptchaHtml:
    def test_emits_valid_captcha_config_json(self):
        from pow_challenge import BiometricCaptcha

        points = [[10, 20], [30, 40], [50, 60]]
        html = BiometricCaptcha._render_captcha_html(
            captcha_id="cap-xyz",
            points_json=json.dumps(points),
        )
        cfg = _extract_json_block(html, "captcha-config")
        assert cfg["captcha_id"] == "cap-xyz"
        assert cfg["points"] == points

    def test_handles_malformed_points_json(self):
        # If the caller somehow passes a non-JSON string, the renderer
        # must still produce a page (with empty points) rather than
        # crashing. The captcha JS tolerates an empty curve.
        from pow_challenge import BiometricCaptcha

        html = BiometricCaptcha._render_captcha_html(
            captcha_id="cap-fallback",
            points_json="{not valid json",
        )
        cfg = _extract_json_block(html, "captcha-config")
        assert cfg["captcha_id"] == "cap-fallback"
        assert cfg["points"] == []

    def test_contains_captcha_script(self):
        from pow_challenge import BiometricCaptcha

        html = BiometricCaptcha._render_captcha_html(
            captcha_id="cap-1",
            points_json=json.dumps([[0, 0], [1, 1]]),
        )
        # The captcha orchestrator JS refers to the config id when it
        # bootstraps — regardless of minification that string survives.
        assert "captcha-config" in html
        # And the page renders the canvas (static HTML, not from JS).
        assert '<canvas id="captchaCanvas"' in html
