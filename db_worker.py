"""Async SQLite write queue (C3 #8).

Design
------
The scoring hot path is 100% in-memory — no DB I/O blocks a request. Every
five minutes (see ``_snapshot_saver`` in realtime_server.py) we persist the
aggregated in-memory ThreatScore dict to SQLite. Before C3 that persistence
went through ``asyncio.to_thread(db.save_scores_batch, ...)`` which uses the
default thread pool; fine for the common case but it has three soft spots:

  1. **Thread-pool coupling.** The scoring server shares its default
     executor with every ``asyncio.to_thread`` call in the process (blocklist
     writes, model loads, file I/O, …). A slow fsync can starve everything.

  2. **Per-snapshot connection churn.** The task re-opens the sqlite3
     connection after any exception — WAL checkpoint state is lost each
     time, and so is any prepared-statement cache we might want later.

  3. **No graceful shutdown.** On SIGTERM the snapshot task just stops;
     anything in flight may or may not have committed.

``DbWriteQueue`` fixes all three by running a **single dedicated thread**
that owns one long-lived sqlite3 connection and pulls work items off a
``queue.Queue``. It exposes:

  * ``submit_batch(scores)`` — non-blocking; drops on a full queue
  * ``submit_score(score)`` — same, single-item path
  * ``shutdown(timeout)`` — sentinel + join; guarantees drain on clean exit

Backpressure is intentional: if the queue fills (e.g. the disk has stalled
and consumers can't keep up), we'd rather drop the oldest aggregate snapshot
than let the queue grow without bound and OOM the process. Drops are
counted in ``metrics`` and logged at WARNING so operators can see it.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover — type hints only
    from bot_engine import ScoreDatabase, ThreatScore


logger = logging.getLogger("bot-engine.db-worker")


# ── Work item protocol ──
# Items on the queue are small tuples:
#   ("batch", [ThreatScore, ...])
#   ("single", ThreatScore)
#   ("shutdown", None)                    ← sentinel; worker exits after draining
# We use tuples (not dataclasses) to keep the object count low — the queue
# can see millions of submits over a long-running process.


@dataclass
class DbQueueMetrics:
    """Lightweight counters for observability.

    These are reported in the scoring server's stats snapshot so operators
    can see queue health alongside scoring metrics.
    """

    queued: int = 0          # total items accepted into the queue
    written: int = 0          # total ThreatScores persisted to SQLite
    batches: int = 0          # total batches committed
    dropped_full: int = 0     # items refused because the queue was full
    errors: int = 0           # exceptions raised by the writer (reconnects trigger)
    shutdown_drained: int = 0 # items written during the shutdown drain
    last_write_ts: float = 0.0
    current_depth: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict:
        """Return a JSON-safe dict of counters."""
        with self._lock:
            return {
                "queued": self.queued,
                "written": self.written,
                "batches": self.batches,
                "dropped_full": self.dropped_full,
                "errors": self.errors,
                "shutdown_drained": self.shutdown_drained,
                "last_write_ts": self.last_write_ts,
                "current_depth": self.current_depth,
            }


class DbWriteQueue:
    """Single-writer SQLite persistence queue.

    The worker thread owns the ``ScoreDatabase`` instance end-to-end: no
    other thread touches it. This sidesteps sqlite3's thread-affinity
    rule (``check_same_thread=True`` by default) without us having to pass
    ``check_same_thread=False`` and deal with manual locking around the
    connection.

    The queue is typed loosely — it holds opaque (kind, payload) tuples
    so we can extend it with new operation kinds later without widening
    a dataclass hierarchy. See the docstring at the top of this module.
    """

    # Poison pill — a private sentinel object so callers can't accidentally
    # forge one by submitting a string. Identity comparison in the worker
    # loop is what closes it down.
    _SHUTDOWN = object()

    def __init__(
        self,
        db_factory,
        *,
        max_queue: int = 10_000,
        name: str = "db-write-queue",
        coalesce: bool = True,
    ):
        """
        Args:
          db_factory: zero-arg callable returning a ``ScoreDatabase``. We
                      defer construction to the worker thread so the
                      connection is owned by that thread — required by
                      sqlite3's default ``check_same_thread=True`` check.
          max_queue:  hard cap on pending items. Submits past this fail
                      fast (``submit_*`` returns False) rather than block
                      or grow without bound.
          coalesce:   if True, when the worker pulls a batch off the
                      queue it greedily pulls any additional batches
                      already waiting and folds them into one
                      transaction. This amortises fsync cost when a
                      caller bursts many small batches.
        """
        self._db_factory = db_factory
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._max_queue = max_queue
        self._name = name
        self._coalesce = coalesce
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._stopping = threading.Event()
        self.metrics = DbQueueMetrics()

    # ── Lifecycle ──

    def start(self) -> None:
        """Start the background writer thread. Idempotent."""
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._run, name=self._name, daemon=True
        )
        self._thread.start()
        logger.info("db-write-queue started (max_queue=%d)", self._max_queue)

    def shutdown(self, timeout: float = 5.0) -> bool:
        """Signal the worker to drain and exit.

        Blocks up to ``timeout`` seconds waiting for the worker thread to
        finish. Returns True if the thread exited cleanly, False if it
        timed out (in which case we leak it — the process is presumably
        dying anyway).
        """
        if not self._started or self._thread is None:
            return True
        self._stopping.set()
        # Use put_nowait; if the queue is full we still want shutdown to
        # proceed — the ``_stopping`` flag will short-circuit the drain
        # loop once the worker clears space.
        try:
            self._q.put_nowait((self._SHUTDOWN, None))
        except queue.Full:
            logger.warning(
                "db-write-queue: queue full during shutdown; relying on flag"
            )
        self._thread.join(timeout=timeout)
        alive = self._thread.is_alive()
        if alive:
            logger.error(
                "db-write-queue: worker did not exit within %.1fs", timeout
            )
        return not alive

    # ── Submission (hot-path, non-blocking) ──

    def submit_batch(self, scores: list) -> bool:
        """Enqueue a batch of ThreatScore for persistence.

        Returns True if accepted, False if the queue was full. Never
        blocks. The caller is expected to treat False as "operator should
        probably investigate disk latency" rather than retrying in a
        tight loop.
        """
        if not scores:
            return True
        return self._put(("batch", list(scores)))

    def submit_score(self, score) -> bool:
        """Enqueue a single ThreatScore. Same semantics as submit_batch."""
        return self._put(("single", score))

    def _put(self, item) -> bool:
        if self._stopping.is_set():
            # Refuse new work once shutdown has started — guarantees a
            # finite drain window.
            return False
        try:
            self._q.put_nowait(item)
        except queue.Full:
            with self.metrics._lock:
                self.metrics.dropped_full += 1
            # Log at WARNING once per ~100 drops to avoid log spam if the
            # disk stays wedged for a while.
            drops = self.metrics.dropped_full
            if drops == 1 or drops % 100 == 0:
                logger.warning(
                    "db-write-queue: dropped %d items (queue at capacity)", drops
                )
            return False
        with self.metrics._lock:
            self.metrics.queued += 1
            self.metrics.current_depth = self._q.qsize()
        return True

    # ── Worker loop (runs in dedicated thread) ──

    def _run(self) -> None:
        """Main loop. Owns the sqlite3 connection for its entire lifetime."""
        db = None
        try:
            db = self._db_factory()
            logger.info("db-write-queue: connection established")
        except Exception:
            # If even the initial connect fails we can't do anything useful
            # — crash the thread. Callers' submits will pile up and drop;
            # the surrounding server should notice the dropped_full metric.
            logger.exception("db-write-queue: initial connect failed")
            return

        try:
            while True:
                kind, payload = self._q.get()
                if kind is self._SHUTDOWN:
                    break

                # Coalesce: opportunistically grab any additional pending
                # items so we commit them in one transaction. This matters
                # when a caller bursts many submit_score() calls — one
                # fsync beats N fsyncs on a spinning disk.
                scores: list = []
                if kind == "batch":
                    scores.extend(payload)
                elif kind == "single":
                    scores.append(payload)

                if self._coalesce:
                    pulled = 0
                    while pulled < 64:  # cap to avoid starving shutdown
                        try:
                            next_kind, next_payload = self._q.get_nowait()
                        except queue.Empty:
                            break
                        if next_kind is self._SHUTDOWN:
                            # Put the sentinel back so the outer loop sees
                            # it on the next iteration (after we commit).
                            self._q.put_nowait((self._SHUTDOWN, None))
                            break
                        if next_kind == "batch":
                            scores.extend(next_payload)
                        elif next_kind == "single":
                            scores.append(next_payload)
                        pulled += 1

                if scores:
                    try:
                        self._write(db, scores)
                    except Exception:
                        logger.exception(
                            "db-write-queue: write failed (%d scores dropped)",
                            len(scores),
                        )
                        with self.metrics._lock:
                            self.metrics.errors += 1
                        # Try to rebuild the connection; if that fails we
                        # continue the loop so the shutdown path still
                        # drains the queue.
                        try:
                            db.close()
                        except Exception:
                            pass
                        try:
                            db = self._db_factory()
                        except Exception:
                            logger.exception(
                                "db-write-queue: reconnect failed; "
                                "discarding subsequent writes"
                            )
                            db = None

                with self.metrics._lock:
                    self.metrics.current_depth = self._q.qsize()
        finally:
            # Final drain: if we're shutting down cleanly, write anything
            # that snuck in between the sentinel and us returning.
            drained_count = 0
            while db is not None:
                try:
                    kind, payload = self._q.get_nowait()
                except queue.Empty:
                    break
                if kind is self._SHUTDOWN:
                    continue
                scores = (
                    list(payload) if kind == "batch"
                    else [payload] if kind == "single"
                    else []
                )
                if not scores:
                    continue
                try:
                    self._write(db, scores)
                    drained_count += len(scores)
                except Exception:
                    logger.exception("db-write-queue: drain write failed")
                    break

            if drained_count:
                with self.metrics._lock:
                    self.metrics.shutdown_drained += drained_count
                logger.info(
                    "db-write-queue: drained %d scores on shutdown",
                    drained_count,
                )

            if db is not None:
                try:
                    db.close()
                except Exception:
                    logger.debug("db-write-queue: close raised", exc_info=True)
            logger.info("db-write-queue: worker exited")

    def _write(self, db, scores: list) -> None:
        """Persist ``scores`` via the owned connection. Worker-thread only."""
        if not scores:
            return
        written = db.save_scores_batch(scores)
        with self.metrics._lock:
            self.metrics.written += written
            self.metrics.batches += 1
            self.metrics.last_write_ts = time.time()

    # ── Introspection helpers (used by tests and /stats) ──

    @property
    def depth(self) -> int:
        """Current queue depth (approximate — qsize isn't exact)."""
        return self._q.qsize()

    @property
    def is_running(self) -> bool:
        return self._started and self._thread is not None and self._thread.is_alive()
