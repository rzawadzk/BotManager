/*
 * PoW challenge orchestrator — reads the per-challenge config from a
 * <script id="challenge-config" type="application/json"> block, spawns
 * the worker, collects telemetry, and POSTs results to /_bot_challenge.
 *
 * Before C3 this code lived as an f-string inside pow_challenge.py with
 * {CHALLENGE_ID} / {BATCHES} / {REDIRECT} / {COLLECT_MS} placeholders
 * interpolated at render time. Now the dynamic values travel as JSON in
 * a single data element and the JS is 100% static → one file for terser
 * to minify, and no template-quoting surprises.
 *
 * Template: the embedding page must include
 *   <script id="challenge-config" type="application/json">
 *     {"challenge_id": "...", "batches": [...], "redirect": "...", "collect_ms": 3000}
 *   </script>
 *   <script id="worker-src" type="text/js-worker">...</script>
 * plus the DOM hooks (#progress, #status, #batchLabel, #spinner).
 */

(function () {
  "use strict";

  // ── Load config from the JSON script tag ──
  var cfgEl = document.getElementById("challenge-config");
  if (!cfgEl) {
    console.error("challenge-config script tag missing");
    return;
  }
  var cfg;
  try {
    cfg = JSON.parse(cfgEl.textContent || "{}");
  } catch (err) {
    console.error("challenge-config JSON parse failed", err);
    return;
  }
  var CHALLENGE_ID = cfg.challenge_id;
  var BATCHES = cfg.batches || [];
  var REDIRECT = cfg.redirect || "/";
  var COLLECT_MS = cfg.collect_ms || 3000;
  var BEACON_URL = "/_bot_challenge";

  var progress = document.getElementById("progress");
  var status = document.getElementById("status");
  var batchLabel = document.getElementById("batchLabel");
  var spinner = document.getElementById("spinner");

  // ── Telemetry collection (runs in parallel with PoW) ──
  var telemetry = {
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
  };

  var t1 = performance.now(), t2 = performance.now();
  telemetry.perfRes = t2 - t1;

  var rafTs = []; var rafN = 0;
  function mRAF(ts) { rafTs.push(ts); if (++rafN < 30) requestAnimationFrame(mRAF); }
  requestAnimationFrame(mRAF);

  try {
    var c = document.createElement("canvas"); c.width = 256; c.height = 64;
    var x = c.getContext("2d");
    x.textBaseline = "top"; x.font = "14px Arial";
    x.fillStyle = "#f60"; x.fillRect(125, 1, 62, 20);
    x.fillStyle = "#069"; x.fillText("BotCk,.+@#$", 2, 15);
    x.fillStyle = "rgba(102,204,0,0.7)"; x.fillText("BotCk,.+@#$", 4, 17);
    x.globalCompositeOperation = "multiply";
    x.fillStyle = "rgb(255,0,255)"; x.beginPath(); x.arc(50, 50, 50, 0, Math.PI * 2, true); x.fill();
    var hv = 0; var d = c.toDataURL();
    for (var i = 0; i < d.length; i++) { hv = ((hv << 5) - hv) + d.charCodeAt(i); hv |= 0; }
    telemetry.canvasHash = hv.toString(16);
  } catch (e) { /* canvas unavailable — drop silently */ }

  try {
    var c2 = document.createElement("canvas");
    var gl = c2.getContext("webgl") || c2.getContext("experimental-webgl");
    if (gl) {
      var dbg = gl.getExtension("WEBGL_debug_renderer_info");
      if (dbg) {
        telemetry.webglRenderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || "";
        telemetry.webglVendor = gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) || "";
      }
    }
  } catch (e) { /* webgl unavailable */ }

  document.addEventListener("mousemove", function (e) {
    if (telemetry.mouseMoves.length < 200)
      telemetry.mouseMoves.push([e.clientX, e.clientY, e.timeStamp | 0]);
  });
  document.addEventListener("click", function (e) {
    telemetry.mouseClicks.push([e.clientX, e.clientY, e.timeStamp | 0]);
  });
  document.addEventListener("scroll", function () {
    telemetry.scrollEvents.push([window.scrollY, Date.now()]);
  });
  document.addEventListener("keydown", function () { telemetry.keyCount++; });

  if (navigator.permissions) {
    navigator.permissions.query({ name: "notifications" }).then(function (r) {
      telemetry.notifPerm = r.state;
    }).catch(function () {});
  }

  ["Bluetooth", "BatteryManager", "Gamepad", "MediaDevices", "Credential",
   "PaymentRequest", "Presentation", "WakeLock", "USB", "Serial", "HID", "XRSystem"
  ].forEach(function (a) { if (a in window || a in navigator) telemetry.apis.push(a); });

  // ── Start Web Worker ──
  status.textContent = "Solving puzzles...";
  batchLabel.textContent = "Solving puzzle 1 of " + BATCHES.length + "...";

  var workerSrc = document.getElementById("worker-src").textContent;
  var blob = new Blob([workerSrc], { type: "application/javascript" });
  var worker = new Worker(URL.createObjectURL(blob));

  var solved = false;
  var solvedNonces = [];
  var solveElapsed = 0;

  worker.onmessage = function (e) {
    var msg = e.data;

    if (msg.done) {
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
    } else if (msg.batchDone) {
      var pct = ((msg.batchIndex + 1) / msg.totalBatches) * 100;
      progress.style.width = pct + "%";
      batchLabel.textContent = "Solved puzzle " + (msg.batchIndex + 1) + " of " + msg.totalBatches;
      if (msg.batchIndex + 1 < msg.totalBatches) {
        status.textContent = "Starting puzzle " + (msg.batchIndex + 2) + "...";
      }
    } else {
      var basePct = (msg.batchIndex / msg.totalBatches) * 100;
      var expectedHashes = Math.pow(2, BATCHES[msg.batchIndex].difficulty);
      var inBatchPct = Math.min(0.95, msg.hashes / expectedHashes);
      var totalPct = basePct + inBatchPct * (100 / msg.totalBatches);
      progress.style.width = Math.min(95, totalPct) + "%";
      batchLabel.textContent = "Solving puzzle " + (msg.batchIndex + 1) + " of " + msg.totalBatches + "...";
      var rate = msg.hashes / (msg.elapsed / 1000);
      status.textContent = "Working... " + (rate / 1000 | 0) + "k hashes/s";
    }
  };

  worker.postMessage({ batches: BATCHES });

  // ── Submit all nonces + telemetry ──
  function submitResult() {
    if (rafTs.length > 2) {
      var intervals = [];
      for (var i = 1; i < rafTs.length; i++) intervals.push(rafTs[i] - rafTs[i-1]);
      var sum = intervals.reduce(function (a, b) { return a + b; }, 0);
      telemetry.rafAvg = sum / intervals.length;
      var sq = intervals.reduce(function (a, b) { return a + Math.pow(b - telemetry.rafAvg, 2); }, 0);
      telemetry.rafStd = Math.sqrt(sq / intervals.length);
    }

    var payload = {
      challenge_id: CHALLENGE_ID,
      nonces: solvedNonces,
      solve_time_ms: solveElapsed,
      telemetry: telemetry,
    };

    fetch(BEACON_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      credentials: "same-origin",
    })
    .then(function (resp) {
      if (resp.ok) return resp.json();
      throw new Error("Verification failed");
    })
    .then(function (data) {
      if (data.verified) {
        setTimeout(function () { window.location.href = REDIRECT; }, 300);
      } else {
        status.textContent = "Verification failed. Retrying...";
        status.className = "status error";
        setTimeout(function () { window.location.reload(); }, 2000);
      }
    })
    .catch(function (err) {
      status.textContent = "Error: " + err.message;
      status.className = "status error";
      setTimeout(function () { window.location.reload(); }, 3000);
    });
  }

  // ── Timeout fallback ──
  setTimeout(function () {
    if (!solved) {
      worker.terminate();
      status.textContent = "Challenge timeout. Reloading...";
      status.className = "status error";
      setTimeout(function () { window.location.reload(); }, 2000);
    }
  }, 30000);
})();
