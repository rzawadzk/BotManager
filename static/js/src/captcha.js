/*
 * Biometric-curve captcha — the user traces a Bezier path with mouse or
 * finger; server-side behavioural analysis decides pass/fail.
 *
 * Config is provided via a JSON script tag:
 *   <script id="captcha-config" type="application/json">
 *     {"captcha_id": "...", "points": [{x,y}, {x,y}, {x,y}, {x,y}]}
 *   </script>
 *
 * C3 #9: extracted from the _render_captcha_html f-string.
 */

(function () {
  "use strict";

  var cfgEl = document.getElementById("captcha-config");
  if (!cfgEl) {
    console.error("captcha-config script tag missing");
    return;
  }
  var cfg;
  try {
    cfg = JSON.parse(cfgEl.textContent || "{}");
  } catch (err) {
    console.error("captcha-config JSON parse failed", err);
    return;
  }
  var CAPTCHA_ID = cfg.captcha_id;
  var POINTS = cfg.points || [];
  var BEACON_URL = "/_bot_captcha";

  var canvas = document.getElementById("captchaCanvas");
  var ctx = canvas.getContext("2d");
  var submitBtn = document.getElementById("submitBtn");
  var statusEl = document.getElementById("status");
  var traceData = [];
  var isTracing = false;
  var hasStarted = false;

  // ── Draw the Bezier curve ──
  function drawCurve() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
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

    // Solid thicker guide (hit target)
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

    // Start / end markers
    ctx.beginPath();
    ctx.arc(POINTS[0].x, POINTS[0].y, 8, 0, Math.PI * 2);
    ctx.fillStyle = "#28a745";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(POINTS[3].x, POINTS[3].y, 8, 0, Math.PI * 2);
    ctx.fillStyle = "#dc3545";
    ctx.fill();
  }

  function drawTrace() {
    if (traceData.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(traceData[0].x, traceData[0].y);
    for (var i = 1; i < traceData.length; i++) {
      ctx.lineTo(traceData[i].x, traceData[i].y);
    }
    ctx.strokeStyle = "#495057";
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.stroke();
  }

  function getPos(e) {
    var rect = canvas.getBoundingClientRect();
    var clientX, clientY, pressure;
    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
      pressure = e.touches[0].force || 0;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
      pressure = e.pressure || 0;
    }
    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
      t: Date.now(),
      pressure: pressure,
    };
  }

  function onStart(e) {
    e.preventDefault();
    isTracing = true;
    if (!hasStarted) {
      hasStarted = true;
      traceData = [];
    }
    traceData.push(getPos(e));
  }

  function onMove(e) {
    e.preventDefault();
    if (!isTracing) return;
    traceData.push(getPos(e));
    drawCurve();
    drawTrace();
  }

  function onEnd(e) {
    e.preventDefault();
    isTracing = false;
    if (traceData.length > 5) {
      submitBtn.disabled = false;
    }
  }

  canvas.addEventListener("mousedown", onStart);
  canvas.addEventListener("mousemove", onMove);
  canvas.addEventListener("mouseup", onEnd);
  canvas.addEventListener("mouseleave", onEnd);
  canvas.addEventListener("touchstart", onStart, { passive: false });
  canvas.addEventListener("touchmove", onMove, { passive: false });
  canvas.addEventListener("touchend", onEnd, { passive: false });
  canvas.addEventListener("touchcancel", onEnd, { passive: false });

  submitBtn.addEventListener("click", function () {
    submitBtn.disabled = true;
    statusEl.textContent = "Verifying...";

    fetch(BEACON_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        captcha_id: CAPTCHA_ID,
        trace_data: traceData,
      }),
      credentials: "same-origin",
    })
    .then(function (resp) {
      if (resp.ok) return resp.json();
      throw new Error("Verification failed");
    })
    .then(function (data) {
      if (data.passed) {
        statusEl.textContent = "Verified!";
        statusEl.className = "status success";
      } else {
        statusEl.textContent = "Please try again.";
        statusEl.className = "status error";
        traceData = [];
        hasStarted = false;
        drawCurve();
        setTimeout(function () {
          submitBtn.disabled = false;
          statusEl.textContent = "";
          statusEl.className = "status";
        }, 1500);
      }
    })
    .catch(function (err) {
      statusEl.textContent = "Error: " + err.message;
      statusEl.className = "status error";
    });
  });

  drawCurve();
})();
