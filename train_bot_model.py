#!/usr/bin/env python3
"""ML training pipeline for the bot management scoring engine.

Generates synthetic training data, trains an XGBoost binary classifier to
distinguish malicious bot traffic from legitimate (human + good-bot) traffic,
and exports the model to ONNX format for sub-millisecond inference.

The exported ONNX model is consumed by bot_engine.py v2.1 MLScorer.

Usage examples
--------------
    # Train with defaults and export
    python train_bot_model.py

    # Custom sample count, output path, and validation
    python train_bot_model.py --samples 100000 --output ./bot_model.onnx --validate

    # Reproducible run with a specific seed
    python train_bot_model.py --seed 123 --validate
"""

import argparse
import datetime as _dt
import hashlib
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import onnxruntime as ort
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# ---------------------------------------------------------------------------
# Feature schema -- must match bot_engine.py v2.1 MLScorer
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "ja4_hash_int",
    "h2_fp_int",
    "h3_present",
    "ua_length",
    "ua_entropy",
    "header_count",
    "path_depth",
    "method_encoded",
    "has_accept_language",
    "has_accept",
    "has_cookie",
    "has_referer",
    "content_length",
    "hour_of_day",
    "temporal_jitter",
    "identity_drift",
    "request_rate",
    "is_tls",
]

NUM_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_samples: int = 50000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic labelled data for three traffic profiles.

    Returns (X, y) where X has shape (n_samples, 18) and y is 0 (legit) or
    1 (bad bot).
    """
    rng = np.random.default_rng(seed)

    n_human = int(n_samples * 0.50)
    n_good_bot = int(n_samples * 0.15)
    n_bad_bot = n_samples - n_human - n_good_bot

    # --- Human traffic (label 0) ----------------------------------------
    human = np.zeros((n_human, NUM_FEATURES))
    human[:, 0] = rng.integers(1_000_000, 2_000_000_000, size=n_human)       # ja4_hash_int
    human[:, 1] = rng.integers(1_000_000, 2_000_000_000, size=n_human)       # h2_fp_int
    human[:, 2] = rng.choice([0, 1], size=n_human, p=[0.6, 0.4])            # h3_present
    human[:, 3] = rng.normal(110, 20, size=n_human).clip(30, 250)            # ua_length
    human[:, 4] = rng.normal(4.2, 0.4, size=n_human).clip(2.5, 5.5)         # ua_entropy
    human[:, 5] = rng.integers(8, 16, size=n_human)                          # header_count
    human[:, 6] = rng.integers(1, 6, size=n_human)                           # path_depth
    human[:, 7] = rng.choice([0, 1], size=n_human, p=[0.7, 0.3])            # method_encoded
    human[:, 8] = rng.choice([0, 1], size=n_human, p=[0.05, 0.95])          # has_accept_language
    human[:, 9] = rng.choice([0, 1], size=n_human, p=[0.02, 0.98])          # has_accept
    human[:, 10] = rng.choice([0, 1], size=n_human, p=[0.15, 0.85])         # has_cookie
    human[:, 11] = rng.choice([0, 1], size=n_human, p=[0.20, 0.80])         # has_referer
    human[:, 12] = np.where(
        human[:, 7] == 1,
        rng.integers(50, 5000, size=n_human),
        0,
    )                                                                         # content_length
    human[:, 13] = rng.integers(0, 24, size=n_human)                         # hour_of_day
    human[:, 14] = rng.uniform(0.5, 5.0, size=n_human)                       # temporal_jitter
    human[:, 15] = rng.choice([1, 2], size=n_human, p=[0.7, 0.3])           # identity_drift
    human[:, 16] = rng.uniform(1.0, 5.0, size=n_human)                       # request_rate
    human[:, 17] = rng.choice([0, 1], size=n_human, p=[0.05, 0.95])         # is_tls
    human_labels = np.zeros(n_human, dtype=np.int32)

    # --- Good bot traffic (label 0) ------------------------------------
    good = np.zeros((n_good_bot, NUM_FEATURES))
    # Known bots use a small set of JA4 hashes
    known_ja4 = rng.integers(100, 500, size=20)
    good[:, 0] = rng.choice(known_ja4, size=n_good_bot)                      # ja4_hash_int
    good[:, 1] = rng.integers(1_000_000, 2_000_000_000, size=n_good_bot)     # h2_fp_int
    good[:, 2] = rng.choice([0, 1], size=n_good_bot, p=[0.8, 0.2])          # h3_present
    good[:, 3] = rng.normal(60, 10, size=n_good_bot).clip(20, 120)           # ua_length (structured)
    good[:, 4] = rng.normal(2.8, 0.3, size=n_good_bot).clip(1.5, 3.8)       # ua_entropy (low)
    good[:, 5] = rng.integers(4, 9, size=n_good_bot)                         # header_count
    good[:, 6] = rng.integers(1, 4, size=n_good_bot)                         # path_depth
    good[:, 7] = rng.choice([0, 4], size=n_good_bot, p=[0.8, 0.2])          # GET or HEAD only
    good[:, 8] = rng.choice([0, 1], size=n_good_bot, p=[0.7, 0.3])          # has_accept_language
    good[:, 9] = rng.choice([0, 1], size=n_good_bot, p=[0.3, 0.7])          # has_accept
    good[:, 10] = 0                                                           # no cookies
    good[:, 11] = rng.choice([0, 1], size=n_good_bot, p=[0.6, 0.4])         # has_referer
    good[:, 12] = 0                                                           # always GET/HEAD
    good[:, 13] = rng.integers(0, 24, size=n_good_bot)                       # hour_of_day
    good[:, 14] = rng.uniform(0.001, 0.1, size=n_good_bot)                   # very low jitter
    good[:, 15] = 1                                                           # consistent identity
    good[:, 16] = rng.uniform(10.0, 50.0, size=n_good_bot)                   # high rate
    good[:, 17] = rng.choice([0, 1], size=n_good_bot, p=[0.3, 0.7])         # is_tls
    good_labels = np.zeros(n_good_bot, dtype=np.int32)

    # --- Bad bot traffic (label 1) -------------------------------------
    bad = np.zeros((n_bad_bot, NUM_FEATURES))
    bad[:, 0] = rng.integers(0, 2_000_000_000, size=n_bad_bot)               # ja4_hash_int
    bad[:, 1] = rng.integers(0, 2_000_000_000, size=n_bad_bot)               # h2_fp_int
    bad[:, 2] = rng.choice([0, 1], size=n_bad_bot, p=[0.85, 0.15])          # h3_present
    # Empty or spoofed UAs: bimodal -- very short or suspiciously long
    ua_len_short = rng.integers(0, 15, size=n_bad_bot)
    ua_len_long = rng.integers(200, 350, size=n_bad_bot)
    short_mask = rng.random(size=n_bad_bot) < 0.6
    bad[:, 3] = np.where(short_mask, ua_len_short, ua_len_long)              # ua_length
    bad[:, 4] = rng.normal(1.8, 0.8, size=n_bad_bot).clip(0.0, 4.0)         # ua_entropy (low)
    bad[:, 5] = rng.integers(2, 7, size=n_bad_bot)                           # header_count (few)
    bad[:, 6] = rng.integers(3, 12, size=n_bad_bot)                          # path_depth (deep probing)
    bad[:, 7] = rng.choice([0, 1, 2, 3, 5], size=n_bad_bot,
                           p=[0.3, 0.25, 0.15, 0.15, 0.15])                 # varied methods
    bad[:, 8] = rng.choice([0, 1], size=n_bad_bot, p=[0.85, 0.15])          # missing accept-language
    bad[:, 9] = rng.choice([0, 1], size=n_bad_bot, p=[0.5, 0.5])            # has_accept
    bad[:, 10] = rng.choice([0, 1], size=n_bad_bot, p=[0.9, 0.1])           # mostly no cookie
    bad[:, 11] = rng.choice([0, 1], size=n_bad_bot, p=[0.8, 0.2])           # missing referer
    bad[:, 12] = np.where(
        bad[:, 7] >= 1,
        rng.integers(0, 10000, size=n_bad_bot),
        0,
    )                                                                         # content_length
    # Odd hours: weight toward late night / early morning
    bad[:, 13] = rng.choice(
        list(range(24)),
        size=n_bad_bot,
        p=[0.08, 0.08, 0.08, 0.08, 0.07, 0.06, 0.04, 0.03, 0.02, 0.02,
           0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04,
           0.05, 0.05, 0.06, 0.02],
    )                                                                         # hour_of_day
    bad[:, 14] = rng.uniform(0.01, 0.1, size=n_bad_bot)                      # very low jitter
    bad[:, 15] = rng.integers(3, 11, size=n_bad_bot)                          # high identity drift
    bad[:, 16] = rng.uniform(15.0, 100.0, size=n_bad_bot)                     # high rate
    bad[:, 17] = rng.choice([0, 1], size=n_bad_bot, p=[0.4, 0.6])           # is_tls
    bad_labels = np.ones(n_bad_bot, dtype=np.int32)

    # --- Combine and add global noise ----------------------------------
    X = np.vstack([human, good, bad]).astype(np.float32)
    y = np.concatenate([human_labels, good_labels, bad_labels])

    # Small Gaussian noise on continuous features to prevent overfitting
    continuous_cols = [0, 1, 3, 4, 12, 14, 15, 16]
    noise_scale = 0.02
    for col in continuous_cols:
        col_std = X[:, col].std()
        if col_std > 0:
            X[:, col] += rng.normal(0, col_std * noise_scale, size=len(X))

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Real-traffic ingestion (C1.3)
# ---------------------------------------------------------------------------
# Parse nginx bot_access.log lines of the form emitted by the `bot_log`
# log_format defined in nginx/bot_engine.conf:
#
#   $remote_addr [$time_local] "$request" $status
#       score=$upstream_http_x_bot_score
#       action=$upstream_http_x_bot_action
#       class=$upstream_http_x_bot_classification
#       ua="$http_user_agent"
#       ja4=$upstream_http_x_ja4_hash
#       rt=$request_time
#
# Each line is auto-labelled with deterministic rules; ambiguous rows are
# dropped. Per-IP aggregation supplies temporal_jitter / identity_drift /
# request_rate. The resulting (X, y) is concatenated with the synthetic
# corpus so the model sees actual production traffic patterns.

_LOG_LINE_RE = re.compile(
    r'^(?P<ip>\S+)\s+'
    r'\[(?P<time>[^\]]+)\]\s+'
    r'"(?P<method>\S+)\s+(?P<path>\S+)\s+(?P<proto>[^"]+)"\s+'
    r'(?P<status>\d+)\s+'
    r'score=(?P<score>\S+)\s+'
    r'action=(?P<action>\S+)\s+'
    r'class=(?P<cls>\S+)\s+'
    r'ua="(?P<ua>.*?)"\s+'
    r'ja4=(?P<ja4>\S+)\s+'
    r'rt=(?P<rt>\S+)'
)

_HONEYPOT_RE = re.compile(r"^/(api/v2/internal|admin/export|backup/db|\.env|debug/config)")

_METHOD_ENCODING = {
    "GET": 0, "POST": 1, "PUT": 2, "DELETE": 3,
    "HEAD": 4, "PATCH": 5, "OPTIONS": 6,
}

# Default channel-bound features we cannot infer from a log line. Matched
# to the centres of the synthetic human distribution so log rows don't get
# artificially pushed away from the legit cluster on unknown axes.
_LOG_DEFAULTS = {
    "h2_fp_int": 0.0,
    "h3_present": 0.0,
    "header_count": 10.0,
    "has_accept_language": 1.0,
    "has_accept": 1.0,
    "has_cookie": 0.0,
    "has_referer": 0.0,
    "content_length": 0.0,
    "is_tls": 1.0,
}


def _parse_nginx_time(s: str) -> float:
    """Parse nginx's $time_local ('15/Apr/2026:12:34:56 +0000') → unix ts."""
    try:
        return _dt.datetime.strptime(s, "%d/%b/%Y:%H:%M:%S %z").timestamp()
    except ValueError:
        return 0.0


def _ua_entropy(ua: str) -> float:
    """Shannon entropy (bits) over the character distribution of a UA string."""
    if not ua:
        return 0.0
    counts = Counter(ua)
    total = len(ua)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _to_float(s: str) -> float:
    """Parse a log field that may be '-' for missing values."""
    if s in ("-", "", None):
        return 0.0
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def parse_bot_access_log(log_path: str) -> Iterator[dict]:
    """Yield parsed rows from a nginx bot_access.log file.

    Unparseable lines are skipped silently. The stream may be gigabytes;
    this is a generator so the caller can stream-process it.
    """
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = _LOG_LINE_RE.match(line)
            if not m:
                continue
            yield {
                "ip": m.group("ip"),
                "ts": _parse_nginx_time(m.group("time")),
                "method": m.group("method").upper(),
                "path": m.group("path"),
                "status": int(m.group("status")),
                "score": _to_float(m.group("score")),
                "action": m.group("action"),
                "cls": m.group("cls"),
                "ua": m.group("ua"),
                "ja4": m.group("ja4"),
                "rt": _to_float(m.group("rt")),
            }


def _compute_ip_stats(rows: list[dict]) -> dict[str, dict]:
    """Group rows by IP and compute per-IP session features."""
    by_ip: dict[str, list[dict]] = {}
    for row in rows:
        by_ip.setdefault(row["ip"], []).append(row)

    stats: dict[str, dict] = {}
    for ip, rlist in by_ip.items():
        rlist.sort(key=lambda r: r["ts"])
        timestamps = [r["ts"] for r in rlist if r["ts"] > 0]

        jitter = 0.0
        if len(timestamps) >= 3:
            deltas = np.diff(timestamps)
            jitter = float(np.std(deltas)) if len(deltas) else 0.0

        if len(timestamps) >= 2:
            duration = max(1.0, timestamps[-1] - timestamps[0])
            rate = len(rlist) / duration * 60.0  # req/min
        else:
            rate = 1.0

        uas = {r["ua"] for r in rlist if r["ua"]}
        stats[ip] = {
            "count": len(rlist),
            "jitter": jitter,
            "identity_drift": max(1, len(uas)),
            "request_rate": rate,
        }
    return stats


def autolabel_row(row: dict, ip_stats: dict) -> Optional[int]:
    """Deterministically label a row as bad (1), good (0), or skip (None).

    Rules (highest-precedence first):
      - Honeypot hit (path or class) → bad
      - class ∈ {verified_good_bot, good_bot} → good
      - action=block AND engine score ≥ 80 → bad
      - high-volume burst (>120 req/min) with low class confidence → bad
      - class=human + action=allow + 200 status + ≥3 same-IP requests → good
      - anything else → skip (ambiguous; don't train on it)
    """
    path = row.get("path", "")
    cls = row.get("cls", "")
    action = row.get("action", "")
    score = row.get("score", 0.0)
    status = row.get("status", 0)
    stats = ip_stats.get(row.get("ip", ""), {})

    # Honeypot access = unambiguous bad-bot signal
    if _HONEYPOT_RE.match(path):
        return 1
    if cls == "honeypot" or action == "honeypot":
        return 1

    # Explicit good-bot verification — trust engine's drDNS outcome
    if cls in ("verified_good_bot", "good_bot"):
        return 0

    # Engine already blocked with very high confidence
    if action == "block" and score >= 80:
        return 1

    # Burst patterns the engine didn't outright block but that look bot-like
    if stats.get("request_rate", 0.0) >= 120.0 and score >= 50:
        return 1

    # Clean human traffic on allow path — must have enough volume to be
    # a real session, not a single probe.
    if (cls == "human" and action == "allow" and status == 200
            and stats.get("count", 0) >= 3):
        return 0

    return None


def _row_to_feature_vec(row: dict, ip_stats: dict) -> np.ndarray:
    """Build an 18-d feature vector matching FEATURE_NAMES order."""
    ua = row.get("ua") or ""
    ja4 = row.get("ja4", "") or ""
    path = row.get("path", "/")
    method = row.get("method", "GET")
    ts = row.get("ts", 0.0)
    ip = row.get("ip", "")
    stats = ip_stats.get(ip, {})

    if ja4 and ja4 != "-":
        ja4_int = int(hashlib.md5(ja4.encode("utf-8")).hexdigest()[:8], 16)
    else:
        ja4_int = 0

    hour = _dt.datetime.fromtimestamp(ts).hour if ts > 0 else 12

    return np.array([
        float(ja4_int),                                   # ja4_hash_int
        _LOG_DEFAULTS["h2_fp_int"],                       # h2_fp_int
        _LOG_DEFAULTS["h3_present"],                      # h3_present
        float(len(ua)),                                   # ua_length
        float(_ua_entropy(ua)),                           # ua_entropy
        _LOG_DEFAULTS["header_count"],                    # header_count
        float(path.count("/")),                           # path_depth
        float(_METHOD_ENCODING.get(method, 0)),           # method_encoded
        _LOG_DEFAULTS["has_accept_language"],             # has_accept_language
        _LOG_DEFAULTS["has_accept"],                      # has_accept
        _LOG_DEFAULTS["has_cookie"],                      # has_cookie
        _LOG_DEFAULTS["has_referer"],                     # has_referer
        _LOG_DEFAULTS["content_length"],                  # content_length
        float(hour),                                      # hour_of_day
        float(stats.get("jitter", 1.0)),                  # temporal_jitter
        float(stats.get("identity_drift", 1)),            # identity_drift
        float(stats.get("request_rate", 1.0)),            # request_rate
        _LOG_DEFAULTS["is_tls"],                          # is_tls
    ], dtype=np.float32)


def ingest_real_traffic(
    log_path: str,
    max_rows: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse a bot_access.log and return (X, y) for training.

    `max_rows` (if given) keeps only the most recent N rows from the log,
    which is useful for capping memory on multi-GB files.

    Returns empty arrays with the correct dtype/shape if the log is empty
    or no rows survived auto-labelling.
    """
    rows = list(parse_bot_access_log(log_path))
    if max_rows is not None and max_rows > 0 and len(rows) > max_rows:
        rows = rows[-max_rows:]

    if not rows:
        return (np.zeros((0, NUM_FEATURES), dtype=np.float32),
                np.zeros(0, dtype=np.int32))

    stats = _compute_ip_stats(rows)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    labelled = 0
    skipped = 0
    for row in rows:
        label = autolabel_row(row, stats)
        if label is None:
            skipped += 1
            continue
        X_list.append(_row_to_feature_vec(row, stats))
        y_list.append(label)
        labelled += 1

    print(f"  Log rows parsed  : {len(rows)}")
    print(f"  Auto-labelled    : {labelled} (skipped {skipped} ambiguous)")
    if labelled == 0:
        return (np.zeros((0, NUM_FEATURES), dtype=np.float32),
                np.zeros(0, dtype=np.int32))

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    pos = int((y == 1).sum())
    print(f"  Real-traffic mix : good={len(y) - pos}, bad={pos}")
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> xgb.XGBClassifier:
    """Train an XGBoost binary classifier and print evaluation metrics."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)

    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}")
    print(f"  Positive ratio   : {pos_count / len(y_train):.3f}")
    print(f"  scale_pos_weight : {scale_pos_weight:.3f}")

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        use_label_encoder=False,
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # --- Evaluation ---
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n  === Evaluation Metrics ===")
    print(f"  AUC       : {auc_score:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

    print("\n  === Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0, 0]}  FP={cm[0, 1]}")
    print(f"  FN={cm[1, 0]}  TP={cm[1, 1]}")

    print("\n  === Feature Importance (gain) ===")
    importances = clf.feature_importances_
    ranked = sorted(
        zip(FEATURE_NAMES, importances), key=lambda t: t[1], reverse=True
    )
    for rank, (name, imp) in enumerate(ranked, 1):
        print(f"  {rank:>2}. {name:<25s} {imp:.4f}")

    return clf


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(model: xgb.XGBClassifier, output_path: str) -> None:
    """Convert the trained XGBoost model to ONNX and write to disk."""

    initial_type = [("features", FloatTensorType([None, NUM_FEATURES]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=15,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"  ONNX model written to {output_path}")


# ---------------------------------------------------------------------------
# ONNX validation
# ---------------------------------------------------------------------------

def validate_onnx(
    onnx_path: str,
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Load the ONNX model and compare its outputs against XGBoost."""

    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name

    # --- Correctness check on 100 samples ---
    sample_idx = np.random.default_rng(0).choice(len(X_test), size=min(100, len(X_test)), replace=False)
    X_sample = X_test[sample_idx].astype(np.float32)

    xgb_probs = model.predict_proba(X_sample)[:, 1]

    onnx_out = sess.run(None, {input_name: X_sample})
    # skl2onnx outputs: [labels, probabilities]; probabilities is a list of
    # dicts or a 2-D array depending on converter version.
    onnx_probs_raw = onnx_out[1]
    if isinstance(onnx_probs_raw, list):
        onnx_probs = np.array([d[1] if isinstance(d, dict) else d for d in onnx_probs_raw], dtype=np.float32)
    else:
        onnx_probs = np.array(onnx_probs_raw, dtype=np.float32)
        if onnx_probs.ndim == 2:
            onnx_probs = onnx_probs[:, 1]

    max_diff = float(np.max(np.abs(xgb_probs - onnx_probs)))
    print(f"  Max probability difference (100 samples): {max_diff:.2e}")
    if max_diff < 1e-5:
        print("  Correctness check PASSED (tolerance 1e-5)")
    else:
        print(f"  WARNING: max diff {max_diff:.2e} exceeds tolerance 1e-5")

    # --- Full test set accuracy comparison ---
    X_full = X_test.astype(np.float32)
    onnx_full_out = sess.run(None, {input_name: X_full})
    onnx_full_raw = onnx_full_out[0]
    onnx_labels = np.array(onnx_full_raw).flatten()

    xgb_labels = model.predict(X_full)

    onnx_acc = accuracy_score(y_test, onnx_labels)
    xgb_acc = accuracy_score(y_test, xgb_labels)
    print(f"  XGBoost accuracy on test set : {xgb_acc:.4f}")
    print(f"  ONNX accuracy on test set    : {onnx_acc:.4f}")

    # --- Inference timing ---
    n_timing = min(1000, len(X_test))
    X_timing = X_test[:n_timing].astype(np.float32)

    # Warm-up
    for _ in range(10):
        sess.run(None, {input_name: X_timing[:1]})

    start = time.perf_counter()
    for i in range(n_timing):
        sess.run(None, {input_name: X_timing[i : i + 1]})
    elapsed = time.perf_counter() - start

    avg_us = (elapsed / n_timing) * 1_000_000
    print(f"  ONNX avg inference time      : {avg_us:.1f} us/sample ({n_timing} samples)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train bot-detection XGBoost model and export to ONNX.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50000,
        help="Number of synthetic training samples (default: 50000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/var/lib/bot-engine/bot_model.onnx",
        help="Output path for the ONNX model (default: /var/lib/bot-engine/bot_model.onnx)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run ONNX validation after export",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help=("Path to nginx bot_access.log (bot_log format) to ingest as real "
              "training traffic. Auto-labelled with deterministic rules and "
              "concatenated with the synthetic corpus."),
    )
    parser.add_argument(
        "--max-log-rows",
        type=int,
        default=500_000,
        help=("Cap on real-traffic rows ingested from --log-file (keeps the "
              "most recent N). Set to 0 to disable the cap. Default: 500000"),
    )

    args = parser.parse_args()

    print(f"[1/4] Generating {args.samples} synthetic samples (seed={args.seed}) ...")
    X, y = generate_synthetic_data(n_samples=args.samples, seed=args.seed)
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Label distribution: 0={int((y == 0).sum())}, 1={int((y == 1).sum())}")

    if args.log_file:
        print(f"\n[1b/4] Ingesting real traffic from {args.log_file} ...")
        cap = args.max_log_rows if args.max_log_rows > 0 else None
        X_real, y_real = ingest_real_traffic(args.log_file, max_rows=cap)
        if len(X_real) > 0:
            X = np.vstack([X, X_real]).astype(np.float32)
            y = np.concatenate([y, y_real]).astype(np.int32)
            # Reshuffle so real rows aren't bunched at the tail during split
            perm = np.random.default_rng(args.seed).permutation(len(X))
            X, y = X[perm], y[perm]
            print(f"  Combined shape    : X={X.shape}, y={y.shape}")
            print(f"  Final distribution: 0={int((y == 0).sum())}, "
                  f"1={int((y == 1).sum())}")
        else:
            print("  No usable real-traffic rows; proceeding with synthetic only.")

    print("\n[2/4] Training XGBoost classifier ...")
    np.random.seed(args.seed)
    model = train_model(X, y, seed=args.seed)

    print(f"\n[3/4] Exporting model to ONNX -> {args.output} ...")
    export_to_onnx(model, args.output)

    if args.validate:
        print("\n[4/4] Validating ONNX model ...")
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, stratify=y,
        )
        validate_onnx(args.output, model, X_test, y_test)
    else:
        print("\n[4/4] Skipping ONNX validation (use --validate to enable)")

    print("\nDone.")


if __name__ == "__main__":
    main()
