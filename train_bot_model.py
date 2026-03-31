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
import time
from pathlib import Path

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

    args = parser.parse_args()

    print(f"[1/4] Generating {args.samples} synthetic samples (seed={args.seed}) ...")
    X, y = generate_synthetic_data(n_samples=args.samples, seed=args.seed)
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Label distribution: 0={int((y == 0).sum())}, 1={int((y == 1).sum())}")

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
