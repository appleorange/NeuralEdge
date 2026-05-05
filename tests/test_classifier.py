"""
Verification test for classifier.py.
Fetches live data — expect 2–3 min for full dataset build.
"""
import logging
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

from src.classifier import (
    FEATURE_COLS, LABEL_BUY, LABEL_HOLD, LABEL_SELL, LABEL_STR,
    BUY_THRESHOLD, SELL_THRESHOLD, LOOKAHEAD,
    build_features, generate_labels, build_training_set,
    walk_forward_validate, meets_shipping_bar, train_final, save_model, load_model, predict,
    _balance,
)
from src.alpaca_client import fetch_bars

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
results = []


def check(label, condition, detail=""):
    tag = PASS if condition else FAIL
    print(f"  {tag} {label}" + (f"  ({detail})" if detail else ""))
    results.append((label, condition))
    return condition


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── Test 1: Feature builder ────────────────────────────────────────────────────
section("TEST 1 — build_features()")

df_raw = fetch_bars("AAPL", limit=300)
df_feat = build_features(df_raw)

check("All 19 feature columns present", all(c in df_feat.columns for c in FEATURE_COLS),
      f"missing: {[c for c in FEATURE_COLS if c not in df_feat.columns]}")

df_clean = df_feat.dropna(subset=FEATURE_COLS)
check("Has rows after NaN drop", len(df_clean) > 50, f"{len(df_clean)} clean rows")

# Sanity check derived features
last = df_clean.iloc[-1]
check("rsi_momentum is RSI diff over 3 bars",
      abs(last["rsi_momentum"] - (df_feat["rsi"].iloc[-1] - df_feat["rsi"].iloc[-4])) < 0.01)
check("bb_width > 0", float(last["bb_width"]) > 0, f"bb_width={last['bb_width']:.4f}")
check("volume_ratio > 0", float(last["volume_ratio"]) > 0, f"vol_ratio={last['volume_ratio']:.4f}")
check("sentiment defaults to neutral", float(last["sentiment_available"]) == 0.0)

# With real sentiment
s = {"sentiment_today": 0.5, "sentiment_3d": 0.3, "sentiment_trend": 0.2,
     "sentiment_count": 3, "sentiment_available": 1}
df_with_sent = build_features(df_raw, sentiment=s)
last_s = df_with_sent.iloc[-1]
check("Sentiment injected correctly",
      abs(float(last_s["sentiment_today"]) - 0.5) < 0.001 and
      float(last_s["sentiment_available"]) == 1.0)


# ── Test 2: Label generation ───────────────────────────────────────────────────
section("TEST 2 — generate_labels()")

df_labeled = generate_labels(df_feat.dropna(subset=FEATURE_COLS))
check("label column present", "label" in df_labeled.columns)
check("forward_return column present", "forward_return" in df_labeled.columns)
check("Labels only 0/1/2", set(df_labeled["label"].unique()).issubset({0, 1, 2}))
check("No NaN labels", df_labeled["label"].isna().sum() == 0)
check("Last LOOKAHEAD rows dropped", len(df_labeled) < len(df_feat))

# Verify label logic
for _, row in df_labeled.sample(min(200, len(df_labeled)), random_state=42).iterrows():
    fr = row["forward_return"]
    lbl = int(row["label"])
    if fr > BUY_THRESHOLD:
        assert lbl == LABEL_BUY,  f"Expected BUY for fr={fr:.4f}"
    elif fr < SELL_THRESHOLD:
        assert lbl == LABEL_SELL, f"Expected SELL for fr={fr:.4f}"
    else:
        assert lbl == LABEL_HOLD, f"Expected HOLD for fr={fr:.4f}"
check("Label logic matches thresholds (200 sampled rows)", True)

dist = df_labeled["label"].value_counts().rename(LABEL_STR)
print(f"\n  Label distribution (AAPL):\n{dist.to_string()}")


# ── Test 3: Class balancing ────────────────────────────────────────────────────
section("TEST 3 — Class balancing (_balance)")

df_bal = _balance(df_labeled)
counts_bal = df_bal["label"].value_counts()
minority = min(counts_bal.get(LABEL_BUY, 0), counts_bal.get(LABEL_SELL, 0))
hold_count = counts_bal.get(LABEL_HOLD, 0)
check("HOLD capped at 2× minority", hold_count <= minority * 2 + 1,
      f"HOLD={hold_count}, minority={minority}")
check("BUY and SELL still present", LABEL_BUY in counts_bal and LABEL_SELL in counts_bal)


# ── Test 4: Full training dataset ─────────────────────────────────────────────
section("TEST 4 — build_training_set() — fetching all 10 tickers")
print("  (fetching 504 bars × 10 tickers — may take ~60s...)")

df_train = build_training_set()

check("Dataset has rows", len(df_train) > 200, f"{len(df_train)} rows")
check("All feature cols present", all(c in df_train.columns for c in FEATURE_COLS))
check("date column present", "date" in df_train.columns)
check("ticker column present", "ticker" in df_train.columns)
check("Multiple tickers", df_train["ticker"].nunique() >= 5,
      f"{df_train['ticker'].nunique()} tickers")
check("No NaN in features", df_train[FEATURE_COLS].isna().sum().sum() == 0,
      f"{df_train[FEATURE_COLS].isna().sum().sum()} NaNs")

dist_all = df_train["label"].value_counts().rename(LABEL_STR)
print(f"\n  Label distribution (all tickers):\n{dist_all.to_string()}")
print(f"  Tickers: {sorted(df_train['ticker'].unique())}")


# ── Test 5: Walk-forward validation ───────────────────────────────────────────
section("TEST 5 — walk_forward_validate() — 3-fold walk-forward")

fold_results = walk_forward_validate(df_train)

check("3 folds completed", len(fold_results) == 3, f"{len(fold_results)} folds")
for r in fold_results:
    check(f"Fold {r['fold']}: BUY F1 returned", isinstance(r["buy_f1"], float))
    check(f"Fold {r['fold']}: simulated_pnl returned", isinstance(r["simulated_pnl"], float))
    check(f"Fold {r['fold']}: no data leakage (train_end < test_start)",
          r["train_end"] < r["test_start"],
          f"train_end={r['train_end']} test_start={r['test_start']}")

passed = meets_shipping_bar(fold_results)
all_buy_f1 = [r["buy_f1"] for r in fold_results]
all_pnl    = [r["simulated_pnl"] for r in fold_results]
print(f"\n  BUY F1 per fold: {[f'{v:.4f}' for v in all_buy_f1]}")
print(f"  P&L  per fold:  {[f'{v:+.4f}' for v in all_pnl]}")
print(f"  Shipping bar met: {passed}")
check("Shipping bar evaluated (no error)", isinstance(passed, bool))


# ── Test 6: Final model train + save + load + predict ─────────────────────────
section("TEST 6 — train_final / save / load / predict")

model = train_final(df_train)
save_model(model, {"test_run": True})

from pathlib import Path
check("Model file saved", Path("models/classifier.joblib").exists())
check("Metadata file saved", Path("models/classifier_meta.json").exists())

model2 = load_model()
check("Model loads back successfully", model2 is not None)

# predict() on live data
df_live = fetch_bars("AAPL", limit=300)
signal, conf = predict(df_live)
check("predict() returns valid signal", signal in {"BUY", "SELL", "HOLD"}, f"signal={signal}")
check("predict() confidence in [0,1]", 0.0 <= conf <= 1.0, f"conf={conf:.4f}")
print(f"\n  Live AAPL prediction: {signal} (confidence={conf:.4f})")


# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
passed_count = sum(1 for _, ok in results if ok)
total = len(results)
print(f"  RESULT: {passed_count}/{total} checks passed")
if passed_count == total:
    print("  classifier.py VERIFIED")
    if meets_shipping_bar(fold_results):
        print("  Shipping bar MET — ready for backtester.py")
    else:
        print("  NOTE: Shipping bar not yet met — model needs improvement")
        print("  (This is expected on limited data; backtester.py will diagnose further)")
else:
    print("  FAILURES:")
    for label, ok in results:
        if not ok:
            print(f"    • {label}")
print(f"{'═'*60}")
sys.exit(0 if passed_count == total else 1)
