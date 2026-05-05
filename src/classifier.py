"""
classifier.py — XGBoost signal classifier for NeuralEdge.

Responsibilities:
  build_features()         — derive 19 features from OHLCV + sentiment
  generate_labels()        — BUY/HOLD/SELL via forward-return thresholds
  build_training_set()     — fetch + combine data for all training tickers
  walk_forward_validate()  — 3-fold walk-forward; F1 + simulated P&L per fold
  train_final()            — train on full dataset for deployment
  predict()                — live inference: signal + confidence
"""
import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import resample
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

LABEL_SELL, LABEL_HOLD, LABEL_BUY = 0, 1, 2
LABEL_STR = {LABEL_SELL: "SELL", LABEL_HOLD: "HOLD", LABEL_BUY: "BUY"}

BUY_THRESHOLD  =  0.02   # approved spec: forward_return > +2% → BUY
SELL_THRESHOLD = -0.03   # approved spec: forward_return < -3% → SELL
LOOKAHEAD      =  5      # trading days

TRAINING_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
                    "NVDA", "META", "JPM", "V", "NFLX"]

FEATURE_COLS = [
    "rsi", "rsi_momentum", "macd_diff", "macd_diff_momentum",
    "bb_pband", "bb_width", "above_sma50", "above_sma200",
    "price_vs_sma50_pct", "price_vs_sma200_pct",
    "return_1d", "return_5d", "return_10d", "volume_ratio",
    "sentiment_today", "sentiment_3d", "sentiment_trend",
    "sentiment_count", "sentiment_available",
]

MODEL_PATH   = Path("models/classifier.joblib")
METADATA_PATH = Path("models/classifier_meta.json")

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
)


# ── Feature Engineering ───────────────────────────────────────────────────────

def build_features(df_bars: pd.DataFrame, sentiment: dict = None) -> pd.DataFrame:
    """
    Takes a sorted OHLCV DataFrame and optional sentiment dict.
    Returns DataFrame with all 19 feature columns appended.
    Rows with NaN in any feature column are dropped.
    """
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.indicators import compute_indicators

    df = compute_indicators(df_bars.copy())
    close  = df["close"]
    volume = df["volume"]

    # Momentum derivatives
    df["rsi_momentum"]       = df["rsi"]       - df["rsi"].shift(3)
    df["macd_diff_momentum"] = df["macd_diff"] - df["macd_diff"].shift(3)

    # Bollinger Band width (relative volatility)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)

    # Price vs moving averages (percentage deviation)
    df["price_vs_sma50_pct"]  = (close - df["sma50"])  / df["sma50"].replace(0, np.nan)
    df["price_vs_sma200_pct"] = (close - df["sma200"]) / df["sma200"].replace(0, np.nan)

    # Price returns
    df["return_1d"]  = close.pct_change(1)
    df["return_5d"]  = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)

    # Relative volume
    vol_avg = volume.rolling(20).mean()
    df["volume_ratio"] = volume / vol_avg.replace(0, np.nan)

    # Sentiment — use provided values or neutral defaults
    s = sentiment or {}
    df["sentiment_today"]     = s.get("sentiment_today", 0.0)
    df["sentiment_3d"]        = s.get("sentiment_3d", 0.0)
    df["sentiment_trend"]     = s.get("sentiment_trend", 0.0)
    df["sentiment_count"]     = float(s.get("sentiment_count", 0))
    df["sentiment_available"] = float(s.get("sentiment_available", 0))

    return df


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append 'label' and 'forward_return' columns.
    BUY  if forward_return >  +2%
    SELL if forward_return <  -3%
    HOLD otherwise
    Drops the final LOOKAHEAD rows (no label possible).
    """
    df = df.copy()
    df["forward_return"] = df["close"].shift(-LOOKAHEAD) / df["close"] - 1

    def _label(r):
        if pd.isna(r):
            return np.nan
        if r > BUY_THRESHOLD:
            return LABEL_BUY
        if r < SELL_THRESHOLD:
            return LABEL_SELL
        return LABEL_HOLD

    df["label"] = df["forward_return"].apply(_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


# ── Dataset Builder ───────────────────────────────────────────────────────────

def build_training_set(
    tickers: list[str] = None,
    bars_per_ticker: int = 504,
) -> pd.DataFrame:
    """
    Fetch bars for all tickers, compute features + labels.
    Sentiment features default to neutral (historical data gap).
    Returns combined DataFrame sorted by date.
    """
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.alpaca_client import fetch_bars

    tickers = tickers or TRAINING_TICKERS
    frames = []

    # Need bars_per_ticker usable rows AFTER ~220-bar warmup, so fetch enough history.
    # 1 trading day ≈ 1.4 calendar days (accounts for weekends + holidays).
    lookback_days = int((bars_per_ticker + 300) * 1.4)
    start_date = datetime.now() - timedelta(days=lookback_days)

    for sym in tickers:
        logger.info("Building features for %s...", sym)
        try:
            df = fetch_bars(sym, start=start_date, limit=bars_per_ticker + 300)
            if df.empty or len(df) < 250:
                logger.warning("%s: only %d bars, skipping", sym, len(df))
                continue

            df = build_features(df)           # adds all 19 feature cols
            df = generate_labels(df)          # adds label + forward_return
            df = df.dropna(subset=FEATURE_COLS + ["label"])
            df["ticker"] = sym
            df["date"]   = pd.to_datetime(df["timestamp"]).dt.date
            frames.append(df)
            logger.info("%s: %d usable rows", sym, len(df))
        except Exception as e:
            logger.error("Failed to build %s: %s", sym, e)

    if not frames:
        raise RuntimeError("No training data built — check API connection")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    logger.info("Training set: %d rows across %d tickers", len(combined), len(frames))
    return combined


# ── Class Imbalance ───────────────────────────────────────────────────────────

def _balance(df: pd.DataFrame) -> pd.DataFrame:
    """Undersample HOLD to at most 2× the minority class count."""
    counts  = df["label"].value_counts()
    minority = counts.min()
    hold_cap = min(counts.get(LABEL_HOLD, 0), minority * 2)

    parts = []
    for lbl in [LABEL_BUY, LABEL_SELL]:
        part = df[df["label"] == lbl]
        if len(part) > 0:
            parts.append(part)
    hold_part = df[df["label"] == LABEL_HOLD]
    if len(hold_part) > hold_cap:
        hold_part = resample(hold_part, n_samples=hold_cap, random_state=42, replace=False)
    parts.append(hold_part)

    return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)


def _sample_weights(y: pd.Series) -> np.ndarray:
    """Inverse-frequency sample weights."""
    counts  = y.value_counts()
    total   = len(y)
    weights = y.map(lambda lbl: total / (len(counts) * counts[lbl]))
    return weights.values


# ── Training ──────────────────────────────────────────────────────────────────

def _train_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    combined = X[FEATURE_COLS].copy()
    combined["label"] = y.values
    df_bal = _balance(combined)
    X_bal  = df_bal[FEATURE_COLS]
    y_bal  = df_bal["label"]
    sw     = _sample_weights(y_bal)

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_bal, y_bal, sample_weight=sw, verbose=False)
    return model


# ── Walk-Forward Validation ───────────────────────────────────────────────────

def walk_forward_validate(df: pd.DataFrame) -> list[dict]:
    """
    3-fold walk-forward validation.
    Each fold: train on all prior dates, test on next ~25% of remaining dates.
    Reports: accuracy, per-class F1, BUY F1, simulated P&L.
    """
    dates     = sorted(df["date"].unique())
    n         = len(dates)
    # Minimum training window = first 50% of dates.
    # The second 50% is divided into 3 equal test windows so each fold has
    # a meaningful test set and no stub window at the end.
    half      = n // 2
    test_size = max(half // 3, 5)   # at least 5 trading days per test window

    folds = []
    for i in range(3):
        train_end  = half + i * test_size
        test_start = train_end
        test_end   = min(test_start + test_size, n)
        if test_end > n or test_start >= n:
            break
        folds.append((dates[:train_end], dates[test_start:test_end]))

    results = []
    for i, (train_dates, test_dates) in enumerate(folds, 1):
        train_df = df[df["date"].isin(set(train_dates))]
        test_df  = df[df["date"].isin(set(test_dates))]

        if len(train_df) < 50 or len(test_df) < 10:
            logger.warning("Fold %d: insufficient data (train=%d, test=%d)", i, len(train_df), len(test_df))
            continue

        X_train = train_df[FEATURE_COLS]
        y_train = train_df["label"]
        X_test  = test_df[FEATURE_COLS]
        y_test  = test_df["label"]

        model  = _train_model(X_train, y_train)
        y_pred = model.predict(X_test)

        # Per-class F1
        labels_present = sorted(y_test.unique())
        report = classification_report(
            y_test, y_pred,
            labels=labels_present,
            target_names=[LABEL_STR[l] for l in labels_present],
            output_dict=True,
            zero_division=0,
        )
        buy_f1 = report.get("BUY", {}).get("f1-score", 0.0)

        # Simulated P&L — take forward_return on every BUY prediction
        buy_mask    = (y_pred == LABEL_BUY)
        sim_returns = test_df["forward_return"].values[buy_mask]
        sim_pnl     = float(sim_returns.sum()) if len(sim_returns) > 0 else 0.0
        num_trades  = int(buy_mask.sum())

        result = {
            "fold":         i,
            "train_rows":   len(train_df),
            "test_rows":    len(test_df),
            "train_start":  str(min(train_dates)),
            "train_end":    str(max(train_dates)),
            "test_start":   str(min(test_dates)),
            "test_end":     str(max(test_dates)),
            "accuracy":     round(float((y_pred == y_test.values).mean()), 4),
            "buy_f1":       round(buy_f1, 4),
            "simulated_pnl": round(sim_pnl, 4),
            "num_buy_trades": num_trades,
            "label_dist":   y_test.value_counts().to_dict(),
            "report":       report,
        }
        results.append(result)

        buy_f1_ok = buy_f1 > 0.52
        pnl_ok    = sim_pnl > 0
        print(f"\n  Fold {i}: train {result['train_start']}→{result['train_end']}  "
              f"test {result['test_start']}→{result['test_end']}")
        print(f"    BUY F1={buy_f1:.4f} {'✓' if buy_f1_ok else '✗ (need >0.52)'}  "
              f"P&L={sim_pnl:+.4f} {'✓' if pnl_ok else '✗ (need >0)'}  "
              f"trades={num_trades}  acc={result['accuracy']:.4f}")

    return results


def meets_shipping_bar(results: list[dict]) -> bool:
    """Approved bar: BUY F1 > 0.52 AND positive P&L on ALL 3 folds."""
    if len(results) < 3:
        logger.warning("Only %d folds completed (need 3)", len(results))
        return False
    return all(r["buy_f1"] > 0.52 and r["simulated_pnl"] > 0 for r in results)


# ── Final Model ───────────────────────────────────────────────────────────────

def train_final(df: pd.DataFrame) -> XGBClassifier:
    """Train on the full dataset. Call after walk_forward_validate passes."""
    logger.info("Training final model on %d rows...", len(df))
    model = _train_model(df[FEATURE_COLS], df["label"])
    return model


def save_model(model: XGBClassifier, metadata: dict = None) -> None:
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    meta = {
        "saved_at": datetime.utcnow().isoformat(),
        "feature_cols": FEATURE_COLS,
        "label_map": LABEL_STR,
        "buy_threshold": BUY_THRESHOLD,
        "sell_threshold": SELL_THRESHOLD,
        "lookahead_days": LOOKAHEAD,
        **(metadata or {}),
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("Model saved to %s", MODEL_PATH)


def load_model() -> XGBClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained model at {MODEL_PATH} — run train_final() first")
    return joblib.load(MODEL_PATH)


# ── Live Inference ────────────────────────────────────────────────────────────

def predict(df_bars: pd.DataFrame, sentiment: dict = None) -> tuple[str, float]:
    """
    Run inference on the most recent bar.
    Returns (signal: 'BUY'|'SELL'|'HOLD', confidence: 0.0–1.0).
    confidence = probability of the predicted class.
    """
    model = load_model()
    df    = build_features(df_bars, sentiment)
    df    = df.dropna(subset=FEATURE_COLS)

    if df.empty:
        logger.warning("predict(): no valid rows after feature build — defaulting to HOLD")
        return "HOLD", 0.0

    X      = df[FEATURE_COLS].iloc[[-1]]   # last row only
    proba  = model.predict_proba(X)[0]      # [P(SELL), P(HOLD), P(BUY)]
    pred   = int(np.argmax(proba))
    conf   = float(proba[pred])
    signal = LABEL_STR[pred]

    logger.info("predict(): %s (conf=%.3f)  proba=%s", signal, conf,
                {LABEL_STR[i]: round(float(p), 3) for i, p in enumerate(proba)})
    return signal, conf


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("Building training dataset (fetching 2 years of data for 10 tickers)...")
    df = build_training_set()

    dist = df["label"].value_counts().rename(LABEL_STR)
    print(f"\nLabel distribution:\n{dist.to_string()}")

    print("\nRunning walk-forward validation...")
    results = walk_forward_validate(df)

    passed = meets_shipping_bar(results)
    print(f"\n{'PASSED' if passed else 'DID NOT PASS'} shipping bar "
          f"(BUY F1 > 0.52 AND positive P&L on all 3 folds)")

    if passed:
        print("\nTraining final model...")
        model = train_final(df)
        save_model(model, {"training_tickers": TRAINING_TICKERS, "num_rows": len(df)})
        print(f"Model saved to {MODEL_PATH}")
