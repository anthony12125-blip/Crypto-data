"""
Model Training Pipeline
=========================
Trains ML models on collected market data to improve:
1. Alt selection (which alts will amplify the most?)
2. Entry timing (when exactly to enter?)
3. Exit timing (when to get out before the drop?)
4. Regime detection (is the bull still running?)

Models:
- LightGBM: Primary classifier for alt ranking (fast, great on tabular data)
- Ensemble: Combines multiple models for robust predictions

Designed for Apple Silicon M4 — trains in minutes, infers in milliseconds.

Run: python3 train_models.py
Requires: At least 7-14 days of data from data_collector.py
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trainer")


class FeatureEngineer:
    """Transform raw data into ML features."""

    @staticmethod
    def price_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        """Generate price-based features from OHLCV data."""
        features = pd.DataFrame(index=df.index)
        close = df["close"]
        volume = df["volume"]

        # Returns at multiple windows
        for w in [1, 3, 6, 12, 24, 48]:
            features[f"{prefix}ret_{w}"] = close.pct_change(w)

        # Volatility
        for w in [12, 24, 48]:
            features[f"{prefix}vol_{w}"] = close.pct_change().rolling(w).std()

        # RSI
        for period in [14, 28]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            features[f"{prefix}rsi_{period}"] = 100 - (100 / (1 + rs))

        # EMAs and crossovers
        ema8 = close.ewm(span=8).mean()
        ema21 = close.ewm(span=21).mean()
        ema50 = close.ewm(span=50).mean()
        features[f"{prefix}ema8_21_cross"] = (ema8 / ema21 - 1) * 100
        features[f"{prefix}ema21_50_cross"] = (ema21 / ema50 - 1) * 100
        features[f"{prefix}price_vs_ema50"] = (close / ema50 - 1) * 100

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        features[f"{prefix}macd_hist"] = macd - signal

        # Volume features
        vol_ma = volume.rolling(20).mean()
        features[f"{prefix}vol_ratio"] = volume / vol_ma.replace(0, np.nan)
        features[f"{prefix}vol_trend"] = volume.rolling(5).mean() / vol_ma.replace(0, np.nan)

        # Bollinger Band width (volatility measure)
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        features[f"{prefix}bb_width"] = (std20 * 2) / ma20.replace(0, np.nan) * 100

        # High-low range as pct of close
        features[f"{prefix}hl_range"] = (df["high"] - df["low"]) / close * 100

        return features

    @staticmethod
    def relative_features(alt_df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
        """Features measuring alt's behavior relative to BTC."""
        features = pd.DataFrame(index=alt_df.index)

        alt_ret = alt_df["close"].pct_change()
        btc_ret = btc_df["close"].reindex(alt_df.index).pct_change()

        # Rolling beta
        for w in [24, 48, 96]:
            cov = alt_ret.rolling(w).cov(btc_ret)
            var = btc_ret.rolling(w).var()
            features[f"beta_{w}"] = cov / var.replace(0, np.nan)

        # Rolling correlation
        for w in [24, 48]:
            features[f"corr_{w}"] = alt_ret.rolling(w).corr(btc_ret)

        # Relative strength
        for w in [6, 12, 24, 48]:
            alt_change = alt_df["close"].pct_change(w)
            btc_change = btc_df["close"].reindex(alt_df.index).pct_change(w)
            features[f"rs_{w}"] = alt_change - btc_change

        # Amplification ratio (key feature)
        for w in [6, 12, 24]:
            alt_abs = alt_ret.rolling(w).apply(lambda x: x[x > 0].sum())
            btc_abs = btc_ret.rolling(w).apply(lambda x: x[x > 0].sum())
            features[f"amp_ratio_{w}"] = alt_abs / btc_abs.replace(0, np.nan)

        return features


class ModelTrainer:
    """Train and manage ML models."""

    def __init__(self, db_path: str = "data/market_data.db", model_dir: str = "models"):
        self.db_path = db_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.feature_eng = FeatureEngineer()

    def load_training_data(self) -> dict:
        """Load data from SQLite database."""
        conn = sqlite3.connect(self.db_path)

        # OHLCV data
        ohlcv = pd.read_sql(
            "SELECT * FROM ohlcv WHERE timeframe='1h' ORDER BY timestamp",
            conn
        )

        # Alt snapshots
        snapshots = pd.read_sql(
            "SELECT * FROM alt_snapshots ORDER BY timestamp",
            conn
        )

        # Fear & Greed
        fg = pd.read_sql(
            "SELECT * FROM fear_greed ORDER BY timestamp",
            conn
        )

        conn.close()

        logger.info(f"Loaded: {len(ohlcv)} OHLCV rows, "
                     f"{len(snapshots)} alt snapshots, {len(fg)} F&G records")

        return {"ohlcv": ohlcv, "snapshots": snapshots, "fear_greed": fg}

    def prepare_features_and_labels(self, data: dict) -> tuple:
        """
        Build feature matrix and labels for the alt ranking model.
        
        Label: future_return_4h > X% AND same_direction_as_btc
        (Did this alt go up significantly in the next 4 hours while BTC was hot?)
        """
        ohlcv = data["ohlcv"]
        if ohlcv.empty:
            logger.error("No OHLCV data available")
            return None, None

        # Get BTC data
        btc_data = ohlcv[ohlcv["symbol"] == "BTC/USDT"].copy()
        if btc_data.empty:
            logger.error("No BTC data")
            return None, None

        btc_data["timestamp"] = pd.to_datetime(btc_data["timestamp"], unit="ms")
        btc_data.set_index("timestamp", inplace=True)
        btc_data = btc_data.sort_index()

        # Get unique alt symbols
        alt_symbols = [s for s in ohlcv["symbol"].unique() if s != "BTC/USDT"]
        logger.info(f"Processing {len(alt_symbols)} alt symbols...")

        all_features = []
        all_labels = []

        for symbol in alt_symbols:
            try:
                alt_data = ohlcv[ohlcv["symbol"] == symbol].copy()
                alt_data["timestamp"] = pd.to_datetime(alt_data["timestamp"], unit="ms")
                alt_data.set_index("timestamp", inplace=True)
                alt_data = alt_data.sort_index()

                if len(alt_data) < 100:
                    continue

                # Generate features
                price_feat = self.feature_eng.price_features(alt_data, prefix="alt_")
                btc_feat = self.feature_eng.price_features(btc_data, prefix="btc_")
                rel_feat = self.feature_eng.relative_features(alt_data, btc_data)

                # Combine features
                features = pd.concat([price_feat, btc_feat.reindex(price_feat.index), rel_feat], axis=1)

                # Label: future 4h return
                future_ret = alt_data["close"].pct_change(4).shift(-4) * 100
                btc_future = btc_data["close"].reindex(alt_data.index).pct_change(4).shift(-4) * 100

                # Binary label: alt goes up > 3% in next 4h while BTC is also positive
                label = ((future_ret > 3) & (btc_future > 0)).astype(int)

                # Add symbol column
                features["symbol"] = symbol

                # Align and clean
                combined = pd.concat([features, label.rename("label")], axis=1).dropna()

                if len(combined) > 50:
                    all_features.append(combined.drop("label", axis=1))
                    all_labels.append(combined["label"])

            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue

        if not all_features:
            logger.error("No valid training data generated")
            return None, None

        X = pd.concat(all_features)
        y = pd.concat(all_labels)

        # Remove non-numeric columns
        X = X.select_dtypes(include=[np.number])

        # Handle infinities and NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Label distribution: {y.value_counts().to_dict()}")

        return X, y

    def train_lightgbm(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train LightGBM classifier with walk-forward validation."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("LightGBM not installed. Run: pip3 install lightgbm --break-system-packages")
            return None

        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        logger.info("Training LightGBM model...")

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 300,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }

        # Walk-forward cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        results = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )

            preds = model.predict(X_val)
            probs = model.predict_proba(X_val)[:, 1]

            acc = accuracy_score(y_val, preds)
            prec = precision_score(y_val, preds, zero_division=0)
            rec = recall_score(y_val, preds, zero_division=0)
            f1 = f1_score(y_val, preds, zero_division=0)

            results.append({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
            logger.info(f"  Fold {fold+1}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

        # Train final model on all data
        final_model = lgb.LGBMClassifier(**params)
        final_model.fit(X, y)

        # Save model
        model_path = self.model_dir / "alt_ranker_lgbm.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)

        # Save feature importance
        importance = pd.DataFrame({
            "feature": X.columns,
            "importance": final_model.feature_importances_
        }).sort_values("importance", ascending=False)

        importance.to_csv(self.model_dir / "feature_importance.csv", index=False)

        # Average metrics
        avg_metrics = {
            k: round(np.mean([r[k] for r in results]), 4)
            for k in results[0].keys()
        }

        logger.info(f"\n  Average metrics across folds:")
        for k, v in avg_metrics.items():
            logger.info(f"    {k}: {v:.4f}")

        # Save metadata
        metadata = {
            "model_type": "LightGBM",
            "trained_at": datetime.utcnow().isoformat(),
            "samples": len(X),
            "features": len(X.columns),
            "metrics": avg_metrics,
            "top_features": importance.head(20).to_dict(orient="records"),
        }
        with open(self.model_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"\nTop 10 features:")
        for _, row in importance.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.0f}")

        return metadata

    def train_all(self):
        """Full training pipeline."""
        logger.info("=" * 60)
        logger.info("  MODEL TRAINING PIPELINE")
        logger.info("=" * 60)

        # Load data
        data = self.load_training_data()
        if data["ohlcv"].empty:
            logger.error("No data available. Run data_collector.py first!")
            logger.info("The collector needs to run for at least 7-14 days to gather enough data.")
            return

        # Prepare features
        X, y = self.prepare_features_and_labels(data)
        if X is None:
            return

        # Train LightGBM
        lgbm_results = self.train_lightgbm(X, y)

        # Summary
        print(f"""
{'='*60}
  TRAINING COMPLETE
{'='*60}
  Models saved to: {self.model_dir}/
  
  Files:
    alt_ranker_lgbm.pkl    — Main ranking model
    feature_importance.csv — Feature importance rankings
    model_metadata.json    — Training metadata & metrics
    
  Next: Run backtest.py to validate performance
{'='*60}
        """)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all()
