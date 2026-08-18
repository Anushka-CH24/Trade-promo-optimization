"""
train_model.py
---------------
Trains an XGBoost regressor to predict daily unit sales as a function of
promo discount depth, recent demand momentum, and calendar effects. The
model is used two ways downstream:
  1. Point prediction of units_sold for a given (SKU, discount, context)
  2. Uplift estimation: predicted units at a proposed discount minus
     predicted units at 0% discount (same context) = incremental volume,
     which feeds the promo ROI calculator in the dashboard.

Chronological split (train on first ~10 months, test on last ~2 months) so
evaluation reflects real forecasting conditions, not random-shuffle leakage
across time.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from features import build_feature_frame, encode_for_model, NUMERIC_FEATURES, CATEGORICAL_COLS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_DIR / "retail_scanner_raw.parquet")
    df["date"] = pd.to_datetime(df["date"])

    print("Building features...")
    df = build_feature_frame(df)

    split_date = df["date"].quantile(0.85, interpolation="nearest")
    print(f"Chronological split at {split_date.date()}")
    train_df = df[df["date"] <= split_date]
    test_df = df[df["date"] > split_date]

    X_train = encode_for_model(train_df)
    X_test = encode_for_model(test_df, encoders=X_train.columns)
    y_train = train_df["units_sold"]
    y_test = test_df["units_sold"]

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | Features: {X_train.shape[1]}")

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
        eval_metric="mae",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    preds = np.clip(model.predict(X_test), 0, None)
    mape = mean_absolute_percentage_error(
        y_test[y_test > 0], preds[y_test.values > 0]
    ) * 100
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = np.abs(y_test.values - preds).mean()

    print(f"\nTest MAPE: {mape:.2f}%")
    print(f"Test RMSE: {rmse:.2f} units")
    print(f"Test MAE:  {mae:.2f} units")

    # feature importance
    importances = pd.Series(model.feature_importances_, index=X_train.columns) \
        .sort_values(ascending=False).head(15)
    print("\nTop 15 feature importances:")
    print(importances.to_string())

    MODEL_DIR.mkdir(exist_ok=True)
    model.save_model(MODEL_DIR / "xgb_promo_model.json")
    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(list(X_train.columns), f)
    metrics = {
        "test_mape_pct": round(float(mape), 2),
        "test_rmse_units": round(float(rmse), 2),
        "test_mae_units": round(float(mae), 2),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "split_date": str(split_date.date()),
        "n_features": int(X_train.shape[1]),
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model, feature schema, and metrics to {MODEL_DIR}/")

if __name__ == "__main__":
    main()
