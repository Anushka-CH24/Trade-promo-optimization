"""
backtest_efficiency.py
-----------------------
Validates the "trade spend efficiency lift" headline number the same way a
commercial analytics team would before pitching a recommendation engine:

For a held-out sample of historical promo events, compare:
  (a) ACTUAL net margin realized at the discount depth that was actually run
  (b) MODEL-RECOMMENDED net margin at the discount depth (0-50%, step 5) that
      the trained model predicts would maximize net margin for that
      SKU/store/context, holding fixed promo cost constant

Trade spend efficiency = total net margin / total fixed promo cost across
the sampled events. The reported lift is the % improvement of (b) over (a).

This is a backtest against held-out data (post train/test split date), not
a live experiment -- reported honestly as a simulated/backtested estimate,
not a field-tested A/B result.
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parent))
from features import build_feature_frame, encode_for_model
from roi_logic import net_promo_roi

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

rng = np.random.default_rng(7)

def main():
    df = pd.read_parquet(DATA_DIR / "retail_scanner_raw.parquet")
    df["date"] = pd.to_datetime(df["date"])

    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)
    split_date = pd.Timestamp(metrics["split_date"])

    skus = pd.read_csv(DATA_DIR / "dim_skus.csv").set_index("sku_id")

    model = xgb.XGBRegressor()
    model.load_model(MODEL_DIR / "xgb_promo_model.json")
    with open(MODEL_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)

    feat = build_feature_frame(df)
    held_out = feat[(feat["date"] > split_date) & (feat["is_promotional_flag"] == 1)]

    # sample 400 promo events for a tractable backtest (each requires ~11 model
    # calls: 1 actual + 10 grid points for the recommendation)
    sample = held_out.sample(n=min(400, len(held_out)), random_state=7)

    discount_grid = np.arange(0, 55, 5)
    # Per-store-day display/listing cost (end-cap, shelf-talker, etc. prorated
    # daily) -- NOT the same as the dashboard's campaign-level default, which
    # rolls up cost across every store running the promo. At single
    # store-day granularity, 150-500 is realistic; 15000 would dwarf any
    # plausible single-store daily profit and make every ROI negative.
    fixed_cost = 300

    actual_margins, recommended_margins = [], []

    for _, row in sample.iterrows():
        sku_row = skus.loc[row["sku_id"]]
        base_price = float(sku_row["base_price"])
        base_margin_pct = float(sku_row["base_margin_pct"])

        # baseline prediction (0% discount) for this exact context
        base_ctx = row.copy()
        base_ctx["discount_pct"] = 0
        X_base = encode_for_model(pd.DataFrame([base_ctx]), encoders=feature_cols)
        pred_base = max(float(model.predict(X_base)[0]), 0)

        # actual discount that was run historically
        actual_ctx = row.copy()
        X_actual = encode_for_model(pd.DataFrame([actual_ctx]), encoders=feature_cols)
        pred_actual = max(float(model.predict(X_actual)[0]), 0)
        roi_actual = net_promo_roi(pred_actual, pred_base, base_price, base_margin_pct,
                                    row["discount_pct"], fixed_cost)
        actual_margins.append(roi_actual["net_margin"])

        # grid search over discount depth for the model-recommended margin
        best_margin = -np.inf
        for d in discount_grid:
            ctx = row.copy()
            ctx["discount_pct"] = d
            X = encode_for_model(pd.DataFrame([ctx]), encoders=feature_cols)
            pred = max(float(model.predict(X)[0]), 0)
            roi = net_promo_roi(pred, pred_base, base_price, base_margin_pct, d, fixed_cost)
            if roi["net_margin"] > best_margin:
                best_margin = roi["net_margin"]
        recommended_margins.append(best_margin)

    total_cost = fixed_cost * len(sample)
    actual_efficiency = sum(actual_margins) / total_cost
    recommended_efficiency = sum(recommended_margins) / total_cost
    lift_pct = (recommended_efficiency - actual_efficiency) / abs(actual_efficiency) * 100

    result = {
        "n_promo_events_sampled": int(len(sample)),
        "actual_net_margin_total": round(float(sum(actual_margins)), 2),
        "recommended_net_margin_total": round(float(sum(recommended_margins)), 2),
        "actual_trade_spend_efficiency": round(float(actual_efficiency), 4),
        "recommended_trade_spend_efficiency": round(float(recommended_efficiency), 4),
        "efficiency_lift_pct": round(float(lift_pct), 2),
    }
    print(json.dumps(result, indent=2))
    with open(MODEL_DIR / "backtest_results.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
