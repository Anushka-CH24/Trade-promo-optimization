"""
features.py
------------
Shared feature engineering so the training script and the Streamlit app
build features identically (avoids train/serve skew).

Leakage guard: lag features use only units_sold from STRICTLY PRIOR days
(shift before rolling), and we never use same-day or future inventory,
same-day true_demand, or post-promo outcomes as inputs.
"""
import pandas as pd
import numpy as np

CATEGORICAL_COLS = ["category", "region", "store_tier", "display_location_type"]
NUMERIC_FEATURES = [
    "discount_pct", "lag_7", "lag_14", "lag_30",
    "is_weekend", "is_salary_day", "is_holiday",
    "dow", "month", "base_price",
]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_id", "sku_id", "date"]).copy()
    # shift(1) first so "today" is never included in its own lag window
    df["_shifted"] = df.groupby(["store_id", "sku_id"])["units_sold"].shift(1)
    g = df.groupby(["store_id", "sku_id"])["_shifted"]
    df["lag_7"] = g.transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["lag_14"] = g.transform(lambda x: x.rolling(14, min_periods=1).mean())
    df["lag_30"] = g.transform(lambda x: x.rolling(30, min_periods=1).mean())
    df[["lag_7", "lag_14", "lag_30"]] = df[["lag_7", "lag_14", "lag_30"]].fillna(0)
    df = df.drop(columns=["_shifted"])
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: calendar + lag features, ready for encoding."""
    df = add_calendar_features(df)
    df = add_lag_features(df)
    return df


def encode_for_model(df: pd.DataFrame, encoders: dict | None = None):
    """One-hot encode categoricals. If `encoders` (list of known category
    values per column) is passed, align columns to match training-time
    schema exactly -- needed when scoring new scenario-planner inputs."""
    work = df.copy()
    dummies = pd.get_dummies(work[CATEGORICAL_COLS], columns=CATEGORICAL_COLS)
    X = pd.concat([work[NUMERIC_FEATURES].reset_index(drop=True),
                    dummies.reset_index(drop=True)], axis=1)
    if encoders is not None:
        X = X.reindex(columns=encoders, fill_value=0)
    return X
