"""
app.py
------
Trade Promotion & Demand Analytics dashboard.

Tab 1 - Commercial Overview : KPIs, top SKUs, promo lift summary
Tab 2 - Scenario Planner    : pick a SKU, slide a discount, see predicted
                              volume, stockout risk, and net promo ROI
Tab 3 - Supply Chain Impact : stores/SKUs at risk of stockout during promos

Run with:  streamlit run app.py
"""
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Dynamic Path Resolution (Prevents FileNotFoundError & ModuleNotFoundError)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

# Search candidates for source code
src_candidates = [
    ROOT_DIR,
    ROOT_DIR / "src",
    ROOT_DIR / "trade-promo-project",
    ROOT_DIR / "trade-promo-project" / "src",
    ROOT_DIR / "trade-promo-optimization",
    ROOT_DIR / "trade-promo-optimization" / "src",
]
for p in src_candidates:
    if p.exists() and str(p) not in sys.path:
        sys.path.append(str(p))

# Search candidates for data files (checks for dim_stores.csv)
data_candidates = [
    ROOT_DIR / "data",
    ROOT_DIR / "trade-promo-project" / "data",
    ROOT_DIR / "trade-promo-optimization" / "data",
    ROOT_DIR.parent / "data",
]
DATA_DIR = next((p for p in data_candidates if (p / "dim_stores.csv").exists()), ROOT_DIR / "data")

# Search candidates for model files (checks for xgb_promo_model.json)
model_candidates = [
    ROOT_DIR / "models",
    ROOT_DIR / "trade-promo-project" / "models",
    ROOT_DIR / "trade-promo-optimization" / "models",
    ROOT_DIR.parent / "models",
]
MODEL_DIR = next((p for p in model_candidates if (p / "xgb_promo_model.json").exists()), ROOT_DIR / "models")

import sqlite3
import json
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import plotly.express as px

# Imports from src/
from features import build_feature_frame, encode_for_model
from roi_logic import net_promo_roi

st.set_page_config(page_title="Trade Promotion & Demand Analytics", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def get_connection():
    return sqlite3.connect(DATA_DIR / "trade_promo.db", check_same_thread=False)


@st.cache_data
def load_dims():
    stores = pd.read_csv(DATA_DIR / "dim_stores.csv")
    skus = pd.read_csv(DATA_DIR / "dim_skus.csv")
    return stores, skus


@st.cache_data
def load_sql_view(view_name: str) -> pd.DataFrame:
    conn = get_connection()
    return pd.read_sql(f"SELECT * FROM {view_name}", conn)


@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_DIR / "xgb_promo_model.json")
    with open(MODEL_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)
    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return model, feature_cols, metrics


@st.cache_data
def load_recent_history():
    """Last 60 days per store/SKU, needed to compute lag features for the
    scenario planner without re-loading the full 1.2M-row fact table."""
    conn = get_connection()
    q = """
        SELECT date, store_id, sku_id, category, region, store_tier,
               base_price, units_sold, discount_pct, display_location_type,
               is_weekend, is_salary_day, is_holiday
        FROM fact_sales
        WHERE date >= (SELECT date(MAX(date), '-60 days') FROM fact_sales)
    """
    df = pd.read_sql(q, conn, parse_dates=["date"])
    return df


stores, skus = load_dims()
model, feature_cols, metrics = load_model()

st.title("Trade Promotion & Demand Analytics")
st.caption(
    "FMCG commercial analytics: promo uplift, price elasticity, "
    "stockout risk, and trade spend ROI across 50 stores x 40 SKUs x 1.2M+ scanner rows."
)

tab1, tab2, tab3 = st.tabs(["Commercial Overview", "Scenario Planner", "Supply Chain Impact"])

# ---------------------------------------------------------------------------
# TAB 1: Commercial Overview
# ---------------------------------------------------------------------------
with tab1:
    perf = load_sql_view("vw_sku_promo_performance")
    region_rev = load_sql_view("vw_region_revenue")

    total_revenue = region_rev["revenue"].sum()
    total_profit = region_rev["profit"].sum()
    avg_uplift = perf["uplift_pct"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"{total_revenue/1e6:,.1f}M")
    c2.metric("Total Profit", f"{total_profit/1e6:,.1f}M")
    c3.metric("Avg. Promo Uplift", f"{avg_uplift:,.1f}%")
    c4.metric("Model MAPE (holdout)", f"{metrics['test_mape_pct']:.1f}%")

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Revenue trend by region")
        trend = region_rev.groupby(["year_month", "region"], as_index=False)["revenue"].sum()
        fig = px.line(trend, x="year_month", y="revenue", color="region")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Top 10 SKUs by promo uplift")
        top10 = perf.sort_values("uplift_pct", ascending=False).head(10)
        fig2 = px.bar(top10, x="uplift_pct", y="sku_id", color="category", orientation="h")
        fig2.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                            yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("SKU promo performance")
    st.dataframe(
        perf.sort_values("promo_profit", ascending=False)
        .rename(columns={
            "uplift_pct": "Uplift %", "avg_discount_pct": "Avg Discount %",
            "promo_revenue": "Promo Revenue", "promo_profit": "Promo Profit",
            "promo_days": "Promo Days",
        }),
        use_container_width=True, height=300,
    )

# ---------------------------------------------------------------------------
# TAB 2: Scenario Planner
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Simulate a proposed promotion")
    history = load_recent_history()

    colA, colB, colC = st.columns(3)
    sku_choice = colA.selectbox("SKU", sorted(skus["sku_id"].unique()))
    store_choice = colB.selectbox("Store", sorted(stores["store_id"].unique()))
    display_choice = colC.selectbox("Display type", ["None", "End-cap", "Shelf-talker", "Floor-stack"])

    discount = st.slider("Proposed discount %", 0, 50, 20, step=5)
    fixed_cost = st.number_input("Fixed promo cost (display, listing fee, etc.)",
                                  min_value=0, value=15000, step=1000)

    sku_row = skus[skus.sku_id == sku_choice].iloc[0]
    store_row = stores[stores.store_id == store_choice].iloc[0]

    hist_slice = history[(history.store_id == store_choice) & (history.sku_id == sku_choice)]
    hist_slice = hist_slice.sort_values("date")

    def build_scenario_row(disc_pct):
        last_date = hist_slice["date"].max() if len(hist_slice) else pd.Timestamp.today()
        scenario_date = last_date + pd.Timedelta(days=1)
        row = pd.DataFrame([{
            "date": scenario_date,
            "store_id": store_choice, "sku_id": sku_choice,
            "category": sku_row["category"], "region": store_row["region"],
            "store_tier": store_row["store_tier"], "base_price": sku_row["base_price"],
            "units_sold": np.nan, "discount_pct": disc_pct,
            "display_location_type": display_choice,
            "is_weekend": int(scenario_date.dayofweek >= 5),
            "is_salary_day": int(scenario_date.day <= 2 or scenario_date.day >= 28),
            "is_holiday": 0,
        }])
        combined = pd.concat([hist_slice, row], ignore_index=True)
        combined = build_feature_frame(combined)
        return combined.iloc[[-1]]

    scenario_promo = build_scenario_row(discount)
    scenario_base = build_scenario_row(0)

    X_promo = encode_for_model(scenario_promo, encoders=feature_cols)
    X_base = encode_for_model(scenario_base, encoders=feature_cols)

    pred_promo = max(float(model.predict(X_promo)[0]), 0)
    pred_base = max(float(model.predict(X_base)[0]), 0)

    roi = net_promo_roi(
        predicted_units_promo=pred_promo,
        predicted_units_baseline=pred_base,
        base_price=float(sku_row["base_price"]),
        base_margin_pct=float(sku_row["base_margin_pct"]),
        discount_pct=discount,
        fixed_promo_cost=fixed_cost,
    )

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted units (promo)", f"{pred_promo:,.1f}")
    m2.metric("Predicted units (no promo)", f"{pred_base:,.1f}")
    m3.metric("Promo lift", f"{roi['promo_lift_units']:+,.1f} units")
    m4.metric("Net Promo ROI", f"{roi['roi_pct']}%" if roi["roi_pct"] is not None else "n/a")

    if roi["margin_dilution_flag"]:
        st.warning(
            f"Margin dilution risk: predicted promo profit ({roi['promo_profit']:,.0f}) "
            f"is *lower* than baseline profit ({roi['baseline_profit']:,.0f}) at this "
            f"store/SKU/discount combination, even though volume increases. "
            f"Consider a shallower discount."
        )
    else:
        st.success(
            f"Net margin impact: {roi['net_margin']:+,.0f} "
            f"(incremental profit {roi['incremental_profit']:,.0f} vs. fixed cost {fixed_cost:,.0f})"
        )

    # Stockout risk check against recent inventory pattern
    avg_recent_units = hist_slice["units_sold"].tail(14).mean() if len(hist_slice) else 0
    if pred_promo > 2.5 * max(avg_recent_units, 1):
        st.error(
            f"Stockout risk: predicted promo demand ({pred_promo:,.0f} units) is "
            f"{pred_promo / max(avg_recent_units,1):.1f}x recent average sell-through "
            f"({avg_recent_units:,.1f} units/day). Confirm replenishment can cover the surge."
        )

# ---------------------------------------------------------------------------
# TAB 3: Supply Chain Impact
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Stores/SKUs at risk of stockout")
    risk = load_sql_view("vw_stockout_risk")
    risk = risk.sort_values("low_stock_days", ascending=False)

    c1, c2 = st.columns(2)
    c1.metric("Store/SKU pairs flagged", len(risk))
    c2.metric("Avg. phantom demand (missed units)", f"{risk['max_phantom_demand'].mean():.1f}")

    region_filter = st.multiselect("Filter by region", sorted(risk["region"].unique()),
                                    default=sorted(risk["region"].unique()))
    filtered = risk[risk["region"].isin(region_filter)]

    fig3 = px.scatter(
        filtered, x="avg_inventory", y="low_stock_days", color="category",
        size="max_phantom_demand", hover_data=["store_id", "sku_id"],
        labels={"avg_inventory": "Avg inventory on hand", "low_stock_days": "Low-stock days"},
    )
    fig3.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(filtered.rename(columns={
        "low_stock_days": "Low Stock Days", "avg_inventory": "Avg Inventory",
        "max_phantom_demand": "Max Phantom Demand",
    }), use_container_width=True, height=300)

st.divider()
st.caption(
    "Data is synthetically generated to mirror real Nielsen/IRI-style retail scanner "
    "panels (price elasticity, promo cannibalization, seasonality, stockouts) since "
    "licensed retailer panel data isn't publicly redistributable. Methodology and "
    "pipeline generalize directly to real scanner/POS data of the same schema."
)
