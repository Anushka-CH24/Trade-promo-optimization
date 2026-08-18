"""
generate_data.py
-----------------
Synthesizes a realistic FMCG retail scanner dataset.

Why synthetic: real retailer scanner data (Nielsen/IRI panels) is licensed and
not publicly downloadable. This generator encodes the same statistical
properties analysts work with in real FMCG commercial-analytics roles:
  - price elasticity per SKU/category
  - promo uplift with diminishing returns at deep discounts
  - cannibalization across SKUs in the same category during promos
  - day-of-week and monthly seasonality
  - holiday and salary-day demand spikes
  - stockouts that cap realized sales below true demand ("phantom demand")

Scale: 50 stores x 40 SKUs x 600 days = 1,200,000 rows, matching the
resume-stated dataset size.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

N_STORES = 50
N_SKUS = 40
N_DAYS = 600
START_DATE = datetime(2023, 1, 1)

CATEGORIES = ["Snacks", "Beverages", "Personal Care", "Home Care", "Dairy"]
REGIONS = ["North", "South", "East", "West"]
STORE_TIERS = ["Metro", "Tier-1", "Tier-2"]

# ---------------------------------------------------------------------------
# 1. Dimension tables: stores and SKUs
# ---------------------------------------------------------------------------
stores = pd.DataFrame({
    "store_id": [f"S{i:03d}" for i in range(N_STORES)],
    "region": rng.choice(REGIONS, N_STORES),
    "store_tier": rng.choice(STORE_TIERS, N_STORES, p=[0.2, 0.35, 0.45]),
})
# base daily footfall multiplier by tier
tier_mult = {"Metro": 1.6, "Tier-1": 1.1, "Tier-2": 0.7}
stores["footfall_index"] = stores["store_tier"].map(tier_mult) * rng.normal(1, 0.15, N_STORES)

skus = pd.DataFrame({
    "sku_id": [f"SKU{i:03d}" for i in range(N_SKUS)],
    "category": rng.choice(CATEGORIES, N_SKUS),
})
skus["base_price"] = rng.uniform(20, 350, N_SKUS).round(2)  # local currency units
# price elasticity: how strongly demand responds to a discount (negative = elastic)
skus["elasticity"] = rng.uniform(-3.2, -1.1, N_SKUS).round(2)
# base daily demand per store at full price
skus["base_daily_units"] = rng.uniform(3, 40, N_SKUS).round(1)
# unit margin at base price (as % of price)
skus["base_margin_pct"] = rng.uniform(0.15, 0.35, N_SKUS).round(3)

# ---------------------------------------------------------------------------
# 2. Promo calendar: each SKU runs promos in irregular windows across stores
# ---------------------------------------------------------------------------
dates = [START_DATE + timedelta(days=int(d)) for d in range(N_DAYS)]
date_df = pd.DataFrame({"date": dates})
date_df["dow"] = date_df["date"].dt.dayofweek
date_df["month"] = date_df["date"].dt.month
date_df["is_weekend"] = date_df["dow"].isin([5, 6]).astype(int)
# salary day bump: 1st and last 3 days of month
date_df["is_salary_day"] = (date_df["date"].dt.day <= 2) | (date_df["date"].dt.day >= 28)
date_df["is_salary_day"] = date_df["is_salary_day"].astype(int)
# a handful of holiday spikes per year
holiday_doy = set(rng.choice(range(N_DAYS), size=24, replace=False))
date_df["is_holiday"] = date_df.index.isin(holiday_doy).astype(int)

promo_flags = {}  # (sku_id) -> boolean array over N_DAYS
promo_depth = {}   # (sku_id) -> discount pct array over N_DAYS
for sku in skus["sku_id"]:
    flags = np.zeros(N_DAYS, dtype=bool)
    depth = np.zeros(N_DAYS)
    n_promos = rng.integers(6, 14)  # promo events per SKU over the 600 days
    for _ in range(n_promos):
        start = rng.integers(0, N_DAYS - 14)
        length = rng.integers(3, 10)
        d = rng.choice([0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
        flags[start:start + length] = True
        depth[start:start + length] = d
    promo_flags[sku] = flags
    promo_depth[sku] = depth

# ---------------------------------------------------------------------------
# 3. Simulate daily sell-out per store x sku x day
# ---------------------------------------------------------------------------
rows = []
for _, srow in stores.iterrows():
    for _, krow in skus.iterrows():
        sku = krow["sku_id"]
        flags = promo_flags[sku]
        depth = promo_depth[sku]
        base_units = krow["base_daily_units"] * srow["footfall_index"]

        seasonal = 1 + 0.25 * date_df["is_weekend"] + 0.18 * date_df["is_salary_day"] \
                   + 0.35 * date_df["is_holiday"] \
                   + 0.06 * np.sin(2 * np.pi * date_df["month"] / 12)

        discount_pct = depth
        # elasticity effect with diminishing returns at deep discounts (sqrt dampening)
        uplift_mult = 1 + np.where(
            discount_pct > 0,
            (-krow["elasticity"]) * np.sqrt(discount_pct) * 0.9,
            0,
        )
        # cannibalization: when a promo is running, ~15% of uplift is pulled from
        # non-promoted SKUs in the same category (applied later as a correction)
        noise = rng.normal(1, 0.18, N_DAYS)
        true_demand = base_units * seasonal.values * uplift_mult * noise
        true_demand = np.clip(true_demand, 0, None)

        # inventory & stockouts: on-hand replenished periodically, can run short
        inventory = np.zeros(N_DAYS)
        stock = rng.uniform(80, 200) * (base_units / max(base_units, 1))
        realized = np.zeros(N_DAYS)
        for t in range(N_DAYS):
            if t % 7 == 0:  # weekly replenishment
                stock += rng.uniform(0.9, 1.3) * base_units * 7
            sell = min(true_demand[t], stock)
            realized[t] = sell
            stock -= sell
            inventory[t] = stock

        promo_price = krow["base_price"] * (1 - discount_pct)
        display = rng.choice(["None", "End-cap", "Shelf-talker", "Floor-stack"],
                              size=N_DAYS, p=[0.55, 0.2, 0.15, 0.10])

        df = pd.DataFrame({
            "date": date_df["date"].values,
            "store_id": srow["store_id"],
            "sku_id": sku,
            "category": krow["category"],
            "region": srow["region"],
            "store_tier": srow["store_tier"],
            "base_price": krow["base_price"],
            "promo_price": promo_price.round(2),
            "is_promotional_flag": flags.astype(int),
            "discount_pct": (discount_pct * 100).round(1),
            "display_location_type": display,
            "units_sold": realized.round().astype(int),
            "true_demand": true_demand.round().astype(int),
            "inventory_on_hand": inventory.round().astype(int),
            "is_weekend": date_df["is_weekend"].values,
            "is_salary_day": date_df["is_salary_day"].values,
            "is_holiday": date_df["is_holiday"].values,
        })
        rows.append(df)

full = pd.concat(rows, ignore_index=True)

# ---------------------------------------------------------------------------
# 4. Apply category-level cannibalization: promoted SKU steals ~12-18% of
#    incremental units from non-promoted SKUs in the same category/store/day
# ---------------------------------------------------------------------------
full = full.sort_values(["store_id", "category", "date"]).reset_index(drop=True)
promo_by_cat_day = full[full.is_promotional_flag == 1].groupby(
    ["store_id", "category", "date"]
)["units_sold"].sum().rename("cat_promo_units").reset_index()
full = full.merge(promo_by_cat_day, on=["store_id", "category", "date"], how="left")
full["cat_promo_units"] = full["cat_promo_units"].fillna(0)
cannib_mask = (full.is_promotional_flag == 0) & (full.cat_promo_units > 0)
cannib_factor = rng.uniform(0.03, 0.07, cannib_mask.sum())
full.loc[cannib_mask, "units_sold"] = (
    full.loc[cannib_mask, "units_sold"] * (1 - cannib_factor)
).round().astype(int)
full["units_sold"] = full["units_sold"].clip(lower=0)
full = full.drop(columns=["cat_promo_units"])

# revenue and margin
full["revenue"] = (full["units_sold"] * full["promo_price"]).round(2)
margin_lookup = skus.set_index("sku_id")["base_margin_pct"]
full["base_margin_pct"] = full["sku_id"].map(margin_lookup)
# promo cuts margin roughly proportional to discount depth
full["realized_margin_pct"] = (full["base_margin_pct"] - full["discount_pct"] / 100 * 0.6).clip(lower=0.02)
full["profit"] = (full["revenue"] * full["realized_margin_pct"]).round(2)

print(f"Generated {len(full):,} rows")
full.to_csv("/home/claude/trade-promo-project/data/retail_scanner_raw.csv", index=False)
stores.to_csv("/home/claude/trade-promo-project/data/dim_stores.csv", index=False)
skus.to_csv("/home/claude/trade-promo-project/data/dim_skus.csv", index=False)
print("Saved data/retail_scanner_raw.csv, dim_stores.csv, dim_skus.csv")
