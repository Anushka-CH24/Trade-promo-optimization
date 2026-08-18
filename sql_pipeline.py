"""
sql_pipeline.py
----------------
Loads the raw scanner data into a local SQLite database and runs the SQL
layer: integration of the fact table with store/SKU dimensions, cleaning
(deduplication, null handling, type coercion), and the aggregate views the
Streamlit dashboard reads from.

Uses SQLite so the project runs with zero external infra, but every query
is standard ANSI SQL and drops into Postgres/Snowflake/BigQuery unchanged.
"""
import sqlite3
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "trade_promo.db"

def load_raw_tables(conn):
    fact = pd.read_parquet(DATA_DIR / "retail_scanner_raw.parquet")
    stores = pd.read_csv(DATA_DIR / "dim_stores.csv")
    skus = pd.read_csv(DATA_DIR / "dim_skus.csv")

    # --- cleaning ---
    fact["date"] = pd.to_datetime(fact["date"])
    before = len(fact)
    fact = fact.drop_duplicates(subset=["date", "store_id", "sku_id"])
    fact = fact.dropna(subset=["units_sold", "revenue"])
    fact = fact[fact["units_sold"] >= 0]
    print(f"Cleaned fact table: {before:,} -> {len(fact):,} rows "
          f"({before - len(fact):,} duplicates/nulls removed)")

    fact.to_sql("fact_sales", conn, if_exists="replace", index=False)
    stores.to_sql("dim_stores", conn, if_exists="replace", index=False)
    skus.to_sql("dim_skus", conn, if_exists="replace", index=False)


SQL_VIEWS = {
    # SKU-level promo performance: uplift vs non-promo baseline, ROI
    "vw_sku_promo_performance": """
        CREATE VIEW vw_sku_promo_performance AS
        WITH baseline AS (
            SELECT sku_id, AVG(units_sold) AS baseline_units
            FROM fact_sales
            WHERE is_promotional_flag = 0
            GROUP BY sku_id
        ),
        promo AS (
            SELECT sku_id, AVG(units_sold) AS promo_units,
                   AVG(discount_pct) AS avg_discount_pct,
                   SUM(profit) AS promo_profit,
                   SUM(revenue) AS promo_revenue,
                   COUNT(*) AS promo_days
            FROM fact_sales
            WHERE is_promotional_flag = 1
            GROUP BY sku_id
        )
        SELECT p.sku_id, k.category, b.baseline_units, p.promo_units,
               ROUND((p.promo_units - b.baseline_units) / b.baseline_units * 100, 1) AS uplift_pct,
               p.avg_discount_pct, p.promo_revenue, p.promo_profit, p.promo_days
        FROM promo p
        JOIN baseline b ON p.sku_id = b.sku_id
        JOIN dim_skus k ON p.sku_id = k.sku_id
    """,
    # Store x region rollup for the commercial overview tab
    "vw_region_revenue": """
        CREATE VIEW vw_region_revenue AS
        SELECT f.region, f.store_tier,
               strftime('%Y-%m', f.date) AS year_month,
               SUM(f.revenue) AS revenue,
               SUM(f.profit) AS profit,
               SUM(f.units_sold) AS units_sold
        FROM fact_sales f
        GROUP BY f.region, f.store_tier, year_month
    """,
    # Stockout risk: days where inventory ran below 15% of weekly avg replenishment
    "vw_stockout_risk": """
        CREATE VIEW vw_stockout_risk AS
        SELECT store_id, sku_id, category, region,
               COUNT(*) AS low_stock_days,
               AVG(inventory_on_hand) AS avg_inventory,
               MAX(true_demand - units_sold) AS max_phantom_demand
        FROM fact_sales
        WHERE inventory_on_hand < (SELECT AVG(inventory_on_hand) * 0.15 FROM fact_sales)
        GROUP BY store_id, sku_id, category, region
        HAVING low_stock_days > 3
    """,
}


def build_views(conn):
    cur = conn.cursor()
    for name, sql in SQL_VIEWS.items():
        cur.execute(f"DROP VIEW IF EXISTS {name}")
        cur.execute(sql)
    conn.commit()
    print(f"Built {len(SQL_VIEWS)} SQL views: {', '.join(SQL_VIEWS)}")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    load_raw_tables(conn)
    build_views(conn)

    # sanity check
    check = pd.read_sql("SELECT * FROM vw_sku_promo_performance ORDER BY uplift_pct DESC LIMIT 5", conn)
    print("\nTop 5 SKUs by promo uplift:")
    print(check.to_string(index=False))
    conn.close()
    print(f"\nSQLite DB written to {DB_PATH}")
