"""
roi_logic.py
------------
Promo ROI math used by the Scenario Planner tab.

Promo Lift      = Predicted units at proposed discount - Predicted units at 0% discount
Net Margin      = (Promo Lift * discounted unit margin) - Fixed Promo Cost
Net Promo ROI % = Net Margin / Fixed Promo Cost * 100

Also flags margin dilution: cases where volume rises but total profit falls
because the discount cut into margin faster than volume grew.
"""
import numpy as np


def promo_lift(predicted_units_promo: float, predicted_units_baseline: float) -> float:
    return predicted_units_promo - predicted_units_baseline


def discounted_unit_margin(base_price: float, base_margin_pct: float, discount_pct: float) -> float:
    """Margin per unit after discount, mirroring the margin-dilution logic
    used at data-generation time (discount eats into margin, not just price)."""
    promo_price = base_price * (1 - discount_pct / 100)
    realized_margin_pct = max(base_margin_pct - (discount_pct / 100) * 0.6, 0.02)
    return promo_price * realized_margin_pct


def net_promo_roi(predicted_units_promo, predicted_units_baseline,
                   base_price, base_margin_pct, discount_pct, fixed_promo_cost):
    lift = promo_lift(predicted_units_promo, predicted_units_baseline)
    unit_margin_promo = discounted_unit_margin(base_price, base_margin_pct, discount_pct)
    unit_margin_base = base_price * base_margin_pct

    incremental_profit = lift * unit_margin_promo
    baseline_profit = predicted_units_baseline * unit_margin_base
    promo_profit = predicted_units_promo * unit_margin_promo

    net_margin = incremental_profit - fixed_promo_cost
    roi_pct = (net_margin / fixed_promo_cost * 100) if fixed_promo_cost > 0 else np.nan
    margin_dilution = promo_profit < baseline_profit  # volume up, profit down

    return {
        "promo_lift_units": round(lift, 1),
        "incremental_profit": round(incremental_profit, 2),
        "net_margin": round(net_margin, 2),
        "roi_pct": round(roi_pct, 1) if not np.isnan(roi_pct) else None,
        "margin_dilution_flag": bool(margin_dilution),
        "baseline_profit": round(baseline_profit, 2),
        "promo_profit": round(promo_profit, 2),
    }
