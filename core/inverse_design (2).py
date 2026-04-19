"""
Stage 3: Inverse Design Engine — Tolerance-based Inverse Design
Purpose: Propose linear mold corrections for predicted dimension deviations within tolerance
Constraints: No non-linear/asymmetric corrections ❌ / No global scale changes ❌ / Linear correction only ✔
Formula: E(x) = L_target(x) - L_predicted(x)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class CorrectionItem:
    """Single correction item"""
    id: str                  # C-01, C-02 ...
    priority: str            # HIGH / MED / LOW
    correction_type: str     # uniform_offset / local_linear / feature_level / process
    feature_name: str
    error_mm: float          # E(x) = target - predicted
    correction_mm: float     # suggested correction (positive: enlarge mold, negative: shrink)
    mold_current: float      # current mold dimension
    mold_corrected: float    # mold dimension after correction
    tolerance_plus: float
    tolerance_minus: float
    within_tolerance: bool   # whether correction falls within tolerance
    note: str


def run_inverse_design(
    dim_df: pd.DataFrame,
    global_shrink_avg: float,
) -> dict:
    """
    Dimension prediction result DataFrame → generate inverse design correction proposals

    Input:
      dim_df : result from predict_part_dimensions()
      global_shrink_avg : overall average shrinkage rate (decimal, e.g. 0.0055)
    """

    corrections = []
    post_correction = []
    c_id = 1

    # ── STEP 1: Global uniform compensation ──────────────────
    # Scale up mold based on average shrinkage across all features
    uniform_comp_pct = global_shrink_avg * 100

    corrections.append(CorrectionItem(
        id=f"C-{c_id:02d}",
        priority="MED",
        correction_type="uniform_offset",
        feature_name="Global Uniform Shrinkage Compensation",
        error_mm=0.0,
        correction_mm=0.0,
        mold_current=0.0,
        mold_corrected=0.0,
        tolerance_plus=0.0,
        tolerance_minus=0.0,
        within_tolerance=True,
        note=f"Scale up entire mold by +{uniform_comp_pct:.3f}% (average shrinkage compensation)",
    ))
    c_id += 1

    # ── STEP 2: Per-feature correction ────────────────────────
    for _, row in dim_df.iterrows():
        verdict = row["Verdict"]
        if verdict == "OK" and row["Tolerance Used (%)"] < 60:
            continue  # sufficient margin → no correction needed

        feature = row["Feature"]
        error = row["Deviation (mm)"]            # predicted - nominal
        mold_dim = row["Mold Dim (mm)"]
        shrinkage = row.get("_shrinkage", global_shrink_avg)
        tol_str = row["Tolerance"]

        try:
            tol_p = row.get("_upper", 0) - row["Nominal (mm)"]
            tol_m = row["Nominal (mm)"] - row.get("_lower", 0)
        except Exception:
            tol_p, tol_m = 0.1, 0.1

        # E(x) = target - predicted = -error
        e_x = -error

        # Correctability check: |E(x)| must be within tolerance range
        max_allow = tol_p + tol_m  # total tolerance width
        correctable = abs(e_x) <= max_allow * 2.0  # attempt correction up to 2× tolerance

        if not correctable:
            priority = "HIGH"
            note = "Deviation exceeds tolerance range — part redesign required"
            within_tol = False
        elif verdict in ("UNDER", "OVER"):
            priority = "HIGH"
            note = _make_correction_note(feature, e_x, verdict)
            within_tol = True
        else:
            priority = "MED"
            note = f"Tolerance used {row['Tolerance Used (%)']}% — proactive correction recommended"
            within_tol = True

        # Mold correction dimension: back-calculate from shrinkage
        # L_mold_new = L_nominal / (1 - S)  ← forward inverse calculation
        nominal = row["Nominal (mm)"]
        if nominal > 0 and (1 - shrinkage) > 0:
            mold_corrected = nominal / (1 - shrinkage)
        else:
            mold_corrected = mold_dim + e_x * (1 / (1 - shrinkage))

        correction_mm = mold_corrected - mold_dim

        corrections.append(CorrectionItem(
            id=f"C-{c_id:02d}",
            priority=priority,
            correction_type="local_linear",
            feature_name=feature,
            error_mm=round(e_x, 4),
            correction_mm=round(correction_mm, 4),
            mold_current=round(mold_dim, 4),
            mold_corrected=round(mold_corrected, 4),
            tolerance_plus=tol_p,
            tolerance_minus=tol_m,
            within_tolerance=within_tol,
            note=note,
        ))
        c_id += 1

        # Predicted result after correction
        new_predicted = mold_corrected * (1 - shrinkage)
        new_error = new_predicted - nominal
        new_verdict = (
            "OK" if abs(new_error) <= min(tol_p, tol_m) * 0.95
            else ("WARN" if abs(new_error) <= max(tol_p, tol_m) else "OVER")
        )
        post_correction.append({
            "Feature": feature,
            "Pre-Deviation": round(error, 4),
            "Post-Deviation": round(new_error, 4),
            "Tolerance": f"+{tol_p} / -{tol_m}",
            "Result": new_verdict,
        })

    # ── STEP 3: Process parameter adjustment (alternative without mold modification) ──
    # Increasing packing pressure → reduces shrinkage
    residual_items = [c for c in corrections if c.correction_type == "local_linear"
                      and not c.within_tolerance]
    if residual_items:
        corrections.append(CorrectionItem(
            id=f"C-{c_id:02d}",
            priority="LOW",
            correction_type="process",
            feature_name="Packing Pressure Adjustment",
            error_mm=0.0,
            correction_mm=0.0,
            mold_current=0.0,
            mold_corrected=0.0,
            tolerance_plus=0.0,
            tolerance_minus=0.0,
            within_tolerance=True,
            note="Increasing packing pressure by +5~10 MPa reduces shrinkage by ~0.05~0.1% (alternative to mold modification)",
        ))

    # ── Cost reduction estimate ────────────────────────────────
    high_count = sum(1 for c in corrections if c.priority == "HIGH")
    trial_before = min(3 + high_count, 6)
    trial_after = max(1, trial_before - 2)
    cost_reduction_pct = round((1 - trial_after / trial_before) * 100)

    return {
        "corrections": corrections,
        "post_correction": pd.DataFrame(post_correction) if post_correction else pd.DataFrame(),
        "uniform_compensation_pct": round(uniform_comp_pct, 3),
        "global_shrink_avg": round(global_shrink_avg * 100, 3),
        "cost_estimate": {
            "trial_before": trial_before,
            "trial_after": trial_after,
            "cost_reduction_pct": cost_reduction_pct,
            "dev_time_saving_weeks": max(1, high_count),
        },
        "summary": _make_summary(corrections),
    }


def _make_correction_note(feature: str, error: float, verdict: str) -> str:
    direction = "Enlarge mold" if error > 0 else "Reduce mold"
    return f"{feature}: {direction} by {abs(error):.4f}mm required ({verdict} correction)"


def _make_summary(corrections: list) -> dict:
    high = sum(1 for c in corrections if c.priority == "HIGH")
    med = sum(1 for c in corrections if c.priority == "MED")
    low = sum(1 for c in corrections if c.priority == "LOW")
    total_correct = sum(1 for c in corrections if c.within_tolerance and c.correction_type == "local_linear")
    return {
        "HIGH": high,
        "MED": med,
        "LOW": low,
        "total_correctable": total_correct,
        "verdict": (
            "✅ All corrections within tolerance" if high == 0
            else f"⚠ {high} item(s) require priority correction"
        ),
    }


def build_error_map(shrink_df: pd.DataFrame, features_df: pd.DataFrame) -> dict:
    """
    Build E(x) map — spatial error distribution (for heatmap)
    shrink_df: includes per-location shrinkage
    """
    if "x" not in shrink_df.columns:
        return {}

    x = shrink_df["x"].values
    y = shrink_df["y"].values
    s = shrink_df["shrinkage"].values

    # Simplified: error = deviation of local shrinkage from mean (actual = vs drawing)
    s_mean = s.mean()
    error = (s - s_mean) * 100  # deviation in % units

    return {
        "x": x.round(2).tolist(),
        "y": y.round(2).tolist(),
        "error": error.round(4).tolist(),
        "error_min": float(error.min()),
        "error_max": float(error.max()),
    }
