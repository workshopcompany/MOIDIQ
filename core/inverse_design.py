"""
Stage 3: Inverse Design Engine — Tolerance-Based Mold Correction
Purpose: Propose linear mold corrections only within tolerance range
Constraints: No nonlinear/asymmetric correction / No global scale change / Linear correction only
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
    correction_mm: float     # Suggested correction (positive: enlarge mold, negative: shrink)
    mold_current: float      # Current mold dimension
    mold_corrected: float    # Corrected mold dimension
    tolerance_plus: float
    tolerance_minus: float
    within_tolerance: bool   # Whether correction brings part within tolerance
    note: str


def run_inverse_design(
    dim_df: pd.DataFrame,
    global_shrink_avg: float,
) -> dict:
    """
    Dimension prediction DataFrame -> Generate inverse design correction proposals

    Input:
      dim_df : result from predict_part_dimensions()
      global_shrink_avg : global average shrinkage rate (decimal, e.g. 0.0055)
    """

    corrections = []
    post_correction = []
    c_id = 1

    # ── STEP 1: Global Uniform Compensation ──────────────────
    # Scale up mold globally based on average shrinkage rate
    uniform_comp_pct = global_shrink_avg * 100

    corrections.append(CorrectionItem(
        id=f"C-{c_id:02d}",
        priority="MED",
        correction_type="uniform_offset",
        feature_name="Uniform Shrinkage Compensation",
        error_mm=0.0,
        correction_mm=0.0,
        mold_current=0.0,
        mold_corrected=0.0,
        tolerance_plus=0.0,
        tolerance_minus=0.0,
        within_tolerance=True,
        note=f"Enlarge entire mold by +{uniform_comp_pct:.3f}% (average shrinkage compensation)",
    ))
    c_id += 1

    # ── STEP 2: Feature-Level Correction ─────────────────────
    for _, row in dim_df.iterrows():
        verdict = row["판정"]
        if verdict == "OK" and row["공차 소진율 (%)"] < 60:
            continue  # Sufficient margin -> no correction needed

        feature = row["Feature"]
        error = row["편차 (mm)"]            # predicted - nominal
        mold_dim = row["금형 치수 (mm)"]
        shrinkage = row.get("_shrinkage", global_shrink_avg)
        tol_str = row["공차"]

        try:
            tol_p = row.get("_upper", 0) - row["도면 공칭 (mm)"]
            tol_m = row["도면 공칭 (mm)"] - row.get("_lower", 0)
        except Exception:
            tol_p, tol_m = 0.1, 0.1

        # E(x) = target - predicted = -error
        e_x = -error

        # Correctable only if |E(x)| is within tolerance range
        max_allow = tol_p + tol_m  # Total tolerance band
        correctable = abs(e_x) <= max_allow * 2.0  # Attempt correction up to 2x tolerance

        if not correctable:
            priority = "HIGH"
            note = "Deviation too large — geometry redesign required"
            within_tol = False
        elif verdict in ("UNDER", "OVER"):
            priority = "HIGH"
            note = _make_correction_note(feature, e_x, verdict)
            within_tol = True
        else:
            priority = "MED"
            note = f"Tolerance consumed {row['공차 소진율 (%)']:.1f}% — preventive correction recommended"
            within_tol = True

        # Corrected mold dimension via inverse shrinkage calculation
        # L_mold_new = L_nominal / (1 - S)
        nominal = row["도면 공칭 (mm)"]
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

        # Post-correction prediction result
        new_predicted = mold_corrected * (1 - shrinkage)
        new_error = new_predicted - nominal
        new_verdict = (
            "OK" if abs(new_error) <= min(tol_p, tol_m) * 0.95
            else ("CAUTION" if abs(new_error) <= max(tol_p, tol_m) else "EXCEED")
        )
        post_correction.append({
            "Feature": feature,
            "Pre-Dev.": round(error, 4),
            "Post-Dev.": round(new_error, 4),
            "Tolerance": f"+{tol_p} / -{tol_m}",
            "Result": new_verdict,
        })

    # ── STEP 3: Process Parameter Correction (alternative to mold rework) ──
    # Increasing packing pressure reduces shrinkage
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
            note="+5~10 MPa packing pressure increase reduces shrinkage by ~0.05~0.1% (alternative to mold rework)",
        ))

    # ── Cost Reduction Estimate ───────────────────────────────
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
    return f"{feature}: {direction} by {abs(error):.4f}mm ({verdict} correction)"


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
    Generate E(x) map — positional error distribution (for heatmap)
    shrink_df: contains position-wise shrinkage data
    """
    if "x" not in shrink_df.columns:
        return {}

    x = shrink_df["x"].values
    y = shrink_df["y"].values
    s = shrink_df["shrinkage"].values

    # Simplified: error = deviation of local shrinkage from mean (vs. drawing in real use)
    s_mean = s.mean()
    error = (s - s_mean) * 100  # deviation in %

    return {
        "x": x.round(2).tolist(),
        "y": y.round(2).tolist(),
        "error": error.round(4).tolist(),
        "error_min": float(error.min()),
        "error_max": float(error.max()),
    }
