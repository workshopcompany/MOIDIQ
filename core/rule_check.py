import numpy as np


# ── Material-Specific Threshold Database ───────────────────────────
MATERIAL_LIMITS = {
    "PC+ABS": {
        "lt_ratio_max": 150,
        "min_thickness": 1.5,
        "max_thickness": 4.0,
        "draft_angle_min": 1.0,
        "thickness_ratio_warn": 1.5,
        "thickness_ratio_fail": 2.0,
        "shrink_range": (0.004, 0.007),
    },
    "ABS": {
        "lt_ratio_max": 130,
        "min_thickness": 1.2,
        "max_thickness": 4.5,
        "draft_angle_min": 1.0,
        "thickness_ratio_warn": 1.6,
        "thickness_ratio_fail": 2.2,
        "shrink_range": (0.005, 0.008),
    },
    "PA66 GF30": {
        "lt_ratio_max": 120,
        "min_thickness": 1.5,
        "max_thickness": 5.0,
        "draft_angle_min": 0.5,
        "thickness_ratio_warn": 1.8,
        "thickness_ratio_fail": 2.5,
        "shrink_range": (0.003, 0.010),
    },
    "PP": {
        "lt_ratio_max": 200,
        "min_thickness": 1.0,
        "max_thickness": 5.0,
        "draft_angle_min": 1.0,
        "thickness_ratio_warn": 2.0,
        "thickness_ratio_fail": 3.0,
        "shrink_range": (0.010, 0.020),
    },
}


def run_feasibility_check(params: dict) -> dict:
    """
    params = {
        "material": "PC+ABS",
        "min_thickness": 1.8,    # mm
        "max_thickness": 3.2,    # mm
        "flow_length": 148.0,    # mm (Max flow path)
        "avg_thickness": 2.4,    # mm
        "draft_angle": 1.5,      # degree
        "gate_count": 2,
        "undercut": False,
        "part_volume": 12.5,     # cm³
    }
    """
    mat = params.get("material", "PC+ABS")
    limits = MATERIAL_LIMITS.get(mat, MATERIAL_LIMITS["PC+ABS"])

    results = []
    overall = "PASS"

    # ── 1. L/t ratio Check ─────────────────────────
    avg_t = params.get("avg_thickness", 2.0)
    flow_l = params.get("flow_length", 100.0)
    lt_ratio = flow_l / avg_t
    lt_max = limits["lt_ratio_max"]

    if lt_ratio > lt_max:
        status = "FAIL"
        action = f"Add gates or increase thickness (Current {lt_ratio:.1f} > Limit {lt_max})"
        overall = "FAIL"
    elif lt_ratio > lt_max * 0.85:
        status = "WARN"
        action = "Short shot risk — Recommended gate position review"
        if overall != "FAIL":
            overall = "WARN"
    else:
        status = "PASS"
        action = "—"

    results.append({
        "Item": "Flow Length / Thickness Ratio",
        "Value": f"{lt_ratio:.1f}",
        "Reference": f"≤ {lt_max}",
        "Verdict": status,
        "Action": action,
    })

    # ── 2. Thickness Uniformity ────────────────────────────
    t_min = params.get("min_thickness", 1.5)
    t_max = params.get("max_thickness", 3.0)
    t_ratio = t_max / t_min if t_min > 0 else 99

    if t_ratio > limits["thickness_ratio_fail"]:
        status = "FAIL"
        action = "Excessive thickness deviation — High risk of Weld line / Sink marks"
        overall = "FAIL"
    elif t_ratio > limits["thickness_ratio_warn"]:
        status = "WARN"
        action = "Recommended to optimize thickness around ribs/bosses"
        if overall != "FAIL":
            overall = "WARN"
    else:
        status = "PASS"
        action = "—"

    results.append({
        "Item": "Thickness Uniformity (t_max / t_min)",
        "Value": f"{t_ratio:.2f}",
        "Reference": f"≤ {limits['thickness_ratio_warn']} (WARN) / {limits['thickness_ratio_fail']} (FAIL)",
        "Verdict": status,
        "Action": action,
    })

    # ── 3. Minimum Thickness Check ──────────────────────────
    t_min_limit = limits["min_thickness"]
    if t_min < t_min_limit:
        status = "FAIL"
        action = f"Min thickness {t_min}mm < Material limit {t_min_limit}mm — Filling impossible"
        overall = "FAIL"
    else:
        status = "PASS"
        action = "—"

    results.append({
        "Item": f"Minimum Thickness ({mat} Limit)",
        "Value": f"{t_min} mm",
        "Reference": f"≥ {t_min_limit} mm",
        "Verdict": status,
        "Action": action,
    })

    # ── 4. Draft Angle Check ───────────────────────────
    draft = params.get("draft_angle", 1.0)
    draft_min = limits["draft_angle_min"]

    if draft < draft_min:
        status = "FAIL"
        action = f"Draft angle {draft}° < Min {draft_min}° — Ejection failure risk"
        overall = "FAIL"
    elif draft < draft_min + 0.5:
        status = "WARN"
        action = "Potential ejection resistance — Increase draft angle recommended"
        if overall != "FAIL":
            overall = "WARN"
    else:
        status = "PASS"
        action = "—"

    results.append({
        "Item": "Draft Angle",
        "Value": f"{draft}°",
        "Reference": f"≥ {draft_min}°",
        "Verdict": status,
        "Action": action,
    })

    # ── 5. Undercut Check ────────────────────────────────
    undercut = params.get("undercut", False)
    if undercut:
        status = "WARN"
        action = "Side Core / Lifter required — Increases mold complexity and cost"
        if overall != "FAIL":
            overall = "WARN"
    else:
        status = "PASS"
        action = "—"

    results.append({
        "Item": "Undercut",
        "Value": "Detected" if undercut else "None",
        "Reference": "—",
        "Verdict": status,
        "Action": action,
    })

    # ── 6. Gate Balance Risk (Heuristic) ────────────
    gate_count = params.get("gate_count", 1)
    if gate_count == 1 and lt_ratio > 80:
        status = "WARN"
        action = "Excessive flow length for single gate — Imbalanced filling risk"
        if overall != "FAIL":
            overall = "WARN"
    else:
        status = "PASS"
        action = "—"

    results.append({
        "Item": "Gate Balance",
        "Value": f"Gates: {gate_count} / L/t: {lt_ratio:.0f}",
        "Reference": "—",
        "Verdict": status,
        "Action": action,
    })

    # ── Estimated Shrinkage Range ──────────────────────────
    shrink_lo, shrink_hi = limits["shrink_range"]

    return {
        "overall": overall,
        "material": mat,
        "lt_ratio": lt_ratio,
        "thickness_ratio": t_ratio,
        "items": results,
        "shrink_range_pct": (shrink_lo * 100, shrink_hi * 100),
        "summary": _make_summary(results, overall),
    }


def _make_summary(results, overall):
    pass_n = sum(1 for r in results if r["Verdict"] == "PASS")
    warn_n = sum(1 for r in results if r["Verdict"] == "WARN")
    fail_n = sum(1 for r in results if r["Verdict"] == "FAIL")
    return {
        "PASS": pass_n,
        "WARN": warn_n,
        "FAIL": fail_n,
        "verdict_text": {
            "PASS": "Feasible — Proceed to Stage 1 Flow Analysis",
            "WARN": "Conditionally Feasible — Review risk items before proceeding",
            "FAIL": "Not Feasible — Geometry modification required",
        }[overall],
    }


# --- Gate Design Optimization Function ───────────────────────────────────
def calculate_gate_dimensions(part_thickness: float, part_volume: float, material: str) -> dict:
    """
    Calculates optimal gate dimensions based on Stage 1 results.
    
    Inputs:
        part_thickness: Average part thickness (mm)
        part_volume: Total part volume (cm³)
        material: Material name
    
    Outputs:
        dict: Recommended gate thickness, width, area, and method
    """
    
    # Material-specific gate coefficients
    gate_coefficients = {
        "PC+ABS": {
            "thickness_factor": 0.8,  # Ratio of gate to part thickness
            "width_factor": 2.5,       # Width = Thickness * factor
            "min_area_mm2": 5.0,
            "max_area_mm2": 100.0,
        },
        "ABS": {
            "thickness_factor": 0.75,
            "width_factor": 2.4,
            "min_area_mm2": 4.5,
            "max_area_mm2": 95.0,
        },
        "PA66 GF30": {
            "thickness_factor": 0.85,
            "width_factor": 2.6,
            "min_area_mm2": 6.0,
            "max_area_mm2": 110.0,
        },
        "PP": {
            "thickness_factor": 0.7,
            "width_factor": 2.3,
            "min_area_mm2": 4.0,
            "max_area_mm2": 90.0,
        },
    }
    
    coeff = gate_coefficients.get(material, gate_coefficients["PC+ABS"])
    
    # 1. Calculate Gate Thickness
    recommended_thickness = part_thickness * coeff["thickness_factor"]
    recommended_thickness = max(0.5, min(recommended_thickness, 3.0))
    
    # 2. Calculate Gate Width
    recommended_width = recommended_thickness * coeff["width_factor"]
    
    # 3. Calculate Gate Area
    gate_area = recommended_thickness * recommended_width
    
    # 4. Volume-based adjustment
    if part_volume > 50:
        gate_area *= 1.15  # Large parts: +15%
    elif part_volume < 5:
        gate_area *= 0.9   # Small parts: -10%
    
    # 5. Apply limits
    gate_area = max(coeff["min_area_mm2"], min(gate_area, coeff["max_area_mm2"]))
    
    calculation_method = "Volume-Adjusted Empirical Method"
    
    return {
        "recommended_thickness_mm": round(recommended_thickness, 2),
        "recommended_width_mm": round(recommended_width, 2),
        "area_mm2": round(gate_area, 2),
        "calculation_method": calculation_method,
        "material": material,
        "part_volume_cm3": part_volume,
        "notes": f"Gate designed for {material} with part volume {part_volume} cm³",
    }


# --- Streamlit UI Component ───────────────────────────────────
def render_gate_design_section(avg_thickness: float, total_volume: float, selected_material: str):
    """
    Handles gate design button and result output in Streamlit app.
    """
    import streamlit as st
    
    st.markdown("### 🎯 Gate Design Optimization")
    
    if st.button("Calculate Optimal Gate Size", use_container_width=True, type="primary"):
        gate_results = calculate_gate_dimensions(
            part_thickness=avg_thickness,
            part_volume=total_volume,
            material=selected_material
        )
        
        st.success("✅ Recommended Gate Design")
        c1, c2, c3 = st.columns(3)
        c1.metric("Thickness", f"{gate_results['recommended_thickness_mm']} mm")
        c2.metric("Width", f"{gate_results['recommended_width_mm']} mm")
        c3.metric("Area", f"{gate_results['area_mm2']} mm²")
        st.caption(f"Method: {gate_results['calculation_method']}")
        
        with st.expander("📋 Gate Design Details"):
            st.write(f"**Material:** {gate_results['material']}")
            st.write(f"**Part Volume:** {gate_results['part_volume_cm3']} cm³")
            st.write(f"**Notes:** {gate_results['notes']}")
# MIM 재료 추가 (17-4PH, 316L, Ti-6Al-4V)
MATERIAL_LIMITS.update({
    "17-4PH": {
        "lt_ratio_max": 100,
        "min_thickness": 0.5,
        "max_thickness": 6.0,
        "draft_angle_min": 0.5,
        "thickness_ratio_warn": 2.0,
        "thickness_ratio_fail": 3.0,
        "shrink_range": (0.012, 0.020),
    },
    "316L": {
        "lt_ratio_max": 100,
        "min_thickness": 0.5,
        "max_thickness": 6.0,
        "draft_angle_min": 0.5,
        "thickness_ratio_warn": 2.0,
        "thickness_ratio_fail": 3.0,
        "shrink_range": (0.013, 0.021),
    },
    "Ti-6Al-4V": {
        "lt_ratio_max": 80,
        "min_thickness": 0.8,
        "max_thickness": 5.0,
        "draft_angle_min": 1.0,
        "thickness_ratio_warn": 1.8,
        "thickness_ratio_fail": 2.5,
        "shrink_range": (0.010, 0.018),
    },
})
