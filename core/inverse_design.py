"""
Stage 3: Inverse Design Engine — 공차 기반 역설계
목적: 예측 치수 편차를 공차 범위 내에서만 선형 보정 제안
제약: 비선형/비대칭 보정 ❌ / 전체 scale 변경 ❌ / Linear correction only ✔
수식: E(x) = L_target(x) - L_predicted(x)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class CorrectionItem:
    """보정 항목 1개"""
    id: str                  # C-01, C-02 ...
    priority: str            # HIGH / MED / LOW
    correction_type: str     # uniform_offset / local_linear / feature_level / process
    feature_name: str
    error_mm: float          # E(x) = target - predicted
    correction_mm: float     # 제안 보정량 (양수: 금형 크게, 음수: 작게)
    mold_current: float      # 현재 금형 치수
    mold_corrected: float    # 보정 후 금형 치수
    tolerance_plus: float
    tolerance_minus: float
    within_tolerance: bool   # 보정 후 공차 내 들어오는지
    note: str


def run_inverse_design(
    dim_df: pd.DataFrame,
    global_shrink_avg: float,
) -> dict:
    """
    치수 예측 결과 DataFrame → 역설계 보정 제안 생성

    입력:
      dim_df : predict_part_dimensions() 결과
      global_shrink_avg : 전체 평균 수축률 (소수, 예: 0.0055)
    """

    corrections = []
    post_correction = []
    c_id = 1

    # ── STEP 1: 전체 균일 보상 계산 ──────────────────
    # 전체적으로 수축률 평균 기반 금형 scale-up
    uniform_comp_pct = global_shrink_avg * 100

    corrections.append(CorrectionItem(
        id=f"C-{c_id:02d}",
        priority="MED",
        correction_type="uniform_offset",
        feature_name="전체 균일 수축 보상",
        error_mm=0.0,
        correction_mm=0.0,
        mold_current=0.0,
        mold_corrected=0.0,
        tolerance_plus=0.0,
        tolerance_minus=0.0,
        within_tolerance=True,
        note=f"전체 금형 +{uniform_comp_pct:.3f}% 확대 (평균 수축률 보상)",
    ))
    c_id += 1

    # ── STEP 2: Feature별 보정 ────────────────────────
    for _, row in dim_df.iterrows():
        verdict = row["판정"]
        if verdict == "OK" and row["공차 소진율 (%)"] < 60:
            continue  # 여유 충분 → 보정 불필요

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

        # 보정 가능 여부: |E(x)| ≤ tolerance 범위 내에서만
        max_allow = tol_p + tol_m  # 전체 공차 폭
        correctable = abs(e_x) <= max_allow * 2.0  # 2배까지는 보정 시도

        if not correctable:
            priority = "HIGH"
            note = "공차 초과 범위가 너무 큼 — 형상 재설계 필요"
            within_tol = False
        elif verdict in ("UNDER", "OVER"):
            priority = "HIGH"
            note = _make_correction_note(feature, e_x, verdict)
            within_tol = True
        else:
            priority = "MED"
            note = f"공차 소진율 {row['공차 소진율 (%)']}% — 선제 보정 권장"
            within_tol = True

        # 금형 보정 치수: 수축률 역산
        # L_mold_new = L_nominal / (1 - S)  ← 정방향 역산
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

        # 보정 후 예측 결과
        new_predicted = mold_corrected * (1 - shrinkage)
        new_error = new_predicted - nominal
        new_verdict = (
            "OK" if abs(new_error) <= min(tol_p, tol_m) * 0.95
            else ("주의" if abs(new_error) <= max(tol_p, tol_m) else "초과")
        )
        post_correction.append({
            "Feature": feature,
            "보정 전 편차": round(error, 4),
            "보정 후 편차": round(new_error, 4),
            "공차": f"+{tol_p} / -{tol_m}",
            "결과": new_verdict,
        })

    # ── STEP 3: 공정 파라미터 보정 (금형 미수정 대안) ──
    # 보압 증가 → 수축률 감소 효과
    residual_items = [c for c in corrections if c.correction_type == "local_linear"
                      and not c.within_tolerance]
    if residual_items:
        corrections.append(CorrectionItem(
            id=f"C-{c_id:02d}",
            priority="LOW",
            correction_type="process",
            feature_name="보압 (Packing Pressure) 조정",
            error_mm=0.0,
            correction_mm=0.0,
            mold_current=0.0,
            mold_corrected=0.0,
            tolerance_plus=0.0,
            tolerance_minus=0.0,
            within_tolerance=True,
            note="보압 +5~10 MPa 증가 시 수축률 약 0.05~0.1% 감소 효과 (금형 미수정 대안)",
        ))

    # ── 비용 절감 추정 ────────────────────────────────
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
    direction = "금형 확대" if error > 0 else "금형 축소"
    return f"{feature}: {direction} {abs(error):.4f}mm 필요 ({verdict} 보정)"


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
            "✅ 전체 공차 내 보정 가능" if high == 0
            else f"⚠ {high}개 항목 우선 보정 필요"
        ),
    }


def build_error_map(shrink_df: pd.DataFrame, features_df: pd.DataFrame) -> dict:
    """
    E(x) map 생성 — 위치별 오차 분포 (heatmap용)
    shrink_df: 위치별 수축률 포함
    """
    if "x" not in shrink_df.columns:
        return {}

    x = shrink_df["x"].values
    y = shrink_df["y"].values
    s = shrink_df["shrinkage"].values

    # 단순화: 오차 = 위치별 수축률과 평균의 차이 (실제는 도면 대비)
    s_mean = s.mean()
    error = (s - s_mean) * 100  # % 단위 편차

    return {
        "x": x.round(2).tolist(),
        "y": y.round(2).tolist(),
        "error": error.round(4).tolist(),
        "error_min": float(error.min()),
        "error_max": float(error.max()),
    }
