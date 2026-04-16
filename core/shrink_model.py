import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ── 재료 PVT 파라미터 (Tait 모델 기반 단순화) ─────────────
MATERIAL_PVT = {
    "PC+ABS": {
        "base_shrink": 0.005,          # 기준 수축률 (무차원)
        "pressure_coeff": -0.000018,   # 압력 증가 → 수축 감소
        "temp_coeff": 0.000045,        # 온도 증가 → 수축 증가
        "thickness_coeff": -0.0003,    # 두꺼울수록 수축 증가 (절대값)
        "shrink_min": 0.003,
        "shrink_max": 0.009,
        "anisotropy": 1.15,            # flow 방향 vs 수직 방향 비율
    },
    "ABS": {
        "base_shrink": 0.006,
        "pressure_coeff": -0.000015,
        "temp_coeff": 0.000050,
        "thickness_coeff": -0.00025,
        "shrink_min": 0.004,
        "shrink_max": 0.010,
        "anisotropy": 1.10,
    },
    "PA66 GF30": {
        "base_shrink": 0.005,
        "pressure_coeff": -0.000020,
        "temp_coeff": 0.000060,
        "thickness_coeff": -0.00040,
        "shrink_min": 0.002,
        "shrink_max": 0.012,
        "anisotropy": 1.40,            # GF 방향 이방성 강함
    },
    "PP": {
        "base_shrink": 0.015,
        "pressure_coeff": -0.000030,
        "temp_coeff": 0.000080,
        "thickness_coeff": -0.00060,
        "shrink_min": 0.008,
        "shrink_max": 0.025,
        "anisotropy": 1.20,
    },
}


@dataclass
class PartFeature:
    """도면의 측정 대상 Feature 정의"""
    name: str           # 예: "전체 길이 X"
    nominal: float      # 도면 공칭치수 (mm)
    tolerance_plus: float   # + 공차
    tolerance_minus: float  # - 공차 (양수값으로 입력)
    mold_dim: float         # 현재 금형 치수 (mm)
    local_pressure: float   # 해당 위치 압력 (MPa)
    local_temp: float       # 해당 위치 온도 (°C)
    local_thickness: float  # 해당 위치 두께 (mm)


def predict_shrinkage_field(
    cae_df: pd.DataFrame,
    material: str = "PC+ABS",
    avg_thickness: float = 2.5,
) -> pd.DataFrame:
    """
    CAE field 전체에 대해 위치별 수축률 예측
    반환: 원본 df + shrinkage 컬럼 추가
    """
    pvt = MATERIAL_PVT.get(material, MATERIAL_PVT["PC+ABS"])

    pressure = cae_df["pressure"].values
    temperature = cae_df["temperature"].values

    # ── Physics-based shrinkage model ──────────────
    # S(x) = S_base + α·P(x) + β·T(x) + γ·thickness + noise
    shrinkage = (
        pvt["base_shrink"]
        + pvt["pressure_coeff"] * pressure          # 압력↑ → 수축↓
        + pvt["temp_coeff"] * (temperature - 240)   # 기준온도 대비
        + pvt["thickness_coeff"] * (avg_thickness - 2.0)
    )

    # 클리핑
    shrinkage = np.clip(shrinkage, pvt["shrink_min"], pvt["shrink_max"])

    result = cae_df.copy()
    result["shrinkage"] = np.round(shrinkage, 6)
    result["shrinkage_pct"] = np.round(shrinkage * 100, 4)

    return result


def predict_part_dimensions(
    features: list[PartFeature],
    material: str = "PC+ABS",
) -> pd.DataFrame:
    """
    도면 Feature별 최종 치수 예측 + 공차 판정
    수식: L_final = L_mold × (1 - S(P, T, thickness))
    """
    pvt = MATERIAL_PVT.get(material, MATERIAL_PVT["PC+ABS"])
    rows = []

    for feat in features:
        # 위치별 수축률 계산
        s = (
            pvt["base_shrink"]
            + pvt["pressure_coeff"] * feat.local_pressure
            + pvt["temp_coeff"] * (feat.local_temp - 240)
            + pvt["thickness_coeff"] * (feat.local_thickness - 2.0)
        )
        s = float(np.clip(s, pvt["shrink_min"], pvt["shrink_max"]))

        # 최종 치수 예측
        predicted_dim = feat.mold_dim * (1 - s)
        deviation = predicted_dim - feat.nominal

        # 공차 판정
        upper = feat.nominal + feat.tolerance_plus
        lower = feat.nominal - feat.tolerance_minus

        if predicted_dim > upper:
            verdict = "OVER"
            margin = predicted_dim - upper
        elif predicted_dim < lower:
            verdict = "UNDER"
            margin = lower - predicted_dim
        else:
            verdict = "OK"
            # 공차 여유
            margin = min(upper - predicted_dim, predicted_dim - lower)

        # 공차 소진율 (0=여유 / 1=한계 / >1=초과)
        tol_range = feat.tolerance_plus + feat.tolerance_minus
        if tol_range > 0:
            consumption = abs(deviation) / tol_range * 2
        else:
            consumption = 0.0

        rows.append({
            "Feature": feat.name,
            "금형 치수 (mm)": round(feat.mold_dim, 4),
            "수축률 (%)": round(s * 100, 4),
            "예측 치수 (mm)": round(predicted_dim, 4),
            "도면 공칭 (mm)": feat.nominal,
            "공차": f"+{feat.tolerance_plus} / -{feat.tolerance_minus}",
            "편차 (mm)": round(deviation, 4),
            "판정": verdict,
            "공차 소진율 (%)": round(consumption * 100, 1),
            "_margin": margin,
            "_shrinkage": s,
            "_upper": upper,
            "_lower": lower,
        })

    return pd.DataFrame(rows)


def build_shrink_map_grid(shrink_df: pd.DataFrame, grid_size: int = 20) -> dict:
    """
    scatter 데이터를 균일 grid로 변환 (heatmap용)
    """
    if "x" not in shrink_df.columns or "shrinkage" not in shrink_df.columns:
        return {}

    x = shrink_df["x"].values
    y = shrink_df["y"].values
    s = shrink_df["shrinkage"].values

    x_bins = np.linspace(x.min(), x.max(), grid_size)
    y_bins = np.linspace(y.min(), y.max(), grid_size)

    grid = np.zeros((grid_size, grid_size))
    count = np.zeros((grid_size, grid_size))

    for xi, yi, si in zip(x, y, s):
        ix = min(int((xi - x.min()) / (x.max() - x.min() + 1e-9) * grid_size), grid_size - 1)
        iy = min(int((yi - y.min()) / (y.max() - y.min() + 1e-9) * grid_size), grid_size - 1)
        grid[iy, ix] += si
        count[iy, ix] += 1

    count[count == 0] = 1
    grid = grid / count

    return {
        "z": (grid * 100).round(4).tolist(),  # % 단위
        "x": x_bins.round(2).tolist(),
        "y": y_bins.round(2).tolist(),
        "z_min": float(s.min() * 100),
        "z_max": float(s.max() * 100),
        "z_avg": float(s.mean() * 100),
    }


def get_sample_features(material: str = "PC+ABS") -> list[PartFeature]:
    """
    예시 파트 Feature 리스트 (실제 도면 기반으로 교체)
    bracket 부품 기준
    """
    return [
        PartFeature("전체 길이 (X)", 148.00, 0.20, 0.20, 148.75, 112.0, 248.0, 2.4),
        PartFeature("폭 (Y)",         62.00, 0.15, 0.15,  62.40,  98.0, 244.0, 2.1),
        PartFeature("홀 간격 A",       32.00, 0.10, 0.10,  32.22, 125.0, 250.0, 2.0),
        PartFeature("홀 간격 B",       28.00, 0.10, 0.10,  28.18,  95.0, 243.0, 2.0),
        PartFeature("리브 두께",         2.00, 0.10, 0.10,   2.02,  88.0, 241.0, 2.0),
        PartFeature("보스 높이",        12.00, 0.15, 0.15,  12.10, 105.0, 245.0, 2.8),
        PartFeature("Flatness",         0.00, 0.15, 0.00,   0.00,   0.0,   0.0, 2.4),
    ]
# MIM 재료 추가
MATERIAL_PVT.update({
    "17-4PH": {
        "base_shrink": 0.016,
        "pressure_coeff": -0.000025,
        "temp_coeff": 0.000060,
        "thickness_coeff": -0.00050,
        "shrink_min": 0.012,
        "shrink_max": 0.022,
        "anisotropy": 1.05,
    },
    "316L": {
        "base_shrink": 0.017,
        "pressure_coeff": -0.000025,
        "temp_coeff": 0.000065,
        "thickness_coeff": -0.00050,
        "shrink_min": 0.013,
        "shrink_max": 0.023,
        "anisotropy": 1.05,
    },
    "Ti-6Al-4V": {
        "base_shrink": 0.014,
        "pressure_coeff": -0.000020,
        "temp_coeff": 0.000055,
        "thickness_coeff": -0.00045,
        "shrink_min": 0.010,
        "shrink_max": 0.020,
        "anisotropy": 1.08,
    },
})
