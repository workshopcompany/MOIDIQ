import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata # 상단에 추가 필요

# ── 재료별 공정 기준 ──────────────────────────────────────
PROCESS_LIMITS = {
    "PC+ABS": {
        "melt_temp": (230, 260),      # °C
        "mold_temp": (60, 80),        # °C
        "inj_speed": (30, 80),        # mm/s
        "pack_pressure": (70, 100),   # MPa
        "pack_time": (5, 15),         # sec
        "max_shear_rate": 50000,      # 1/s
        "max_pressure": 160,          # MPa
        "freeze_temp": 125,           # °C (Tg 근처)
    },
    "ABS": {
        "melt_temp": (210, 250),
        "mold_temp": (50, 80),
        "inj_speed": (25, 70),
        "pack_pressure": (60, 90),
        "pack_time": (5, 12),
        "max_shear_rate": 40000,
        "max_pressure": 140,
        "freeze_temp": 100,
    },
    "PA66 GF30": {
        "melt_temp": (270, 300),
        "mold_temp": (70, 90),
        "inj_speed": (20, 60),
        "pack_pressure": (80, 120),
        "pack_time": (8, 20),
        "max_shear_rate": 30000,
        "max_pressure": 180,
        "freeze_temp": 220,
    },
    "PP": {
        "melt_temp": (200, 240),
        "mold_temp": (20, 60),
        "inj_speed": (40, 100),
        "pack_pressure": (50, 80),
        "pack_time": (4, 10),
        "max_shear_rate": 60000,
        "max_pressure": 120,
        "freeze_temp": 130,
    },
}


def load_cae_data(file_or_path) -> pd.DataFrame:
    """
    CSV 컬럼 예시:
      x, y, pressure, temperature, fill_time, velocity_x, velocity_y
    x,y : 그리드 좌표 (mm)
    pressure : 압력 (MPa)
    temperature : 온도 (°C)
    fill_time : 충진 시간 (s)
    """
    if hasattr(file_or_path, "read"):
        df = pd.read_csv(file_or_path)
    else:
        df = pd.read_csv(file_or_path)

    required = ["x", "y", "pressure", "temperature", "fill_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    return df


def analyze_cae(df: pd.DataFrame, material: str = "PC+ABS") -> dict:
    """
    CAE 결과 전체 분석 → 결함 위험도 + 최적 조건 반환
    """
    limits = PROCESS_LIMITS.get(material, PROCESS_LIMITS["PC+ABS"])

    pressure = df["pressure"].values
    temperature = df["temperature"].values
    fill_time = df["fill_time"].values

    # ── 기본 통계 ──────────────────────────────────
    stats = {
        "max_pressure_MPa": float(pressure.max()),
        "avg_pressure_MPa": float(pressure.mean()),
        "max_temperature_C": float(temperature.max()),
        "avg_temperature_C": float(temperature.mean()),
        "fill_time_s": float(fill_time.max()),
        "pressure_gradient": float(_calc_gradient(df)),
    }

    # ── 결함 위험 점수 계산 ────────────────────────
    defect_risks = _score_defect_risks(df, limits, stats)

    # ── 최적 조건 도출 ─────────────────────────────
    optimal = _derive_optimal_conditions(stats, limits, defect_risks)

    # ── 그리드 맵 생성 (시각화용) ──────────────────
    grid_maps = _build_grid_maps(df)

    return {
        "stats": stats,
        "defect_risks": defect_risks,
        "optimal_conditions": optimal,
        "grid_maps": grid_maps,
        "material": material,
    }


def _calc_gradient(df: pd.DataFrame) -> float:
    """압력 기울기 추정 (gate → end of fill)"""
    if "x" not in df.columns:
        return 0.0
    sorted_df = df.sort_values("x")
    if len(sorted_df) < 2:
        return 0.0
    p_range = sorted_df["pressure"].max() - sorted_df["pressure"].min()
    x_range = sorted_df["x"].max() - sorted_df["x"].min()
    return float(p_range / x_range) if x_range > 0 else 0.0


def _score_defect_risks(df, limits, stats) -> dict:
    """
    각 결함 위험도를 0~1 스코어로 변환
    0 = 안전 / 1 = 매우 위험
    """
    risks = {}

    # Short shot 위험 (압력 부족)
    p_max = stats["max_pressure_MPa"]
    p_limit = limits["max_pressure"]
    if p_max < p_limit * 0.4:
        risks["short_shot"] = {"score": 0.8, "level": "HIGH",
                               "detail": f"최대 압력 {p_max:.0f} MPa — 충진 부족 가능"}
    elif p_max < p_limit * 0.6:
        risks["short_shot"] = {"score": 0.4, "level": "MED",
                               "detail": f"충진 경계 — 사출속도 증가 검토"}
    else:
        risks["short_shot"] = {"score": 0.1, "level": "LOW", "detail": "정상"}

    # Weld line 위험 (충진 끝단 온도 저하 추정)
    temp_std = float(df["temperature"].std())
    if temp_std > 20:
        risks["weld_line"] = {"score": 0.7, "level": "HIGH",
                              "detail": f"온도 편차 {temp_std:.1f}°C — Weld line 발생 예상"}
    elif temp_std > 10:
        risks["weld_line"] = {"score": 0.4, "level": "MED",
                              "detail": "온도 편차 주의 — 게이트 위치 재검토"}
    else:
        risks["weld_line"] = {"score": 0.1, "level": "LOW", "detail": "정상"}

    # Sink mark 위험 (고압 + 두꺼운 단면 추정)
    p_avg = stats["avg_pressure_MPa"]
    if p_avg < p_limit * 0.3:
        risks["sink_mark"] = {"score": 0.6, "level": "HIGH",
                              "detail": "보압 부족 가능 — Sink 위험"}
    else:
        risks["sink_mark"] = {"score": 0.2, "level": "LOW", "detail": "정상"}

    # Warpage 위험 (온도 불균일)
    t_max = stats["max_temperature_C"]
    t_avg = stats["avg_temperature_C"]
    if (t_max - t_avg) > 30:
        risks["warpage"] = {"score": 0.7, "level": "HIGH",
                            "detail": f"Hot spot {t_max:.0f}°C — 불균일 냉각으로 Warpage 위험"}
    elif (t_max - t_avg) > 15:
        risks["warpage"] = {"score": 0.4, "level": "MED",
                            "detail": "냉각 채널 위치 검토 권장"}
    else:
        risks["warpage"] = {"score": 0.1, "level": "LOW", "detail": "정상"}

    # Air trap (충진 끝단 판정 — 단순 추정)
    fill_uniformity = float(df["fill_time"].std() / df["fill_time"].mean()) if df["fill_time"].mean() > 0 else 0
    if fill_uniformity > 0.4:
        risks["air_trap"] = {"score": 0.6, "level": "HIGH",
                             "detail": "충진 불균일 — Air trap 위험 구간 존재"}
    else:
        risks["air_trap"] = {"score": 0.2, "level": "LOW", "detail": "정상"}

    return risks


def _derive_optimal_conditions(stats, limits, defect_risks) -> dict:
    """
    결함 위험 스코어 + 재료 한계 기반으로 최적 공정 조건 도출
    Rule-based weighted optimization
    """
    # 기준: 각 파라미터를 중간값에서 시작해 위험 보정
    melt_lo, melt_hi = limits["melt_temp"]
    mold_lo, mold_hi = limits["mold_temp"]
    spd_lo, spd_hi = limits["inj_speed"]
    pack_lo, pack_hi = limits["pack_pressure"]
    ptime_lo, ptime_hi = limits["pack_time"]

    # Warpage 위험 높으면 → 금형 온도 낮추고 pack time 늘림
    warpage_score = defect_risks.get("warpage", {}).get("score", 0)
    sink_score = defect_risks.get("sink_mark", {}).get("score", 0)
    short_score = defect_risks.get("short_shot", {}).get("score", 0)

    # 최적 melt temp: warpage 위험 낮추려면 낮게
    melt_opt = melt_hi - (melt_hi - melt_lo) * warpage_score * 0.5
    mold_opt = mold_lo + (mold_hi - mold_lo) * 0.4  # 중하단

    # 사출속도: short shot 위험 있으면 올림
    spd_opt = spd_lo + (spd_hi - spd_lo) * (0.5 + short_score * 0.3)

    # 보압: sink 위험 있으면 올림
    pack_opt = pack_lo + (pack_hi - pack_lo) * (0.5 + sink_score * 0.4)

    # 보압 시간: warpage/sink 위험 시 늘림
    ptime_opt = ptime_lo + (ptime_hi - ptime_lo) * (0.4 + max(warpage_score, sink_score) * 0.4)

    return {
        "melt_temperature": {
            "range": limits["melt_temp"],
            "optimal": round(melt_opt, 1),
            "unit": "°C",
        },
        "mold_temperature": {
            "range": limits["mold_temp"],
            "optimal": round(mold_opt, 1),
            "unit": "°C",
        },
        "injection_speed": {
            "range": limits["inj_speed"],
            "optimal": round(spd_opt, 1),
            "unit": "mm/s",
        },
        "packing_pressure": {
            "range": limits["pack_pressure"],
            "optimal": round(pack_opt, 1),
            "unit": "MPa",
        },
        "packing_time": {
            "range": limits["pack_time"],
            "optimal": round(ptime_opt, 1),
            "unit": "sec",
        },
    }


def _build_grid_maps(df: pd.DataFrame) -> dict:
    """
    x,y 좌표 기반으로 매끄러운 2D grid 맵 생성 (보간법 적용)
    """
    if "x" not in df.columns or "y" not in df.columns:
        return {}

    try:
        # 1. 그리드 생성 범위 설정 (실제 데이터보다 약간 여유 있게)
        points = df[['x', 'y']].values
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        
        # 해상도 설정 (예: 100x100 그리드)
        grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

        maps = {}
        for col in ["pressure", "temperature", "fill_time"]:
            if col not in df.columns:
                continue
            
            values = df[col].values
            
            # 2. 선형 보간(Linear Interpolation)으로 빈틈 메우기
            # 데이터가 없는 지점(0)을 억지로 만드는 대신 주변 값을 참조해 계산합니다.
            grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
            
            # 3. 보간 후 끝부분의 NaN 값을 가장 가까운 값으로 채움
            grid_z = pd.DataFrame(grid_z).fillna(method='ffill').fillna(method='bfill').values
            
            # 4. 시각화를 위해 다시 부드럽게 스무딩
            grid_z = gaussian_filter(grid_z.astype(float), sigma=1.0)
            
            # Plotly Heatmap은 [y, x] 구조를 기대하므로 T(전치) 처리
            maps[col] = {
                "z": grid_z.T.tolist(), 
                "x": np.linspace(x_min, x_max, 100).tolist(),
                "y": np.linspace(y_min, y_max, 100).tolist(),
            }
        return maps
    except Exception as e:
        print(f"Grid mapping error: {e}")
        return {}


def generate_sample_cae_csv(n_points: int = 200, material: str = "PC+ABS") -> pd.DataFrame:
    """
    실제 CAE 없을 때 사용할 샘플 데이터 생성
    물리적으로 그럴듯한 pressure/temperature 분포 시뮬레이션
    """
    np.random.seed(42)
    limits = PROCESS_LIMITS.get(material, PROCESS_LIMITS["PC+ABS"])

    # 그리드 좌표 (100mm x 60mm 부품 가정)
    x = np.random.uniform(0, 100, n_points)
    y = np.random.uniform(0, 60, n_points)

    # Gate at (0, 30) 가정
    gate_x, gate_y = 0, 30
    dist_from_gate = np.sqrt((x - gate_x) ** 2 + (y - gate_y) ** 2)
    max_dist = dist_from_gate.max()

    # Pressure: gate에서 멀수록 낮아짐 + 노이즈
    p_max = limits["max_pressure"] * 0.85
    pressure = p_max * (1 - dist_from_gate / max_dist * 0.75) + np.random.normal(0, 3, n_points)
    pressure = np.clip(pressure, 5, limits["max_pressure"])

    # Temperature: gate 근처 높고, 끝단 낮음 + hot spot
    t_lo, t_hi = limits["melt_temp"]
    base_temp = (t_lo + t_hi) / 2
    temperature = base_temp - dist_from_gate / max_dist * 25 + np.random.normal(0, 5, n_points)
    # Hot spot 추가 (좌상단)
    hot_mask = (x > 70) & (y > 45)
    temperature[hot_mask] += 15
    temperature = np.clip(temperature, limits["mold_temp"][0] + 20, t_hi + 10)

    # Fill time: 거리 비례 + 두께 변동
    fill_time = dist_from_gate / max_dist * 1.8 + np.random.normal(0, 0.05, n_points)
    fill_time = np.clip(fill_time, 0.01, 2.5)

    df = pd.DataFrame({
        "x": np.round(x, 2),
        "y": np.round(y, 2),
        "pressure": np.round(pressure, 2),
        "temperature": np.round(temperature, 2),
        "fill_time": np.round(fill_time, 3),
    })

    return df
# MIM 재료 추가
PROCESS_LIMITS.update({
    "17-4PH": {
        "melt_temp": (175, 200),
        "mold_temp": (30, 50),
        "inj_speed": (15, 50),
        "pack_pressure": (80, 140),
        "pack_time": (5, 20),
        "max_shear_rate": 20000,
        "max_pressure": 150,
        "freeze_temp": 120,
    },
    "316L": {
        "melt_temp": (175, 200),
        "mold_temp": (30, 50),
        "inj_speed": (15, 50),
        "pack_pressure": (80, 130),
        "pack_time": (5, 20),
        "max_shear_rate": 18000,
        "max_pressure": 140,
        "freeze_temp": 120,
    },
    "Ti-6Al-4V": {
        "melt_temp": (180, 210),
        "mold_temp": (35, 55),
        "inj_speed": (10, 40),
        "pack_pressure": (90, 150),
        "pack_time": (8, 25),
        "max_shear_rate": 15000,
        "max_pressure": 160,
        "freeze_temp": 130,
    },
})
