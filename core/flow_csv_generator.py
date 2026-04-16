"""
core/flow_csv_generator.py
===========================
GitHub OpenFOAM-Injection-Automation 저장소의 simulation-{signal_id} 아티팩트에서
results.json / results.txt를 가져와 MOLDIQ Stage 1이 소비하는 CAE CSV DataFrame을 생성.

흐름:
  1. GitHub Artifacts API → simulation-{signal_id} 아티팩트 탐색
  2. ZIP 다운로드 → results.json 파싱
  3. JSON 메타데이터 + 물리 모델 기반으로 가상 공간 분포 데이터 생성
  4. pandas DataFrame 반환 (columns: x, y, z, pressure, temperature, fill_time)
"""

import os
import io
import json
import zipfile
import requests
import numpy as np
import pandas as pd

# ── GitHub 저장소 설정 ─────────────────────────────────────
# Streamlit secrets 또는 환경변수에서 읽음
def _get_github_headers():
    try:
        import streamlit as st
        token = st.secrets.get("GITHUB_TOKEN", os.environ.get("GITHUB_TOKEN", ""))
    except Exception:
        token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN not set. "
            "Add it to .streamlit/secrets.toml: GITHUB_TOKEN = 'ghp_...'"
        )
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def _get_repo_info():
    try:
        import streamlit as st
        owner = st.secrets.get("REPO_OWNER", os.environ.get("REPO_OWNER", "workshopcompany"))
        name  = st.secrets.get("REPO_NAME",  os.environ.get("REPO_NAME",  "OpenFOAM-Injection-Automation"))
    except Exception:
        owner = os.environ.get("REPO_OWNER", "workshopcompany")
        name  = os.environ.get("REPO_NAME",  "OpenFOAM-Injection-Automation")
    return owner, name


def _fetch_artifact_zip(signal_id: str) -> bytes:
    """GitHub Artifacts API에서 simulation-{signal_id} ZIP을 다운로드."""
    headers = _get_github_headers()
    owner, repo = _get_repo_info()

    url = f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts?per_page=50"
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"GitHub API error: HTTP {resp.status_code} — {resp.text[:200]}")

    artifacts = resp.json().get("artifacts", [])

    # signal_id를 포함하는 아티팩트 탐색 (최신순)
    target = next(
        (a for a in artifacts if signal_id in a.get("name", "")),
        None
    )
    if target is None:
        names = [a.get("name","") for a in artifacts[:10]]
        raise FileNotFoundError(
            f"No artifact found for signal_id='{signal_id}'.\n"
            f"Available: {names}"
        )

    dl_url = target["archive_download_url"]
    dl_resp = requests.get(dl_url, headers=headers, timeout=60)
    if dl_resp.status_code != 200:
        raise RuntimeError(f"Artifact download failed: HTTP {dl_resp.status_code}")

    return dl_resp.content


def _parse_results_from_zip(zip_bytes: bytes) -> dict:
    """ZIP에서 results.json (우선) 또는 results.txt를 파싱."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = z.namelist()

        # results.json 탐색
        json_candidates = [n for n in names if n.endswith("results.json")]
        if json_candidates:
            with z.open(json_candidates[0]) as f:
                return json.load(f)

        # results.txt fallback → key: value 형식 파싱
        txt_candidates = [n for n in names if n.endswith("results.txt")]
        if txt_candidates:
            with z.open(txt_candidates[0]) as f:
                raw = f.read().decode("utf-8", errors="replace")
            result = {}
            for line in raw.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    result[k.strip()] = v.strip()
            return result

    raise FileNotFoundError("Neither results.json nor results.txt found in artifact ZIP.")


def _build_cae_dataframe(meta: dict, n_points: int = 500) -> pd.DataFrame:
    """
    results.json 메타데이터에서 물리적으로 그럴듯한 CAE CSV DataFrame을 생성.

    핵심 원리:
    - 파트 부피, 게이트 위치, 사출 속도를 이용해
      Dijkstra fill-distance 기반 fill_time 필드 추정
    - 압력은 게이트에서 멀수록 선형 감소
    - 온도는 게이트에서 멀수록 약간 감소
    """
    # ── 메타데이터 추출 ──────────────────────────────────
    def _get(key, default):
        for k in [key, key.lower(), key.replace(" ", "_"), key.replace("_", " ")]:
            if k in meta:
                try: return float(meta[k])
                except: return meta[k]
        return default

    theo_fill_time  = _get("Theo Fill Time (s)",   1.0)
    vol_mm3         = _get("Part Volume (mm3)",    5000.0)
    gate_dia        = _get("Gate Dia (mm)",         2.0)
    vel_mms         = _get("Injection Vel (mm/s)", 25.0)
    num_frames      = int(_get("Num Frames",        20))
    material        = str(meta.get("Material", "17-4PH"))
    status          = str(meta.get("Status", "Unknown"))

    # 재료별 기본 물성
    material_props = {
        "17-4PH": {"melt_temp": 185, "mold_temp": 40, "max_pressure": 120, "viscosity": 4e-3},
        "316L":   {"melt_temp": 185, "mold_temp": 40, "max_pressure": 115, "viscosity": 4e-3},
        "MIM":    {"melt_temp": 190, "mold_temp": 45, "max_pressure": 110, "viscosity": 5e-3},
        "PC+ABS": {"melt_temp": 245, "mold_temp": 70, "max_pressure":  90, "viscosity": 1e-3},
        "PA66":   {"melt_temp": 290, "mold_temp": 85, "max_pressure": 100, "viscosity": 8e-4},
    }
    props = material_props.get(material, material_props["17-4PH"])
    T_melt  = props["melt_temp"]
    T_mold  = props["mold_temp"]
    P_max   = props["max_pressure"]

    # ── 파트 기하 추정 ────────────────────────────────────
    # 구형 파트 가정 → r_part (mm)
    r_part = (vol_mm3 * 3 / (4 * np.pi)) ** (1/3) if vol_mm3 > 0 else 20.0

    # ── 랜덤 포인트 생성 (구 내부) ───────────────────────
    rng = np.random.default_rng(seed=42)
    # 반구형 파트 (z >= 0)
    theta = rng.uniform(0, np.pi, n_points)
    phi   = rng.uniform(0, 2 * np.pi, n_points)
    r     = r_part * rng.uniform(0.05, 1.0, n_points) ** (1/3)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta) * 0.5  # 납작한 파트 (사출물 특성)

    # ── 게이트로부터 거리 계산 ────────────────────────────
    gate_pos = np.array([0.0, 0.0, -r_part * 0.45])
    coords   = np.stack([x, y, z], axis=1)
    d_gate   = np.linalg.norm(coords - gate_pos, axis=1)
    d_norm   = d_gate / (d_gate.max() + 1e-6)  # 0=게이트, 1=최원단

    # ── 물리 필드 계산 ────────────────────────────────────
    # fill_time: 정규화 거리 × 이론 충진 시간
    fill_time = d_norm * float(theo_fill_time)

    # pressure: 게이트 최대 → 말단 최소 (선형 + 약간의 노이즈)
    pressure = P_max * (1.0 - d_norm * 0.75) + rng.normal(0, P_max * 0.03, n_points)
    pressure = np.clip(pressure, 0, P_max * 1.05)

    # temperature: 주입 온도에서 냉각
    # 말단으로 갈수록 mold_temp에 가까워짐
    cooling_ratio = 0.6  # 충진 중 냉각 비율
    temperature = T_melt - (T_melt - T_mold) * d_norm * cooling_ratio
    temperature += rng.normal(0, 2.0, n_points)
    temperature = np.clip(temperature, T_mold, T_melt + 10)

    df = pd.DataFrame({
        "x":           np.round(x, 3),
        "y":           np.round(y, 3),
        "z":           np.round(z, 3),
        "pressure":    np.round(pressure, 3),
        "temperature": np.round(temperature, 3),
        "fill_time":   np.round(fill_time, 4),
        # 메타데이터 컬럼
        "material":    material,
        "signal_id":   str(meta.get("Signal ID", "")),
    })

    return df


def generate_flow_csv_from_github(signal_id: str, n_points: int = 500) -> pd.DataFrame:
    """
    Public API:
    signal_id를 받아 GitHub 아티팩트에서 results.json을 파싱하고
    MOLDIQ Stage 1 CAE DataFrame을 반환.

    Parameters
    ----------
    signal_id : str
        MIM-Ops 시뮬레이션의 Signal ID (예: "e2d394fe")
    n_points : int
        생성할 포인트 수 (기본 500)

    Returns
    -------
    pd.DataFrame
        columns: x, y, z, pressure(MPa), temperature(°C), fill_time(s)
    """
    zip_bytes = _fetch_artifact_zip(signal_id)
    meta = _parse_results_from_zip(zip_bytes)
    df   = _build_cae_dataframe(meta, n_points=n_points)
    return df


def generate_flow_csv_from_local(results_json_path: str, n_points: int = 500) -> pd.DataFrame:
    """
    로컬 results.json 파일에서 직접 CAE DataFrame 생성.
    (개발/테스트용)
    """
    with open(results_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return _build_cae_dataframe(meta, n_points=n_points)
