"""
core/flow_csv_generator.py
===========================
GitHub OpenFOAM-Injection-Automation 저장소 아티팩트에서
results.json / results.txt를 가져와 MOLDIQ Stage 1용 CAE CSV DataFrame 생성.

변경 이력:
  v2 — 아티팩트 탐색 로직 완전 재작성
       - 다양한 signal_id 입력 형식 모두 허용
       - 진단 정보(실제 아티팩트 목록) 에러에 포함
       - 만료된 아티팩트 제외
       - REPO_NAME 환경변수 우선, 하드코딩 fallback
"""

import os
import io
import json
import zipfile
import requests
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════
#  설정 헬퍼
# ══════════════════════════════════════════════════════════

def _get_token() -> str:
    """GITHUB_TOKEN을 secrets 또는 환경변수에서 읽기."""
    token = ""
    try:
        import streamlit as st
        token = st.secrets.get("GITHUB_TOKEN", "")
    except Exception:
        pass
    if not token:
        token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError(
            "❌ GITHUB_TOKEN이 설정되지 않았습니다.\n"
            "Streamlit Cloud → 앱 Settings → Secrets 탭에서 아래를 추가하세요:\n\n"
            "  GITHUB_TOKEN = \"ghp_xxxxxxxxxxxx\"\n\n"
            "토큰에는 repo 권한이 필요합니다."
        )
    return token


def _get_repo_info() -> tuple[str, str]:
    """저장소 owner/name을 secrets 또는 환경변수에서 읽기."""
    owner = ""
    name  = ""
    try:
        import streamlit as st
        owner = st.secrets.get("OPENFOAM_REPO_OWNER", st.secrets.get("REPO_OWNER", ""))
        name  = st.secrets.get("OPENFOAM_REPO_NAME",  st.secrets.get("REPO_NAME",  ""))
    except Exception:
        pass
    if not owner:
        owner = os.environ.get("OPENFOAM_REPO_OWNER",
                os.environ.get("REPO_OWNER", "workshopcompany"))
    if not name:
        name = os.environ.get("OPENFOAM_REPO_NAME",
               os.environ.get("REPO_NAME",  "OpenFOAM-Injection-Automation"))
    return owner, name


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


# ══════════════════════════════════════════════════════════
#  아티팩트 목록 조회 (진단용 public 함수)
# ══════════════════════════════════════════════════════════

def list_artifacts(per_page: int = 30) -> list[dict]:
    """
    저장소의 아티팩트 목록 반환 (만료되지 않은 것만).
    Streamlit UI에서 진단/디버그용으로 직접 호출 가능.

    Returns
    -------
    list of dict with keys: id, name, created_at, size_in_bytes, expired
    """
    owner, repo = _get_repo_info()
    url = (
        f"https://api.github.com/repos/{owner}/{repo}"
        f"/actions/artifacts?per_page={per_page}"
    )
    resp = requests.get(url, headers=_headers(), timeout=15)

    if resp.status_code == 401:
        raise RuntimeError(
            "❌ GitHub 인증 실패 (HTTP 401).\n"
            "GITHUB_TOKEN이 만료되었거나 repo 권한이 없습니다.\n"
            "새 토큰을 발급하고 Streamlit Secrets를 업데이트하세요."
        )
    if resp.status_code == 404:
        raise RuntimeError(
            f"❌ 저장소를 찾을 수 없습니다 (HTTP 404).\n"
            f"현재 설정: {owner}/{repo}\n"
            "Secrets의 OPENFOAM_REPO_OWNER / OPENFOAM_REPO_NAME 을 확인하세요."
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"❌ GitHub API 오류: HTTP {resp.status_code}\n{resp.text[:300]}"
        )

    all_artifacts = resp.json().get("artifacts", [])
    # 만료되지 않은 것만 반환
    return [a for a in all_artifacts if not a.get("expired", False)]


# ══════════════════════════════════════════════════════════
#  signal_id → 아티팩트 매칭
# ══════════════════════════════════════════════════════════

def _normalize_signal_id(raw: str) -> str:
    """
    사용자가 입력한 다양한 형식을 정규화.
    - "simulation-47664275" → "47664275"
    - "47664275"            → "47664275"
    - "b4727fd"             → "b4727fd"   (workflow run sha)
    - " 47664275 "          → "47664275"  (공백 제거)
    """
    s = raw.strip()
    # "simulation-" 접두어가 있으면 제거
    if s.startswith("simulation-"):
        s = s[len("simulation-"):]
    return s


def _find_artifact(signal_id: str, artifacts: list[dict]) -> dict | None:
    """
    여러 매칭 전략으로 아티팩트를 탐색. 최신 것 우선.

    전략 1: 아티팩트 name이 signal_id를 포함
    전략 2: signal_id가 아티팩트 name을 포함 (역방향)
    전략 3: "simulation" 이름을 가진 것 중 가장 최신
    """
    sig = signal_id.lower()

    # 전략 1
    for a in artifacts:
        if sig in a.get("name", "").lower():
            return a

    # 전략 2
    for a in artifacts:
        if a.get("name", "").lower() in sig:
            return a

    # 전략 3: 입력값이 없거나 "latest"이면 simulation 이름 최신것
    if not sig or sig == "latest":
        for a in artifacts:
            if "simulation" in a.get("name", "").lower():
                return a

    return None


# ══════════════════════════════════════════════════════════
#  ZIP 다운로드 & 파싱
# ══════════════════════════════════════════════════════════

def _download_artifact_zip(artifact: dict) -> bytes:
    """아티팩트 ZIP 다운로드."""
    dl_url = artifact["archive_download_url"]
    resp = requests.get(dl_url, headers=_headers(), timeout=60, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(
            f"❌ 아티팩트 다운로드 실패: HTTP {resp.status_code}\n"
            f"아티팩트: {artifact.get('name')}"
        )
    return resp.content


def _parse_results_from_zip(zip_bytes: bytes) -> dict:
    """
    ZIP에서 results.json (우선) 또는 results.txt 파싱.
    results.json 안에 voxel_coords / flow_weights 배열이 있으면 함께 반환.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = z.namelist()

        # results.json 탐색 (경로 무관)
        json_files = [n for n in names if n.endswith("results.json")]
        if json_files:
            with z.open(json_files[0]) as f:
                return json.load(f)

        # results.txt fallback (voxel 데이터 없음)
        txt_files = [n for n in names if n.endswith("results.txt")]
        if txt_files:
            with z.open(txt_files[0]) as f:
                raw = f.read().decode("utf-8", errors="replace")
            result = {}
            for line in raw.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    result[k.strip()] = v.strip()
            return result

        raise FileNotFoundError(
            f"❌ ZIP 안에 results.json / results.txt 가 없습니다.\n"
            f"포함된 파일: {names[:15]}"
        )


# ══════════════════════════════════════════════════════════
#  CAE DataFrame 생성
# ══════════════════════════════════════════════════════════

def _build_cae_from_voxels(meta: dict) -> pd.DataFrame:
    """
    results.json에 voxel_coords + flow_weights 배열이 있을 때
    실제 솔버 좌표 그대로 DataFrame 생성 (복셀화/재계산 없음).
    """
    coords  = np.array(meta["voxel_coords"],  dtype=np.float32)  # (N,3)
    weights = np.array(meta["flow_weights"],  dtype=np.float32)  # (N,)

    def _val(key, default):
        for k in [key, key.lower(), key.replace(" ", "_"), key.replace("_", " ")]:
            if k in meta:
                try:
                    return float(meta[k])
                except (ValueError, TypeError):
                    pass
        return default

    theo_fill_time = _val("Theo Fill Time (s)", 1.0)
    material       = str(meta.get("Material", "17-4PH"))

    _props = {
        "17-4PH":   {"melt_temp": 185, "mold_temp": 40, "max_pressure": 120},
        "316L":     {"melt_temp": 185, "mold_temp": 40, "max_pressure": 115},
        "Ti-6Al-4V":{"melt_temp": 195, "mold_temp": 45, "max_pressure": 130},
        "PC+ABS":   {"melt_temp": 245, "mold_temp": 70, "max_pressure":  90},
        "PA66":     {"melt_temp": 290, "mold_temp": 85, "max_pressure": 100},
        "ABS":      {"melt_temp": 230, "mold_temp": 60, "max_pressure":  85},
    }
    props  = _props.get(material, _props["17-4PH"])
    T_melt = props["melt_temp"]
    T_mold = props["mold_temp"]
    P_max  = props["max_pressure"]

    # flow_weights: 0=gate, 1=최원단 → 물리량 역산
    w = weights
    fill_time   = w * theo_fill_time
    pressure    = np.clip(P_max * (1.0 - w * 0.75), 0, P_max * 1.05)
    temperature = np.clip(T_melt - (T_melt - T_mold) * w * 0.6, T_mold, T_melt + 10)

    return pd.DataFrame({
        "x":           np.round(coords[:, 0], 3),
        "y":           np.round(coords[:, 1], 3),
        "z":           np.round(coords[:, 2], 3),
        "pressure":    np.round(pressure,    3),
        "temperature": np.round(temperature, 3),
        "fill_time":   np.round(fill_time,   4),
        "flow_weight": np.round(w,           4),   # 원본 Dijkstra 가중치 보존
        "material":    material,
        "signal_id":   str(meta.get("Signal ID", "")),
    })

def _build_cae_dataframe(meta: dict, n_points: int = 500) -> pd.DataFrame:
    """
    results.json 메타데이터 → 물리 기반 CAE CSV DataFrame.

    압력: 게이트에서 멀수록 선형 감소
    온도: 게이트에서 멀수록 mold_temp에 수렴
    fill_time: 정규화 거리 × 이론 충진 시간
    """
    def _val(key, default):
        for k in [key, key.lower(), key.replace(" ", "_"), key.replace("_", " ")]:
            if k in meta:
                try:
                    return float(meta[k])
                except (ValueError, TypeError):
                    return meta[k]
        return default

    theo_fill_time = _val("Theo Fill Time (s)",   1.0)
    vol_mm3        = _val("Part Volume (mm3)",    5000.0)
    gate_dia       = _val("Gate Dia (mm)",         2.0)
    vel_mms        = _val("Injection Vel (mm/s)", 25.0)
    material       = str(meta.get("Material", "17-4PH"))

    # 재료별 물성
    _props = {
        "17-4PH":  {"melt_temp": 185, "mold_temp": 40, "max_pressure": 120},
        "316L":    {"melt_temp": 185, "mold_temp": 40, "max_pressure": 115},
        "Ti-6Al-4V":{"melt_temp": 195,"mold_temp": 45, "max_pressure": 130},
        "PC+ABS":  {"melt_temp": 245, "mold_temp": 70, "max_pressure":  90},
        "PA66":    {"melt_temp": 290, "mold_temp": 85, "max_pressure": 100},
        "ABS":     {"melt_temp": 230, "mold_temp": 60, "max_pressure":  85},
    }
    props  = _props.get(material, _props["17-4PH"])
    T_melt = props["melt_temp"]
    T_mold = props["mold_temp"]
    P_max  = props["max_pressure"]

    r_part = max((vol_mm3 * 3 / (4 * np.pi)) ** (1 / 3), 5.0)

    rng   = np.random.default_rng(seed=42)
    theta = rng.uniform(0, np.pi,     n_points)
    phi   = rng.uniform(0, 2 * np.pi, n_points)
    r     = r_part * rng.uniform(0.05, 1.0, n_points) ** (1 / 3)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta) * 0.5

    gate_pos = np.array([0.0, 0.0, -r_part * 0.45])
    coords   = np.stack([x, y, z], axis=1)
    d_gate   = np.linalg.norm(coords - gate_pos, axis=1)
    d_norm   = d_gate / (d_gate.max() + 1e-6)

    fill_time   = d_norm * float(theo_fill_time)
    pressure    = P_max * (1.0 - d_norm * 0.75) + rng.normal(0, P_max * 0.03, n_points)
    pressure    = np.clip(pressure, 0, P_max * 1.05)
    temperature = T_melt - (T_melt - T_mold) * d_norm * 0.6
    temperature += rng.normal(0, 2.0, n_points)
    temperature = np.clip(temperature, T_mold, T_melt + 10)

    return pd.DataFrame({
        "x":           np.round(x, 3),
        "y":           np.round(y, 3),
        "z":           np.round(z, 3),
        "pressure":    np.round(pressure,    3),
        "temperature": np.round(temperature, 3),
        "fill_time":   np.round(fill_time,   4),
        "material":    material,
        "signal_id":   str(meta.get("Signal ID", "")),
    })


# ══════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════

def generate_flow_csv_from_github(
    signal_id: str,
    n_points: int = 500,
) -> pd.DataFrame:
    """
    signal_id를 받아 GitHub 아티팩트에서 CAE DataFrame 반환.

    Parameters
    ----------
    signal_id : str
        다음 형식 모두 허용:
        - "47664275"              (숫자 ID만)
        - "simulation-47664275"  (아티팩트 전체 이름)
        - "latest"               (가장 최근 simulation 아티팩트)
    n_points : int
        생성할 공간 포인트 수 (기본 500)

    Returns
    -------
    pd.DataFrame  columns: x, y, z, pressure(MPa), temperature(°C), fill_time(s)
    """
    normalized = _normalize_signal_id(signal_id)

    # 1. 아티팩트 목록 조회
    artifacts = list_artifacts(per_page=50)

    # 2. 매칭
    target = _find_artifact(normalized, artifacts)
    if target is None:
        available = [a.get("name", "") for a in artifacts[:15]]
        raise FileNotFoundError(
            f"❌ signal_id='{signal_id}' 에 해당하는 아티팩트를 찾지 못했습니다.\n\n"
            f"입력 형식 예시:\n"
            f"  • 숫자만:  47664275\n"
            f"  • 전체명:  simulation-47664275\n"
            f"  • 최신:    latest\n\n"
            f"현재 저장소에 있는 아티팩트 ({len(artifacts)}개):\n"
            + "\n".join(f"  • {n}" for n in available)
        )

    # 3. 다운로드 & 파싱
    zip_bytes = _download_artifact_zip(target)
    meta      = _parse_results_from_zip(zip_bytes)

    # 4. voxel_coords + flow_weights 있으면 실제 솔버 좌표 사용
    #    없으면 기존 방식(메타데이터 기반 근사 생성) fallback
    if "voxel_coords" in meta and "flow_weights" in meta:
        df = _build_cae_from_voxels(meta)
    else:
        df = _build_cae_dataframe(meta, n_points=n_points)

    return df


def generate_flow_csv_from_local(results_json_path: str, n_points: int = 500) -> pd.DataFrame:
    """로컬 results.json 파일에서 직접 CAE DataFrame 생성 (개발/테스트용)."""
    with open(results_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return _build_cae_dataframe(meta, n_points=n_points)
