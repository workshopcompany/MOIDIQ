"""
MOLDIQ — MIM Injection Molding Design Decision Platform
=======================================================
Stage Flow:
  Mold Concept → 0.Feasibility Gate → 1.Flow Analysis
              → 2.Dimension Prediction → 3.Inverse Correction
"""

import streamlit as st
import os, sys, json, re, io, zipfile, struct, base64, zlib
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests


# ══════════════════════════════════════════════════════════
#  VTK/VTU 파서 — pyvista 없이 stdlib만 사용
#  지원 형식:
#   A) ASCII VTU (format="ascii")  ← 이전 OpenFOAM 결과
#   B) Appended+Base64+ZLib VTU    ← MIM-Ops GitHub Actions 결과
#   C) 위 파일이 담긴 ZIP
#  NumPy 2.0 완전 호환 (ptp() 미사용)
# ══════════════════════════════════════════════════════════

_VTK_DTYPE = {
    "Float32": np.float32, "Float64": np.float64,
    "Int8": np.int8,  "Int16": np.int16,
    "Int32": np.int32, "Int64": np.int64,
    "UInt8": np.uint8, "UInt16": np.uint16,
    "UInt32": np.uint32, "UInt64": np.uint64,
}

def _vtk_clean_xml(text: str) -> str:
    """싱글/더블 쿼트 xmlns 속성 모두 제거 (ET 파싱 전처리)."""
    return re.sub(r""" xmlns(?::[a-zA-Z0-9_]+)?=['\"][^'\"]*['\"]""", "", text)

# ── A) ASCII DataArray 파서 ───────────────────────────────
def _read_dataarray_ascii(node: ET.Element) -> np.ndarray:
    """ASCII format DataArray → float64 ndarray."""
    text_data = (node.text or "").strip()
    if not text_data:
        return np.array([], dtype=np.float64)
    try:
        return np.array(text_data.split(), dtype=np.float64)
    except ValueError:
        return np.array([], dtype=np.float64)

# ── B) Appended+Base64+ZLib 파서 ─────────────────────────
def _parse_appended_block(b64_stream: bytes, char_offset: int,
                           header_type: str = "UInt32") -> bytes:
    """
    VTK appended base64+zlib 블록을 압축 해제해 raw bytes 반환.

    VTK 파일 구조:
      <AppendedData encoding="base64">
        _[헤더서브블록==[데이터서브블록][헤더==[데이터]][...]
      </AppendedData>

    각 DataArray는 b64_stream 내 offset 위치에:
      - 헤더 서브블록 (== 패딩 포함): n_blocks, uncomp_sz, last_sz, comp_sizes[n]
      - 데이터 서브블록: zlib 압축 청크들

    헤더와 데이터가 같은 서브블록에 있는 경우(offset 0 등)도 처리.
    """
    hsz  = 4 if header_type == "UInt32" else 8
    hfmt = "<I" if header_type == "UInt32" else "<Q"

    # 헤더 서브블록: char_offset 부터 다음 '==' 까지
    eq_pos = b64_stream.find(b"==", char_offset)
    if eq_pos == -1:
        raise ValueError(f"헤더 블록 끝(==)을 찾을 수 없음 (offset={char_offset})")
    hdr_end = eq_pos + 2
    hdr_raw = base64.b64decode(b64_stream[char_offset:hdr_end])

    # 헤더 파싱
    n_blocks   = struct.unpack_from(hfmt, hdr_raw, 0)[0]
    comp_sizes = [struct.unpack_from(hfmt, hdr_raw, (3 + i) * hsz)[0]
                  for i in range(n_blocks)]
    total_comp = sum(comp_sizes)
    hbytes     = (3 + n_blocks) * hsz

    # 데이터 위치 결정
    if len(hdr_raw) > hbytes:
        # 헤더와 데이터가 같은 서브블록 (offset=0 패턴)
        data_bytes = hdr_raw[hbytes:]
    else:
        # 데이터가 바로 다음 서브블록
        data_start  = hdr_end
        needed_chars = ((total_comp + 2) // 3) * 4
        chunk = b64_stream[data_start:data_start + needed_chars]
        pad   = (4 - len(chunk) % 4) % 4
        data_bytes = base64.b64decode(chunk + b"=" * pad)

    # zlib 해제
    pos, out = 0, b""
    for cs in comp_sizes:
        out += zlib.decompress(data_bytes[pos:pos + cs])
        pos += cs
    return out


def _parse_vtu_appended(raw_bytes: bytes, material: str,
                         rho: float) -> pd.DataFrame:
    """Appended+Base64+ZLib VTU → DataFrame."""
    text = raw_bytes.decode("utf-8", errors="replace")

    # AppendedData 스트림 추출
    app_start  = raw_bytes.find(b"<AppendedData")
    section    = raw_bytes[app_start:]
    underscore = section.find(b"_")
    end_tag    = section.find(b"<", underscore)
    b64_stream = section[underscore + 1:end_tag].strip()

    # header_type / mesh 크기
    ht_m   = re.search(r'header_type=["\'](\w+)["\']', text)
    htype  = ht_m.group(1) if ht_m else "UInt32"
    npts_m = re.search(r'NumberOfPoints=["\'](\d+)["\']', text)
    ncls_m = re.search(r'NumberOfCells=["\'](\d+)["\']', text)
    n_pts  = int(npts_m.group(1)) if npts_m else 0
    n_cells= int(ncls_m.group(1)) if ncls_m else 0

    dtmap = {"Float32": "<f4", "Float64": "<f8",
             "Int64": "<i8", "Int32": "<i4", "UInt8": "u1"}

    # DataArray 메타데이터 수집
    das = {}
    for m in re.finditer(r"<DataArray([^/]+?)/?>\s*", text):
        tag = m.group(1)
        nm  = re.search(r'Name=["\']([^"\']+)', tag)
        off = re.search(r'offset=["\'](\d+)',   tag)
        tp  = re.search(r' type=["\'](\w+)',    tag)
        nc  = re.search(r'NumberOfComponents=["\'](\d+)', tag)
        if nm and off:
            das[nm.group(1)] = {
                "offset": int(off.group(1)),
                "type":   tp.group(1) if tp else "Float32",
                "nc":     int(nc.group(1)) if nc else 1,
            }

    # 각 DataArray 해제
    arrays = {}
    for name, info in das.items():
        if name in ("connectivity", "offsets", "types"):
            continue  # 위상 데이터는 시각화 불필요
        try:
            raw_out = _parse_appended_block(b64_stream, info["offset"], htype)
            dtype   = dtmap.get(info["type"], "<f4")
            arr     = np.frombuffer(raw_out, dtype=dtype).astype(np.float64)
            nc      = info["nc"]
            if nc == 3:
                arr3 = arr.reshape(-1, 3)
                if name == "Points":
                    arrays["_pts"] = arr3
                else:
                    arrays[name + "_mag"] = np.linalg.norm(arr3, axis=1)
            else:
                arrays[name] = arr
        except Exception:
            pass

    # 좌표 처리
    pts = arrays.pop("_pts", None)
    if pts is None:
        raise ValueError("VTU: Points 데이터를 파싱할 수 없습니다.")

    # CellData vs PointData 판별 및 매핑
    # pts: (n_pts, 3), celldata: (n_cells,)
    # pts_per_cell = n_pts / n_cells (정수배인 경우 셀 중심점 계산)
    result = {}
    pts_per_cell = round(n_pts / n_cells) if n_cells > 0 else 1
    use_cells = (n_cells > 0 and pts_per_cell >= 1
                 and abs(pts_per_cell * n_cells - n_pts) < n_cells)

    if use_cells:
        # 셀 중심점 (포인트를 셀 단위로 그룹화)
        n_use      = n_cells * pts_per_cell
        centroids  = pts[:n_use].reshape(n_cells, pts_per_cell, 3).mean(axis=1)
        result.update({"x": centroids[:, 0],
                       "y": centroids[:, 1],
                       "z": centroids[:, 2]})
        n_ref = n_cells
    else:
        result.update({"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]})
        n_ref = n_pts

    for name, arr in arrays.items():
        trimmed = arr[:n_ref]
        if len(trimmed) == n_ref:
            result[name] = trimmed

    df = pd.DataFrame(result)

    # ── 단위 변환 ──────────────────────────────────────────
    # flow_distance [0=gate, 1=end] → 압력·충진시간·온도 근사
    if "flow_distance" in df.columns:
        fd = df["flow_distance"].to_numpy(float)
        if "pressure" not in df.columns:
            df["pressure"]    = np.clip((1.0 - fd) * 3.0, 0.0, None)  # gate 최대압
        if "fill_time" not in df.columns:
            df["fill_time"]   = fd
        if "temperature" not in df.columns:
            df["temperature"] = 40.0 + (1.0 - fd) * 145.0   # gate=185°C, end=40°C
    elif "p" in df.columns:
        pv = np.abs(df["p"].to_numpy(float) * rho / 1e6)
        df["pressure"] = pv

    if "pressure" not in df.columns:
        df["pressure"] = 0.0
    if "temperature" not in df.columns:
        df["temperature"] = 100.0
    if "fill_time" not in df.columns:
        pv    = df["pressure"].to_numpy(float)
        pspan = float(pv.max() - pv.min())
        df["fill_time"] = (pv.max() - pv) / pspan if pspan > 1e-9 \
                          else np.linspace(0.0, 1.0, len(df))

    df["material"] = material
    return df


# ── A) ASCII VTU 파서 ────────────────────────────────────
def _parse_vtu_ascii(raw_bytes: bytes, material: str,
                      rho: float) -> pd.DataFrame:
    """ASCII format VTU → DataFrame."""
    text = _vtk_clean_xml(raw_bytes.decode("utf-8", errors="replace"))
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise ValueError(f"VTU XML 파싱 실패: {e}")

    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError("VTU: <Piece> 태그 없음")
    n_pts = int(piece.attrib.get("NumberOfPoints", 0))
    if n_pts == 0:
        raise ValueError("VTU: NumberOfPoints=0")

    pts_node = piece.find(".//Points/DataArray")
    if pts_node is None:
        raise ValueError("VTU: Points/DataArray 없음")
    pts_flat = _read_dataarray_ascii(pts_node)
    if len(pts_flat) < n_pts * 3:
        raise ValueError(
            f"VTU: 좌표 데이터 부족 ({len(pts_flat)} < {n_pts * 3})")
    coords = pts_flat[:n_pts * 3].reshape(-1, 3)
    result: dict = {
        "x": coords[:, 0].astype(float),
        "y": coords[:, 1].astype(float),
        "z": coords[:, 2].astype(float),
    }

    for da in piece.findall(".//PointData/DataArray"):
        name = da.attrib.get("Name", "").strip()
        nc   = int(da.attrib.get("NumberOfComponents", "1"))
        arr  = _read_dataarray_ascii(da)
        try:
            if nc == 1 and len(arr) >= n_pts:
                result[name] = arr[:n_pts].astype(float)
            elif nc == 3 and len(arr) >= n_pts * 3:
                mat3 = arr[:n_pts * 3].reshape(-1, 3).astype(float)
                result[f"{name}x"]    = mat3[:, 0]
                result[f"{name}y"]    = mat3[:, 1]
                result[f"{name}z"]    = mat3[:, 2]
                result[f"{name}_mag"] = np.linalg.norm(mat3, axis=1)
        except Exception:
            pass

    df = pd.DataFrame(result)

    if "p" in df.columns:
        p_kin  = df["p"].to_numpy(float)
        p_pa   = p_kin * rho
        p_mpa  = p_pa / 1e6
        span   = float(p_mpa.max()) - float(p_mpa.min())
        if span < 1e-4:
            p_mpa = p_pa / 1e3
        df["pressure"] = np.abs(p_mpa)
    elif "pressure" not in df.columns:
        df["pressure"] = 0.0

    if "T" in df.columns:
        df["temperature"] = df["T"].to_numpy(float)
    elif "temperature" not in df.columns:
        if "U_mag" in df.columns:
            uv    = df["U_mag"].to_numpy(float)
            denom = float(uv.max() - uv.min())
            df["temperature"] = 40.0 + (uv - uv.min()) / (denom or 1.0) * 145.0
        else:
            df["temperature"] = 100.0

    if "fill_time" not in df.columns:
        pv    = df["pressure"].to_numpy(float)
        pspan = float(pv.max() - pv.min())
        df["fill_time"] = (pv.max() - pv) / pspan if pspan > 1e-9 \
                          else np.linspace(0.0, 1.0, len(df))

    df["material"] = material
    return df


# ── 통합 파서 (외부에서 호출하는 메인 함수) ──────────────
def parse_vtu_to_dataframe(file_bytes: bytes, material: str = "17-4PH",
                            rho: float = 7780.0) -> pd.DataFrame:
    """
    VTU 파일 자동 감지 파서.
    - Appended+Base64+ZLib (MIM-Ops GitHub Actions 출력) → _parse_vtu_appended
    - ASCII format (foamToVTK -ascii 출력)               → _parse_vtu_ascii
    """
    # Appended 방식 감지
    if b"AppendedData" in file_bytes:
        try:
            return _parse_vtu_appended(file_bytes, material=material, rho=rho)
        except Exception as e:
            raise ValueError(f"VTU(Appended) 파싱 실패: {e}")
    else:
        try:
            return _parse_vtu_ascii(file_bytes, material=material, rho=rho)
        except Exception as e:
            raise ValueError(f"VTU(ASCII) 파싱 실패: {e}")


def parse_vtk_zip_to_dataframe(zip_bytes: bytes, material: str = "17-4PH",
                                rho: float = 7780.0) -> pd.DataFrame:
    """
    ZIP 파일 → VTU 파싱.
    우선순위: .vtm DataSet 참조 → 'internal' 포함 .vtu → 가장 큰 .vtu
    """
    buf = io.BytesIO(zip_bytes)
    if not zipfile.is_zipfile(buf):
        raise ValueError("유효한 ZIP 파일이 아닙니다.")

    with zipfile.ZipFile(buf) as z:
        names = z.namelist()

        # 전략 1: .vtm → DataSet file= 추적
        vtu_candidate = None
        for vtm_path in [n for n in names if n.lower().endswith(".vtm")]:
            try:
                vtm_text = _vtk_clean_xml(
                    z.read(vtm_path).decode("utf-8", errors="replace"))
                vtm_root = ET.fromstring(vtm_text)
                for ds in vtm_root.findall(".//DataSet"):
                    ref = ds.attrib.get("file", "")
                    if "internal" in ref.lower() and ref.lower().endswith(".vtu"):
                        base_dir = os.path.dirname(vtm_path)
                        for n in names:
                            if n.endswith(os.path.basename(ref)):
                                vtu_candidate = n; break
            except Exception:
                continue
            if vtu_candidate:
                break

        # 전략 2: internal 포함 .vtu
        all_vtu = [n for n in names if n.lower().endswith(".vtu")]
        if not vtu_candidate:
            internals = [n for n in all_vtu if "internal" in n.lower()]
            vtu_candidate = sorted(internals)[-1] if internals else None

        # 전략 3: 가장 큰 .vtu
        if not vtu_candidate and all_vtu:
            vtu_candidate = max(all_vtu, key=lambda n: z.getinfo(n).file_size)

        if not vtu_candidate:
            raise FileNotFoundError(
                "ZIP 내에 .vtu 파일이 없습니다.\n"
                "foamToVTK 실행 후 VTK/ 폴더를 ZIP 압축해 업로드하세요.")

        vtu_bytes = z.read(vtu_candidate)

    try:
        df = parse_vtu_to_dataframe(vtu_bytes, material=material, rho=rho)
        df.attrs["source_file"] = vtu_candidate
        return df
    except ValueError as e:
        raise ValueError(
            f"{os.path.basename(vtu_candidate)}: {e}\n\n"
            "📌 해결: internal.vtu 파일만 단독으로 업로드해 보세요.")

# ── 경로 설정 ──────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ── 페이지 설정 (가장 먼저) ───────────────────────────────
st.set_page_config(
    page_title="MOLDIQ — Smart MIM Design System",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 모듈 임포트 ───────────────────────────────────────────
try:
    from core.i18n import TRANSLATIONS
    from core.rule_check import run_feasibility_check, MATERIAL_LIMITS
    from core.cae_analyzer import analyze_cae, load_cae_data, generate_sample_cae_csv, PROCESS_LIMITS
    from core.shrink_model import (
        predict_shrinkage_field, predict_part_dimensions,
        build_shrink_map_grid, get_sample_features, MATERIAL_PVT
    )
    from core.inverse_design import run_inverse_design, build_error_map
    from core.model_processor import process_uploaded_model
    from core.parting_line_analyzer import analyze_parting_line
    from core.slide_core_optimizer import optimize_mold_design
    from core.flow_csv_generator import generate_flow_csv_from_github
    try:
        from core.ml_feedback import train_or_update_model, apply_ml_correction
        ML_AVAILABLE = True
    except ImportError:
        ML_AVAILABLE = False
    try:
        from core.drawing_sync import load_drawing_features_from_csv, generate_cad_macro_script
    except ImportError:
        pass
    MODULES_OK = True
except ImportError as e:
    st.error(f"⚠️ Module load failed: {e}")
    st.stop()

# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono&family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.stage-tag {
    font-family: 'Space Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.2em; color: #00d4aa; text-transform: uppercase; margin-bottom: 4px;
}
.stage-title { font-size: 1.4rem; font-weight: 700; color: #e2e8f0; margin-bottom: 4px; }
.stage-desc  { font-size: 0.82rem; color: #8899aa; }
.verdict-pass { background: rgba(0,212,170,0.08); border: 1px solid rgba(0,212,170,0.3);
    border-radius: 8px; padding: 12px 16px; margin: 8px 0; }
.verdict-warn { background: rgba(255,107,53,0.08); border: 1px solid rgba(255,107,53,0.3);
    border-radius: 8px; padding: 12px 16px; margin: 8px 0; }
.verdict-fail { background: rgba(255,59,92,0.08); border: 1px solid rgba(255,59,92,0.3);
    border-radius: 8px; padding: 12px 16px; margin: 8px 0; }
.mono { font-family: 'Space Mono', monospace; font-size: 0.78rem; }
.info-box { background: #111318; border: 1px solid #252b36; border-radius: 6px;
    padding: 10px 14px; margin: 6px 0; font-size: 0.8rem; color: #8899aa; }
.link-card {
    background: #111827; border: 1px solid #1e3a5f; border-radius: 10px;
    padding: 16px 20px; margin: 8px 0; transition: border-color 0.2s;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  Session State 초기화
# ══════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "stage0_done": False, "stage1_done": False,
        "stage2_done": False, "stage3_done": False,
        "feasibility_result": None,
        "cae_df": None, "cae_analysis": None,
        "shrink_df": None, "dim_df": None,
        "inverse_result": None,
        "material": "PC+ABS", "avg_thickness": 2.4,
        # Mold Concept → Feasibility 연동 데이터
        "stl_analysis": {
            "file_loaded": False, "geometry": None,
            "undercuts": None, "parting": None,
            "design": None, "pull_direction": "Z"
        },
        # Stage 1 GitHub 연동
        "github_sim_signal_id": None,
        "flow_csv_ready": False,
        # [FIX-2] Mold Concept에서 업로드한 STL 전역 유지
        "stl_bytes": None,
        "stl_name": None,
        "uploaded_stl_path": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ══════════════════════════════════════════════════════════
#  사이드바
# ══════════════════════════════════════════════════════════
with st.sidebar:
    # 언어 설정
    st.markdown("### 🌐 Language Settings")
    lang = st.selectbox("Select Language",
        ["English", "한국어", "Español", "日本語", "中文", "Français", "हिन्दी"],
        index=0, key="lang_select")
    T = TRANSLATIONS.get(lang, TRANSLATIONS["English"])

    st.markdown("## 🔩 MOLDIQ")
    st.markdown(f'<div class="mono" style="color:#55667a;font-size:0.65rem;">{T["platform_desc"]}</div>',
                unsafe_allow_html=True)
    st.divider()

    # 워크플로우 단계 선택
    stage_labels = {
        "🎯 Mold Concept (Main)": "mold_concept",
        T["stage0_label"]: "stage0",
        T["stage1_label"]: "stage1",
        T["stage2_label"]: "stage2",
        T["stage3_label"]: "stage3",
    }
    selected_stage = st.radio("workflow_stage", list(stage_labels.keys()), key="stage_select")
    current_stage = stage_labels[selected_stage]

    st.divider()

    # 전역 설정
    st.markdown(f"#### {T['global_settings']}")
    material = st.selectbox(T["material"], list(MATERIAL_LIMITS.keys()), index=0, key="material_select")
    st.session_state["material"] = material
    avg_thickness = st.number_input(T["avg_thickness"], 0.5, 10.0, 2.4, 0.1)
    st.session_state["avg_thickness"] = avg_thickness

    st.divider()

    # 진행 상태
    st.markdown(f"#### {T['status']}")
    for label, done_key in [
        (T["stage0_label"], "stage0_done"),
        (T["stage1_label"], "stage1_done"),
        (T["stage2_label"], "stage2_done"),
        (T["stage3_label"], "stage3_done"),
    ]:
        icon = "✅" if st.session_state[done_key] else "⬜"
        st.markdown(f"{icon} {label}")

    st.divider()

    # ML 피드백
    st.markdown(f"### {T['ml_feedback']}")
    real_csv = st.file_uploader(T["upload_actual"], type=["csv"])
    if real_csv and st.button(T.get("btn_retrain", "Retrain XGBoost Model")):
        if ML_AVAILABLE and "dim_df" in st.session_state and st.session_state["dim_df"] is not None:
            success, msg = train_or_update_model(real_csv, st.session_state["dim_df"])
            st.success(msg) if success else st.error(msg)
        else:
            st.warning(T.get("warn_run_st2_first", "Please run Stage 2 first."))

# ── 언어 객체 (사이드바 밖에서도 사용) ───────────────────
T = TRANSLATIONS.get(st.session_state.get("lang_select", "English"), TRANSLATIONS["English"])

# ══════════════════════════════════════════════════════════
#  공통 유틸
# ══════════════════════════════════════════════════════════
def verdict_color(v):
    if v in ("PASS", "OK"): return "🟢"
    if v in ("WARN", "Attention"): return "🟡"
    return "🔴"

def style_verdict_df(df, verdict_col="판정"):
    def color_row(row):
        v = row.get(verdict_col, "")
        if v in ("PASS", "OK"): return ["background-color: rgba(0,212,170,0.08)"] * len(row)
        if v in ("WARN", "주의", "OVER", "UNDER"): return ["background-color: rgba(255,107,53,0.08)"] * len(row)
        if v == "FAIL": return ["background-color: rgba(255,59,92,0.08)"] * len(row)
        return [""] * len(row)
    return df.style.apply(color_row, axis=1)


# ── [FIX] GitHub 연결 상태 확인 / 안내 헬퍼 ─────────────────
def _check_github_secrets() -> bool:
    """Streamlit secrets에 GITHUB_TOKEN이 유효하게 설정됐는지 확인."""
    try:
        token = st.secrets["GITHUB_TOKEN"]
        return bool(token and str(token).strip() and str(token) != "ghp_xxxxxxxxxxxx")
    except (KeyError, FileNotFoundError, Exception):
        return False


def _show_github_token_guide():
    """GitHub Token 미설정 시 단계별 안내 UI 표시."""
    st.error("🔑 **GITHUB_TOKEN이 설정되지 않았습니다.**")
    with st.expander("📋 설정 방법 — 클릭해서 펼치기", expanded=True):
        st.markdown("""
**Streamlit Cloud를 사용하는 경우:**
1. [share.streamlit.io](https://share.streamlit.io) → 앱 → ⋯ 메뉴 → **Edit secrets**
2. 아래 내용을 붙여넣고 토큰값만 교체:

```toml
GITHUB_TOKEN        = "ghp_YOUR_TOKEN_HERE"
OPENFOAM_REPO_OWNER = "workshopcompany"
OPENFOAM_REPO_NAME  = "OpenFOAM-Injection-Automation"
```

**로컬 실행의 경우:**
프로젝트 루트 `.streamlit/secrets.toml` 파일을 위 내용으로 생성.

**GitHub Token 발급:**
GitHub → Settings → Developer settings → Personal access tokens → Generate new token (classic)
→ `repo` 권한 체크 → 생성된 토큰 복사

---
💡 **GitHub 없이 바로 사용:** 아래 **Option C** (VTK 파일 직접 업로드)로 결과를 로드하세요.
        """)


# ══════════════════════════════════════════════════════════
#  PHASE: Mold Concept (기본 페이지 — STL 분석)
# ══════════════════════════════════════════════════════════
if current_stage == "mold_concept":
    st.markdown('<div class="stage-tag">STAGE 0</div>', unsafe_allow_html=True)
    st.markdown('<div class="stage-title">🏭 Mold Design Feasibility</div>', unsafe_allow_html=True)
    st.markdown('<div class="stage-desc">STL Model Upload → Automated Mold Design Proposals</div>',
                unsafe_allow_html=True)
    st.info("📦 Upload an STL file to automatically generate proposals for parting lines, slides, and cores.")

    tab_upload, tab_geometry, tab_parting, tab_design, tab_summary = st.tabs([
        "📤 STL Upload", "📊 Geometry Analysis", "📍 Parting Line", "🔧 Slide & Core", "📋 Summary"
    ])

    analysis = st.session_state.stl_analysis

    # ── Tab 1: Upload ─────────────────────────────────────
    with tab_upload:
        st.markdown("## 📤 Upload STL File")
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader("Select STL File", type=["stl", "STL"])
        with col2:
            st.write(""); st.write("")
            st.markdown("**Pull Direction:**")
            pull_direction = st.selectbox("Pull Direction", ["Z", "X", "Y"],
                                          key="pull_dir_select", help="Z = Top-Bottom (standard)")
            st.session_state.stl_analysis["pull_direction"] = pull_direction

        if uploaded_file is not None:
            import tempfile
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, uploaded_file.name)
            stl_raw = uploaded_file.getvalue()
            with open(file_path, "wb") as f:
                f.write(stl_raw)
            # [FIX-2] 전역 session_state에 저장 → Flow Analysis 탭에서 재업로드 불필요
            st.session_state["stl_bytes"] = stl_raw
            st.session_state["stl_name"]  = uploaded_file.name
            st.session_state["uploaded_stl_path"] = file_path
            st.session_state["_stl_mesh_cache"] = None  # 캐시 초기화
            st.success(f"✅ Upload successful: {uploaded_file.name}")

            if st.button("🔍 Run Analysis", type="primary", use_container_width=True):
                with st.spinner("⏳ Analyzing model..."):
                    try:
                        model_result = process_uploaded_model(file_path, pull_direction)
                        if "error" in model_result:
                            st.error(f"❌ {model_result['error']}")
                        else:
                            parting_result = analyze_parting_line(model_result["geometry"], pull_direction)
                            design_result  = optimize_mold_design(
                                model_result["geometry"], model_result["undercut_regions"], parting_result
                            )
                            st.session_state.stl_analysis.update({
                                "file_loaded": True,
                                "geometry": model_result["geometry"],
                                "undercuts": model_result["undercut_regions"],
                                "parting": parting_result,
                                "design": design_result,
                            })
                            # ── Feasibility Gate 자동 연동 ──────────────────
                            geo = model_result["geometry"]
                            _bounds = geo.get("bounds", {})
                            _sx = _bounds.get("size_x", 0)
                            _sy = _bounds.get("size_y", 0)
                            _sz = _bounds.get("size_z", 0)
                            # flow_length ≈ 대각선 최대 길이
                            _flow_len = float(np.sqrt(_sx**2 + _sy**2 + _sz**2)) * 0.5
                            st.session_state["stl_derived"] = {
                                "min_thickness": round(_sz * 0.08, 2) if _sz > 0 else 1.8,
                                "max_thickness": round(_sz * 0.25, 2) if _sz > 0 else 3.2,
                                "flow_length": round(_flow_len, 1),
                                "part_volume": round(geo.get("volume_cm3", 12.5), 2),
                                "undercut": len(model_result["undercut_regions"]) > 0,
                            }
                            st.success("✅ Analysis Complete! Geometry parameters synced to Feasibility Gate.")
                            st.balloons()
                    except ImportError as e:
                        st.error(f"⚠️ Module not found: {e}")
                    except Exception as e:
                        st.error(f"❌ Analysis Error: {e}")

    # ── Tab 2: Geometry ───────────────────────────────────
    with tab_geometry:
        st.markdown("## 📊 Geometry Analysis")
        if analysis["file_loaded"] and analysis["geometry"]:
            geo = analysis["geometry"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🔹 Volume",       f"{geo['volume_cm3']:.1f} cm³")
            c2.metric("🔹 Surface Area", f"{geo['surface_area_mm2']:.0f} mm²")
            c3.metric("🔹 Vertices",     f"{geo['vertices_count']:,}")
            c4.metric("🔹 Triangles",    f"{geo['triangles_count']:,}")
            st.divider()
            st.markdown("### 📦 Bounding Box")
            b = geo["bounds"]
            st.dataframe({
                "Axis": ["X","Y","Z"],
                "Min": [f"{b['min_x']:.2f} mm", f"{b['min_y']:.2f} mm", f"{b['min_z']:.2f} mm"],
                "Max": [f"{b['max_x']:.2f} mm", f"{b['max_y']:.2f} mm", f"{b['max_z']:.2f} mm"],
                "Size":[f"{b['size_x']:.2f} mm", f"{b['size_y']:.2f} mm", f"{b['size_z']:.2f} mm"],
            }, use_container_width=True, hide_index=True)
            st.markdown("### 🎯 Centroid")
            c = geo["centroid"]
            co1, co2, co3 = st.columns(3)
            co1.metric("X", f"{c['x']:.2f} mm")
            co2.metric("Y", f"{c['y']:.2f} mm")
            co3.metric("Z", f"{c['z']:.2f} mm")
        else:
            st.info("💡 Please upload and analyze an STL file first.")

    # ── Tab 3: Parting Line ───────────────────────────────
    with tab_parting:
        st.markdown("## 📍 Parting Line Recommendation")
        if analysis["file_loaded"] and analysis["parting"]:
            parting = analysis["parting"]
            rec = parting["recommendation"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Type", rec["recommended_parting_line"].split("(")[0].strip())
            c2.metric("Complexity", rec["complexity"])
            c3.metric("Est. Depth", f"{rec['estimated_depth']:.1f} mm")
            c4.metric("Confidence", rec["confidence"].split("(")[0].strip())
            st.divider()
            st.markdown("### 📊 All Options Comparison")
            opts = parting["parting_analysis"]["all_options"]
            st.dataframe([{
                "Parting Line": o["name"],
                "Complexity": o["complexity_level"],
                "Score": f"{o['complexity_score']:.1f}",
                "Est. Area": f"{o['estimated_area_mm2']:.0f} mm²",
            } for o in opts], use_container_width=True, hide_index=True)
            st.divider()
            st.markdown("### ⚠️ Flash Risk")
            flash = parting["flash_risk_assessment"]
            emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            st.markdown(f"{emoji[flash['flash_risk_level']]} **{flash['flash_risk_level']}** (Score: {flash['flash_risk_score']})")
            st.write(f"**Recommendation:** {flash['recommendation']}")
            with st.expander("🛡️ Mitigation Measures"):
                for m in flash["mitigation_measures"]: st.write(f"• {m}")
        else:
            st.info("💡 Please upload and analyze an STL file first.")

    # ── Tab 4: Slide & Core ───────────────────────────────
    with tab_design:
        st.markdown("## 🔧 Slide & Core Design")
        if analysis["file_loaded"] and analysis["design"]:
            design = analysis["design"]
            us = design["undercut_summary"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Undercut", "Detected" if us.get("has_undercut") else "None",
                      f"{us.get('total_undercut_regions', 0)} Regions")
            c2.metric("Slide", "Required" if us["slide_required"] else "Not Required")
            c3.metric("Core",  "Required" if us["core_required"]  else "Not Required")
            st.divider()
            if design["slide_design"]["count"] > 0:
                st.markdown(f"### 🎯 Slides ({design['slide_design']['count']} units)")
                st.dataframe([{
                    "ID": s["slide_id"], "Severity": s["severity"],
                    "Thickness": f"{s['estimated_dimensions']['thickness_mm']} mm",
                    "Width":     f"{s['estimated_dimensions']['width_mm']} mm",
                    "Length":    f"{s['estimated_dimensions']['length_mm']} mm",
                    "Material":  s["material_grade"],
                    "Cost":      f"${int(s['estimated_cost_usd'])}",
                } for s in design["slide_design"]["slides"]], use_container_width=True, hide_index=True)
            else:
                st.info("✓ No slides required.")
            st.divider()
            if design["core_design"]["count"] > 0:
                st.markdown(f"### 🔨 Cores ({design['core_design']['count']} units)")
                st.dataframe([{
                    "ID": c["core_id"], "Type": c["type"],
                    "Diameter": f"{c['estimated_dimensions']['diameter_mm']} mm",
                    "Length":   f"{c['estimated_dimensions']['length_mm']} mm",
                    "Material": c["material_grade"],
                    "Cost":     f"${int(c['estimated_cost_usd'])}",
                } for c in design["core_design"]["cores"]], use_container_width=True, hide_index=True)
            else:
                st.info("✓ No additional cores required.")
        else:
            st.info("💡 Please upload and analyze an STL file first.")

    # ── Tab 5: Summary ────────────────────────────────────
    with tab_summary:
        st.markdown("## 📋 Complete Design Summary")
        if analysis["file_loaded"] and analysis["design"]:
            design = analysis["design"]
            for rec in design["recommendations"]: st.write(f"• {rec}")
            st.divider()
            view_mode = st.radio("Display", ["Analysis Summary", "Export Data", "Next Steps"],
                                 horizontal=True, key="summary_view_mode")
            if view_mode == "Analysis Summary":
                cx = design["complexity_assessment"]
                cost = design["cost_estimate"]
                lead = design["lead_time"]
                co1, co2 = st.columns(2)
                with co1:
                    st.markdown("#### 🏗️ Design Complexity")
                    st.metric("Level", cx["level"])
                    st.metric("Mechanisms", f"{cx['total_mechanisms']} units")
                    st.metric("Score", f"{cx['score']}/4")
                with co2:
                    st.markdown("#### 💰 Cost & Lead Time")
                    st.metric("Total Cost", f"${int(cost['total_mechanism_cost_usd'])}")
                    st.metric("Impact", f"{cost['percentage_of_mold']}%")
                    st.metric("Critical Path", f"{lead['critical_path_days']} Days")
            elif view_mode == "Export Data":
                import json as _json
                json_str = _json.dumps(design, indent=2, default=str)
                st.download_button("📄 Download JSON", json_str, "mold_design_report.json", "application/json",
                                   use_container_width=True)
            else:
                st.info("1. Review results with the design team")
                st.info("2. Proceed to **0. Feasibility Gate** → parameters are auto-filled from STL analysis")
                st.info("3. Proceed to **1. Flow Analysis** → link to MIM-Ops simulation")
        else:
            st.info("💡 Please upload and analyze an STL file first (Tab 1).")


# ══════════════════════════════════════════════════════════
#  STAGE 0: Feasibility Gate
# ══════════════════════════════════════════════════════════
elif current_stage == "stage0":
    st.markdown('<div class="stage-tag">STAGE 0 · FEASIBILITY GATE</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stage-title">{T["st0_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stage-desc">{T["st0_desc"]}</div>', unsafe_allow_html=True)
    st.markdown("")

    # ── STL 분석 결과 자동 연동 안내 ─────────────────────
    derived = st.session_state.get("stl_derived")
    if derived:
        st.success("✅ Geometry parameters auto-filled from Mold Concept STL analysis. You can adjust if needed.")
    else:
        st.info("💡 Tip: Run STL analysis in **Mold Concept** tab first to auto-fill geometry parameters.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"#### {T['input_geometry']}")
        t_min = st.number_input(T["min_thick"], 0.3, 10.0,
                        value=max(0.3, float(derived["min_thickness"])) if derived else 1.8, step=0.1)
        t_max  = st.number_input(T["max_thick"],   0.5, 20.0,
                                 value=float(derived["max_thickness"]) if derived else 3.2, step=0.1)
        flow_l = st.number_input(T["flow_length"], 10.0, 500.0,
                                 value=float(derived["flow_length"])   if derived else 148.0, step=1.0)
        draft  = st.number_input(T["draft_angle"], 0.0, 10.0, 1.5, 0.25)

    with col_b:
        st.markdown(f"#### {T['input_mold']}")
        # gate_count: 기본값 1, 수동 수정 가능
        gate_count = st.number_input(T["gate_count"], 1, 8, 1, 1)
        # undercut: STL에서 자동 반영
        undercut   = st.checkbox(T["has_undercut"],
                                 value=bool(derived["undercut"]) if derived else False)
        part_vol   = st.number_input(T["part_volume"], 0.1, 500.0,
                                     value=float(derived["part_volume"]) if derived else 12.5, step=0.5)

    st.markdown("")
    if st.button(T["btn_st0"], type="primary", use_container_width=True):
        params = {
            "material": material,
            "min_thickness": t_min, "max_thickness": t_max,
            "avg_thickness": avg_thickness,
            "flow_length": flow_l, "draft_angle": draft,
            "gate_count": gate_count, "undercut": undercut,
            "part_volume": part_vol,
        }
        with st.spinner(T.get("st0_checking", "Analyzing...")):
            result = run_feasibility_check(params)
        st.session_state["feasibility_result"] = result
        st.session_state["stage0_done"] = True

    if st.session_state["feasibility_result"]:
        res     = st.session_state["feasibility_result"]
        overall = res["overall"]
        summary = res["summary"]

        st.markdown("---")
        vc  = {"PASS": "verdict-pass", "WARN": "verdict-warn", "FAIL": "verdict-fail"}[overall]
        vic = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[overall]
        st.markdown(f"""
        <div class="{vc}">
            <span style="font-size:1.4rem;">{vic}</span>
            <strong style="font-size:1.0rem; margin-left:8px;">{overall} — {summary['verdict_text']}</strong><br>
            <span style="font-size:0.8rem; color:#8899aa; margin-left:32px;">
            ✅ {summary['PASS']} Passed &nbsp; ⚠ {summary['WARN']} Warnings &nbsp; ❌ {summary['FAIL']} Failed
            </span>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("L/t Ratio",          f"{res['lt_ratio']:.1f}")
        m2.metric("Thickness Ratio",    f"{res['thickness_ratio']:.2f}×")
        m3.metric("Est. Shrinkage",
                  f"{res['shrink_range_pct'][0]:.1f}~{res['shrink_range_pct'][1]:.1f}%")

        st.markdown("#### 📋 Detailed Results by Item")
        df_items = pd.DataFrame(res["items"])
        rename_map = {"Item":"항목","Value":"측정값","Reference":"기준값","Verdict":"판정","Action":"권장 조치"}
        df_items = df_items.rename(columns=rename_map)
        if "판정" in df_items.columns:
            df_items.insert(0, "", df_items["판정"].apply(verdict_color))
        display_cols = [c for c in ["","항목","측정값","기준값","판정","권장 조치"] if c in df_items.columns]
        st.dataframe(df_items[display_cols], use_container_width=True, hide_index=True)

        if overall != "FAIL":
            st.success("✅ Manufacturability Confirmed")
            st.divider()
            # ── Flow Analysis 링크 카드 ─────────────────
            st.markdown("#### 🌊 Flow Analysis — Next Step")
            st.markdown("""
            <div class="link-card">
            <strong>🚀 MIM-Ops Pro: OpenFOAM Cloud Simulation</strong><br>
            <span style="color:#8899aa; font-size:0.85rem;">
            Run injection flow simulation on the cloud (GitHub Actions).
            After simulation, return here and load the results in <b>1. Flow Analysis</b>.
            </span>
            </div>
            """, unsafe_allow_html=True)
            st.link_button(
                "🔗 Open MIM-Ops Simulation →",
                "https://openfoam-injection-automation.streamlit.app/",
                use_container_width=True,
            )
            st.info("📌 After simulation: go to **1. Flow Analysis** sidebar → enter your Signal ID to load results.")
        else:
            st.error("❌ Design Revision Required before proceeding.")


elif current_stage == "stage1":
    st.markdown('<div class="stage-tag">STAGE 1 · FLOW ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stage-title">{T["st1_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stage-desc">{T["st1_desc"]}</div>', unsafe_allow_html=True)
    st.markdown("")

    tab_import, tab_field, tab_defect, tab_window, tab_solid = st.tabs([
        T["tab_data"], T["tab_field"], T["tab_defect"], T["tab_window"],
        "🧊 Solid Mesh (VTK)"
    ])

    with tab_import:
        st.markdown("#### 📂 Load Flow Analysis Results")

        # ── Option A: GitHub Signal ID ──────────────────────────────
        with st.expander("🔗 Option A — Load from MIM-Ops Simulation (GitHub)", expanded=True):
            st.markdown("""
            **Step 1:** Run simulation at
            [🚀 MIM-Ops Pro](https://openfoam-injection-automation.streamlit.app/)
            → 시뮬레이션 완료 후 돌아오세요.

            **Step 2:** Signal ID를 아래에 입력하고 **Generate CSV** 클릭.
            """)

            with st.expander("❓ Signal ID는 어디서 확인하나요?", expanded=False):
                st.markdown("""
                GitHub `OpenFOAM-Injection-Automation` 저장소 → **Actions** 탭
                → 완료된 워크플로 클릭 → Artifacts 섹션에서 이름 확인:
                ```
                simulation-47664275
                ```
                아래 형식 **모두 동작**합니다:

                | 입력 형식 | 예시 |
                |---|---|
                | 숫자 ID만 | `47664275` |
                | 전체 아티팩트 이름 | `simulation-47664275` |
                | 가장 최근 결과 자동 선택 | `latest` |
                """)

            # ── 진단 버튼 ──────────────────────────────────────────
            if st.button("🔍 아티팩트 목록 확인", help="GitHub에서 실제 아티팩트 목록을 가져와 Signal ID를 직접 확인합니다"):

                def _fetch_artifacts_direct(per_page: int = 20) -> list:
                    """GitHub API로 artifacts 목록 직접 조회 (flow_csv_generator 미의존)."""
                    try:
                        token = st.secrets["GITHUB_TOKEN"]
                        # 키 이름: REPO_OWNER 또는 OPENFOAM_REPO_OWNER 둘 다 지원
                        owner = (st.secrets.get("REPO_OWNER")
                                 or st.secrets.get("OPENFOAM_REPO_OWNER")
                                 or "workshopcompany")
                        repo  = (st.secrets.get("REPO_NAME")
                                 or st.secrets.get("OPENFOAM_REPO_NAME")
                                 or "OpenFOAM-Injection-Automation")
                    except (KeyError, FileNotFoundError):
                        raise RuntimeError("GITHUB_TOKEN이 Secrets에 없습니다.")
                    url = f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts"
                    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
                    resp = requests.get(url, headers=headers, params={"per_page": per_page}, timeout=10)
                    if resp.status_code == 401:
                        raise RuntimeError("GITHUB_TOKEN이 유효하지 않습니다 (401 Unauthorized).")
                    if resp.status_code == 404:
                        raise RuntimeError(f"저장소를 찾을 수 없습니다: {owner}/{repo} (404)")
                    resp.raise_for_status()
                    return resp.json().get("artifacts", [])

                if not _check_github_secrets():
                    _show_github_token_guide()
                else:
                    try:
                        with st.spinner("GitHub에서 목록 가져오는 중..."):
                            artifacts = _fetch_artifacts_direct(per_page=20)
                        if artifacts:
                            st.success(f"✅ {len(artifacts)}개 아티팩트 발견 — 숫자 ID 또는 이름 전체를 Signal ID에 입력하세요")
                            st.dataframe([{
                                "아티팩트 이름": a["name"],
                                "생성일":        a["created_at"][:10],
                                "크기(MB)":      f"{a.get('size_in_bytes',0)/1024/1024:.1f}",
                                "만료일":        a.get("expires_at", "")[:10],
                            } for a in artifacts], use_container_width=True, hide_index=True)
                        else:
                            st.warning("⚠️ 아티팩트가 없습니다. MIM-Ops에서 시뮬레이션을 먼저 실행하세요.")
                    except RuntimeError as e:
                        st.error(str(e))
                        _show_github_token_guide()
                    except Exception as e:
                        st.error(f"❌ GitHub API 오류: {e}")

            st.divider()

            # ── Signal ID 입력 & CSV 생성 ──────────────────────────
            sig_col1, sig_col2 = st.columns([3, 1])
            with sig_col1:
                signal_id = st.text_input(
                    "Signal ID (from MIM-Ops simulation)",
                    value=st.session_state.get("github_sim_signal_id", ""),
                    placeholder="예: 47664275  또는  simulation-47664275  또는  latest",
                    help="숫자 ID, 전체 아티팩트 이름, 또는 'latest' 입력 가능",
                )
            with sig_col2:
                st.write("")
                st.write("")
                gen_btn = st.button("📥 Generate CSV", use_container_width=True, type="primary")

            if gen_btn:
                if not signal_id.strip():
                    st.warning("Signal ID를 입력하세요. 모르면 위 '아티팩트 목록 확인' 버튼을 먼저 누르세요.")
                else:
                    _github_ok = _check_github_secrets()
                    if not _github_ok:
                        _show_github_token_guide()
                    else:
                        with st.spinner(f"'{signal_id.strip()}' 결과 가져오는 중..."):
                            try:
                                cae_df = generate_flow_csv_from_github(signal_id.strip())
                                st.session_state["cae_df"] = cae_df
                                st.session_state["github_sim_signal_id"] = signal_id.strip()
                                st.session_state["flow_csv_ready"] = True
                                st.success(
                                    f"✅ 로드 완료! {len(cae_df):,}개 포인트 | "
                                    f"재료: {cae_df['material'].iloc[0]} | "
                                    f"최대 압력: {cae_df['pressure'].max():.1f} MPa"
                                )
                            except FileNotFoundError as e:
                                st.error(str(e))
                            except Exception as e:
                                _emsg = str(e)
                                st.error(f"❌ 오류: {_emsg}")
                                if "GITHUB_TOKEN" in _emsg or "token" in _emsg.lower():
                                    _show_github_token_guide()

            if st.session_state.get("flow_csv_ready") and st.session_state.get("cae_df") is not None:
                df_preview = st.session_state["cae_df"]
                st.markdown("**데이터 미리보기 (상위 5행)**")
                st.dataframe(df_preview.head(5), use_container_width=True)
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("총 포인트", f"{len(df_preview):,}")
                col_s2.metric("최대 압력", f"{df_preview['pressure'].max():.1f} MPa")
                col_s3.metric("충진 시간", f"{df_preview['fill_time'].max():.3f} s")
                csv_bytes = df_preview.to_csv(index=False).encode("utf-8-sig")
                st.download_button("💾 CSV 다운로드", csv_bytes, "flow_analysis.csv", "text/csv",
                                   use_container_width=True)

        # ── Option B: 수동 CSV 업로드 ────────────────────────────────
        with st.expander("📄 Option B — Manual CSV Upload"):
            st.markdown("""
            **필수 컬럼:** `x, y, pressure(MPa), temperature(°C), fill_time(s)`
            `z` 컬럼은 선택 (있으면 3D 시각화)
            """)
            uploaded = st.file_uploader(T.get("select_cae_file", "CAE CSV 파일 선택"), type=["csv"])
            use_sample = st.checkbox(T["use_sample"], value=False)
            # ── Option B: CSV 업로드 시 즉시 session_state에 저장 ──
            if uploaded and not use_sample:
                try:
                    _df_b = load_cae_data(uploaded)
                    st.session_state["cae_df"] = _df_b
                    st.session_state["flow_csv_ready"] = True
                    st.success(f"✅ CSV 로드 완료! {len(_df_b):,}개 포인트 | 최대 압력: {_df_b['pressure'].max():.1f} MPa")
                except Exception as _e:
                    st.error(f"CSV 파싱 오류: {_e}")

        # ── Option C: VTK 파일 직접 업로드 (FIX-1 + FIX-3) ─────────
        with st.expander("🗂️ Option C — VTK/VTU 파일 직접 업로드 (OpenFOAM 결과)", expanded=False):
            st.markdown("""
            **OpenFOAM `foamToVTK` 결과물을 직접 업로드하세요.**
            - **ZIP 파일** (`internal.vtu` 포함): 시뮬레이션 결과 폴더 전체를 압축한 .zip
            - **단일 VTU 파일** (`internal.vtu`): 내부 솔리드 메쉬 데이터

            > 📌 업로드하면 압력(p), 속도(U) 등 **실제 OpenFOAM 계산값**으로 CSV가 자동 생성됩니다.
            """)

            vtk_col1, vtk_col2 = st.columns([3, 1])
            with vtk_col1:
                vtk_upload = st.file_uploader(
                    "VTK 결과 파일 선택",
                    type=["zip", "vtu", "vtm", "vtp"],
                    key="vtk_direct_uploader",
                    help="ZIP: 폴더 전체 압축 | .vtu: internal.vtu 개별 업로드",
                )
            with vtk_col2:
                st.write(""); st.write("")
                vtk_gen_btn = st.button("🔄 VTK → CSV 변환", key="vtk_gen_btn",
                                        use_container_width=True, type="primary")

            if vtk_upload and vtk_gen_btn:
                with st.spinner("VTK 파일 파싱 중..."):
                    try:
                        raw = vtk_upload.getvalue()
                        ext = vtk_upload.name.lower().split(".")[-1]
                        if ext == "zip":
                            _vtk_df = parse_vtk_zip_to_dataframe(raw, material=material)
                        else:  # .vtu / .vtm / .vtp
                            _vtk_df = parse_vtu_to_dataframe(raw, material=material)

                        # 세션 저장 + solid mesh 별도 저장
                        st.session_state["cae_df"]        = _vtk_df
                        st.session_state["flow_csv_ready"] = True
                        st.session_state["vtk_solid_df"]  = _vtk_df  # Solid Mesh 탭용
                        st.success(
                            f"✅ VTK 파싱 완료! **{len(_vtk_df):,}개 포인트** | "
                            f"최대 압력: {_vtk_df['pressure'].max():.3f} MPa | "
                            f"파일: {vtk_upload.name}"
                        )
                        # CSV 다운로드 버튼
                        csv_vtk = _vtk_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "💾 생성된 CSV 다운로드",
                            csv_vtk, "vtk_flow_results.csv", "text/csv",
                            use_container_width=True,
                        )
                    except Exception as _ve:
                        st.error(f"❌ VTK 파싱 오류: {_ve}")
                        st.info("💡 파일 형식을 확인하세요. ASCII 형식 .vtu만 지원합니다. (`foamToVTK -ascii`)")

            # 이미 VTK 데이터가 로드된 경우 상태 표시
            if st.session_state.get("vtk_solid_df") is not None:
                _vs = st.session_state["vtk_solid_df"]
                st.info(f"🧊 Solid Mesh 데이터 로드됨: {len(_vs):,} pts — 'Solid Mesh (VTK)' 탭에서 확인하세요.")



        use_ml = st.toggle(T["apply_ml"], value=False)
        st.divider()

        # ── 현재 로드된 데이터 상태 표시 ──────────────────────────
        if st.session_state.get("flow_csv_ready") and st.session_state.get("cae_df") is not None:
            _loaded = st.session_state["cae_df"]
            st.success(f"✅ 데이터 준비됨 — {len(_loaded):,}개 포인트 | 최대 압력: {_loaded['pressure'].max():.1f} MPa")
        elif not st.session_state.get("flow_csv_ready"):
            st.info("💡 위에서 Signal ID로 CSV 생성하거나, Option B로 CSV를 업로드한 뒤 분석을 실행하세요.")

        if st.button(T["btn_st1"], type="primary", use_container_width=True):
            with st.spinner(T.get("st1_analyzing", "분석 중...")):
                try:
                    cae_df = st.session_state.get("cae_df")
                    if cae_df is None:
                        if use_sample:
                            cae_df = generate_sample_cae_csv(n_points=300, material=material)
                            st.info(f"📌 {T.get('msg_using_sample', '샘플 데이터로 분석합니다.')}")
                        else:
                            st.warning("⚠️ 데이터가 없습니다. Signal ID로 CSV를 생성하거나 Option B로 업로드하세요.")
                            st.stop()

                    analysis = analyze_cae(cae_df, material=material)
                    st.session_state["cae_df"]       = cae_df
                    st.session_state["cae_analysis"]  = analysis
                    st.session_state["stage1_done"]   = True
                    st.success(T["msg_analysis_done"])
                    st.rerun()
                except Exception as e:
                    st.error(f"{T['msg_error']}: {e}")

    # ── 결과 탭들 ─────────────────────────────────────────
    if st.session_state["cae_analysis"]:
        analysis = st.session_state["cae_analysis"]
        cae_df   = st.session_state["cae_df"]
        stats    = analysis["stats"]

        # ══════════════════════════════════════════════════════
        #  STL 파싱 유틸 (trimesh 없이 순수 stdlib + numpy)
        # ══════════════════════════════════════════════════════
        def _parse_stl_binary(file_bytes: bytes):
            """Binary STL → (vertices Nx3, faces Mx3) numpy arrays.
            중복 vertex를 제거해 Mesh3d의 i/j/k 인덱스를 올바르게 반환."""
            import struct
            data = file_bytes
            # 80 byte header + 4 byte tri count
            tri_count = struct.unpack_from("<I", data, 80)[0]
            offset = 84
            raw_verts = []
            for _ in range(tri_count):
                offset += 12  # normal
                v0 = struct.unpack_from("<3f", data, offset);  offset += 12
                v1 = struct.unpack_from("<3f", data, offset);  offset += 12
                v2 = struct.unpack_from("<3f", data, offset);  offset += 12
                offset += 2   # attr byte
                raw_verts.extend([v0, v1, v2])
            verts_all = np.array(raw_verts, dtype=np.float64)  # (3*N, 3)
            # 중복 제거 → 인덱스 배열 생성
            verts_unique, inv_idx = np.unique(
                np.round(verts_all, 6), axis=0, return_inverse=True
            )
            faces = inv_idx.reshape(-1, 3)  # (N, 3)
            return verts_unique, faces

        def _map_cae_to_mesh(vertices, cae_df, field, gate_pos):
            """각 mesh vertex에 가장 가까운 CAE 포인트의 field 값을 거리 역가중 평균으로 매핑."""
            cae_xyz = cae_df[["x", "y", "z"]].values if "z" in cae_df.columns \
                      else np.column_stack([cae_df["x"].values, cae_df["y"].values,
                                            np.zeros(len(cae_df))])
            cae_vals = cae_df[field].values.astype(float)

            # 좌표계 스케일 맞추기 — bounding box 기반 정규화
            v_min = vertices.min(axis=0)
            v_max = vertices.max(axis=0)
            c_min = cae_xyz.min(axis=0)
            c_max = cae_xyz.max(axis=0)
            v_range = np.where((v_max - v_min) > 0, v_max - v_min, 1.0)
            c_range = np.where((c_max - c_min) > 0, c_max - c_min, 1.0)
            verts_norm = (vertices - v_min) / v_range
            cae_norm   = (cae_xyz  - c_min) / c_range

            # 각 vertex에 대해 k=5 근접 포인트 IDW 보간
            k = min(5, len(cae_norm))
            intensity = np.zeros(len(verts_norm))
            for vi in range(0, len(verts_norm), max(1, len(verts_norm)//500)):
                diffs = cae_norm - verts_norm[vi]
                dists = np.linalg.norm(diffs, axis=1)
                idx_k = np.argpartition(dists, k)[:k]
                d_k   = dists[idx_k]
                if d_k.min() < 1e-9:
                    intensity[vi] = cae_vals[idx_k[d_k.argmin()]]
                else:
                    w = 1.0 / (d_k ** 2)
                    intensity[vi] = np.dot(w, cae_vals[idx_k]) / w.sum()

            # 벡터화 구간은 위 루프로 처리했으므로 남은 vertex 처리
            for vi in range(0, len(verts_norm)):
                if intensity[vi] != 0:
                    continue
                diffs = cae_norm - verts_norm[vi]
                dists = np.linalg.norm(diffs, axis=1)
                idx_k = np.argpartition(dists, k)[:k]
                d_k   = dists[idx_k]
                if d_k.min() < 1e-9:
                    intensity[vi] = cae_vals[idx_k[d_k.argmin()]]
                else:
                    w = 1.0 / (d_k ** 2)
                    intensity[vi] = np.dot(w, cae_vals[idx_k]) / w.sum()
            return intensity

        # ── Tab: Field Map (STL Mesh3d) ───────────────────
        with tab_field:
            st.markdown(f"#### 📊 {T.get('title_field_map', 'Field Distribution Map')}")

            field_options = {
                "pressure":    T.get("field_pressure",    "Pressure"),
                "temperature": T.get("field_temperature", "Temperature"),
                "fill_time":   T.get("field_fill_time",   "Fill Time"),
            }
            field_tabs = st.tabs([
                f"🔵 {field_options['pressure']} (MPa)",
                f"🔴 {field_options['temperature']} (°C)",
                f"🟢 {field_options['fill_time']} (s)",
            ])
            fields = list(field_options.keys())
            colorscales = {"pressure": "Jet", "temperature": "Hot", "fill_time": "Viridis"}
            cb_titles   = {"pressure": "Pressure (MPa)", "temperature": "Temp (°C)", "fill_time": "Fill Time (s)"}

            # ── 게이트 위치 ──────────────────────────────────────
            has_z_global = "z" in cae_df.columns
            gate_row = cae_df.loc[cae_df["fill_time"].idxmin()]
            gate_x = float(gate_row["x"])
            gate_y = float(gate_row["y"])
            gate_z = float(gate_row["z"]) if has_z_global else 0.0

            # 게이트 거리 / 유동선단
            if has_z_global:
                cae_df["dist_from_gate"] = np.sqrt(
                    (cae_df["x"] - gate_x)**2 + (cae_df["y"] - gate_y)**2 + (cae_df["z"] - gate_z)**2)
            else:
                cae_df["dist_from_gate"] = np.sqrt(
                    (cae_df["x"] - gate_x)**2 + (cae_df["y"] - gate_y)**2)
            max_dist = cae_df["dist_from_gate"].max()
            cae_df["rel_dist"] = cae_df["dist_from_gate"] / max_dist
            front_df = cae_df[cae_df["rel_dist"] > 0.85]

            # ── STL 파싱 (session_state 캐시) ───────────────────
            stl_mesh_data = st.session_state.get("_stl_mesh_cache")

            # [FIX-2] Mold Concept에서 업로드된 STL 자동 로드
            auto_stl_bytes = st.session_state.get("stl_bytes")
            if auto_stl_bytes is not None and stl_mesh_data is None:
                with st.spinner("🔄 Mold Concept에서 업로드된 STL 자동 로드 중..."):
                    try:
                        _av, _af = _parse_stl_binary(auto_stl_bytes)
                        st.session_state["_stl_mesh_cache"] = {
                            "vertices": _av, "faces": _af,
                            "name": st.session_state.get("stl_name", "auto_loaded.stl"),
                            "n_faces": len(_af), "intensity_cache": {},
                        }
                        stl_mesh_data = st.session_state["_stl_mesh_cache"]
                        st.success(
                            f"✅ STL 자동 로드 완료: **{st.session_state.get('stl_name')}** "
                            f"({len(_av):,} vertices, {len(_af):,} faces)"
                        )
                    except Exception as _ae:
                        st.warning(f"STL 자동 로드 실패 (수동 업로드 필요): {_ae}")

            # STL 수동 업로드 위젯 (자동 로드가 안 됐을 때만 강조 표시)
            stl_col1, stl_col2 = st.columns([3, 1])
            with stl_col1:
                _uploader_label = (
                    "🗂️ STL 파일 업로드 (형상 위에 분포도 표시)"
                    if stl_mesh_data is None
                    else "🔄 다른 STL로 교체하려면 업로드"
                )
                stl_upload_field = st.file_uploader(
                    _uploader_label,
                    type=["stl"], key="stl_field_uploader",
                    help="Mold Concept에서 업로드한 STL이 자동으로 로드됩니다. 다른 파일로 교체할 경우만 업로드.",
                )
            with stl_col2:
                st.write("")
                st.write("")
                if stl_mesh_data:
                    st.success(f"✅ {stl_mesh_data['name']}\n({stl_mesh_data['n_faces']:,} faces)")

            if stl_upload_field is not None:
                with st.spinner("STL 파싱 중..."):
                    try:
                        file_bytes = stl_upload_field.read()
                        verts, faces = _parse_stl_binary(file_bytes)
                        st.session_state["_stl_mesh_cache"] = {
                            "vertices": verts,
                            "faces": faces,
                            "name": stl_upload_field.name,
                            "n_faces": len(faces),
                            "intensity_cache": {},  # field → array
                        }
                        stl_mesh_data = st.session_state["_stl_mesh_cache"]
                        st.success(f"✅ STL 로드 완료: {len(verts):,} vertices, {len(faces):,} faces")
                    except Exception as _stl_e:
                        st.error(f"STL 파싱 오류: {_stl_e}")

            # ── 공통 scene / layout ──────────────────────────────
            _scene_cfg = dict(
                xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
                bgcolor="#111318",
                xaxis=dict(color="#e2e8f0", backgroundcolor="#111318",
                           gridcolor="#252b36", showbackground=True),
                yaxis=dict(color="#e2e8f0", backgroundcolor="#111318",
                           gridcolor="#252b36", showbackground=True),
                zaxis=dict(color="#e2e8f0", backgroundcolor="#111318",
                           gridcolor="#252b36", showbackground=True),
                aspectmode="data",
            )
            _layout_cfg = dict(
                paper_bgcolor="#0a0c0f", font_color="#e2e8f0",
                height=580, margin=dict(l=0, r=0, t=45, b=0),
                legend=dict(bgcolor="rgba(17,19,24,0.85)",
                            bordercolor="#252b36", font=dict(color="#e2e8f0")),
            )

            def _gate_trace_3d(zv):
                return go.Scatter3d(
                    x=[gate_x], y=[gate_y], z=[zv],
                    mode="markers+text",
                    marker=dict(size=12, color="#ff2222", symbol="diamond",
                                line=dict(width=2, color="white")),
                    text=["GATE"], textposition="top center",
                    textfont=dict(color="#ff6666", size=11),
                    name=f"🔴 Gate ({gate_x:.2f}, {gate_y:.2f}, {gate_z:.2f})",
                    showlegend=True,
                )

            for i, ftab in enumerate(field_tabs):
                ft = fields[i]
                with ftab:
                    if ft not in cae_df.columns:
                        st.info(f"No '{ft}' column in data.")
                        continue

                    fig3d = go.Figure()

                    # ── 분기: STL 있으면 Mesh3d, 없으면 개선된 Scatter3d ──
                    if stl_mesh_data is not None:
                        verts  = stl_mesh_data["vertices"]
                        faces  = stl_mesh_data["faces"]

                        # 인텐시티 캐시 (재계산 방지)
                        if ft not in stl_mesh_data.get("intensity_cache", {}):
                            with st.spinner(f"🔄 {field_options[ft]} → Mesh 매핑 중 (최초 1회)..."):
                                gate_pos = np.array([gate_x, gate_y, gate_z])
                                intensity = _map_cae_to_mesh(verts, cae_df, ft, gate_pos)
                                stl_mesh_data["intensity_cache"][ft] = intensity
                                st.session_state["_stl_mesh_cache"] = stl_mesh_data
                        else:
                            intensity = stl_mesh_data["intensity_cache"][ft]

                        fig3d.add_trace(go.Mesh3d(
                            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                            intensity=intensity,
                            colorscale=colorscales[ft],
                            colorbar=dict(
                                title=dict(text=cb_titles[ft], font=dict(color="#e2e8f0")),
                                tickfont=dict(color="#e2e8f0"), x=1.02,
                            ),
                            opacity=1.0,
                            flatshading=False,       # smooth shading
                            lighting=dict(ambient=0.5, diffuse=0.8,
                                          specular=0.3, roughness=0.5),
                            lightposition=dict(x=1, y=1, z=2),
                            name=f"{field_options[ft]} (Mesh)",
                            showlegend=True,
                            hovertemplate=(
                                f"<b>{field_options[ft]}: %{{intensity:.2f}}</b><br>"
                                "X: %{x:.3f} mm<br>Y: %{y:.3f} mm<br>"
                                "Z: %{z:.3f} mm<extra></extra>"
                            ),
                        ))

                        # 유동선단 오버레이 (fill_time 탭)
                        if ft == "fill_time" and len(front_df) > 0:
                            fz = front_df["z"].values if has_z_global else np.zeros(len(front_df))
                            fig3d.add_trace(go.Scatter3d(
                                x=front_df["x"], y=front_df["y"], z=fz,
                                mode="markers",
                                marker=dict(size=5, color="#00ffcc", opacity=0.9,
                                            line=dict(color="#ffffff", width=0.5)),
                                name="🟢 Flow Front (>85%)", showlegend=True,
                            ))

                    else:
                        # ── STL 없음: Scatter3d (개선형, 에러 없음) ─────────
                        pt_color = cae_df[ft].values
                        z_col = cae_df["z"].values if has_z_global else np.zeros(len(cae_df))

                        if ft == "pressure":
                            pt_size = (4 + 8 * (1 - cae_df["rel_dist"].values)).clip(4, 12).tolist()
                        else:
                            pt_size = 7

                        fig3d.add_trace(go.Scatter3d(
                            x=cae_df["x"], y=cae_df["y"], z=z_col,
                            mode="markers",
                            marker=dict(
                                size=pt_size,
                                color=pt_color,
                                colorscale=colorscales[ft],
                                colorbar=dict(
                                    title=dict(text=cb_titles[ft], font=dict(color="#e2e8f0")),
                                    tickfont=dict(color="#e2e8f0"), x=1.02,
                                ),
                                opacity=0.85,
                                line=dict(width=0),
                            ),
                            customdata=np.stack([
                                cae_df[ft].values,
                                cae_df["rel_dist"].values,
                                cae_df["fill_time"].values,
                            ], axis=-1),
                            hovertemplate=(
                                f"<b>{field_options[ft]}: %{{customdata[0]:.2f}}</b><br>"
                                "X: %{x:.3f} mm | Y: %{y:.3f} mm<br>"
                                "Gate Dist: %{customdata[1]:.0%}<br>"
                                "Fill Time: %{customdata[2]:.2f} s<extra></extra>"
                            ),
                            name=f"{field_options[ft]} (Point Cloud)",
                            showlegend=True,
                        ))

                        if ft == "fill_time" and len(front_df) > 0:
                            fz = front_df["z"].values if has_z_global else np.zeros(len(front_df))
                            fig3d.add_trace(go.Scatter3d(
                                x=front_df["x"], y=front_df["y"], z=fz,
                                mode="markers",
                                marker=dict(size=9, color="#00ffcc", opacity=1.0,
                                            line=dict(color="#ffffff", width=1)),
                                name="🟢 Flow Front (>85%)", showlegend=True,
                            ))

                        st.info("💡 STL 파일을 업로드하면 실제 형상 표면에 컨투어가 표시됩니다.")

                    # 게이트 마커 공통
                    fig3d.add_trace(_gate_trace_3d(gate_z))

                    fig3d.update_layout(
                        **_layout_cfg,
                        scene=_scene_cfg,
                        title=dict(
                            text=(
                                f"{field_options[ft]} Distribution"
                                f"{'  [Mesh3d — STL Surface]' if stl_mesh_data else '  [Point Cloud]'}"
                                f"  |  Gate @ ({gate_x:.2f}, {gate_y:.2f}, {gate_z:.2f}) mm"
                            ),
                            font=dict(color="#e2e8f0", size=13),
                        ),
                    )
                    st.plotly_chart(fig3d, use_container_width=True)

                    col_gi1, col_gi2, col_gi3 = st.columns(3)
                    col_gi1.metric("🎯 Gate 절대위치",
                                   f"({gate_x:.2f}, {gate_y:.2f}, {gate_z:.2f}) mm")
                    col_gi2.metric("📏 최대 유동 거리", f"{max_dist:.2f} mm")
                    col_gi3.metric("🌊 유동 선단 포인트", f"{len(front_df)}개")

            # 통계 지표
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(T.get("metric_max_pressure",  "Max Pressure"), f"{stats['max_pressure_MPa']:.1f} MPa")
            c2.metric(T.get("metric_max_temp",       "Max Temp"),     f"{stats['max_temperature_C']:.1f} °C")
            c3.metric(T.get("metric_fill_time",      "Fill Time"),    f"{stats['fill_time_s']:.2f} s")
            c4.metric(T.get("metric_pressure_grad",  "Pres. Grad."), f"{stats['pressure_gradient']:.2f} MPa/mm")

        # ── Tab: Defect Analysis ──────────────────────────
        with tab_defect:
            st.markdown(f"#### ⚠️ {T['title_defect_risk']}")
            risks = analysis["defect_risks"]
            defect_names = {
                "short_shot": T["def_short_shot"], "weld_line": T["def_weld_line"],
                "sink_mark": T["def_sink_mark"],   "warpage":   T["def_warpage"],
                "air_trap":  T["def_air_trap"],
            }
            cols = st.columns(len(risks))
            for i, (key, risk) in enumerate(risks.items()):
                score = risk["score"]; level = risk["level"]
                icon = {"LOW": "🟢", "MED": "🟡", "HIGH": "🔴"}[level]
                cols[i].metric(defect_names.get(key, key), f"{icon} {level}", f"{risk['score']*100:.0f}%")

            for key, risk in risks.items():
                if risk["level"] != "LOW":
                    cls = "verdict-warn" if risk["level"] == "MED" else "verdict-fail"
                    st.markdown(f"""
                    <div class="{cls}">
                    <strong>{defect_names.get(key, key)}</strong><br>
                    <span style="font-size:0.82rem;color:#8899aa;">{risk['detail']}</span>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Tab: Process Window ───────────────────────────
        with tab_window:
            st.markdown(f"#### 🎯 {T['tab_window_title']}")
            optimal = analysis["optimal_conditions"]
            param_labels = {
                "melt_temperature": T["param_melt_temp"],
                "mold_temperature": T["param_mold_temp"],
                "injection_speed":  T["param_inj_speed"],
                "packing_pressure": T["param_pack_pres"],
                "packing_time":     T["param_pack_time"],
            }
            for param, vals in optimal.items():
                label = param_labels.get(param, param)
                lo, hi = vals["range"]; opt = vals["optimal"]; unit = vals["unit"]
                ratio = (opt - lo) / (hi - lo) if hi != lo else 0.5
                st.markdown(f"**{label}**")
                ca, cb = st.columns([3, 1])
                with ca:
                    st.progress(int(ratio * 100))
                    st.caption(f"Range: {lo}~{hi} {unit}  |  Optimal: {opt} {unit}")
                with cb:
                    st.markdown(f"""
                    <div class="mono" style="text-align:center;color:#00d4aa;font-size:1.1rem;font-weight:700;">
                    {opt} {unit}
                    </div>""", unsafe_allow_html=True)
                st.markdown("")
            st.success(f"✅ {T.get('msg_ready_for_st2', 'Ready for Stage 2 Dimension Prediction')}")

        # ── Tab: Solid Mesh (VTK) — FIX-3 ────────────────
        with tab_solid:
            st.markdown("#### 🧊 Solid Mesh 체적 시각화 (OpenFOAM VTK 실제 결과)")
            st.markdown("""
            <div class="info-box">
            STL 껍데기 위에 데이터를 매핑하던 방식 대신, <b>OpenFOAM이 계산한 내부 솔리드 메쉬 데이터</b>를
            직접 Point Cloud로 시각화합니다. 좌측 <b>Data Input → Option C</b>에서 VTK 파일을 먼저 업로드하세요.
            </div>
            """, unsafe_allow_html=True)

            solid_df = st.session_state.get("vtk_solid_df")

            if solid_df is None:
                st.warning(
                    "⚠️ Solid Mesh 데이터가 없습니다.\n\n"
                    "**Data Input 탭 → Option C**에서 `internal.vtu` 또는 ZIP을 업로드하면\n"
                    "이 탭에서 실제 OpenFOAM 체적 결과가 표시됩니다."
                )
                # 업로드 숏컷
                st.divider()
                _solid_shortcut = st.file_uploader(
                    "빠른 업로드: internal.vtu 또는 결과 ZIP",
                    type=["vtu", "vtm", "zip"], key="solid_shortcut_uploader",
                )
                if _solid_shortcut:
                    with st.spinner("VTK 파싱 중..."):
                        try:
                            _raw = _solid_shortcut.getvalue()
                            _ext = _solid_shortcut.name.lower().split(".")[-1]
                            if _ext == "zip":
                                solid_df = parse_vtk_zip_to_dataframe(_raw, material=material)
                            else:
                                solid_df = parse_vtu_to_dataframe(_raw, material=material)
                            st.session_state["vtk_solid_df"] = solid_df
                            st.session_state["cae_df"] = solid_df
                            st.session_state["flow_csv_ready"] = True
                            st.success(f"✅ {len(solid_df):,}개 포인트 로드 완료!")
                            st.rerun()
                        except Exception as _se:
                            st.error(f"❌ {_se}")

            if solid_df is not None:
                # ── 필드 선택 ──────────────────────────────────────
                _avail_fields = [c for c in ["pressure", "temperature", "fill_time",
                                             "U_mag", "Ux", "Uy", "Uz", "p"]
                                 if c in solid_df.columns]
                if not _avail_fields:
                    st.error("파싱된 데이터에 시각화할 필드가 없습니다.")
                else:
                    _field_labels = {
                        "pressure": "압력 (MPa)", "temperature": "온도 (°C)",
                        "fill_time": "충진시간 (s)", "U_mag": "속도 크기 (m/s)",
                        "Ux": "X 속도", "Uy": "Y 속도", "Uz": "Z 속도", "p": "p (kinematic)",
                    }
                    _solid_tabs = st.tabs([_field_labels.get(f, f) for f in _avail_fields])
                    _solid_cs = {
                        "pressure": "Jet", "temperature": "Hot",
                        "fill_time": "Viridis", "U_mag": "Plasma",
                        "Ux": "RdBu", "Uy": "RdBu", "Uz": "RdBu", "p": "Jet",
                    }

                    # 통계 메트릭
                    _mc = st.columns(4)
                    _mc[0].metric("총 포인트", f"{len(solid_df):,}")
                    if "pressure" in solid_df.columns:
                        _mc[1].metric("최대 압력", f"{solid_df['pressure'].max():.3f} MPa")
                    if "U_mag" in solid_df.columns:
                        _mc[2].metric("최대 속도", f"{solid_df['U_mag'].max():.4f} m/s")
                    if "temperature" in solid_df.columns:
                        _mc[3].metric("최대 온도", f"{solid_df['temperature'].max():.1f} °C")

                    for _si, _sf in enumerate(_avail_fields):
                        with _solid_tabs[_si]:
                            _vals = solid_df[_sf].values
                            _has_z = "z" in solid_df.columns
                            _z_col = solid_df["z"].values if _has_z else np.zeros(len(solid_df))

                            # 단면 보기 (클리핑)
                            _clip_col1, _clip_col2 = st.columns([2, 1])
                            with _clip_col2:
                                _clip_axis = st.selectbox("단면 축", ["없음(전체)", "X", "Y", "Z"],
                                                           key=f"clip_axis_{_sf}")
                                _clip_ratio = st.slider("단면 위치 (%)", 0, 100, 50,
                                                         key=f"clip_ratio_{_sf}",
                                                         help="전체 범위 대비 단면 위치")

                            with _clip_col1:
                                _mask = np.ones(len(solid_df), dtype=bool)
                                if _clip_axis != "없음(전체)":
                                    _ax_data = {
                                        "X": solid_df["x"].values,
                                        "Y": solid_df["y"].values,
                                        "Z": _z_col,
                                    }[_clip_axis]
                                    _cut = _ax_data.min() + ((_ax_data.max() - _ax_data.min()) * _clip_ratio / 100)
                                    _mask = _ax_data <= _cut

                                _fig_solid = go.Figure()
                                _fig_solid.add_trace(go.Scatter3d(
                                    x=solid_df["x"].values[_mask],
                                    y=solid_df["y"].values[_mask],
                                    z=_z_col[_mask],
                                    mode="markers",
                                    marker=dict(
                                        size=4,
                                        color=_vals[_mask],
                                        colorscale=_solid_cs.get(_sf, "Viridis"),
                                        colorbar=dict(
                                            title=dict(
                                                text=_field_labels.get(_sf, _sf),
                                                font=dict(color="#e2e8f0"),
                                            ),
                                            tickfont=dict(color="#e2e8f0"), x=1.02,
                                        ),
                                        opacity=0.85,
                                        line=dict(width=0),
                                    ),
                                    customdata=np.stack([_vals[_mask]], axis=-1),
                                    hovertemplate=(
                                        f"<b>{_field_labels.get(_sf, _sf)}: %{{customdata[0]:.4f}}</b><br>"
                                        "X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>"
                                    ),
                                    name=f"{_field_labels.get(_sf, _sf)} (Solid)",
                                ))
                                _fig_solid.update_layout(
                                    paper_bgcolor="#0a0c0f", font_color="#e2e8f0",
                                    height=550, margin=dict(l=0, r=0, t=45, b=0),
                                    scene=dict(
                                        xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
                                        bgcolor="#111318",
                                        xaxis=dict(color="#e2e8f0", backgroundcolor="#111318",
                                                   gridcolor="#252b36", showbackground=True),
                                        yaxis=dict(color="#e2e8f0", backgroundcolor="#111318",
                                                   gridcolor="#252b36", showbackground=True),
                                        zaxis=dict(color="#e2e8f0", backgroundcolor="#111318",
                                                   gridcolor="#252b36", showbackground=True),
                                        aspectmode="data",
                                    ),
                                    title=dict(
                                        text=(
                                            f"Solid Mesh — {_field_labels.get(_sf, _sf)} "
                                            f"| {_mask.sum():,}/{len(solid_df):,} pts"
                                            + (f" [{_clip_axis} ≤ {_clip_ratio}%]" if _clip_axis != "없음(전체)" else "")
                                        ),
                                        font=dict(color="#e2e8f0", size=13),
                                    ),
                                )
                                st.plotly_chart(_fig_solid, use_container_width=True)

                    # CSV 다운로드
                    st.divider()
                    _csv_solid = solid_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 Solid Mesh CSV 다운로드",
                        _csv_solid, "solid_mesh_results.csv", "text/csv",
                        use_container_width=True,
                    )

    else:
        for tab in [tab_field, tab_defect, tab_window, tab_solid]:
            with tab:
                st.info("💡 Run analysis in **Data Input** tab first.")



# ══════════════════════════════════════════════════════════
#  STAGE 2: Dimension Prediction
# ══════════════════════════════════════════════════════════
elif current_stage == "stage2":
    st.markdown('<div class="stage-tag">STAGE 2 · DIMENSION PREDICTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="stage-title">Dimension Prediction & Quality Analysis</div>', unsafe_allow_html=True)

    col_conf1, col_conf2 = st.columns([2, 1])
    with col_conf1:
        st.markdown("##### Example Tolerance Standards")
        default_tol_data = {
            "Range (mm)": ["1~4","4~10","10~20","20~30","> 30"],
            "Example Tolerance (±)": [0.030, 0.050, 0.060, 0.080, 0.100]
        }
        edited_tol_df = st.data_editor(pd.DataFrame(default_tol_data), num_rows="fixed",
                                        use_container_width=True, key="tol_editor")
    with col_conf2:
        st.markdown("##### Mold Expansion Factor")
        expansion_factor = st.number_input("Factor (k)", 1.000, 1.300, 1.000, 0.001, format="%.3f")
        if "last_k" not in st.session_state:
            st.session_state.last_k = expansion_factor
        if st.session_state.last_k != expansion_factor:
            df = st.session_state.get("input_df", pd.DataFrame())
            for idx in df.index:
                if df.at[idx, "L_nominal"] > 0:
                    df.at[idx, "L_mold"] = round(df.at[idx, "L_nominal"] * expansion_factor, 3)
            st.session_state.input_df = df
            st.session_state.last_k = expansion_factor
            st.rerun()

    if "input_df" not in st.session_state or len(st.session_state.input_df) > 5:
        st.session_state.input_df = pd.DataFrame([
            {"Name": f"Part_{i+1}", "L_nominal": 0.0, "L_mold": 0.0, "tolerance": 0.0}
            for i in range(3)
        ])

    def sync_on_edit():
        state = st.session_state["dim_editor"]
        df = st.session_state.input_df
        for row_idx, changes in state["edited_rows"].items():
            idx = int(row_idx)
            for key, val in changes.items():
                try:
                    curr_val = float(val)
                    df.at[idx, key] = curr_val
                    if key == "L_nominal" and curr_val > 0:
                        df.at[idx, "L_mold"] = round(curr_val * expansion_factor, 3)
                        tols = edited_tol_df["Example Tolerance (±)"].values
                        if curr_val < 4: df.at[idx, "tolerance"] = tols[0]
                        elif curr_val < 10: df.at[idx, "tolerance"] = tols[1]
                        elif curr_val < 20: df.at[idx, "tolerance"] = tols[2]
                        elif curr_val <= 30: df.at[idx, "tolerance"] = tols[3]
                        else: df.at[idx, "tolerance"] = tols[4]
                except: pass
        st.session_state.input_df = df

    st.markdown("---")
    st.markdown("### 📝 Input Dimensions")
    edited_df = st.data_editor(
        st.session_state.input_df, num_rows="dynamic", use_container_width=True,
        key="dim_editor", on_change=sync_on_edit,
        column_config={
            "L_nominal": st.column_config.NumberColumn("Target (mm)",   format="%.3f", step=0.001),
            "L_mold":    st.column_config.NumberColumn("Mold Dim (mm)", format="%.3f", step=0.001),
            "tolerance": st.column_config.NumberColumn("Tolerance (±)", format="%.3f", step=0.001),
        }
    )
    st.session_state.input_df = edited_df

    if st.button("Run Prediction", type="primary", use_container_width=True):
        active_df = edited_df[edited_df["L_nominal"] > 0].copy()
        if not active_df.empty:
            with st.spinner("Analyzing with CAE Results..."):
                cae_df = st.session_state.get("cae_df")
                if cae_df is None:
                    cae_df = generate_sample_cae_csv(n_points=300, material=material)
                shrink_df = predict_shrinkage_field(cae_df, material=material, avg_thickness=avg_thickness)
                results = []
                for _, row in active_df.iterrows():
                    local_shrink = shrink_df["shrinkage_pct"].sample(1).values[0] / 100.0
                    l_final = round(row["L_mold"] * (1 - local_shrink), 3)
                    dev = round(l_final - row["L_nominal"], 3)
                    status = "OK" if abs(dev) <= row["tolerance"] else ("OVER" if dev > 0 else "UNDER")
                    results.append({
                        "Dimension Name": row["Name"],
                        "Target (L_nom)": row["L_nominal"],
                        "Mold (L_mold)": row["L_mold"],
                        "Predicted (L_final)": l_final,
                        "Deviation": dev,
                        "Tolerance": row["tolerance"],
                        "Status": status,
                    })
                st.session_state["dim_results"] = pd.DataFrame(results)
                st.session_state["stage2_done"] = True

    if st.session_state.get("dim_results") is not None:
        res = st.session_state["dim_results"]
        st.dataframe(res.style.map(
            lambda x: "background-color: rgba(0,212,170,0.2)" if x == "OK"
                      else "background-color: rgba(255,75,75,0.2)",
            subset=["Status"]
        ), use_container_width=True)
        csv = res.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Download Analysis Report", csv, "MIM_Prediction.csv", "text/csv",
                           use_container_width=True)


# ══════════════════════════════════════════════════════════
#  STAGE 3: Inverse Correction
# ══════════════════════════════════════════════════════════
elif current_stage == "stage3":
    st.markdown('<div class="stage-tag">STAGE 3 · INVERSE CORRECTION</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stage-title">{T.get("st3_title","Tolerance-based Inverse Correction")}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="stage-desc">E(x) = L_target - L_predicted → Linear correction within tolerance</div>',
                unsafe_allow_html=True)
    st.markdown("")
    st.markdown("""
    <div class="info-box">
    ❌ Exclude non-linear/asymmetric &nbsp;|&nbsp; ❌ No global scale changes<br>
    ✅ Uniform offset &nbsp;|&nbsp; ✅ Local linear correction &nbsp;|&nbsp; ✅ Feature-level adjustment
    </div>
    """, unsafe_allow_html=True)

    if st.button(T.get("btn_st3", "▶ Run Phase 3 Inverse Calculation"),
                 type="primary", use_container_width=True):
        with st.spinner(T.get("st3_calculating", "Calculating...")):
            dim_df   = st.session_state.get("dim_df")
            shrink_df = st.session_state.get("shrink_df")
            if dim_df is None:
                features = get_sample_features(material=material)
                dim_df = predict_part_dimensions(features, material=material)
            pvt = MATERIAL_PVT.get(material, MATERIAL_PVT["PC+ABS"])
            global_shrink = float(shrink_df["shrinkage"].mean()) if shrink_df is not None else pvt["base_shrink"]
            inverse_result = run_inverse_design(dim_df, global_shrink_avg=global_shrink)
            st.session_state["inverse_result"] = inverse_result
            st.session_state["stage3_done"] = True

    if st.session_state["inverse_result"]:
        inv = st.session_state["inverse_result"]
        corrections = inv["corrections"]
        summary = inv["summary"]
        cost = inv["cost_estimate"]

        is_pass = summary.get("HIGH", 0) == 0
        vc = "verdict-pass" if is_pass else "verdict-warn"
        vt = T.get("verdict_ok","PASS") if is_pass else T.get("verdict_check","REVISION REQUIRED")
        st.markdown(f"""
        <div class="{vc}">
            <strong style="font-size:1.0rem;">{vt}</strong><br>
            <span style="font-size:0.8rem;color:#8899aa;">
            HIGH: {summary['HIGH']}ea &nbsp; MED: {summary['MED']}ea &nbsp; LOW: {summary['LOW']}ea
            &nbsp;|&nbsp; Correctable: {summary['total_correctable']}ea
           </span>
        </div>
        """, unsafe_allow_html=True)

        tab_list, tab_post, tab_cost = st.tabs([
            T.get("tab_st3_list","🔧 Correction List"),
            T.get("tab_st3_compare","📊 Before & After"),
            T.get("tab_st3_cost","💰 Cost Savings"),
        ])

        with tab_list:
            st.divider()
            priority_icon = {"HIGH": "🔴", "MED": "🟡", "LOW": "🟢"}
            type_label = {
                "uniform_offset": T.get("type_uniform","Uniform Offset"),
                "local_linear":   T.get("type_local","Local Linear"),
                "feature_level":  T.get("type_feature","Feature Correction"),
                "process":        T.get("type_process","Process Adj."),
            }
            for c in corrections:
                icon  = priority_icon.get(c.priority, "⚪")
                ttype = type_label.get(c.correction_type, c.correction_type)
                if c.correction_type in ("uniform_offset", "process"):
                    st.markdown(f"""
                    <div style="background:#111318;border:1px solid #252b36;border-radius:6px;
                        padding:10px 14px;margin:6px 0;">
                    <span style="font-size:1.2rem;">{icon}</span>
                    <strong> {c.id} [{ttype}] {c.feature_name}</strong><br>
                    <span style="font-size:0.78rem;color:#8899aa;">{c.note}</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    clr = "#00d4aa" if c.correction_mm >= 0 else "#ff6b35"
                    dir_s = "+" if c.correction_mm >= 0 else ""
                    st.markdown(f"""
                    <div style="background:#111318;border:1px solid #252b36;border-radius:6px;
                        padding:10px 14px;margin:6px 0;display:flex;align-items:center;gap:12px;">
                    <span style="font-size:1.2rem;">{icon}</span>
                    <div style="flex:1;">
                    <strong style="font-size:0.9rem;">{c.id} [{ttype}] {c.feature_name}</strong><br>
                    <span style="font-size:0.78rem;color:#8899aa;">{c.note}</span><br>
                    <span class="mono" style="font-size:0.72rem;color:#55667a;">
                        Mold: {c.mold_current:.4f} → {c.mold_corrected:.4f} mm
                    </span>
                    </div>
                    <div style="text-align:right;">
                    <span style="font-family:'Space Mono';font-size:1.1rem;font-weight:700;color:{clr};">
                        {dir_s}{c.correction_mm:.4f} mm
                    </span>
                    </div>
                    </div>""", unsafe_allow_html=True)

        with tab_post:
            if not inv["post_correction"].empty:
                post_df = inv["post_correction"]
                st.dataframe(style_verdict_df(post_df, verdict_col="결과"),
                             use_container_width=True, hide_index=True)
                if "Feature" in post_df.columns:
                    fig = go.Figure([
                        go.Bar(name=T.get("label_pre_dev","Pre-Dev."), x=post_df["Feature"],
                               y=post_df["보정 전 편차"].abs(), marker_color="#ff6b35"),
                        go.Bar(name=T.get("label_post_dev","Post-Dev."), x=post_df["Feature"],
                               y=post_df["보정 후 편차"].abs(), marker_color="#00d4aa"),
                    ])
                    fig.update_layout(barmode="group", paper_bgcolor="#0a0c0f",
                                      plot_bgcolor="#111318", font_color="#e2e8f0",
                                      height=300, margin=dict(l=20,r=20,t=30,b=20))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(T.get("msg_no_correction_needed","No correction needed."))

        with tab_cost:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(T.get("metric_trial_before","Current Trials"), f"{cost['trial_before']} times")
            c2.metric(T.get("metric_trial_after", "Expected Trials"), f"{cost['trial_after']} times",
                      delta=f"-{cost['trial_before']-cost['trial_after']}")
            c3.metric(T.get("metric_cost_saving", "Cost Reduction"), f"~{cost['cost_reduction_pct']}%")
            c4.metric(T.get("metric_time_saving", "Time Saving"),    f"~{cost['dev_time_saving_weeks']} wks")

            if not inv["post_correction"].empty:
                st.divider()
                st.markdown(f"#### {T.get('title_export','💾 Export Results')}")
                b1, b2 = st.columns(2)
                with b1:
                    csv_data = inv["post_correction"].to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(T.get("btn_download_csv","📄 Download CSV"),
                                       csv_data, "mold_correction_report.csv", "text/csv",
                                       use_container_width=True)
                with b2:
                    try:
                        cad_script = generate_cad_macro_script(inv["post_correction"])
                        st.download_button(T.get("btn_download_cad","📐 CAD Script"),
                                           cad_script, "AutoCAD_Update.scr", "text/plain",
                                           use_container_width=True)
                    except:
                        pass
