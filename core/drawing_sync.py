# core/drawing_sync.py
import pandas as pd
import io

def load_drawing_features_from_csv(file_buffer):
    """
    사용자가 업로드한 도면 피처 CSV를 파싱 (옵션 D)
    필수 컬럼: Feature, 공칭치수, +공차, -공차, 현재금형치수, x좌표, y좌표
    """
    df = pd.read_csv(file_buffer)
    required = ['Feature', '공칭치수', '+공차', '-공차', '현재금형치수']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 다음 컬럼이 부족해: {missing}")
        
    return df

def generate_cad_macro_script(post_correction_df):
    """
    보정된 치수를 AutoCAD 등에서 자동으로 텍스트 수정할 수 있는 매크로 스크립트 생성 (옵션 C 대안)
    예: AutoCAD LISP 또는 Script 포맷
    """
    script_lines = []
    script_lines.append(";; AutoCAD Dimension Update Script")
    
    for _, row in post_correction_df.iterrows():
        feature = row['Feature']
        corrected_dim = row.get('보정 후 금형치수', row.get('보정 후 편차', 0)) # 역설계 결과값 매핑 필요
        
        # 가상의 CAD 명령어 (도면의 특정 블록이나 텍스트를 찾아 바꾸는 스크립트)
        script_lines.append(f'-FIND "{feature}_DIM" "" "{corrected_dim:.3f}"')
        
    return "\n".join(script_lines)