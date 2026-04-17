"""
Stage 0-1: 3D Model Processing & Geometric Analysis
목적: STL/STEP 파일 업로드 후 형상 정보 추출 및 기본 분석
입력: STL/STEP 파일
출력: 부피, 표면적, 바운딩박스, 주요 축 정보, 언더컷 위치 예측
"""

import numpy as np
import struct
from typing import Dict, List, Tuple, Optional


class ModelProcessor:
    """3D 모델 처리 및 분석 클래스"""
    
    def __init__(self, file_path: str):
        """
        Args:
            file_path: STL/STEP 파일 경로
        """
        self.file_path = file_path
        self.vertices = []
        self.triangles = []
        self.bounds = None
        self.volume = 0.0
        self.surface_area = 0.0
        self.centroid = None
        
    def load_stl(self) -> bool:
        """STL 파일 로드 (Binary/ASCII 모두 지원)"""
        try:
            # Binary STL 시도
            with open(self.file_path, 'rb') as f:
                header = f.read(80)
                n_triangles = struct.unpack('I', f.read(4))[0]
                
                for _ in range(n_triangles):
                    f.read(12)  # normal vector (skip)
                    v1 = struct.unpack('fff', f.read(12))
                    v2 = struct.unpack('fff', f.read(12))
                    v3 = struct.unpack('fff', f.read(12))
                    f.read(2)   # attribute byte count
                    
                    self.triangles.append([len(self.vertices), 
                                         len(self.vertices)+1, 
                                         len(self.vertices)+2])
                    self.vertices.extend([v1, v2, v3])
            
            self.vertices = np.array(self.vertices)
            self.triangles = np.array(self.triangles)
            return True
            
        except:
            # ASCII STL 시도
            try:
                vertices = []
                triangles = []
                with open(self.file_path, 'r') as f:
                    for line in f:
                        if 'vertex' in line:
                            parts = line.split()
                            vertices.append([float(x) for x in parts[-3:]])
                        elif 'endfacet' in line and len(vertices) % 3 == 0:
                            idx = len(self.vertices)
                            triangles.append([idx, idx+1, idx+2])
                
                self.vertices = np.array(vertices)
                self.triangles = np.array(triangles)
                return True
            except:
                return False
    
    def analyze_geometry(self) -> Dict:
        """기하학적 특성 분석"""
        if len(self.vertices) == 0:
            return {}
        
        # 바운딩박스
        self.bounds = {
            'min': self.vertices.min(axis=0),
            'max': self.vertices.max(axis=0),
            'size': self.vertices.max(axis=0) - self.vertices.min(axis=0),
        }
        
        # 무게중심
        self.centroid = self.vertices.mean(axis=0)
        
        # 부피 계산 (Signed Volume Method)
        self.volume = self._calculate_volume()
        
        # 표면적 계산
        self.surface_area = self._calculate_surface_area()
        
        return {
            'vertices_count': len(self.vertices),
            'triangles_count': len(self.triangles),
            'bounds': {
                'min_x': float(self.bounds['min'][0]),
                'min_y': float(self.bounds['min'][1]),
                'min_z': float(self.bounds['min'][2]),
                'max_x': float(self.bounds['max'][0]),
                'max_y': float(self.bounds['max'][1]),
                'max_z': float(self.bounds['max'][2]),
                'size_x': float(self.bounds['size'][0]),
                'size_y': float(self.bounds['size'][1]),
                'size_z': float(self.bounds['size'][2]),
            },
            'centroid': {
                'x': float(self.centroid[0]),
                'y': float(self.centroid[1]),
                'z': float(self.centroid[2]),
            },
            'volume_mm3': float(self.volume),
            'volume_cm3': float(self.volume / 1000),
            'surface_area_mm2': float(self.surface_area),
        }
    
    def _calculate_volume(self) -> float:
        """Divergence theorem을 이용한 부피 계산"""
        if len(self.triangles) == 0:
            return 0.0
        
        volume = 0.0
        for tri in self.triangles:
            v0 = self.vertices[tri[0]]
            v1 = self.vertices[tri[1]]
            v2 = self.vertices[tri[2]]
            
            # Signed volume of tetrahedron
            volume += np.dot(v0, np.cross(v1, v2))
        
        return abs(volume) / 6.0
    
    def _calculate_surface_area(self) -> float:
        """표면적 계산"""
        if len(self.triangles) == 0:
            return 0.0
        
        area = 0.0
        for tri in self.triangles:
            v0 = self.vertices[tri[0]]
            v1 = self.vertices[tri[1]]
            v2 = self.vertices[tri[2]]
            
            # 삼각형 면적 = |cross product| / 2
            edge1 = v1 - v0
            edge2 = v2 - v0
            area += np.linalg.norm(np.cross(edge1, edge2)) / 2.0
        
        return area
    
    def detect_draft_surfaces(self, pull_direction: str = "Z") -> Dict:
        """
        따기 적절한 면과 부적절한 면 감지
        pull_direction: "X", "Y", "Z" (사출 방향)
        """
        if len(self.triangles) == 0:
            return {"good_draft": 0, "poor_draft": 0, "critical": 0}
        
        # Pull direction 벡터
        pull_vec = np.zeros(3)
        if pull_direction == "X":
            pull_vec = np.array([1, 0, 0])
        elif pull_direction == "Y":
            pull_vec = np.array([0, 1, 0])
        else:  # Z
            pull_vec = np.array([0, 0, 1])
        
        good_draft = 0
        poor_draft = 0
        critical = 0
        
        for tri in self.triangles:
            v0 = self.vertices[tri[0]]
            v1 = self.vertices[tri[1]]
            v2 = self.vertices[tri[2]]
            
            # 삼각형 법선 계산
            normal = np.cross(v1 - v0, v2 - v0)
            if np.linalg.norm(normal) > 0:
                normal = normal / np.linalg.norm(normal)
            
            # Pull direction과의 각도
            angle = np.degrees(np.arccos(np.clip(np.dot(normal, pull_vec), -1, 1)))
            
            # Draft angle 판정 (90도 기준)
            if abs(angle - 90) <= 1.0:  # ±1도
                critical += 1
            elif abs(angle - 90) <= 5.0:  # ±5도
                poor_draft += 1
            else:
                good_draft += 1
        
        return {
            "good_draft": good_draft,
            "poor_draft": poor_draft,
            "critical": critical,
            "total_triangles": len(self.triangles),
            "good_draft_pct": round(100 * good_draft / len(self.triangles), 1),
        }
    
    def estimate_undercut_regions(self, pull_direction: str = "Z") -> List[Dict]:
        """
        언더컷 가능성이 높은 영역 추정
        반환: 언더컷 위치별 정보 리스트
        """
        pull_vec = np.zeros(3)
        axis_idx = {"X": 0, "Y": 1, "Z": 2}.get(pull_direction, 2)
        pull_vec[axis_idx] = 1
        
        undercut_regions = []
        
        # 샘플링을 통한 언더컷 감지
        for i, tri in enumerate(self.triangles[:min(100, len(self.triangles))]):
            v0 = self.vertices[tri[0]]
            v1 = self.vertices[tri[1]]
            v2 = self.vertices[tri[2]]
            
            normal = np.cross(v1 - v0, v2 - v0)
            if np.linalg.norm(normal) > 0:
                normal = normal / np.linalg.norm(normal)
            
            # Undercut: normal의 pull direction 성분이 음수
            pull_component = np.dot(normal, pull_vec)
            
            if pull_component < -0.3:  # 강한 언더컷 지시자
                tri_center = (v0 + v1 + v2) / 3
                undercut_regions.append({
                    "location": {
                        "x": float(tri_center[0]),
                        "y": float(tri_center[1]),
                        "z": float(tri_center[2]),
                    },
                    "severity": "HIGH" if pull_component < -0.6 else "MEDIUM",
                    "pull_component": float(pull_component),
                })
        
        return undercut_regions
    
    def get_optimal_pull_direction(self) -> Dict:
        """부품의 기하학적 특성을 고려한 최적 사출 방향 제안"""
        if self.bounds is None:
            return {}
        
        size = self.bounds['size']
        sizes = {
            'X': float(size[0]),
            'Y': float(size[1]),
            'Z': float(size[2]),
        }
        
        # 가장 작은 면적이 facing하는 방향을 선택
        min_axis = min(sizes, key=sizes.get)
        
        return {
            'recommended_pull': min_axis,
            'reason': f'최소 크기: {min_axis} 축 ({sizes[min_axis]:.1f}mm)',
            'alternative': [ax for ax in ['X', 'Y', 'Z'] if ax != min_axis],
            'dimensions': sizes,
        }


def process_uploaded_model(file_path: str, pull_direction: str = "Z") -> Dict:
    """
    업로드된 모델 파일 처리 및 분석 (app.py 통합용)
    
    Args:
        file_path: STL/STEP 파일 경로
        pull_direction: 사출 방향 ("X", "Y", "Z")
    
    Returns:
        dict: 전체 분석 결과
    """
    processor = ModelProcessor(file_path)
    
    if not processor.load_stl():
        return {"error": "Failed to load model file"}
    
    geometry = processor.analyze_geometry()
    draft_analysis = processor.detect_draft_surfaces(pull_direction)
    undercut_regions = processor.estimate_undercut_regions(pull_direction)
    optimal_pull = processor.get_optimal_pull_direction()
    
    return {
        "geometry": geometry,
        "draft_analysis": draft_analysis,
        "undercut_regions": undercut_regions,
        "optimal_pull_direction": optimal_pull,
        "pull_direction_used": pull_direction,
    }
