"""
Stage 1: Parting Line Analysis & Auto-Recommendation
Purpose: Analyze 3D model geometry to propose optimal parting line locations and shapes.
Input: Model geometry information, Pull direction
Output: Parting line locations, Complexity assessment, Undercut/Slide necessity
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class PartingLineAnalyzer:
    """Automated Parting Line Analysis & Recommendation Engine"""
    
    def __init__(self, model_geometry: Dict, pull_direction: str = "Z"):
        """
        Args:
            model_geometry: Output from ModelProcessor.analyze_geometry()
            pull_direction: Pull direction ("X", "Y", "Z")
        """
        self.geometry = model_geometry
        self.pull_direction = pull_direction
        self.parting_options = []
        
    def analyze_parting_lines(self) -> Dict:
        """
        Analyzes and evaluates potential parting line positions on the bounding box.
        """
        bounds = self.geometry.get('bounds', {})
        
        if not bounds:
            return {"error": "No geometry data available"}
        
        # Generate 3 basic parting line options
        options = []
        
        # Option 1: XY-Plane (Standard for Z-axis pull)
        xy_plane = self._evaluate_parting_plane(
            name="XY-Plane (Standard)",
            plane_normal=[0, 0, 1],
            bounds=bounds
        )
        options.append(xy_plane)
        
        # Option 2: XZ-Plane
        xz_plane = self._evaluate_parting_plane(
            name="XZ-Plane (Alternative)",
            plane_normal=[0, 1, 0],
            bounds=bounds
        )
        options.append(xz_plane)
        
        # Option 3: YZ-Plane
        yz_plane = self._evaluate_parting_plane(
            name="YZ-Plane (Alternative)",
            plane_normal=[1, 0, 0],
            bounds=bounds
        )
        options.append(yz_plane)
        
        # Sort options by complexity score (lower is better)
        options.sort(key=lambda x: x['complexity_score'])
        
        return {
            "primary_recommendation": options[0],
            "alternatives": options[1:],
            "all_options": options,
        }
    
    def _evaluate_parting_plane(self, name: str, plane_normal: List, bounds: Dict) -> Dict:
        """Evaluates a single parting line option"""
        plane_normal = np.array(plane_normal)
        
        # Calculate depth from boundaries
        min_coords = np.array([bounds['min_x'], bounds['min_y'], bounds['min_z']])
        max_coords = np.array([bounds['max_x'], bounds['max_y'], bounds['max_z']])
        
        # Intersection distance from the plane
        depth = np.dot(max_coords - min_coords, plane_normal)
        
        # Complexity assessment (lower score is better)
        complexity_score = abs(depth) + self._estimate_complexity_factor(name)
        
        return {
            "name": name,
            "plane_normal": plane_normal.tolist(),
            "depth_mm": float(depth),
            "estimated_area_mm2": float(
                bounds['size_x'] * bounds['size_y'] if 'XY' in name else
                bounds['size_x'] * bounds['size_z'] if 'XZ' in name else
                bounds['size_y'] * bounds['size_z']
            ),
            "complexity_score": float(complexity_score),
            "complexity_level": self._complexity_label(complexity_score),
            "advantages": self._get_advantages(name),
            "disadvantages": self._get_disadvantages(name),
        }
    
    def _estimate_complexity_factor(self, name: str) -> float:
        """Additional complexity factor based on parting line type"""
        factors = {
            "XY-Plane (Standard)": 0.8,      # Most standard
            "XZ-Plane (Alternative)": 1.2,
            "YZ-Plane (Alternative)": 1.2,
        }
        return factors.get(name, 1.0)
    
    def _complexity_label(self, score: float) -> str:
        """Complexity level label"""
        if score < 50:
            return "Very Simple"
        elif score < 100:
            return "Simple"
        elif score < 200:
            return "Moderate"
        elif score < 400:
            return "Complex"
        else:
            return "Very Complex"
    
    def _get_advantages(self, name: str) -> List[str]:
        """Advantages per parting line type"""
        advantages = {
            "XY-Plane (Standard)": [
                "Standard design methodology",
                "Minimum mold manufacturing cost",
                "High productivity",
            ],
            "XZ-Plane (Alternative)": [
                "Utilizes part height",
                "Minimum draft requirements possible",
            ],
            "YZ-Plane (Alternative)": [
                "Utilizes lateral symmetry",
                "Minimum undercut potential",
            ],
        }
        return advantages.get(name, [])
    
    def _get_disadvantages(self, name: str) -> List[str]:
        """Disadvantages per parting line type"""
        disadvantages = {
            "XY-Plane (Standard)": [
                "Potential for edging at top/bottom separation",
            ],
            "XZ-Plane (Alternative)": [
                "Complex mold structure",
                "Increased manufacturing difficulty",
            ],
            "YZ-Plane (Alternative)": [
                "Increased manufacturing cost",
                "Difficult precision management",
            ],
        }
        return disadvantages.get(name, [])
    
    def recommend_parting_line(self) -> Dict:
        """Provides a single optimal parting line recommendation"""
        analysis = self.analyze_parting_lines()
        
        if "error" in analysis:
            return analysis
        
        primary = analysis['primary_recommendation']
        
        return {
            "recommended_parting_line": primary['name'],
            "plane_normal": primary['plane_normal'],
            "estimated_depth": primary['depth_mm'],
            "estimated_parting_area": primary['estimated_area_mm2'],
            "complexity": primary['complexity_level'],
            "confidence": self._calculate_confidence(primary['complexity_score']),
            "next_steps": [
                "Execute Undercut Analysis",
                "Verify Slide/Core Necessity",
                "Validate Mold Strength",
            ]
        }
    
    def _calculate_confidence(self, complexity_score: float) -> str:
        """Parting line recommendation confidence score"""
        if complexity_score < 50:
            return "Very High (95%+)"
        elif complexity_score < 100:
            return "High (85-95%)"
        elif complexity_score < 200:
            return "Medium (70-85%)"
        else:
            return "Low (<70%)"
    
    def predict_flash_risk(self) -> Dict:
        """
        Predicts flash risk at the parting line.
        Higher complexity in the parting surface increases flash risk.
        """
        analysis = self.analyze_parting_lines()
        primary = analysis['primary_recommendation']
        
        complexity = primary['complexity_score']
        area = primary['estimated_area_mm2']
        
        # Risk calculation
        flash_risk = (complexity / 200) * 0.6 + (area / 10000) * 0.4
        
        if flash_risk < 0.3:
            risk_level = "Low"
            recommendation = "Standard cleaning sufficient"
        elif flash_risk < 0.6:
            risk_level = "Medium"
            recommendation = "Excellent parting surface finish mandatory"
        else:
            risk_level = "High"
            recommendation = "Special undercut treatment or guide pin addition recommended"
        
        return {
            "flash_risk_level": risk_level,
            "flash_risk_score": round(flash_risk, 2),
            "recommendation": recommendation,
            "mitigation_measures": self._get_mitigation_measures(risk_level),
        }
    
    def _get_mitigation_measures(self, risk_level: str) -> List[str]:
        """Mitigation measures per risk level"""
        measures = {
            "Low": [
                "Standard parting surface cleaning",
                "Standard clamping pressure",
            ],
            "Medium": [
                "Precision parting surface finishing (Ra < 0.8 μm)",
                "Use high-quality mold release agents",
                "Monitor clamping pressure",
            ],
            "High": [
                "Ultra-precision parting surface machining",
                "Special coatings (TiN, CrN, etc.)",
                "Guide pin / Undercut treatment",
                "High clamping pressure setting",
            ],
        }
        return measures.get(risk_level, [])


def analyze_parting_line(model_geometry: Dict, pull_direction: str = "Z") -> Dict:
    """
    Integrated Parting Line Analysis Function (For app.py integration)
    
    Args:
        model_geometry: Geometry output from ModelProcessor
        pull_direction: Pull direction
    
    Returns:
        dict: Complete parting line analysis results
    """
    analyzer = PartingLineAnalyzer(model_geometry, pull_direction)
    
    parting_analysis = analyzer.analyze_parting_lines()
    recommendation = analyzer.recommend_parting_line()
    flash_risk = analyzer.predict_flash_risk()
    
    return {
        "parting_analysis": parting_analysis,
        "recommendation": recommendation,
        "flash_risk_assessment": flash_risk,
    }