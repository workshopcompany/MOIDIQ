"""
Stage 1: Slide & Core Optimization Engine
Purpose: Analyze undercut locations to propose slide and core designs.
Input: Model geometry, Undercut information, Parting line
Output: Required number of slides, directions, dimensions, and cost estimates.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class SlideCoreOptimizer:
    """Optimization Engine for Slide and Core Design"""
    
    def __init__(self, model_geometry: Dict, undercut_regions: List[Dict], 
                 parting_analysis: Dict):
        """
        Args:
            model_geometry: Output from ModelProcessor.analyze_geometry()
            undercut_regions: Output from ModelProcessor.estimate_undercut_regions()
            parting_analysis: Analysis result from PartingLineAnalyzer
        """
        self.geometry = model_geometry
        self.undercut_regions = undercut_regions
        self.parting_analysis = parting_analysis
        self.slides = []
        self.cores = []
    
    def analyze_undercut_requirements(self) -> Dict:
        """
        Analyzes undercut distribution and determines necessity for slides/cores.
        """
        if not self.undercut_regions:
            return {
                "has_undercut": False,
                "slide_required": False,
                "core_required": False,
                "recommendation": "No sliding mechanism required",
            }
        
        # Categorize undercuts by severity
        high_severity = [u for u in self.undercut_regions if u['severity'] == 'HIGH']
        medium_severity = [u for u in self.undercut_regions if u['severity'] == 'MEDIUM']
        
        has_undercut = len(self.undercut_regions) > 0
        slide_required = len(high_severity) > 0
        core_required = len(medium_severity) > 0
        
        return {
            "has_undercut": has_undercut,
            "total_undercut_regions": len(self.undercut_regions),
            "high_severity_count": len(high_severity),
            "medium_severity_count": len(medium_severity),
            "slide_required": slide_required,
            "core_required": core_required,
            "high_severity_regions": high_severity,
            "medium_severity_regions": medium_severity,
        }
    
    def design_slides(self) -> List[Dict]:
        """
        Designs slides based on undercut locations.
        """
        undercut_analysis = self.analyze_undercut_requirements()
        
        if not undercut_analysis['slide_required']:
            return []
        
        slides = []
        high_severity = undercut_analysis['high_severity_regions']
        
        # Design slides for each undercut location
        for idx, undercut in enumerate(high_severity):
            slide = self._design_single_slide(idx + 1, undercut)
            slides.append(slide)
        
        # Check for interference between slides
        slides = self._check_slide_interference(slides)
        
        self.slides = slides
        return slides
    
    def _design_single_slide(self, slide_id: int, undercut: Dict) -> Dict:
        """Designs a single slide mechanism"""
        location = undercut['location']
        severity = undercut['severity']
        
        # Determine slide size based on undercut severity
        size_factor = 1.5 if severity == 'HIGH' else 1.0
        
        # Estimated slide dimensions (mm)
        base_thickness = 8 + (size_factor * 4)
        base_width = 30 + (size_factor * 20)
        base_length = 40 + (size_factor * 25)
        
        return {
            "slide_id": f"SLD-{slide_id:02d}",
            "location": location,
            "severity": severity,
            "estimated_dimensions": {
                "thickness_mm": round(base_thickness, 1),
                "width_mm": round(base_width, 1),
                "length_mm": round(base_length, 1),
            },
            "pull_direction": self._estimate_pull_direction(undercut),
            "material_grade": "SKD61" if severity == 'HIGH' else "SKD11",
            "estimated_cost_usd": self._estimate_slide_cost(base_thickness, base_width, base_length),
            "lead_time_days": 10 + (int(base_length) // 10),
        }
    
    def _estimate_pull_direction(self, undercut: Dict) -> str:
        """Estimates slide travel direction based on undercut location"""
        # Simple heuristic based on absolute value of pull component
        pull_component = abs(undercut['pull_component'])
        
        if pull_component > 0.7:
            return "Front-Back (Preferred)"
        elif pull_component > 0.4:
            return "Side-to-Side"
        else:
            return "Diagonal"
    
    def _estimate_slide_cost(self, thickness: float, width: float, length: float) -> float:
        """Estimates slide manufacturing cost (USD)"""
        volume = thickness * width * length / 1000  # cm³
        base_cost = 150  # Base cost
        cost_per_volume = 2.5  # Cost per cm³
        return round(base_cost + volume * cost_per_volume, 0)
    
    def _check_slide_interference(self, slides: List[Dict]) -> List[Dict]:
        """Checks and adjusts for interference between slides"""
        for i, slide1 in enumerate(slides):
            for j, slide2 in enumerate(slides[i+1:], start=i+1):
                distance = self._calculate_slide_distance(slide1, slide2)
                if distance < 5:  # Requires a minimum clearance of 5mm
                    # Add interference flags
                    slide1['interference_warning'] = f"Slide #{j+1} within proximity"
                    slide2['interference_warning'] = f"Slide #{i+1} within proximity"
        
        return slides
    
    def _calculate_slide_distance(self, slide1: Dict, slide2: Dict) -> float:
        """Calculates distance between two slides"""
        loc1 = slide1['location']
        loc2 = slide2['location']
        distance = np.sqrt(
            (loc1['x'] - loc2['x'])**2 + 
            (loc1['y'] - loc2['y'])**2 + 
            (loc1['z'] - loc2['z'])**2
        )
        return distance
    
    def design_cores(self) -> List[Dict]:
        """
        Designs cores based on undercut locations.
        """
        undercut_analysis = self.analyze_undercut_requirements()
        
        if not undercut_analysis['core_required']:
            return []
        
        cores = []
        medium_severity = undercut_analysis['medium_severity_regions']
        
        for idx, undercut in enumerate(medium_severity):
            core = self._design_single_core(idx + 1, undercut)
            cores.append(core)
        
        self.cores = cores
        return cores
    
    def _design_single_core(self, core_id: int, undercut: Dict) -> Dict:
        """Designs a single core mechanism"""
        location = undercut['location']
        
        # Basic core dimensions
        core_diameter = 12  # mm
        core_length = 25   # mm
        
        return {
            "core_id": f"CORE-{core_id:02d}",
            "type": "Lifter" if undercut['severity'] == 'MEDIUM' else "Side Core",
            "location": location,
            "estimated_dimensions": {
                "diameter_mm": core_diameter,
                "length_mm": core_length,
            },
            "material_grade": "SKD11",
            "cooling_required": False,
            "estimated_cost_usd": round(80 + (core_diameter * core_length * 0.5), 0),
        }
    
    def get_design_summary(self) -> Dict:
        """Comprehensive summary of slide/core design"""
        undercut_analysis = self.analyze_undercut_requirements()
        slides = self.design_slides()
        cores = self.design_cores()
        
        total_cost = sum(s['estimated_cost_usd'] for s in slides) + \
                     sum(c['estimated_cost_usd'] for c in cores)
        
        max_lead_time = max(
            [s['lead_time_days'] for s in slides] + [0],
            default=0
        )
        
        return {
            "undercut_summary": undercut_analysis,
            "slide_design": {
                "count": len(slides),
                "slides": slides,
            },
            "core_design": {
                "count": len(cores),
                "cores": cores,
            },
            "cost_estimate": {
                "slides_cost_usd": round(sum(s['estimated_cost_usd'] for s in slides), 0),
                "cores_cost_usd": round(sum(c['estimated_cost_usd'] for c in cores), 0),
                "total_mechanism_cost_usd": round(total_cost, 0),
                "percentage_of_mold": round(total_cost / 5000 * 100, 1),  # Assumed total mold cost is 5000
            },
            "lead_time": {
                "critical_path_days": max_lead_time,
                "final_assembly_days": max_lead_time + 3,
            },
            "complexity_assessment": self._assess_complexity(slides, cores),
            "recommendations": self._get_design_recommendations(slides, cores),
        }
    
    def _assess_complexity(self, slides: List[Dict], cores: List[Dict]) -> Dict:
        """Evaluates design complexity"""
        total_mechanisms = len(slides) + len(cores)
        
        if total_mechanisms == 0:
            complexity = "Simple"
            score = 1
        elif total_mechanisms <= 2:
            complexity = "Moderate"
            score = 2
        elif total_mechanisms <= 4:
            complexity = "Complex"
            score = 3
        else:
            complexity = "Very Complex"
            score = 4
        
        return {
            "level": complexity,
            "score": score,
            "total_mechanisms": total_mechanisms,
            "total_moving_parts": total_mechanisms * 2,
        }
    
    def _get_design_recommendations(self, slides: List[Dict], cores: List[Dict]) -> List[str]:
        """Design recommendations"""
        recommendations = []
        
        if len(slides) > 0:
            recommendations.append(f"✓ {len(slides)} Slide(s) Required — Semi-auto execution system recommended")
        
        if len(cores) > 0:
            recommendations.append(f"✓ {len(cores)} Lifter/Core design(s) required")
        
        if len(slides) > 2:
            recommendations.append("⚠️ High slide count — Part design review for optimization recommended")
        
        if len(self.undercut_regions) == 0:
            recommendations.append("✓ No undercuts — Standard 2-plate mold is sufficient")
        
        return recommendations


def optimize_mold_design(model_geometry: Dict, undercut_regions: List[Dict], 
                         parting_analysis: Dict) -> Dict:
    """
    Integrated Slide/Core Optimization Function (For app.py integration)
    
    Args:
        model_geometry: Output from ModelProcessor
        undercut_regions: Output from ModelProcessor
        parting_analysis: Output from PartingLineAnalyzer
    
    Returns:
        dict: Complete slide/core optimization results
    """
    optimizer = SlideCoreOptimizer(model_geometry, undercut_regions, parting_analysis)
    return optimizer.get_design_summary()