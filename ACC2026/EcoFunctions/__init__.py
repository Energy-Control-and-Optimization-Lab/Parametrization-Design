# EcoFunctions/__init__.py (English)
"""
EcoFunctions package - Hydrodynamic analysis tools
Author: Pablo Antonio Matamala Carvajal
"""

__version__ = "2.0.0"
__author__ = "Pablo Antonio Matamala Carvajal"

from .Eco_StlRev import generate_revolution_solid_stl
from .Eco_Cap2B import analyze_two_body_hydrodynamics
from .Eco_Cap1B import analyze_single_body_hydrodynamics

__all__ = [
    'generate_revolution_solid_stl',
    'analyze_two_body_hydrodynamics',
    'analyze_single_body_hydrodynamics',
]
