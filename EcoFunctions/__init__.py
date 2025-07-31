# EcoFunctions/__init__.py (English)
"""
EcoFunctions package - Hydrodynamic analysis tools
Author: Pablo Antonio Matamala Carvajal
"""

__version__ = "1.0.0"
__author__ = "Pablo Antonio Matamala Carvajal"

from .Eco_StlRev import generate_revolution_solid_stl
from .Eco_Cap2B import analyze_two_body_hydrodynamics
from .Eco_Bemio import run_bemio_postprocessing, check_bemio_availability
from .Eco_Spectrum import create_spectrum, SEA_STATES

__all__ = [
    'generate_revolution_solid_stl',
    'analyze_two_body_hydrodynamics',
    'run_bemio_postprocessing',
    'check_bemio_availability',
    'create_spectrum',
    'SEA_STATES'
]