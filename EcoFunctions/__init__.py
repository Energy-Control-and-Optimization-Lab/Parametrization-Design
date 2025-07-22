# EcoFunctions package initialization
# This file makes the EcoFunctions directory a Python package

__version__ = "1.0.0"
__author__ = "Pablo Antonio Matamala Carvajal"

# Import main functions for easy access
from .Eco_StlRev import generate_revolution_solid_stl
from .Eco_Cap2B import analyze_two_body_hydrodynamics
from .Eco_Bemio import run_bemio_postprocessing, check_bemio_availability

__all__ = [
    'generate_revolution_solid_stl',
    'analyze_two_body_hydrodynamics',
    'run_bemio_postprocessing',
    'check_bemio_availability'
]