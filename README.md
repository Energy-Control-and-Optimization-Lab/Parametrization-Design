# EcoFunctions - Hydrodynamic Analysis Tools with DOE

**Version:** 2.1.0  
**Author:** Pablo Antonio Matamala Carvajal  
**Date:** January 2025

---

## 📋 Overview

**EcoFunctions** is a Python package for parametric hydrodynamic analysis of Wave Energy Converters (WECs) with integrated Design of Experiments (DOE) capabilities. The package provides tools for:

- ✅ **Automated STL geometry generation** from parametric profiles
- ✅ **Boundary Element Method (BEM)** hydrodynamic analysis using Capytaine
- ✅ **Design of Experiments (DOE)** for systematic parametric studies
- ✅ **Single and two-body** floating structure analysis
- ✅ **Response Amplitude Operator (RAO)** calculations
- ✅ **Multi-format data export** (NPZ, PKL, MAT, NetCDF)

---

## 🚀 Quick Start

### Installation

1. **Clone or download** the repository
2. **Install required packages:**

```bash
pip install numpy matplotlib scipy pandas capytaine xarray netCDF4 h5netcdf
```

3. **Verify installation:**

```bash
python -c "import capytaine; print('Capytaine OK')"
```

### Basic Usage

```python
from EcoFunctions import generate_revolution_solid_stl, analyze_two_body_hydrodynamics, generate_doe_vectors

# Generate geometry
P = [[3,0,-5], [10,0,-5], [10,0,2], [3,0,2]]
result = generate_revolution_solid_stl(P, filename="float.stl")

# Run hydrodynamic analysis
frequencies = np.linspace(0.2, 1.8, 50)
results = analyze_two_body_hydrodynamics(
    mesh1_path="float.stl",
    mesh2_path="spar.stl",
    frequency_range=frequencies
)

# Generate DOE experiments
wec_params = {'D1': [5,15], 'D2': [0,2], 'D3': [6,13]}
doe_results = generate_doe_vectors(wec_params, method='Box-Behnken')
```

---

## 📦 Package Structure

```
EcoFunctions/
├── __init__.py              # Package initialization
├── Eco_StlRev.py           # STL geometry generation
├── Eco_Cap1B.py            # Single-body hydrodynamic analysis
├── Eco_Cap2B.py            # Two-body hydrodynamic analysis
└── Eco_DOE.py              # Design of Experiments tools

Main scripts/
├── Main.py                  # Standard parametric analysis
├── Main_DOE.py             # DOE-based parametric analysis
└── Example_DOE_Usage.py    # DOE usage examples
```

---

## 🔧 Core Modules

### 1. Eco_StlRev.py - Geometry Generation

Generate revolution solids (STL format) from 2D profiles with adaptive mesh refinement.

**Key Features:**
- Parametric solid of revolution around Z-axis
- Configurable circumferential and Z-axis subdivisions
- **Adaptive subdivision** for short segments (automatic mesh refinement)
- Profile visualization with original and subdivided points

**Function:**
```python
generate_revolution_solid_stl(
    points,                    # List of [x, y, z] coordinates
    filename="solid.stl",      # Output STL filename
    num_segments=36,           # Circumferential segments
    z_subdivisions=4,          # Z-axis subdivisions per segment
    height_threshold=2.0,      # Height threshold for adaptive subdivision
    min_subdivisions=2,        # Min subdivisions for short segments
    visualize=False,           # Show profile plot
    save_plot_path=None,       # Directory to save plot
    plot_filename="profile.png" # Custom plot filename
)
```

**Example:**
```python
P_float = np.array([
    [3, 0, -10],   # P1
    [8, 0, -10],   # P2
    [8, 0, 2],     # P3
    [3, 0, 2]      # P4
])

result = generate_revolution_solid_stl(
    points=P_float,
    filename="float.stl",
    num_segments=60,
    z_subdivisions=10
)
```

---

### 2. Eco_Cap2B.py - Two-Body Hydrodynamic Analysis

Perform coupled two-body hydrodynamic analysis using Capytaine BEM solver.

**Key Features:**
- Radiation and diffraction problem solving
- Added mass and radiation damping calculation
- Excitation forces (Froude-Krylov + diffraction)
- RAO computation for all degrees of freedom
- Relative motion analysis between bodies
- Multi-format export (NPZ, PKL, MAT, NetCDF)

**Function:**
```python
analyze_two_body_hydrodynamics(
    mesh1_path,                # Path to first body STL
    mesh2_path,                # Path to second body STL
    frequency_range,           # Array of frequencies [rad/s]
    mesh1_position=[0,0,0],    # First body position
    mesh2_position=[0,0,0],    # Second body position
    body_names=["Float","Spar"],
    output_directory="hydroData",
    save_plots=True,
    show_plots=False
)
```

**Returns:**
- Added mass matrices (A)
- Radiation damping matrices (B)
- Excitation forces (Fe)
- RAO for all DOFs
- Inertia and stiffness matrices
- Complete dataset in NetCDF format

---

### 3. Eco_Cap1B.py - Single-Body Hydrodynamic Analysis

Similar to Eco_Cap2B but for single floating bodies (6 DOF).

**Function:**
```python
analyze_single_body_hydrodynamics(
    mesh_path,                 # Path to body STL
    frequency_range,           # Array of frequencies [rad/s]
    mesh_position=[0,0,0],
    body_name="Body",
    output_directory="hydroData"
)
```

---

### 4. Eco_DOE.py - Design of Experiments

Generate optimal experiment designs for parametric studies.

**Key Features:**
- Multiple DOE methods (Box-Behnken, CCD, Full-Factorial, LHS)
- Automatic scaling to physical parameter ranges
- Flexible parameter definition (list or dictionary)
- Comprehensive metadata and design information

**Function:**
```python
generate_doe_vectors(
    parameter_ranges,          # Dict or list of parameter ranges
    method='Box-Behnken',      # DOE method
    n_center_points=None,      # Number of center points (auto-calculated)
    seed=None,                 # Random seed for reproducibility
    parameter_names=None       # Custom parameter names
)
```

**Supported Methods:**
- **Box-Behnken**: Efficient for 3+ parameters, good for response surfaces
- **CCD (Central Composite)**: Classical design, includes axial points
- **Full-Factorial**: Complete coverage (use for ≤4 parameters)
- **LHS (Latin Hypercube)**: Space-filling for many parameters

**Example:**
```python
# Define parameters
wec_parameters = {
    'D1': [5, 15],      # float_base_depth [m]
    'D2': [0, 2],       # additional_offset [m]  
    'D3': [6, 13],      # float_radius [m]
    'D4': [25, 35],     # spar_depth [m]
    'D5': [10, 20],     # spar_top_radius [m]
    'D6': [5, 12]       # spar_top_height [m]
}

# Generate Box-Behnken design
doe_results = generate_doe_vectors(
    parameter_ranges=wec_parameters,
    method='Box-Behnken',
    n_center_points=6,
    seed=42
)

# Use in loop
for experiment in doe_results['design_matrix']:
    D1, D2, D3, D4, D5, D6 = experiment
    # Use parameters to generate geometry and run analysis
```

---

## 🎯 Main Scripts

### Main.py - Standard Parametric Analysis

Traditional nested loop approach for parametric studies.

**Features:**
- Manual parameter grid definition (R_values × Dint_values)
- Generates float and spar geometries
- Runs hydrodynamic analysis for each combination
- Organized output folders

**Use when:** You want simple grid-based parameter exploration

---

### Main_DOE.py - DOE-Based Analysis ⭐ **RECOMMENDED**

Advanced parametric analysis using Design of Experiments.

**Features:**
- **Box-Behnken design** for optimal parameter space coverage
- **6 design parameters** (D1-D6) mapped to geometry
- Automatic experiment generation (63 experiments for 6 parameters)
- JSON summaries for each experiment
- Ready for response surface modeling

**Workflow:**
1. Define parameter ranges
2. Generate DOE design (Box-Behnken)
3. Loop through experiments
4. Create parametric geometries using D1-D6
5. Run hydrodynamic analysis
6. Save results with experiment metadata

**Use when:** You want efficient, systematic optimization with fewer experiments

**Number of Experiments:**
| Parameters | Box-Behnken | Full-Factorial |
|------------|-------------|----------------|
| 3          | 15          | 8              |
| 4          | 27          | 16             |
| 5          | 46          | 32             |
| 6          | 63          | 64             |
| 7          | 90          | 128            |

---

## 📊 Typical Workflow

### 1. Define WEC Design Parameters

```python
wec_parameters = {
    'D1': [5, 15],      # Depth parameter
    'D2': [0, 2],       # Offset parameter
    'D3': [6, 13],      # Radius parameter
    # ... up to D6
}
```

### 2. Generate DOE Experiments

```python
doe_results = generate_doe_vectors(
    wec_parameters, 
    method='Box-Behnken',
    n_center_points=6
)
```

### 3. Run Parametric Analysis

```python
for experiment in doe_results['design_matrix']:
    D1, D2, D3, D4, D5, D6 = experiment
    
    # Create geometry using DOE parameters
    P_float = np.array([[3,0,-D1-D2], [D3,0,-D1], ...])
    P_spar = np.array([[0,0,-D4], [D5,0,-D4], ...])
    
    # Generate STL files
    generate_revolution_solid_stl(P_float, "float.stl")
    generate_revolution_solid_stl(P_spar, "spar.stl")
    
    # Run hydrodynamic analysis
    results = analyze_two_body_hydrodynamics(...)
    
    # Save results
```

### 4. Post-Processing

- Collect all `experiment_summary.json` files
- Build response surface model (e.g., Gaussian Process, Polynomial)
- Perform optimization
- Validate optimal design with full simulation

---

## 📁 Output Structure

```
batch_DOE_WEC/
├── DOE_Exp_001/
│   ├── geometry/
│   │   ├── float.stl
│   │   ├── spar.stl
│   │   ├── float_profile_plot.png
│   │   └── spar_profile_plot.png
│   ├── hydroData/
│   │   ├── HydCoeff.npz
│   │   ├── HydCoeff.pkl
│   │   ├── HydCoeff.mat
│   │   ├── rm3.nc
│   │   ├── RAO_heave_comparison.png
│   │   ├── radiation_damping_coefficients.png
│   │   ├── added_mass_coefficients.png
│   │   └── geometry_lateral_view.png
│   └── experiment_summary.json
├── DOE_Exp_002/
│   └── ...
...
└── DOE_Exp_063/
    └── ...
```

---

## 🔬 Technical Details

### Hydrodynamic Coefficients

**Computed matrices:**
- **A(ω)**: Added mass [kg, kg·m²]
- **B(ω)**: Radiation damping [kg/s, kg·m²/s]
- **Fe(ω)**: Excitation force [N, N·m]
- **M**: Inertia matrix [kg, kg·m²]
- **C**: Hydrostatic stiffness [N/m, N·m/rad]

**Equation of motion:**
```
[M + A(ω)] ẍ + B(ω) ẋ + C x = Fe(ω)
```

**RAO Computation:**
```
RAO(ω) = |Z(ω)⁻¹ Fe(ω)|
where Z(ω) = -ω²[M + A(ω)] + iω B(ω) + C
```

---

## ⚙️ Configuration Tips

### Mesh Quality

**Float geometry:**
```python
NUM_SEGMENTS_float = 60        # Good balance
Z_SUBDIVISIONS_float = 10      # Adequate for most cases
```

**Spar geometry:**
```python
NUM_SEGMENTS_spar = 70         # Finer mesh
Z_SUBDIVISIONS_spar = 20       # With adaptive subdivision
height_threshold = 2.0         # Segments < 2m use fewer subdivisions
min_subdivisions = 3           # Minimum for short segments
```

### Frequency Range

```python
# Standard range for WEC analysis
frequencies = np.linspace(0.2, 1.8, 500)  # [rad/s]

# Quick testing
frequencies = np.linspace(0.2, 1.8, 5)
```

### DOE Configuration

**Center points:**
- **1 point**: Minimum (not recommended)
- **3 points**: Standard minimum
- **6 points**: ⭐ **Recommended** for robust error estimation
- **9+ points**: Overkill for most cases

---

## 📚 Dependencies

**Required packages:**
```
numpy >= 1.20.0
matplotlib >= 3.3.0
scipy >= 1.7.0
pandas >= 1.3.0
capytaine >= 2.0
xarray >= 0.19.0
netCDF4 >= 1.5.7  (or h5netcdf >= 0.14.0)
```

**Install all at once:**
```bash
pip install numpy matplotlib scipy pandas capytaine xarray netCDF4 h5netcdf
```

---

## 🐛 Troubleshooting

### Capytaine installation fails
**Solution:** Install build tools first
```bash
# Windows: Install Visual Studio Build Tools
# Linux:
sudo apt-get install build-essential gfortran
pip install capytaine
```

### NetCDF4 not working
**Solution:** Use h5netcdf as fallback (already implemented)
```bash
pip install h5netcdf
```

### Memory issues with large meshes
**Solution:** Reduce mesh density or frequency points
```python
NUM_SEGMENTS = 40  # Instead of 70
frequencies = np.linspace(0.2, 1.8, 100)  # Instead of 500
```

---

## 📖 Examples

See `Example_DOE_Usage.py` for comprehensive examples including:
- Basic DOE generation
- Integration with Main.py workflow
- Comparison of different DOE methods
- Complete WEC analysis example

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:
1. Document all functions with docstrings
2. Include examples for new features
3. Test with different parameter combinations
4. Update README.md with new functionality

---

## 📄 License

[Add your license information here]

---

## 📧 Contact

**Author:** Pablo Antonio Matamala Carvajal  
**Email:** [Add your email]  
**Institution:** [Add your institution]

---

## 🔗 References

- **Capytaine Documentation:** https://capytaine.github.io/
- **Box-Behnken Design:** Box, G. E. P., & Behnken, D. W. (1960). Some new three level designs for the study of quantitative variables. Technometrics, 2(4), 455-475.
- **BEM Theory:** [Add relevant BEM references]

---

## 📝 Version History

**v2.1.0 (January 2025)**
- ✨ Added DOE capabilities (Eco_DOE.py)
- ✨ Added adaptive mesh subdivision
- ✨ Added Main_DOE.py for DOE-based analysis
- 🐛 Improved NetCDF export compatibility
- 📚 Enhanced documentation

**v2.0.0 (July 2024)**
- 🎉 Initial public release
- ✨ Single and two-body hydrodynamic analysis
- ✨ STL geometry generation
- ✨ Multi-format data export

---

**⭐ If this package helps your research, please consider citing it in your publications!**
