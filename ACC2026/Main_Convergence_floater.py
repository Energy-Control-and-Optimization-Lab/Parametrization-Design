#%%     MAIN SCRIPT FOR EXECUTION OF ECO FUNCTIONS
import numpy as np
import shutil
import os
import sys
import matplotlib.pyplot as plt

# Add EcoFunctions to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'EcoFunctions'))

from EcoFunctions.Eco_StlRev import generate_revolution_solid_stl
from EcoFunctions.Eco_Cap2B import analyze_two_body_hydrodynamics
from EcoFunctions.Eco_Cap1B import analyze_single_body_hydrodynamics

#%%     MAIN CONFIGURATION
# ==================
FOLDER_NAME = "batch_convergence"  # Main results folder name
os.makedirs(FOLDER_NAME, exist_ok=True)
print(f"\nMain batch folder: {FOLDER_NAME}")

#%%     PARAMETERS
R = 10  # R parameter values
#R_values = np.array([10])  # R parameter values
D = 273 / ((R**2) - 9)             # D calculated for each R
frequencies = np.linspace(0.2, 1.8, 500)  # Frequency range [rad/s]
#frequencies = np.linspace(0.2, 1.8, 5)  # Frequency range [rad/s]

# STL Generation Parameters
NUM_SEGMENTS = np.array([40, 50, 60, 70, 80])         # Revolution segments (circumferential)
Z_SUBDIVISIONS = np.array([6, 8, 10, 12, 14])        # Z-axis subdivisions per segment (configurable mesh density in Z direction)


#%%  HYDRODYNAMICAL ANALYSIS
for i, (NS, ZS) in enumerate(zip(NUM_SEGMENTS, Z_SUBDIVISIONS)):
    # Create folder name with R value
    folder_name = f"MESH_Seg{NS}_Zsub{ZS}"
    folder_path = os.path.join(FOLDER_NAME, folder_name)
    
    
    # Remove existing folder
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Existing folder {folder_path} removed")
    
    # Copy Default folder
    if os.path.exists("Default"):
        shutil.copytree("Default", folder_path)
        print(f"Default folder copied as {folder_path}")
    else:
        print("⚠ Default folder not found")
        continue
    
    # Define geometry points with current R and D values
    P = np.array([
        [3, 0, -D],   # P1
        [R, 0, -D],   # P2
        [R, 0, 2],    # P3
        [3, 0, 2],    # P4
    ])
    
    
    # Create geometry folder
    geometry_path = os.path.join(folder_path, "geometry")
    os.makedirs(geometry_path, exist_ok=True)
    
    # Remove existing float.stl
    float_stl_path = os.path.join(geometry_path, "float.stl")
    if os.path.exists(float_stl_path):
        try:
            os.remove(float_stl_path)
            print(f"File {float_stl_path} removed")
        except PermissionError:
            print(f"Could not remove {float_stl_path} - file in use")
    
    # Change directory temporarily to save STL
    current_dir = os.getcwd()
    os.chdir(geometry_path)
    
    try:
        # Generate revolution solid with configurable Z subdivisions
        result = generate_revolution_solid_stl(
            points=P,
            filename="float.stl",
            num_segments=NS,
            z_subdivisions=ZS,  # Now configurable!
            visualize=False,  # No plots during batch processing
            save_plot_path=os.getcwd()  # Save plot in current geometry folder
        )
        
        print(f"✅ STL file generated: {result['filename']}")
        print(f"   Vertices: {result['num_vertices']:,}")
        print(f"   Triangles: {result['num_triangles']:,}")
        print(f"   Original points: {result['num_original_points']}")
        print(f"   Profile points after subdivision: {result['num_profile_points']}")
        print(f"   Z subdivisions per segment: {result['subdivision_factor']}")
        
    except Exception as e:
        print(f"⚠ Error generating STL: {e}")
        os.chdir(current_dir)
        continue
    
    # Return to original directory
    os.chdir(current_dir)
    
    # Run hydrodynamic analysis
    try:
        print("🌊 Starting hydrodynamic analysis...")
        
        # Define mesh paths
        mesh1_path = os.path.join(folder_path, "geometry", "float.stl")

        # Verify both files exist
        if not os.path.exists(mesh1_path):
            print(f"⚠ Error: {mesh1_path} not found")
            continue

        
        # Output directory for hydrodynamic data
        hydro_output_dir = os.path.join(folder_path, "hydroData")
        
        # Run hydrodynamic analysis
        results = analyze_single_body_hydrodynamics(
            mesh_path=mesh1_path,  # Solo necesitas un mesh path
            frequency_range=frequencies,
            mesh_position=[0.0, 0.0, 0.0],  # Solo una posición
            body_name="Float",  # Solo un nombre de cuerpo
            output_directory=hydro_output_dir,
            nc_filename="single_body.nc",  # Cambiado el nombre por defecto
            plot_xlim=[-15, 15],
            plot_ylim=[-5, 5],
            save_plots=True,
            show_plots=False,  # No plots during batch processing
            logging_level="WARNING",  # Reduce output for batch processing
        )
        
        # Show key results
        RAO = results['RAO']
        
        print(f"✅ Analysis completed for {len(frequencies)} frequencies")
        
        # Verify output files after analysis
        hydro_files = ['HydCoeff.npz', 'HydCoeff.pkl', 'HydCoeff.mat']
        print(f"🗂 Verifying output files...")
        files_found = []
        for file in hydro_files:
            file_path = os.path.join(hydro_output_dir, file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"   ✅ {file}: {file_size:,} bytes")
                files_found.append(file)
            else:
                print(f"   ⚠ {file}: NOT FOUND")
        
        if files_found:
            print(f"✅ {len(files_found)} output files created successfully")
        else:
            print(f"⚠ NO output files were created")
            print(f"   Verify that Eco_Cap2B.py is updated with corrected version")
        
    except Exception as e:
        print(f"⚠ CRITICAL ERROR in hydrodynamic analysis for {folder_path}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print(f"   Continuing with next case...")
        continue

# Final summary

print(f"\n🔧 Mesh Generation Parameters:")
print(f"  - Circumferential segments: {NUM_SEGMENTS}")
print(f"  - Z-axis subdivisions per segment: {Z_SUBDIVISIONS}")

print(f"\n✅ Analysis complete! Check results in '{FOLDER_NAME}/' folder")
