import numpy as np
import shutil
import os
import sys
import matplotlib.pyplot as plt

# Add EcoFunctions to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'EcoFunctions'))

from EcoFunctions.Eco_StlRev import generate_revolution_solid_stl
from EcoFunctions.Eco_Cap2B import analyze_two_body_hydrodynamics

# MAIN CONFIGURATION
# ==================
FOLDER_NAME = "batch"  # Main results folder name

# BEMIO configuration (MATLAB post-processing)
RUN_BEMIO = False  # Set True to run BEMIO after BEM analysis
BEMIO_SCRIPT_PATH = None  # Custom bemio.m script path (None = use internal)

# Parameter ranges
R_values = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]  # Complete list
D_values = [-1.0, -1.4, -1.8, -2.2, -2.6, -3.0]  # Complete list

# For testing, use reduced configurations:
R_values = [3.5]  # Single geometry for testing
D_values = [-1.0]  # Single geometry for testing

# Hydrodynamic analysis configuration
frequencies = np.linspace(0.1, 2.0, 10)  # Reduced to 10 frequencies for testing

# Verify Eco_Cap2B version
print("Verifying Eco_Cap2B version...")
import inspect
lines = inspect.getsource(analyze_two_body_hydrodynamics)
if "Save hydrodynamic coefficients" in lines:
    print("✅ Updated version detected - HydCoeff files will be saved")
else:
    print("❌ Old version detected - NEED TO REPLACE Eco_Cap2B.py")
    print("   HydCoeff files will NOT be saved until code is updated")

# Create main batch folder
os.makedirs(FOLDER_NAME, exist_ok=True)
print(f"Main batch folder: {FOLDER_NAME}")

# Iterate over all R and D combinations
for R in R_values:
    for D in D_values:
        # Create folder name
        R_str = str(int(round(float(R) * 10)))
        D_str = str(int(round(abs(float(D)) * 10)))
        folder_name = f"Geometry_R{R_str}_D{D_str}"
        folder_path = os.path.join(FOLDER_NAME, folder_name)
        
        print(f"\n--- Processing R={R}, D={D} -> {folder_path} ---")
        
        # Remove existing folder
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Existing folder {folder_path} removed")
        
        # Copy Default folder
        if os.path.exists("Default"):
            shutil.copytree("Default", folder_path)
            print(f"Default folder copied as {folder_path}")
        else:
            print("Default folder not found")
            continue
        
        # Define points P with current R and D values
        P = np.array([
            [3, 0, D],   # P1
            [R, 0, D],   # P2
            [10, 0, -1], # P3
            [10, 0, 3],  # P4
            [3, 0, 3]    # P5
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
            # Generate revolution solid
            result = generate_revolution_solid_stl(
                points=P,
                filename="float.stl",
                num_segments=36,
                visualize=False,  # No plots during batch processing
                save_plot_path=os.getcwd()  # Save plot in current geometry folder
            )
            
            print(f"STL file generated: {result['filename']}")
            print(f"Vertices: {result['num_vertices']}")
            print(f"Triangles: {result['num_triangles']}")
            
        except Exception as e:
            print(f"Error generating STL: {e}")
            os.chdir(current_dir)
            continue
        
        # Return to original directory
        os.chdir(current_dir)
        
        # Run hydrodynamic analysis
        try:
            print("Starting hydrodynamic analysis...")
            
            # Define mesh paths
            mesh1_path = os.path.join(folder_path, "geometry", "float.stl")
            mesh2_path = os.path.join(folder_path, "geometry", "plate.stl")
            
            # Verify both files exist
            if not os.path.exists(mesh1_path):
                print(f"Error: {mesh1_path} not found")
                continue
            if not os.path.exists(mesh2_path):
                print(f"Error: {mesh2_path} not found")
                continue
            
            # Output directory for hydrodynamic data
            hydro_output_dir = os.path.join(folder_path, "hydroData")
            
            # Run hydrodynamic analysis
            results = analyze_two_body_hydrodynamics(
                mesh1_path=mesh1_path,
                mesh2_path=mesh2_path,
                frequency_range=frequencies,
                mesh1_position=[0.0, 0.0, 0.0],
                mesh2_position=[0.0, 0.0, -20.0],
                body_names=["Float", "Plate"],
                output_directory=hydro_output_dir,
                nc_filename="rm3.nc",
                plot_xlim=[-20, 20],
                plot_ylim=[-35, 15],
                save_plots=True,
                show_plots=False,  # No plots during batch processing
                logging_level="INFO",
                run_bemio=RUN_BEMIO,
                bemio_script_path=BEMIO_SCRIPT_PATH
            )
            
            # Show key results
            RAO = results['RAO']
            relative_heave = results['relative_heave_RAO']
            
            print(f"Analysis completed for {len(frequencies)} frequencies")
            print(f"Max Float heave RAO: {np.max(RAO[2, :]):.3f}")
            print(f"Max Plate heave RAO: {np.max(RAO[8, :]):.3f}")
            print(f"Max relative RAO: {np.max(relative_heave):.3f}")
            
            # Verify HydCoeff files after analysis
            hydro_files = ['HydCoeff.npz', 'HydCoeff.pkl', 'HydCoeff.mat']
            print(f"\nVerifying HydCoeff files in {hydro_output_dir}:")
            files_found = []
            for file in hydro_files:
                file_path = os.path.join(hydro_output_dir, file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"✅ {file}: {file_size:,} bytes")
                    files_found.append(file)
                else:
                    print(f"❌ {file}: NOT FOUND")
            
            if files_found:
                print(f"✅ {len(files_found)} HydCoeff files created successfully")
            else:
                print(f"❌ NO HydCoeff files were created")
                print(f"   Verify that Eco_Cap2B.py is updated with corrected version")
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR in hydrodynamic analysis for {folder_path}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Message: {str(e)}")
            print(f"   Continuing with next iteration...")
            continue

print("\n--- Process completed ---")
print(f"Total combinations processed: {len(R_values)} × {len(D_values)} = {len(R_values) * len(D_values)}")
print(f"\nAll results saved in folder: {FOLDER_NAME}/")
print(f"File structure in each subfolder {FOLDER_NAME}/Geometry_R*_D*:")
print("- geometry/float.stl: Generated geometry")
print("- geometry/profile_plot.png: Revolution solid profile") 
print("- hydroData/rm3.nc: Hydrodynamic data in NetCDF format")
print("- hydroData/HydCoeff.npz: Hydrodynamic coefficients (NumPy)")
print("- hydroData/HydCoeff.pkl: Hydrodynamic coefficients (Pickle)")
print("- hydroData/HydCoeff.mat: Hydrodynamic coefficients (MATLAB)")
print("- hydroData/RAO_heave_comparison.png: RAO comparison plot")
print("- hydroData/geometry_lateral_view.png: Lateral geometry view")
if RUN_BEMIO:
    print("- hydroData/*.h5: BEMIO files for WEC-Sim (if BEMIO enabled)")
    print("- hydroData/*.png: Additional BEMIO plots (if BEMIO enabled)")
