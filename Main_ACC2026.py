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

#%%     MAIN CONFIGURATION
# ==================
FOLDER_NAME = "batch_ACC2026"  # Main results folder name
os.makedirs(FOLDER_NAME, exist_ok=True)
print(f"\nMain batch folder: {FOLDER_NAME}")

#%%     PARAMETERS
R_values = np.array([6, 7, 8, 9, 10, 11, 12, 13])  # R parameter values
#R_values = np.array([10])  # R parameter values
D_values = 273 / ((R_values**2) - 9)             # D calculated for each R
Dint_values = np.array([0, 0.5, 1])
frequencies = np.linspace(0.2, 1.8, 500)  # Frequency range [rad/s]
#frequencies = np.linspace(0.2, 1.8, 5)  # Frequency range [rad/s]

# STL Generation Parameters
NUM_SEGMENTS = 50         # Revolution segments (circumferential)
Z_SUBDIVISIONS = 8        # Z-axis subdivisions per segment (configurable mesh density in Z direction)

print("=== Parameter Configuration ===")
print("R-D relationship: D = 273 / (R² - 9)")
print("Cases to analyze:")
for i, (R, D) in enumerate(zip(R_values, D_values)):
    print(f"  R = {R:2d}, D = {D:6.3f}")
print(f"Total R cases: {len(R_values)}")
print(f"Dint values: {Dint_values}")
print(f"Total Dint cases: {len(Dint_values)}")
print(f"Total combinations: {len(R_values) * len(Dint_values)}")
print(f"Frequencies: {len(frequencies)} points from {frequencies[0]:.1f} to {frequencies[-1]:.1f} rad/s")
print(f"STL Mesh Parameters:")
print(f"  - Circumferential segments: {NUM_SEGMENTS}")
print(f"  - Z-axis subdivisions per segment: {Z_SUBDIVISIONS}")
print("="*40)


#%%  HYDRODYNAMICAL ANALYSIS
case_counter = 0
total_cases = len(R_values) * len(Dint_values)

for i, (R, D) in enumerate(zip(R_values, D_values)):
    for j, Dint in enumerate(Dint_values):
        case_counter += 1
        
        # Create folder name with R, D, and Dint values
        folder_name = f"Geometry_R{R}_Dint{int(Dint*10)}"
        folder_path = os.path.join(FOLDER_NAME, folder_name)
        
        print(f"\n--- Processing Case {case_counter}/{total_cases}: R={R}, D={D:.3f}, Dint={Dint} -> {folder_path} ---")
        
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
        
        # Define geometry points with current R, D, and Dint values
        P = np.array([
            [3, 0, -D-Dint],   # P1 - Modified with Dint
            [R, 0, -D],        # P2
            [R, 0, 2],         # P3
            [3, 0, 2],         # P4
        ])
        
        print(f"Geometry points for R={R}, D={D:.3f}, Dint={Dint}:")
        for k, point in enumerate(P):
            print(f"  P{k+1}: [{point[0]:6.3f}, {point[1]:6.3f}, {point[2]:6.3f}]")
        
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
                num_segments=NUM_SEGMENTS,
                z_subdivisions=Z_SUBDIVISIONS,  # Now configurable!
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
            mesh2_path = os.path.join(folder_path, "geometry", "plate_refined.GDF")
            
            # Verify both files exist
            if not os.path.exists(mesh1_path):
                print(f"⚠ Error: {mesh1_path} not found")
                continue
            if not os.path.exists(mesh2_path):
                print(f"⚠ Error: {mesh2_path} not found")
                continue
            
            # Output directory for hydrodynamic data
            hydro_output_dir = os.path.join(folder_path, "hydroData")
            
            # Run hydrodynamic analysis
            results = analyze_two_body_hydrodynamics(
                mesh1_path=mesh1_path,
                mesh2_path=mesh2_path,
                frequency_range=frequencies,
                mesh1_position=[0.0, 0.0, 0.0],
                mesh2_position=[0.0, 0.0, 0.0],
                body_names=["Float", "Plate"],
                output_directory=hydro_output_dir,
                nc_filename="rm3.nc",
                plot_xlim=[-20, 20],
                plot_ylim=[-35, 15],
                save_plots=True,
                show_plots=False,  # No plots during batch processing
                logging_level="WARNING",  # Reduce output for batch processing
            )
            
            # Show key results
            RAO = results['RAO']
            relative_heave = results['relative_heave_RAO']
            
            print(f"✅ Analysis completed for {len(frequencies)} frequencies")
            print(f"   Max Float heave RAO: {np.max(RAO[2, :]):.3f}")
            print(f"   Max Plate heave RAO: {np.max(RAO[8, :]):.3f}")
            print(f"   Max relative RAO: {np.max(relative_heave):.3f}")
            
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
print("\n" + "="*60)
print("🎉 BATCH PROCESSING COMPLETED")
print("="*60)
print(f"Total cases processed: {len(R_values)} R values × {len(Dint_values)} Dint values = {total_cases} cases")
print(f"Results saved in folder: {FOLDER_NAME}/")
print(f"\nGenerated folders:")
for R, D in zip(R_values, D_values):
    for Dint in Dint_values:
        folder_name = f"Geometry_R{R}_D{D:.0f}_Dint{int(Dint*10)}"
        folder_path = os.path.join(FOLDER_NAME, folder_name)
        if os.path.exists(folder_path):
            print(f"  ✅ {folder_name}")
        else:
            print(f"  ⚠ {folder_name} (failed)")

print("\n🔬 Parameter Summary:")
print("R-D relationship: D = 273 / (R² - 9)")
print("Cases analyzed:")
for i, (R, D) in enumerate(zip(R_values, D_values)):
    for j, Dint in enumerate(Dint_values):
        print(f"  R = {R:2d} → D = {D:6.3f}, Dint = {Dint}")

print(f"\n🔧 Mesh Generation Parameters:")
print(f"  - Circumferential segments: {NUM_SEGMENTS}")
print(f"  - Z-axis subdivisions per segment: {Z_SUBDIVISIONS}")

print(f"\n✅ Analysis complete! Check results in '{FOLDER_NAME}/' folder")
