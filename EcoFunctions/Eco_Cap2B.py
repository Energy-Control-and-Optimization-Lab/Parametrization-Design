"""
Two-body hydrodynamic analysis with Capytaine and NetCDF export
Author: Pablo Antonio Matamala Carvajal
Date: 2025-07-21
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import capytaine as cpt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from .Eco_Bemio import run_bemio_postprocessing
except ImportError:
    from Eco_Bemio import run_bemio_postprocessing

def analyze_two_body_hydrodynamics(mesh1_path, mesh2_path, frequency_range, 
                                 mesh1_position=[0.0, 0.0, 0.0], 
                                 mesh2_position=[0.0, 0.0, 0.0],
                                 wave_direction=0.0,
                                 output_directory="hydroData",
                                 nc_filename="rm3.nc",
                                 body_names=["Float", "Plate"],
                                 plot_xlim=[-20, 20],
                                 plot_ylim=[-35, 15],
                                 save_plots=True,
                                 show_plots=True,
                                 logging_level="INFO",
                                 run_bemio=False,
                                 bemio_script_path=None):
    """
    Performs hydrodynamic analysis for two floating bodies with coupled interaction.
    
    Parameters:
    -----------
    mesh1_path : str
        STL file path for first body
    mesh2_path : str
        STL file path for second body
    frequency_range : array_like
        Frequency range to analyze [rad/s]
    mesh1_position : list
        First body position [x, y, z]
    mesh2_position : list
        Second body position [x, y, z]
    wave_direction : float
        Incident wave direction [rad]
    output_directory : str
        Output directory
    nc_filename : str
        NetCDF filename
    body_names : list
        Body names
    save_plots : bool
        Save plots
    show_plots : bool
        Show plots
    logging_level : str
        Capytaine logging level
    run_bemio : bool
        Run BEMIO post-processing
    bemio_script_path : str
        BEMIO script path
    
    Returns:
    --------
    dict : Analysis results
    """
    
    cpt.set_logging(level=logging_level)
    
    # Validate STL files
    if not os.path.exists(mesh1_path):
        raise FileNotFoundError(f"STL file not found: {mesh1_path}")
    if not os.path.exists(mesh2_path):
        raise FileNotFoundError(f"STL file not found: {mesh2_path}")
    
    # Load and configure first body
    mesh1 = cpt.load_mesh(mesh1_path)
    mesh1.translate(mesh1_position)
    body1 = cpt.FloatingBody(
        mesh=mesh1,
        dofs=cpt.rigid_body_dofs(rotation_center=mesh1.center_of_buoyancy),
        center_of_mass=mesh1.center_of_buoyancy,
        name=body_names[0].lower()
    )
    body1.inertia_matrix = body1.compute_rigid_body_inertia()
    body1.hydrostatic_stiffness = body1.immersed_part().compute_hydrostatic_stiffness()
    
    # Load and configure second body
    mesh2 = cpt.load_mesh(mesh2_path)
    mesh2.translate(mesh2_position)
    body2 = cpt.FloatingBody(
        mesh=mesh2,
        dofs=cpt.rigid_body_dofs(rotation_center=mesh2.center_of_buoyancy),
        center_of_mass=mesh2.center_of_buoyancy,
        name=body_names[1].lower()
    )
    body2.inertia_matrix = body2.compute_rigid_body_inertia()
    body2.hydrostatic_stiffness = body2.immersed_part().compute_hydrostatic_stiffness()
    
    # Combine bodies
    all_bodies = body1 + body2
    
    # Create BEM problems
    w = np.array(frequency_range)
    problems = []
    
    for omega in w:
        # Radiation problems for each DOF
        for dof in all_bodies.dofs:
            problems.append(cpt.RadiationProblem(body=all_bodies, radiating_dof=dof, omega=omega))
        # Diffraction problem
        problems.append(cpt.DiffractionProblem(body=all_bodies, omega=omega, wave_direction=wave_direction))
    
    # Solve BEM problems
    solver = cpt.BEMSolver()
    results = solver.solve_all(problems)
    dataset = cpt.assemble_dataset(results)
    
    # Calculate RAO
    A = dataset.added_mass.values
    B = dataset.radiation_damping.values
    Ffk = dataset.Froude_Krylov_force.values[:, 0, :].T
    Fe = dataset.excitation_force.values[:, 0, :].T
    
    M = all_bodies.inertia_matrix
    C = all_bodies.hydrostatic_stiffness
    
    Ndof = 12
    N = len(w)
    R = np.zeros((Ndof, N), dtype=complex)
    
    for i in range(N):
        A_r = A[i, :, :]
        B_r = B[i, :, :]
        F_r = Fe[:, i]
        w_r = w[i]
        
        # Hydrodynamic impedance Z(ω)
        Z = -w_r**2 * (M + A_r) + 1j * w_r * B_r + C
        
        # Solve system: Z * X = F => X = Z⁻¹ * F
        X = np.linalg.solve(Z, F_r)
        R[:, i] = np.squeeze(X)
    
    RAO = np.abs(R)
    RAO_real = R.real
    RAO_imag = R.imag
    RAO_phase = np.angle(R) - np.angle(Fe)
    RAO_phase_deg = np.rad2deg(RAO_phase)
    
    # Relative heave RAO
    R_Rel_Heave = R[2, :] - R[8, :]
    RAO_Rel_Heav = np.abs(R_Rel_Heave)
    
    # Save hydrodynamic coefficients
    os.makedirs(output_directory, exist_ok=True)
    
    hydro_coeffs = {
        'A': A,                          # Added mass
        'B': B,                          # Radiation damping
        'Fe': Fe,                        # Excitation force
        'Ffk': Ffk,                      # Froude-Krylov force
        'w': w,                          # Frequencies
        'RAO': RAO,                      # RAO magnitude
        'RAO_real': RAO_real,            # RAO real part
        'RAO_imag': RAO_imag,            # RAO imaginary part
        'RAO_phase': RAO_phase,          # RAO phase [rad]
        'RAO_phase_deg': RAO_phase_deg,  # RAO phase [deg]
        'R_Rel_Heave': R_Rel_Heave,      # Relative heave complex
        'RAO_Rel_Heav': RAO_Rel_Heav,    # Relative heave magnitude
        'metadata': {
            'body_names': body_names,
            'wave_direction': wave_direction,
            'mesh1_position': mesh1_position,
            'mesh2_position': mesh2_position,
            'frequency_range': [float(w[0]), float(w[-1])],
            'num_frequencies': int(len(w)),
            'description': 'Hydrodynamic coefficients from Capytaine BEM analysis'
        }
    }
    
    # Save in multiple formats
    files_created = []
    
    # NumPy format
    try:
        npz_path = os.path.join(output_directory, "HydCoeff.npz")
        np_data = {k: v for k, v in hydro_coeffs.items() if k != 'metadata' and isinstance(v, np.ndarray)}
        np_data['w'] = w
        np.savez_compressed(npz_path, **np_data)
        files_created.append(npz_path)
    except Exception as e:
        print(f"Error saving NPZ: {e}")
    
    # Pickle format
    try:
        import pickle
        pkl_path = os.path.join(output_directory, "HydCoeff.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(hydro_coeffs, f, protocol=pickle.HIGHEST_PROTOCOL)
        files_created.append(pkl_path)
    except Exception as e:
        print(f"Error saving PKL: {e}")
    
    # MATLAB format
    try:
        from scipy.io import savemat
        mat_path = os.path.join(output_directory, "HydCoeff.mat")
        matlab_data = {}
        for key, value in hydro_coeffs.items():
            if key == 'metadata':
                for meta_key, meta_value in value.items():
                    matlab_data[f"meta_{meta_key}"] = meta_value
            elif isinstance(value, np.ndarray) and np.iscomplexobj(value):
                matlab_data[f"{key}_real"] = value.real
                matlab_data[f"{key}_imag"] = value.imag
            elif isinstance(value, np.ndarray):
                matlab_data[key] = value
            else:
                matlab_data[key] = value
        
        savemat(mat_path, matlab_data, do_compression=True)
        files_created.append(mat_path)
    except Exception as e:
        print(f"Error saving MAT: {e}")
    
    # Export to NetCDF
    nc_filename_final = nc_filename.replace('.h5', '.nc') if nc_filename.endswith('.h5') else nc_filename
    nc_path = os.path.join(output_directory, nc_filename_final)
    
    try:
        try:
            dataset.to_netcdf(nc_path, engine='netcdf4')
        except (ImportError, ValueError):
            try:
                dataset.to_netcdf(nc_path, engine='h5netcdf')
            except (ImportError, ValueError):
                # Separate complex variables
                dataset_real = dataset.copy(deep=True)
                for var_name in list(dataset.data_vars.keys()):
                    var_data = dataset[var_name]
                    if np.iscomplexobj(var_data.values):
                        dataset_real[f"{var_name}_real"] = var_data.real.astype(np.float64)
                        dataset_real[f"{var_name}_imag"] = var_data.imag.astype(np.float64)
                        dataset_real = dataset_real.drop_vars(var_name)
                
                dataset_real.to_netcdf(nc_path)
        
        files_created.append(nc_path)
    except Exception as e:
        print(f"Error exporting NetCDF: {e}")
    
    # BEMIO post-processing
    if run_bemio:
        success = run_bemio_postprocessing(
            nc_path=nc_path,
            output_directory=output_directory,
            bemio_script_path=bemio_script_path,
            verbose=True
        )
    
    # Generate plots
    if save_plots or show_plots:
        # RAO plot
        plt.figure(figsize=(10, 6))
        plt.plot(w, RAO[2, :], label=f'Heave {body_names[0]}')
        plt.plot(w, RAO[8, :], label=f'Heave {body_names[1]}')
        plt.plot(w, RAO_Rel_Heav, label='Relative Heave')
        plt.title("RAO HEAVE")
        plt.xlabel("Wave frequency [rad/s]")
        plt.ylabel("[m/m]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        if save_plots:
            rao_plot_path = os.path.join(output_directory, "RAO_heave_comparison.png")
            plt.savefig(rao_plot_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Geometry plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        
        # Plot mesh1 (Float) - XZ projection
        verts1 = mesh1.vertices
        faces1 = mesh1.faces
        for face in faces1:
            face_verts = [verts1[idx] for idx in face]
            x_coords = [v[0] for v in face_verts] + [face_verts[0][0]]
            z_coords = [v[2] for v in face_verts] + [face_verts[0][2]]
            ax.fill(x_coords, z_coords, facecolor='skyblue', edgecolor='k', 
                   linewidth=0.1, alpha=0.9)
        
        # Plot mesh2 (Plate) - XZ projection
        verts2 = mesh2.vertices
        faces2 = mesh2.faces
        for face in faces2:
            face_verts = [verts2[idx] for idx in face]
            x_coords = [v[0] for v in face_verts] + [face_verts[0][0]]
            z_coords = [v[2] for v in face_verts] + [face_verts[0][2]]
            ax.fill(x_coords, z_coords, facecolor='lightcoral', edgecolor='k', 
                   linewidth=0.1, alpha=0.9)
        
        ax.set_xlim(plot_xlim)
        ax.set_ylim(plot_ylim)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title(f"Lateral view (XZ plane): {body_names[0]} and {body_names[1]}")
        ax.set_aspect('equal')
        ax.grid(True)
        plt.tight_layout()
        
        if save_plots:
            geo_plot_path = os.path.join(output_directory, "geometry_lateral_view.png")
            plt.savefig(geo_plot_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    return {
        'dataset': dataset,
        'RAO': RAO,
        'frequencies': w,
        'relative_heave_RAO': RAO_Rel_Heav,
        'RAO_phase': RAO_phase,
        'RAO_phase_deg': RAO_phase_deg,
        'bodies': all_bodies,
        'response_complex': R,
        'added_mass': A,
        'radiation_damping': B,
        'excitation_force': Fe
    }


if __name__ == "__main__":
    frequencies = np.linspace(0.1, 2.0, 20)
    
    results = analyze_two_body_hydrodynamics(
        mesh1_path="geometry/float.stl",
        mesh2_path="geometry/plate.stl",
        frequency_range=frequencies,
        mesh1_position=[0.0, 0.0, 0.0],
        mesh2_position=[0.0, 0.0, -20.0],
        body_names=["Float", "Plate"],
        output_directory="hydroData",
        nc_filename="rm3.nc"
    )
    
    RAO = results['RAO']
    frequencies = results['frequencies']
    relative_heave = results['relative_heave_RAO']
    
    print(f"Analysis completed for {len(frequencies)} frequencies")
    print(f"Max Float heave RAO: {np.max(RAO[2, :]):.3f}")
    print(f"Max Plate heave RAO: {np.max(RAO[8, :]):.3f}")
    print(f"Max relative RAO: {np.max(relative_heave):.3f}")
