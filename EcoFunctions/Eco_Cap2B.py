"""
Función de análisis hidrodinámico para dos cuerpos con interacción acoplada (Capytaine) y exportación a NetCDF
Author: Pablo Antonio Matamala Carvajal
Date: 2025-07-06
UPDATED: Convertido a función con parámetros de entrada y guardado de coeficientes hidrodinámicos
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import capytaine as cpt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Importar función BEMIO
try:
    from .Eco_Bemio import run_bemio_postprocessing
except ImportError:
    # Fallback si no funciona la importación relativa
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
    Realiza análisis hidrodinámico para dos cuerpos flotantes con interacción acoplada.
    
    Parámetros:
    -----------
    mesh1_path : str
        Ruta al archivo STL del primer cuerpo
    mesh2_path : str
        Ruta al archivo STL del segundo cuerpo
    frequency_range : array-like
        Rango de frecuencias a analizar [rad/s]
    mesh1_position : list, opcional
        Posición del primer cuerpo [x, y, z] (default: [0.0, 0.0, 0.0])
    mesh2_position : list, opcional
        Posición del segundo cuerpo [x, y, z] (default: [0.0, 0.0, 0.0])
    wave_direction : float, opcional
        Dirección de la onda incidente en radianes (default: 0.0)
    output_directory : str, opcional
        Directorio de salida para archivos (default: "hydroData")
    nc_filename : str, opcional
        Nombre del archivo NetCDF para Capytaine (default: "rm3.nc")
    body_names : list, opcional
        Nombres de los cuerpos (default: ["Float", "Plate"])
    plot_xlim : list, opcional
        Límites del eje X para visualización (default: [-20, 20])
    plot_ylim : list, opcional
        Límites del eje Y para visualización (default: [-35, 15])
    save_plots : bool, opcional
        Guardar gráficas (default: True)
    show_plots : bool, opcional
        Mostrar gráficas (default: True)
    logging_level : str, opcional
        Nivel de logging de Capytaine (default: "INFO")
    run_bemio : bool, opcional
        Ejecutar post-procesamiento con BEMIO (default: False)
    bemio_script_path : str, opcional
        Ruta al script bemio.m (default: None, usa script interno)
    
    Retorna:
    --------
    dict
        Diccionario con los siguientes elementos:
        - 'dataset': Dataset completo con coeficientes hidrodinámicos
        - 'RAO': Operador de amplitud de respuesta
        - 'frequencies': Array de frecuencias utilizadas
        - 'relative_heave_RAO': RAO del movimiento relativo de heave
        - 'RAO_phase': Fase del RAO
        - 'bodies': Objeto con ambos cuerpos
    """
    
    # Configurar logging
    cpt.set_logging(level=logging_level)
    
    print(f"🔧 Iniciando análisis hidrodinámico...")
    print(f"   Mesh1: {mesh1_path}")
    print(f"   Mesh2: {mesh2_path}")
    print(f"   Frecuencias: {len(frequency_range)} puntos de {frequency_range[0]:.2f} a {frequency_range[-1]:.2f} rad/s")
    
    # ===============================
    # GEOMETRIAS Y PROPIEDADES FISICAS
    # ===============================
    
    # Verificar que los archivos STL existen
    if not os.path.exists(mesh1_path):
        raise FileNotFoundError(f"No se encontró el archivo STL: {mesh1_path}")
    if not os.path.exists(mesh2_path):
        raise FileNotFoundError(f"No se encontró el archivo STL: {mesh2_path}")
    
    print(f"✅ Archivos STL verificados")
    
    # Cargar y configurar primer cuerpo
    print(f"🔧 Cargando mesh1: {mesh1_path}")
    mesh1 = cpt.load_mesh(mesh1_path)
    print(f"✅ Mesh1 cargado: {len(mesh1.vertices)} vértices, {len(mesh1.faces)} caras")
    
    mesh1.translate(mesh1_position)
    body1 = cpt.FloatingBody(
        mesh=mesh1,
        dofs=cpt.rigid_body_dofs(rotation_center=mesh1.center_of_buoyancy),
        center_of_mass=mesh1.center_of_buoyancy,
        name=body_names[0].lower()
    )
    body1.inertia_matrix = body1.compute_rigid_body_inertia()
    body1.hydrostatic_stiffness = body1.immersed_part().compute_hydrostatic_stiffness()
    print(f"✅ Body1 ({body_names[0]}) creado correctamente")
    
    # Cargar y configurar segundo cuerpo
    print(f"🔧 Cargando mesh2: {mesh2_path}")
    mesh2 = cpt.load_mesh(mesh2_path)
    print(f"✅ Mesh2 cargado: {len(mesh2.vertices)} vértices, {len(mesh2.faces)} caras")
    
    mesh2.translate(mesh2_position)
    body2 = cpt.FloatingBody(
        mesh=mesh2,
        dofs=cpt.rigid_body_dofs(rotation_center=mesh2.center_of_buoyancy),
        center_of_mass=mesh2.center_of_buoyancy,
        name=body_names[1].lower()
    )
    body2.inertia_matrix = body2.compute_rigid_body_inertia()
    body2.hydrostatic_stiffness = body2.immersed_part().compute_hydrostatic_stiffness()
    print(f"✅ Body2 ({body_names[1]}) creado correctamente")
    
    # Combinar ambos cuerpos
    all_bodies = body1 + body2
    print(f"✅ Cuerpos combinados: {len(all_bodies.mesh.vertices)} vértices totales")
    print(f"✅ DOFs totales: {len(all_bodies.dofs)}")
    
    # ===============================
    # SOLUCION DEL PROBLEMA HIDRODINAMICO
    # ===============================
    
    w = np.array(frequency_range)
    problems = []
    
    print(f"🔧 Creando problemas BEM...")
    for i, omega in enumerate(w):
        # Problemas de radiación para cada DOF
        for dof in all_bodies.dofs:
            problems.append(cpt.RadiationProblem(body=all_bodies, radiating_dof=dof, omega=omega))
        # Problema de difracción
        problems.append(cpt.DiffractionProblem(body=all_bodies, omega=omega, wave_direction=wave_direction))
        
        if i % 5 == 0:  # Mostrar progreso cada 5 frecuencias
            print(f"   Frecuencia {i+1}/{len(w)}: {omega:.3f} rad/s")
    
    print(f"✅ Problemas BEM creados: {len(problems)} problemas")
    radiation_problems = len([p for p in problems if isinstance(p, cpt.RadiationProblem)])
    diffraction_problems = len([p for p in problems if isinstance(p, cpt.DiffractionProblem)])
    print(f"   - Radiación: {radiation_problems} problemas")
    print(f"   - Difracción: {diffraction_problems} problemas")
    
    print(f"🔧 Resolviendo problemas BEM... (esto puede tomar tiempo)")
    solver = cpt.BEMSolver()
    results = solver.solve_all(problems)
    print(f"✅ Solver BEM completado: {len(results)} resultados")
    
    print(f"🔧 Ensamblando dataset...")
    dataset = cpt.assemble_dataset(results)
    print(f"✅ Dataset creado:")
    print(f"   - Dimensiones: {dict(dataset.dims)}")
    print(f"   - Variables principales: {list(dataset.data_vars.keys())}")
    if 'omega' in dataset.coords:
        print(f"   - Rango frecuencias: {dataset.coords['omega'].values[0]:.3f} - {dataset.coords['omega'].values[-1]:.3f} rad/s")
    
    # ===============================
    # CALCULO DEL RAO
    # ===============================
    
    print(f"🔧 Calculando RAO...")
    
    try:
        A = dataset.added_mass.values
        B = dataset.radiation_damping.values
        Ffk = dataset.Froude_Krylov_force.values[:, 0, :].T
        Fe = dataset.excitation_force.values[:, 0, :].T
        print(f"✅ Matrices hidrodinámicas extraídas correctamente")
        print(f"   - Masa añadida: {A.shape}")
        print(f"   - Amortiguamiento: {B.shape}")
        print(f"   - Fuerza excitación: {Fe.shape}")
    except Exception as e:
        print(f"❌ Error extrayendo matrices del dataset: {e}")
        print(f"   Variables disponibles: {list(dataset.data_vars.keys())}")
        print(f"   Dimensiones: {dict(dataset.dims)}")
        raise
    
    M = all_bodies.inertia_matrix
    C = all_bodies.hydrostatic_stiffness
    
    Ndof = 12
    N = len(w)
    R = np.zeros((Ndof, N), dtype=complex)
    
    print(f"🔧 Resolviendo sistema de ecuaciones para RAO...")
    for i in range(N):
        A_r = A[i, :, :]
        B_r = B[i, :, :]
        F_r = Fe[:, i]
        w_r = w[i]
        
        # Impedancia hidrodinámica Z(ω)
        Z = -w_r**2 * (M + A_r) + 1j * w_r * B_r + C
        
        # Resolver el sistema: Z * X = F ⇒ X = Z⁻¹ * F
        X = np.linalg.solve(Z, F_r)
        R[:, i] = np.squeeze(X)
        
        if i % 5 == 0:  # Mostrar progreso
            print(f"   RAO frecuencia {i+1}/{N}: {w_r:.3f} rad/s")
    
    RAO = np.abs(R)
    RAO_real = R.real
    RAO_imag = R.imag
    RAO_phase = np.angle(R) - np.angle(Fe)
    RAO_phase_deg = np.rad2deg(RAO_phase)
    
    # RAO relativo de heave
    R_Rel_Heave = R[2, :] - R[8, :]
    RAO_Rel_Heav = np.abs(R_Rel_Heave)
    
    print(f"✅ RAO calculado correctamente")
    print(f"   - RAO máximo {body_names[0]} (heave): {np.max(RAO[2, :]):.3f}")
    print(f"   - RAO máximo {body_names[1]} (heave): {np.max(RAO[8, :]):.3f}")
    print(f"   - RAO máximo relativo: {np.max(RAO_Rel_Heav):.3f}")
    
    # ===============================
    # GUARDAR COEFICIENTES HIDRODINÁMICOS
    # ===============================
    
    print(f"🔧 Guardando coeficientes hidrodinámicos...")
    
    # CREAR EL DIRECTORIO PRIMERO antes de intentar guardar archivos
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"   Directorio de salida: {output_directory}")
    print(f"   ¿Directorio existe? {os.path.exists(output_directory)}")
    
    # Verificar que las variables existen y tienen el formato correcto
    print(f"🔍 Verificando variables:")
    print(f"   A shape: {A.shape}, type: {type(A)}")
    print(f"   B shape: {B.shape}, type: {type(B)}")
    print(f"   Fe shape: {Fe.shape}, type: {type(Fe)}")
    print(f"   w shape: {w.shape}, type: {type(w)}")
    print(f"   RAO shape: {RAO.shape}, type: {type(RAO)}")
    
    # Crear diccionario con todas las variables hidrodinámicas
    try:
        hydro_coeffs = {
            'A': A,                          # Masa añadida (Nw, 12, 12)
            'B': B,                          # Amortiguamiento por radiación (Nw, 12, 12)
            'Fe': Fe,                        # Fuerza de excitación (12, Nw)
            'Ffk': Ffk,                      # Fuerza de Froude-Krylov (12, Nw)
            'w': w,                          # Frecuencias (Nw,)
            'RAO': RAO,                      # RAO magnitud (12, Nw)
            'RAO_real': RAO_real,            # RAO parte real (12, Nw)
            'RAO_imag': RAO_imag,            # RAO parte imaginaria (12, Nw)
            'RAO_phase': RAO_phase,          # RAO fase en radianes (12, Nw)
            'RAO_phase_deg': RAO_phase_deg,  # RAO fase en grados (12, Nw)
            'R_Rel_Heave': R_Rel_Heave,      # RAO relativo complejo (Nw,)
            'RAO_Rel_Heav': RAO_Rel_Heav,    # RAO relativo magnitud (Nw,)
            'metadata': {
                'body_names': body_names,
                'wave_direction': wave_direction,
                'mesh1_position': mesh1_position,
                'mesh2_position': mesh2_position,
                'frequency_range': [float(w[0]), float(w[-1])],
                'num_frequencies': int(len(w)),
                'description': 'Coeficientes hidrodinámicos de análisis BEM con Capytaine'
            }
        }
        print(f"✅ Diccionario hydro_coeffs creado con {len(hydro_coeffs)} elementos")
        
    except Exception as e:
        print(f"❌ Error creando diccionario hydro_coeffs: {e}")
        print(f"   Tipo de error: {type(e).__name__}")
        return None
    
    # Intentar guardar en múltiples formatos
    archivos_creados = []
    
    # Formato 1: NumPy (.npz) - Más simple y confiable
    try:
        npz_path = os.path.join(output_directory, "HydCoeff.npz")
        print(f"🔧 Intentando guardar NPZ en: {npz_path}")
        
        # Preparar datos para NumPy (solo arrays)
        np_data = {}
        for key, value in hydro_coeffs.items():
            if key != 'metadata' and isinstance(value, np.ndarray):
                np_data[key] = value
            elif key == 'w':
                np_data[key] = value
        
        print(f"   Variables para NPZ: {list(np_data.keys())}")
        np.savez_compressed(npz_path, **np_data)
        
        if os.path.exists(npz_path):
            file_size = os.path.getsize(npz_path)
            print(f"✅ Archivo NPZ creado: {file_size:,} bytes")
            archivos_creados.append(npz_path)
        else:
            print(f"❌ Archivo NPZ no se creó")
            
    except Exception as e:
        print(f"❌ Error guardando NPZ: {e}")
        print(f"   Tipo de error: {type(e).__name__}")
    
    # Formato 2: Pickle (.pkl) - Para datos completos
    try:
        pkl_path = os.path.join(output_directory, "HydCoeff.pkl")
        print(f"🔧 Intentando guardar PKL en: {pkl_path}")
        
        import pickle
        with open(pkl_path, 'wb') as f:
            pickle.dump(hydro_coeffs, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if os.path.exists(pkl_path):
            file_size = os.path.getsize(pkl_path)
            print(f"✅ Archivo PKL creado: {file_size:,} bytes")
            archivos_creados.append(pkl_path)
        else:
            print(f"❌ Archivo PKL no se creó")
            
    except Exception as e:
        print(f"❌ Error guardando PKL: {e}")
        print(f"   Tipo de error: {type(e).__name__}")
    
    # Formato 3: MATLAB (.mat) - Si está disponible
    try:
        from scipy.io import savemat
        
        mat_path = os.path.join(output_directory, "HydCoeff.mat")
        print(f"🔧 Intentando guardar MAT en: {mat_path}")
        
        # Preparar datos para MATLAB (convertir complejos y metadatos)
        matlab_data = {}
        for key, value in hydro_coeffs.items():
            if key == 'metadata':
                # Convertir metadatos a formato compatible con MATLAB
                for meta_key, meta_value in value.items():
                    matlab_data[f"meta_{meta_key}"] = meta_value
            elif isinstance(value, np.ndarray) and np.iscomplexobj(value):
                # Separar partes real e imaginaria para MATLAB
                matlab_data[f"{key}_real"] = value.real
                matlab_data[f"{key}_imag"] = value.imag
            elif isinstance(value, np.ndarray):
                matlab_data[key] = value
            else:
                matlab_data[key] = value
        
        print(f"   Variables para MAT: {list(matlab_data.keys())}")
        savemat(mat_path, matlab_data, do_compression=True)
        
        if os.path.exists(mat_path):
            file_size = os.path.getsize(mat_path)
            print(f"✅ Archivo MAT creado: {file_size:,} bytes")
            archivos_creados.append(mat_path)
        else:
            print(f"❌ Archivo MAT no se creó")
            
    except ImportError:
        print("⚠️  scipy.io no disponible, formato .mat omitido")
    except Exception as e:
        print(f"❌ Error guardando MAT: {e}")
        print(f"   Tipo de error: {type(e).__name__}")
    
    # Resumen final
    if archivos_creados:
        print(f"✅ Coeficientes hidrodinámicos guardados exitosamente")
        print(f"   Archivos creados: {len(archivos_creados)}")
        for archivo in archivos_creados:
            nombre = os.path.basename(archivo)
            tamaño = os.path.getsize(archivo)
            print(f"   - {nombre}: {tamaño:,} bytes")
    else:
        print(f"❌ NO SE PUDO CREAR NINGÚN ARCHIVO HydCoeff")
        print(f"   Verificar permisos de escritura en: {output_directory}")
        
        # Intentar crear un archivo de prueba
        try:
            test_path = os.path.join(output_directory, "test_write.txt")
            with open(test_path, 'w') as f:
                f.write("test")
            if os.path.exists(test_path):
                os.remove(test_path)
                print(f"✅ Permisos de escritura verificados")
            else:
                print(f"❌ No se puede escribir en el directorio")
        except Exception as e:
            print(f"❌ Error de permisos: {e}")
    
    # ===============================
    # EXPORTACION A NETCDF
    # ===============================
    
    print(f"🔧 Exportando a NetCDF...")
    os.makedirs(output_directory, exist_ok=True)
    print(f"✅ Directorio de salida verificado: {output_directory}")
    print(f"✅ ¿Directorio existe? {os.path.exists(output_directory)}")
    
    # Cambiar extensión a .nc para formato NetCDF si es necesario
    nc_filename_final = nc_filename.replace('.h5', '.nc') if nc_filename.endswith('.h5') else nc_filename
    nc_path = os.path.join(output_directory, nc_filename_final)
    print(f"🔧 Ruta completa archivo NC: {nc_path}")
    
    try:
        # Intentar con el backend netcdf4 (recomendado para Capytaine)
        try:
            dataset.to_netcdf(nc_path, engine='netcdf4')
            print(f"✅ Dataset exportado con engine netcdf4: {nc_path}")
        except (ImportError, ValueError):
            print("⚠️  Engine netcdf4 no disponible, intentando con h5netcdf...")
            try:
                dataset.to_netcdf(nc_path, engine='h5netcdf')
                print(f"✅ Dataset exportado con engine h5netcdf: {nc_path}")
            except (ImportError, ValueError):
                print("⚠️  Engine h5netcdf no disponible, separando datos complejos...")
                
                # Separar partes complejas como último recurso
                dataset_real = dataset.copy(deep=True)
                
                # Convertir variables complejas a reales
                complex_vars_found = []
                for var_name in list(dataset.data_vars.keys()):
                    var_data = dataset[var_name]
                    if np.iscomplexobj(var_data.values):
                        complex_vars_found.append(var_name)
                        # Crear variables separadas para parte real e imaginaria
                        dataset_real[f"{var_name}_real"] = var_data.real.astype(np.float64)
                        dataset_real[f"{var_name}_imag"] = var_data.imag.astype(np.float64)
                        # Eliminar la variable compleja original
                        dataset_real = dataset_real.drop_vars(var_name)
                
                print(f"   Variables complejas separadas: {complex_vars_found}")
                
                # Exportar con scipy (backend por defecto)
                dataset_real.to_netcdf(nc_path)
                print(f"✅ Dataset exportado con datos complejos separados: {nc_path}")
        
        # Verificar que el archivo se creó correctamente
        if os.path.exists(nc_path):
            file_size = os.path.getsize(nc_path)
            print(f"✅ Archivo .nc creado exitosamente: {file_size:,} bytes")
        else:
            print(f"❌ ERROR: Archivo .nc no se creó en: {nc_path}")
            raise FileNotFoundError(f"No se pudo crear el archivo {nc_path}")
            
    except Exception as e:
        print(f"❌ ERROR CRÍTICO al exportar NetCDF: {e}")
        print(f"   Tipo de error: {type(e).__name__}")
        
        # Como último último recurso, intentar guardar como pickle temporalmente
        print("   Intentando guardar como archivo temporal...")
        try:
            import pickle
            pkl_path = nc_path.replace('.nc', '_backup.pkl')
            
            # Crear un diccionario con los datos más importantes
            backup_data = {
                'added_mass': dataset.added_mass.values,
                'radiation_damping': dataset.radiation_damping.values,
                'excitation_force': dataset.excitation_force.values,
                'frequencies': dataset.coords['omega'].values,
                'metadata': {
                    'body_names': body_names,
                    'wave_direction': wave_direction,
                    'mesh1_position': mesh1_position,
                    'mesh2_position': mesh2_position
                }
            }
            
            with open(pkl_path, 'wb') as f:
                pickle.dump(backup_data, f)
            
            print(f"✅ Datos respaldados en formato pickle: {pkl_path}")
            print("   Nota: Archivo .nc no se pudo crear, pero los datos están guardados")
            
        except Exception as e_final:
            print(f"❌ ERROR FINAL también en backup: {e_final}")
            raise
    
    # ===============================
    # POST-PROCESAMIENTO CON BEMIO (OPCIONAL)
    # ===============================
    
    if run_bemio:
        print("\n" + "="*50)
        print("INICIANDO POST-PROCESAMIENTO BEMIO")
        print("="*50)
        
        success = run_bemio_postprocessing(
            nc_path=nc_path,
            output_directory=output_directory,
            bemio_script_path=bemio_script_path,
            verbose=True
        )
        
        if success:
            print("✅ POST-PROCESAMIENTO BEMIO COMPLETADO")
            print("="*50)
        else:
            print("⚠️  POST-PROCESAMIENTO BEMIO FALLÓ")
            print("Continuando sin archivos BEMIO...")
            print("="*50)
    
    # ===============================
    # VISUALIZACION
    # ===============================
    
    if save_plots or show_plots:
        print(f"🔧 Generando visualizaciones...")
        
        try:
            # Gráfica del RAO
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
                print(f"✅ Gráfica RAO guardada: {rao_plot_path}")
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        except Exception as e:
            print(f"❌ Error generando gráfica RAO: {e}")
        
        try:
            # Gráfica de geometría
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111)
            
            # Graficar mesh1 (Float) - proyección en XZ
            verts1 = mesh1.vertices
            faces1 = mesh1.faces
            for face in faces1:
                face_verts = [verts1[idx] for idx in face]
                x_coords = [v[0] for v in face_verts] + [face_verts[0][0]]
                z_coords = [v[2] for v in face_verts] + [face_verts[0][2]]
                ax.fill(x_coords, z_coords, facecolor='skyblue', edgecolor='k', 
                       linewidth=0.1, alpha=0.9)
            
            # Graficar mesh2 (Plate) - proyección en XZ
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
            ax.set_title(f"Vista lateral (Plano XZ): {body_names[0]} y {body_names[1]}")
            ax.set_aspect('equal')
            ax.grid(True)
            plt.tight_layout()
            
            if save_plots:
                geo_plot_path = os.path.join(output_directory, "geometry_lateral_view.png")
                plt.savefig(geo_plot_path, dpi=300, bbox_inches='tight')
                print(f"✅ Gráfica geometría guardada: {geo_plot_path}")
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        except Exception as e:
            print(f"❌ Error generando gráfica de geometría: {e}")
    
        print(f"✅ Visualizaciones completadas")
    
    print(f"\n🎉 ANÁLISIS HIDRODINÁMICO COMPLETADO EXITOSAMENTE")
    print(f"📁 Archivos generados en: {output_directory}")
    print(f"📊 Datos listos para WEC-Sim o análisis posterior")
    
    # ===============================
    # RETORNO DE RESULTADOS
    # ===============================
    
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


# ===============================
# EJEMPLO DE USO
# ===============================

if __name__ == "__main__":
    # Definir rango de frecuencias
    frequencies = np.linspace(0.1, 2.0, 20)
    
    # Ejecutar análisis
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
    
    # Acceder a los resultados
    RAO = results['RAO']
    frequencies = results['frequencies']
    relative_heave = results['relative_heave_RAO']
    
    print(f"Análisis completado para {len(frequencies)} frecuencias")
    print(f"RAO máximo en heave del Float: {np.max(RAO[2, :]):.3f}")
    print(f"RAO máximo en heave del Plate: {np.max(RAO[8, :]):.3f}")
    print(f"RAO máximo relativo: {np.max(relative_heave):.3f}")