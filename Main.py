import numpy as np
import shutil
import os
import sys
import matplotlib.pyplot as plt

# Agregar la carpeta EcoFunctions al path para permitir importaciones internas
sys.path.append(os.path.join(os.path.dirname(__file__), 'EcoFunctions'))

from EcoFunctions.Eco_StlRev import generate_revolution_solid_stl
from EcoFunctions.Eco_Cap2B import analyze_two_body_hydrodynamics

# CONFIGURACIÓN PRINCIPAL
# ======================
# Nombre de la carpeta principal donde se guardarán todos los resultados
FOLDER_NAME = "batch"  # Modificar aquí para cambiar el nombre

# Configuración de BEMIO (post-procesamiento MATLAB)
RUN_BEMIO = False  # Cambiar a True para ejecutar BEMIO después del análisis BEM
BEMIO_SCRIPT_PATH = None  # Ruta a script bemio.m personalizado (None = usa script interno)

# Rangos de variación
R_values = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]  # Lista completa
D_values = [-1.2, -1.4, -1.6, -1.8, -2.0, -2.2, -2.4, -2.6, -2.8, -3.0]  # Lista completa

# Configuración del análisis hidrodinámico
frequencies = np.linspace(0.25, 2.0, 50)  # Reducido a 10 frecuencias para diagnóstico

# Verificar versión de Eco_Cap2B
print(f"🔍 Verificando versión de Eco_Cap2B...")
import inspect
lines = inspect.getsource(analyze_two_body_hydrodynamics)
if "Guardando coeficientes hidrodinámicos" in lines:
    print("✅ Versión actualizada detectada - HydCoeff debería guardarse")
else:
    print("❌ Versión antigua detectada - NECESITAS REEMPLAZAR Eco_Cap2B.py")
    print("   El archivo HydCoeff NO se guardará hasta que actualices el código")

# Crear carpeta principal del batch si no existe
os.makedirs(FOLDER_NAME, exist_ok=True)
print(f"Carpeta principal del batch: {FOLDER_NAME}")

# Iterar sobre todas las combinaciones de R y D
for R in R_values:
    for D in D_values:
        # Crear nombre de carpeta de manera más simple
        R_str = str(int(round(float(R) * 10)))
        D_str = str(int(round(abs(float(D)) * 10)))
        folder_name = f"Geometry_R{R_str}_D{D_str}"
        folder_path = os.path.join(FOLDER_NAME, folder_name)
        
        print(f"\n--- Procesando R={R}, D={D} -> {folder_path} ---")
        
        # Eliminar carpeta si existe
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Carpeta {folder_path} preexistente eliminada")
        
        # Copiar carpeta Default
        if os.path.exists("Default"):
            shutil.copytree("Default", folder_path)
            print(f"Carpeta Default copiada como {folder_path}")
        else:
            print("No se encontró la carpeta Default")
            continue
        
        # Definir puntos P con los valores actuales de R y D
        P = np.array([
            [3, 0, D],   # P1
            [R, 0, D],   # P2
            [10, 0, -1],  # P3
            [10, 0, 3],   # P4
            [3, 0, 3]     # P5
        ])
        
        # Crear carpeta geometry si no existe
        geometry_path = os.path.join(folder_path, "geometry")
        os.makedirs(geometry_path, exist_ok=True)
        
        # Eliminar archivo float.stl si ya existe
        float_stl_path = os.path.join(geometry_path, "float.stl")
        if os.path.exists(float_stl_path):
            try:
                os.remove(float_stl_path)
                print(f"Archivo {float_stl_path} eliminado")
            except PermissionError:
                print(f"No se pudo eliminar {float_stl_path} - archivo en uso")
        
        # Cambiar directorio temporalmente para guardar el STL
        current_dir = os.getcwd()
        os.chdir(geometry_path)
        
        try:
            # Generar el sólido de revolución
            result = generate_revolution_solid_stl(
                points=P,
                filename="float.stl",
                num_segments=36,
                visualize=False,  # No mostrar gráficos durante procesamiento masivo
                save_plot_path=os.getcwd()  # Guardar el gráfico en la carpeta geometry actual
            )
            
            print(f"Archivo STL generado: {result['filename']}")
            print(f"Vértices: {result['num_vertices']}")
            print(f"Triángulos: {result['num_triangles']}")
            
        except Exception as e:
            print(f"Error generando STL: {e}")
            # Restaurar directorio antes de continuar
            os.chdir(current_dir)
            continue
        
        # Volver al directorio original
        os.chdir(current_dir)
        
        # Ejecutar análisis hidrodinámico usando la función importada
        try:
            print("Iniciando análisis hidrodinámico...")
            
            # Definir las rutas de los meshes
            mesh1_path = os.path.join(folder_path, "geometry", "float.stl")
            mesh2_path = os.path.join(folder_path, "geometry", "plate.stl")
            
            # Verificar que ambos archivos existen
            if not os.path.exists(mesh1_path):
                print(f"Error: No se encontró {mesh1_path}")
                continue
            if not os.path.exists(mesh2_path):
                print(f"Error: No se encontró {mesh2_path}")
                continue
            
            # Directorio de salida para datos hidrodinámicos
            hydro_output_dir = os.path.join(folder_path, "hydroData")
            
            # Ejecutar análisis hidrodinámico
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
                show_plots=False,  # No mostrar gráficos durante el procesamiento masivo
                logging_level="INFO",  # Cambiado de WARNING a INFO para más información
                run_bemio=RUN_BEMIO,  # Ejecutar BEMIO si está habilitado
                bemio_script_path=BEMIO_SCRIPT_PATH  # Script personalizado (opcional)
            )
            
            # Mostrar resultados clave
            RAO = results['RAO']
            relative_heave = results['relative_heave_RAO']
            
            print(f"Análisis completado para {len(frequencies)} frecuencias")
            print(f"RAO máximo en heave del Float: {np.max(RAO[2, :]):.3f}")
            print(f"RAO máximo en heave del Plate: {np.max(RAO[8, :]):.3f}")
            print(f"RAO máximo relativo: {np.max(relative_heave):.3f}")
            
            # Verificar archivos HydCoeff después del análisis
            hydro_files = ['HydCoeff.npz', 'HydCoeff.pkl', 'HydCoeff.mat']
            print(f"\n🔍 Verificando archivos HydCoeff en {hydro_output_dir}:")
            archivos_encontrados = []
            for archivo in hydro_files:
                file_path = os.path.join(hydro_output_dir, archivo)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"✅ {archivo}: {file_size:,} bytes")
                    archivos_encontrados.append(archivo)
                else:
                    print(f"❌ {archivo}: NO ENCONTRADO")
            
            if archivos_encontrados:
                print(f"✅ {len(archivos_encontrados)} archivos HydCoeff creados exitosamente")
            else:
                print(f"❌ NINGÚN archivo HydCoeff fue creado")
                print(f"   Verificar que Eco_Cap2B.py está actualizado con la versión corregida")
            
        except Exception as e:
            print(f"❌ ERROR CRÍTICO en análisis hidrodinámico para {folder_path}")
            print(f"   Tipo de error: {type(e).__name__}")
            print(f"   Mensaje: {str(e)}")
            print(f"   Continuando con la siguiente iteración...")
            continue

print("\n--- Proceso completado ---")
print(f"Total de combinaciones procesadas: {len(R_values)} × {len(D_values)} = {len(R_values) * len(D_values)}")
print(f"\nTodos los resultados se guardaron en la carpeta: {FOLDER_NAME}/")
print(f"Estructura de archivos en cada subcarpeta {FOLDER_NAME}/Geometry_R*_D*:")
print("- geometry/float.stl: Geometría generada")
print("- geometry/profile_plot.png: Perfil del sólido de revolución") 
print("- hydroData/rm3.nc: Datos hidrodinámicos en formato NetCDF")
print("- hydroData/HydCoeff.npz: Coeficientes hidrodinámicos (NumPy)")
print("- hydroData/HydCoeff.pkl: Coeficientes hidrodinámicos (Pickle)")
print("- hydroData/HydCoeff.mat: Coeficientes hidrodinámicos (MATLAB)")
print("- hydroData/RAO_heave_comparison.png: Gráfica del RAO")
print("- hydroData/geometry_lateral_view.png: Vista lateral de la geometría")
if RUN_BEMIO:
    print("- hydroData/*.h5: Archivos BEMIO para WEC-Sim (si BEMIO está habilitado)")
    print("- hydroData/*.png: Gráficas adicionales de BEMIO (si BEMIO está habilitado)")