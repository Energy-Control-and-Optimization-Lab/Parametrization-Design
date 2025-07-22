"""
Módulo para post-procesamiento de datos BEM usando BEMIO (MATLAB)
Archivo: Eco_Bemio.py
Author: Pablo Antonio Matamala Carvajal
Date: 2025-07-21
"""

import os
import subprocess
import shutil

def run_bemio_postprocessing(nc_path, output_directory=None, bemio_script_path=None, 
                           radiation_time=60, excitation_time=157, state_space_option=1.9,
                           verbose=True):
    """
    Ejecuta post-procesamiento con BEMIO usando MATLAB
    
    Parameters:
    -----------
    nc_path : str
        Ruta completa al archivo .nc generado por Capytaine
    output_directory : str, opcional
        Directorio donde están los archivos y donde se guardarán los resultados de BEMIO.
        Si es None, usa el directorio del archivo nc_path
    bemio_script_path : str, opcional
        Ruta al script bemio.m personalizado. Si es None, usa el script interno
    radiation_time : float, opcional
        Tiempo para IRF de radiación (default: 60)
    excitation_time : float, opcional
        Tiempo para IRF de excitación (default: 157)
    state_space_option : float, opcional
        Parámetro para aproximación en espacio de estados (default: 1.9)
    verbose : bool, opcional
        Mostrar mensajes detallados (default: True)
    
    Returns:
    --------
    bool : True si se ejecutó exitosamente, False en caso contrario
    """
    
    if output_directory is None:
        output_directory = os.path.dirname(nc_path)
    
    if verbose:
        print(f"Iniciando post-procesamiento BEMIO...")
        print(f"  Archivo NC: {nc_path}")
        print(f"  Directorio salida: {output_directory}")
    
    # Intentar primero con MATLAB Engine API
    if _run_with_matlab_engine(nc_path, output_directory, bemio_script_path, 
                              radiation_time, excitation_time, state_space_option, verbose):
        return True
    
    # Si falla, intentar con subprocess
    if _run_with_subprocess(nc_path, output_directory, bemio_script_path,
                           radiation_time, excitation_time, state_space_option, verbose):
        return True
    
    if verbose:
        print("❌ No se pudo ejecutar BEMIO con ningún método")
    return False

def _run_with_matlab_engine(nc_path, output_directory, bemio_script_path,
                           radiation_time, excitation_time, state_space_option, verbose):
    """
    Ejecuta BEMIO usando MATLAB Engine API
    """
    try:
        import matlab.engine
        
        if verbose:
            print("🔧 Usando MATLAB Engine API para ejecutar BEMIO...")
        
        # Extraer solo el nombre del archivo .nc (sin ruta completa)
        nc_filename = os.path.basename(nc_path)
        
        # Crear script BEMIO dinámico o usar personalizado
        bemio_script_content = _generate_bemio_script(nc_filename, bemio_script_path,
                                                     radiation_time, excitation_time, 
                                                     state_space_option)
        
        # Crear archivo bemio.m temporal en el directorio de salida
        bemio_temp_path = os.path.join(output_directory, 'bemio_temp.m')
        with open(bemio_temp_path, 'w') as f:
            f.write(bemio_script_content)
        
        # Iniciar MATLAB engine
        eng = matlab.engine.start_matlab()
        
        # Cambiar al directorio de trabajo
        eng.cd(output_directory, nargout=0)
        
        # Ejecutar script BEMIO
        if verbose:
            print("⚙️  Ejecutando script BEMIO en MATLAB...")
        
        eng.bemio_temp(nargout=0)
        
        # Cerrar MATLAB engine
        eng.quit()
        
        # Limpiar archivo temporal
        if os.path.exists(bemio_temp_path):
            os.remove(bemio_temp_path)
        
        if verbose:
            print("✅ Post-procesamiento BEMIO completado exitosamente con Engine API")
            print(f"   Archivos BEMIO guardados en: {output_directory}")
        
        return True
        
    except ImportError:
        if verbose:
            print("⚠️  MATLAB Engine API no disponible")
        return False
    
    except Exception as e:
        if verbose:
            print(f"❌ Error ejecutando BEMIO con Engine API: {e}")
        return False

def _run_with_subprocess(nc_path, output_directory, bemio_script_path,
                        radiation_time, excitation_time, state_space_option, verbose):
    """
    Ejecuta BEMIO usando subprocess como alternativa
    """
    try:
        if verbose:
            print("🔧 Usando subprocess para ejecutar BEMIO...")
        
        # Extraer solo el nombre del archivo .nc
        nc_filename = os.path.basename(nc_path)
        
        # Crear script BEMIO
        bemio_script_content = _generate_bemio_script(nc_filename, bemio_script_path,
                                                     radiation_time, excitation_time, 
                                                     state_space_option)
        
        # Agregar comando exit al final
        if 'exit' not in bemio_script_content.lower():
            bemio_script_content += '\nexit;\n'
        
        # Crear archivo bemio temporal
        bemio_temp_path = os.path.join(output_directory, 'bemio_temp.m')
        with open(bemio_temp_path, 'w') as f:
            f.write(bemio_script_content)
        
        # Ejecutar MATLAB usando subprocess
        if verbose:
            print("⚙️  Ejecutando BEMIO con subprocess...")
        
        # Cambiar al directorio de trabajo
        original_dir = os.getcwd()
        os.chdir(output_directory)
        
        # Ejecutar MATLAB
        result = subprocess.run(['matlab', '-batch', 'bemio_temp'], 
                              capture_output=True, text=True, timeout=300)
        
        # Volver al directorio original
        os.chdir(original_dir)
        
        # Limpiar archivo temporal
        if os.path.exists(bemio_temp_path):
            os.remove(bemio_temp_path)
        
        if result.returncode == 0:
            if verbose:
                print("✅ BEMIO ejecutado exitosamente con subprocess")
            return True
        else:
            if verbose:
                print(f"⚠️  BEMIO terminó con código: {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr}")
            return False
    
    except FileNotFoundError:
        if verbose:
            print("❌ MATLAB no encontrado en el PATH del sistema")
            print("   Asegúrate de que MATLAB esté instalado y accesible desde línea de comandos")
        return False
    
    except subprocess.TimeoutExpired:
        if verbose:
            print("⏱️  BEMIO excedió el tiempo límite (5 minutos)")
        return False
    
    except Exception as e:
        if verbose:
            print(f"❌ Error ejecutando BEMIO con subprocess: {e}")
        return False

def _generate_bemio_script(nc_filename, bemio_script_path, radiation_time, 
                          excitation_time, state_space_option):
    """
    Genera el contenido del script BEMIO
    """
    
    # Usar script personalizado si se proporciona
    if bemio_script_path and os.path.exists(bemio_script_path):
        with open(bemio_script_path, 'r') as f:
            return f.read()
    
    # Script BEMIO por defecto
    bemio_script_content = f"""% Script BEMIO generado automáticamente
% Archivo: {nc_filename}
% Fecha: {os.path.basename(__file__)}

hydro = struct();

% Leer datos de Capytaine
hydro = readCAPYTAINE(hydro,'{nc_filename}');

% Calcular funciones de respuesta al impulso de radiación
hydro = radiationIRF(hydro,{radiation_time},[],[],[],{state_space_option});

% Aproximación en espacio de estados para radiación
hydro = radiationIRFSS(hydro,[],[]);

% Calcular funciones de respuesta al impulso de excitación
hydro = excitationIRF(hydro,{excitation_time},[],[],[],{state_space_option});

% Escribir archivos HDF5 para WEC-Sim
writeBEMIOH5(hydro);

% Generar gráficas
plotBEMIO(hydro);

% Mostrar resumen
fprintf('\\n=== BEMIO Post-procesamiento completado ===\\n');
fprintf('Archivo procesado: {nc_filename}\\n');
fprintf('Tiempo IRF radiación: {radiation_time} s\\n');
fprintf('Tiempo IRF excitación: {excitation_time} s\\n');
fprintf('===========================================\\n\\n');
"""
    
    return bemio_script_content

def check_bemio_availability():
    """
    Verifica si BEMIO está disponible en el sistema
    
    Returns:
    --------
    dict : Diccionario con información sobre disponibilidad
    """
    availability = {
        'matlab_engine': False,
        'matlab_subprocess': False,
        'bemio_detected': False,
        'recommendations': []
    }
    
    # Verificar MATLAB Engine API
    try:
        import matlab.engine
        availability['matlab_engine'] = True
    except ImportError:
        availability['recommendations'].append(
            "Instalar MATLAB Engine API: cd(fullfile(matlabroot,'extern','engines','python')); system('python setup.py install')"
        )
    
    # Verificar MATLAB subprocess
    try:
        result = subprocess.run(['matlab', '-batch', 'exit'], 
                              capture_output=True, timeout=30)
        availability['matlab_subprocess'] = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        availability['recommendations'].append(
            "Asegurar que MATLAB esté en el PATH del sistema"
        )
    
    # Verificar si BEMIO está en el path de MATLAB (requiere MATLAB disponible)
    if availability['matlab_engine'] or availability['matlab_subprocess']:
        availability['recommendations'].append(
            "Verificar que BEMIO esté instalado y en el path de MATLAB"
        )
    
    return availability

# Ejemplo de uso y prueba
if __name__ == "__main__":
    print("=== Eco_Bemio - Módulo de post-procesamiento BEMIO ===\n")
    
    # Verificar disponibilidad
    availability = check_bemio_availability()
    print("Disponibilidad del sistema:")
    print(f"  MATLAB Engine API: {'✅' if availability['matlab_engine'] else '❌'}")
    print(f"  MATLAB Subprocess: {'✅' if availability['matlab_subprocess'] else '❌'}")
    
    if availability['recommendations']:
        print("\nRecomendaciones:")
        for rec in availability['recommendations']:
            print(f"  - {rec}")
    
    print("\nPara usar este módulo:")
    print("  from Eco_Bemio import run_bemio_postprocessing")
    print("  success = run_bemio_postprocessing('path/to/file.nc')")
