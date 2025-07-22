"""
Módulo simple para espectros de olas
Archivo: Eco_Spectrum.py
Author: Pablo Antonio Matamala Carvajal
Date: 2025-07-21
"""

import numpy as np

def jonswap_spectrum(frequencies, Hs, Tp, gamma=3.3):
    """
    Calcula el espectro JONSWAP
    
    Parameters:
    -----------
    frequencies : array
        Frecuencias [Hz] o [rad/s]
    Hs : float
        Altura significativa [m]
    Tp : float
        Período pico [s]
    gamma : float
        Factor de mejora pico (default: 3.3)
    
    Returns:
    --------
    S : array
        Densidad espectral [m²·s]
    """
    
    f = np.array(frequencies)
    
    # Si las frecuencias están en rad/s, convertir a Hz
    if np.max(f) > 10:  # Asume que si max > 10, están en rad/s
        f = f / (2 * np.pi)
    
    # Evitar división por cero
    f = np.where(f <= 0, 1e-6, f)
    
    # Parámetros
    fp = 1.0 / Tp  # Frecuencia pico [Hz]
    alpha = 5.0 * (Hs**2) / (16.0 * Tp**4)
    g = 9.81
    
    # Sigma para factor de mejora
    sigma = np.where(f <= fp, 0.07, 0.09)
    
    # Espectro base
    S = (alpha * g**2 * (2*np.pi)**(-4) * f**(-5) * 
         np.exp(-1.25 * (fp/f)**4))
    
    # Factor de mejora JONSWAP
    enhancement = gamma**np.exp(-0.5 * ((f - fp) / (sigma * fp))**2)
    
    # Espectro final
    S = S * enhancement
    
    # Si entrada era en rad/s, ajustar unidades
    if np.max(frequencies) > 10:
        S = S / (2 * np.pi)
    
    return S

def create_spectrum(frequencies, Hs, Tp, spectrum_type='jonswap', gamma=3.3):
    """
    Crea un espectro de olas (extensible para otros tipos)
    
    Parameters:
    -----------
    frequencies : array
        Array de frecuencias
    Hs : float
        Altura significativa [m]
    Tp : float
        Período pico [s]
    spectrum_type : str
        Tipo de espectro ('jonswap' por ahora)
    gamma : float
        Factor de mejora pico para JONSWAP
    
    Returns:
    --------
    S : array
        Densidad espectral
    """
    
    if spectrum_type.lower() == 'jonswap':
        return jonswap_spectrum(frequencies, Hs, Tp, gamma)
    else:
        raise ValueError(f"Espectro '{spectrum_type}' no implementado")

# Estados de mar típicos
SEA_STATES = {
    'calm': {'Hs': 0.5, 'Tp': 6.0},
    'moderate': {'Hs': 2.0, 'Tp': 8.0},
    'rough': {'Hs': 4.0, 'Tp': 10.0},
    'extreme': {'Hs': 8.0, 'Tp': 14.0}
}

# Ejemplo de uso
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Crear frecuencias
    f = np.linspace(0.01, 0.5, 200)  # Hz
    
    # Crear espectro
    S = create_spectrum(f, Hs=2.5, Tp=8.0)
    
    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(f, S, 'b-', linewidth=2)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Densidad espectral [m²·s]')
    plt.title('Espectro JONSWAP - Hs=2.5m, Tp=8.0s')
    plt.grid(True)
    plt.show()
    
    print("Espectro JONSWAP creado exitosamente!")
