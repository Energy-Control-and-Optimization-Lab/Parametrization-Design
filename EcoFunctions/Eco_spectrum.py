"""
Wave spectrum generation module
Author: Pablo Antonio Matamala Carvajal
Date: 2025-07-21
"""

import numpy as np

def jonswap_spectrum(frequencies, Hs, Tp, gamma=3.3):
    """
    Calculate JONSWAP spectrum
    
    Parameters:
    -----------
    frequencies : array
        Frequencies [Hz] or [rad/s]
    Hs : float
        Significant wave height [m]
    Tp : float
        Peak period [s]
    gamma : float
        Peak enhancement factor
    
    Returns:
    --------
    S : array
        Spectral density [m²·s]
    """
    
    f = np.array(frequencies)
    
    # Convert rad/s to Hz if needed
    if np.max(f) > 10:  # Assume rad/s if max > 10
        f = f / (2 * np.pi)
    
    # Avoid division by zero
    f = np.where(f <= 0, 1e-6, f)
    
    # Parameters
    fp = 1.0 / Tp  # Peak frequency [Hz]
    alpha = 5.0 * (Hs**2) / (16.0 * Tp**4)
    g = 9.81
    
    # Sigma for enhancement factor
    sigma = np.where(f <= fp, 0.07, 0.09)
    
    # Base spectrum
    S = (alpha * g**2 * (2*np.pi)**(-4) * f**(-5) * 
         np.exp(-1.25 * (fp/f)**4))
    
    # JONSWAP enhancement factor
    enhancement = gamma**np.exp(-0.5 * ((f - fp) / (sigma * fp))**2)
    
    # Final spectrum
    S = S * enhancement
    
    # Adjust units if input was in rad/s
    if np.max(frequencies) > 10:
        S = S / (2 * np.pi)
    
    return S

def create_spectrum(frequencies, Hs, Tp, spectrum_type='jonswap', gamma=3.3):
    """
    Create wave spectrum (extensible for other types)
    
    Parameters:
    -----------
    frequencies : array
        Frequency array
    Hs : float
        Significant wave height [m]
    Tp : float
        Peak period [s]
    spectrum_type : str
        Spectrum type ('jonswap' for now)
    gamma : float
        Peak enhancement factor for JONSWAP
    
    Returns:
    --------
    S : array
        Spectral density
    """
    
    if spectrum_type.lower() == 'jonswap':
        return jonswap_spectrum(frequencies, Hs, Tp, gamma)
    else:
        raise ValueError(f"Spectrum '{spectrum_type}' not implemented")

# Typical sea states
SEA_STATES = {
    'calm': {'Hs': 0.5, 'Tp': 6.0},
    'moderate': {'Hs': 2.0, 'Tp': 8.0},
    'rough': {'Hs': 4.0, 'Tp': 10.0},
    'extreme': {'Hs': 8.0, 'Tp': 14.0}
}

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create frequencies
    f = np.linspace(0.01, 0.5, 200)  # Hz
    
    # Create spectrum
    S = create_spectrum(f, Hs=2.5, Tp=8.0)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(f, S, 'b-', linewidth=2)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Spectral density [m²·s]')
    plt.title('JONSWAP Spectrum - Hs=2.5m, Tp=8.0s')
    plt.grid(True)
    plt.show()
    
    print("JONSWAP spectrum created successfully!")
