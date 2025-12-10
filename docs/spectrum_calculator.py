"""
Spectrum Analysis Calculator

This module provides functions to analyze optical spectra for pulse duration calculations.
"""

import numpy as np


def to_python_scalar(value):
    """
    Convert numpy scalar to Python native type for Pyodide compatibility.
    
    Parameters
    ----------
    value : numeric
        Value to convert (can be numpy scalar or Python native)
        
    Returns
    -------
    int or float
        Python native type
    """
    if hasattr(value, 'item'):
        # It's a numpy scalar, use .item() for proper conversion
        return value.item()
    else:
        # Already a Python native type
        return value


def parse_spectrum_file(file_content):
    """
    Parse spectrum data from text file content.
    
    Automatically detects and skips header lines. Expects two columns:
    wavelength (nm) and intensity (pixel counts).
    
    Parameters
    ----------
    file_content : str
        The content of the spectrum file
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'wavelength': array of wavelengths in nm
        - 'intensity': array of intensity values
        - 'lines_skipped': number of header lines skipped
    """
    lines = file_content.strip().split('\n')
    wavelengths = []
    intensities = []
    lines_skipped = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try to parse as numeric data
        # Split by tab, comma, or whitespace
        parts = line.replace(',', '\t').split()
        if len(parts) < 2:
            lines_skipped += 1
            continue
            
        try:
            wl = float(parts[0])
            intensity = float(parts[1])
            wavelengths.append(wl)
            intensities.append(intensity)
        except ValueError:
            # This is likely a header line
            lines_skipped += 1
            continue
    
    if len(wavelengths) == 0:
        raise ValueError("No valid numeric data found in file")
    
    return {
        'wavelength': np.array(wavelengths),
        'intensity': np.array(intensities),
        'lines_skipped': lines_skipped
    }


def calculate_spectral_width_at_threshold(wavelengths, intensities, threshold_fraction):
    """
    Calculate the spectral width at a given threshold fraction.
    
    Parameters
    ----------
    wavelengths : array-like
        Array of wavelengths in nm
    intensities : array-like
        Array of intensity values
    threshold_fraction : float
        The fraction of maximum intensity to measure width at (e.g., 0.5 for FWHM)
        
    Returns
    -------
    float
        The spectral width in nm at the specified threshold
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    
    # Find the maximum value and its position
    max_val = np.max(intensities)
    max_idx = np.argmax(intensities)
    
    # Validate that we have a meaningful signal
    if max_val <= 0 or np.isnan(max_val) or np.isinf(max_val):
        print(f"[ERROR] Invalid maximum value in spectrum: max_val={max_val}")
        return 0.0
    
    # Calculate threshold value
    threshold = max_val * threshold_fraction
    
    # Find the indices where the spectrum crosses threshold
    # Left side
    left_idx = max_idx
    while left_idx > 0 and intensities[left_idx] > threshold:
        left_idx -= 1
    
    # Interpolate for more accurate position
    if left_idx > 0 and intensities[left_idx] != intensities[left_idx + 1]:
        # Linear interpolation
        frac = (threshold - intensities[left_idx]) / (intensities[left_idx + 1] - intensities[left_idx])
        left_wl = wavelengths[left_idx] + frac * (wavelengths[left_idx + 1] - wavelengths[left_idx])
    else:
        left_wl = wavelengths[left_idx]
    
    # Right side
    right_idx = max_idx
    while right_idx < len(intensities) - 1 and intensities[right_idx] > threshold:
        right_idx += 1
    
    # Interpolate for more accurate position
    if 0 < right_idx < len(intensities) and intensities[right_idx] != intensities[right_idx - 1]:
        frac = (threshold - intensities[right_idx - 1]) / (intensities[right_idx] - intensities[right_idx - 1])
        right_wl = wavelengths[right_idx - 1] + frac * (wavelengths[right_idx] - wavelengths[right_idx - 1])
    else:
        right_wl = wavelengths[right_idx]
    
    width = right_wl - left_wl
    
    # Validate result
    if np.isnan(width) or np.isinf(width) or width < 0:
        print(f"[ERROR] Invalid width calculated: width={width}")
        return 0.0
    
    return float(width)


def calculate_transform_limited_pulse(wavelength_nm, bandwidth_nm, pulse_shape='gaussian'):
    """
    Calculate the transform-limited pulse duration from spectral bandwidth.
    
    Uses the time-bandwidth product relationship:
    Δν × Δt ≥ K
    
    where K depends on the pulse shape (0.441 for Gaussian, 0.315 for sech²).
    
    Parameters
    ----------
    wavelength_nm : float
        Center wavelength in nm
    bandwidth_nm : float
        Spectral bandwidth (FWHM) in nm
    pulse_shape : str, optional
        Pulse shape: 'gaussian' or 'sech2' (default: 'gaussian')
        
    Returns
    -------
    float
        Transform-limited pulse duration in femtoseconds (FWHM)
    """
    # Speed of light in nm/s
    c = 299792458e9  # nm/s
    
    # Convert wavelength bandwidth to frequency bandwidth
    # Δν = c × Δλ / λ²
    freq_bandwidth_hz = c * bandwidth_nm / (wavelength_nm ** 2)
    
    # Time-bandwidth product constants
    if pulse_shape.lower() == 'sech2':
        K = 0.315
    else:  # gaussian
        K = 0.441
    
    # Calculate transform-limited pulse duration
    # Δt = K / Δν
    pulse_duration_s = K / freq_bandwidth_hz
    
    # Convert to femtoseconds
    pulse_duration_fs = pulse_duration_s * 1e15
    
    return float(pulse_duration_fs)


def subtract_background_from_edges(intensities, edge_fraction=0.1):
    """
    Estimate and subtract background from spectrum using edge values.
    
    Parameters
    ----------
    intensities : array-like
        Array of intensity values
    edge_fraction : float, optional
        Fraction of spectrum length to use from each edge (default: 0.1)
        
    Returns
    -------
    array
        Background-subtracted intensity values
    """
    intensities = np.asarray(intensities, dtype=float)
    n = len(intensities)
    edge_n = max(1, int(n * edge_fraction))
    
    # Calculate average of edge regions
    left_edge = np.mean(intensities[:edge_n])
    right_edge = np.mean(intensities[-edge_n:])
    
    # Use the minimum of the two edges as background
    background = min(left_edge, right_edge)
    
    print(f"[DEBUG] Background subtraction from edges: left={left_edge:.2f}, right={right_edge:.2f}, bg={background:.2f}")
    
    # Subtract background and ensure non-negative values
    subtracted = intensities - background
    subtracted = np.maximum(subtracted, 0)
    
    return subtracted


def subtract_background_from_file(wavelengths, intensities, bg_file_content):
    """
    Subtract background spectrum from main spectrum.
    
    Parameters
    ----------
    wavelengths : array-like
        Array of wavelengths in nm
    intensities : array-like
        Array of intensity values
    bg_file_content : str
        Content of background spectrum file
        
    Returns
    -------
    array
        Background-subtracted intensity values
    """
    # Parse background file
    bg_parsed = parse_spectrum_file(bg_file_content)
    bg_wavelengths = bg_parsed['wavelength']
    bg_intensities = bg_parsed['intensity']
    
    print(f"[DEBUG] Background file: {len(bg_wavelengths)} points")
    
    # Check if wavelength arrays match
    if len(wavelengths) != len(bg_wavelengths):
        print(f"[WARNING] Wavelength arrays have different lengths: main={len(wavelengths)}, bg={len(bg_wavelengths)}")
        # Interpolate background to match main spectrum wavelengths
        bg_intensities = np.interp(wavelengths, bg_wavelengths, bg_intensities)
    elif not np.allclose(wavelengths, bg_wavelengths, rtol=1e-5):
        print(f"[WARNING] Wavelength arrays don't match exactly, interpolating background")
        bg_intensities = np.interp(wavelengths, bg_wavelengths, bg_intensities)
    
    # Subtract background and ensure non-negative values
    subtracted = intensities - bg_intensities
    subtracted = np.maximum(subtracted, 0)
    
    print(f"[DEBUG] Background subtraction complete: max={np.max(subtracted):.2f}")
    
    return subtracted


def analyze_spectrum(file_content, pulse_shape='gaussian', bg_subtraction='none', bg_file_content=None):
    """
    Analyze a spectrum file and calculate key parameters.
    
    Parameters
    ----------
    file_content : str
        The content of the spectrum file
    pulse_shape : str, optional
        Pulse shape for transform limit calculation: 'gaussian' or 'sech2'
        (default: 'gaussian')
    bg_subtraction : str, optional
        Background subtraction method: 'none', 'edges', or 'file'
        (default: 'none')
    bg_file_content : str, optional
        Content of background spectrum file (required if bg_subtraction='file')
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'wavelength': array of wavelengths in nm
        - 'intensity': array of intensity values (after background subtraction)
        - 'center_wavelength': center wavelength in nm
        - 'fwhm_nm': spectral FWHM in nm
        - 'width_e2_nm': spectral width at 1/e² in nm
        - 'transform_limit_fs': transform-limited pulse duration in fs
        - 'lines_skipped': number of header lines skipped
        - 'bg_subtracted': boolean indicating if background was subtracted
    """
    # Parse the file
    parsed = parse_spectrum_file(file_content)
    wavelengths = parsed['wavelength']
    intensities = parsed['intensity']
    
    # Apply background subtraction if requested
    bg_subtracted = False
    if bg_subtraction == 'edges':
        intensities = subtract_background_from_edges(intensities)
        bg_subtracted = True
    elif bg_subtraction == 'file':
        if bg_file_content is None:
            raise ValueError("bg_file_content is required when bg_subtraction='file'")
        intensities = subtract_background_from_file(wavelengths, intensities, bg_file_content)
        bg_subtracted = True
    
    # Find center wavelength (peak)
    center_idx = np.argmax(intensities)
    center_wavelength = float(wavelengths[center_idx])
    
    # Calculate FWHM
    fwhm_nm = calculate_spectral_width_at_threshold(wavelengths, intensities, 0.5)
    
    # Calculate 1/e² width
    # For Gaussian beams, 1/e² is the intensity level where I = I₀ × e^(-2) ≈ 0.1353 × I₀
    e2_threshold = np.exp(-2)  # ≈ 0.1353
    width_e2_nm = calculate_spectral_width_at_threshold(wavelengths, intensities, e2_threshold)
    
    # Calculate transform-limited pulse duration
    transform_limit_fs = 0.0
    if fwhm_nm > 0:
        transform_limit_fs = calculate_transform_limited_pulse(
            center_wavelength, fwhm_nm, pulse_shape
        )
    
    # Convert to Python native types for Pyodide
    result = {
        'wavelength': wavelengths.tolist(),
        'intensity': intensities.tolist(),
        'center_wavelength': float(to_python_scalar(center_wavelength)),
        'fwhm_nm': float(to_python_scalar(fwhm_nm)),
        'width_e2_nm': float(to_python_scalar(width_e2_nm)),
        'transform_limit_fs': float(to_python_scalar(transform_limit_fs)),
        'lines_skipped': parsed['lines_skipped'],
        'bg_subtracted': bg_subtracted
    }
    
    return result
