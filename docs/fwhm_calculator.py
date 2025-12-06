"""
FWHM (Full Width at Half Maximum) Calculator for Focal Spot Analysis

This module provides functions to calculate the FWHM of a focal spot image.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def calculate_fwhm_1d(profile):
    """
    Calculate the FWHM of a 1D intensity profile.
    
    Parameters
    ----------
    profile : array-like
        1D array of intensity values
        
    Returns
    -------
    float
        The FWHM in pixels
    """
    profile = np.asarray(profile, dtype=float)
    
    # Find the maximum value and its position
    max_val = np.max(profile)
    max_idx = np.argmax(profile)
    
    # Calculate half maximum
    half_max = max_val / 2.0
    
    # Find the indices where the profile crosses half maximum
    # Left side
    left_idx = max_idx
    while left_idx > 0 and profile[left_idx] > half_max:
        left_idx -= 1
    
    # Interpolate for more accurate position
    if left_idx > 0 and profile[left_idx] != profile[left_idx + 1]:
        left_pos = left_idx + (half_max - profile[left_idx]) / (profile[left_idx + 1] - profile[left_idx])
    else:
        left_pos = left_idx
    
    # Right side
    right_idx = max_idx
    while right_idx < len(profile) - 1 and profile[right_idx] > half_max:
        right_idx += 1
    
    # Interpolate for more accurate position
    if 0 < right_idx < len(profile) and profile[right_idx] != profile[right_idx - 1]:
        right_pos = right_idx - 1 + (half_max - profile[right_idx - 1]) / (profile[right_idx] - profile[right_idx - 1])
    else:
        right_pos = right_idx
    
    fwhm = right_pos - left_pos
    return fwhm


def calculate_fwhm_2d(image, smooth_sigma=1.0):
    """
    Calculate the FWHM of a 2D focal spot image in both X and Y directions.
    
    Parameters
    ----------
    image : 2D array-like
        The focal spot image
    smooth_sigma : float, optional
        Gaussian smoothing sigma to reduce noise (default: 1.0)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'fwhm_x': FWHM in the X direction (pixels)
        - 'fwhm_y': FWHM in the Y direction (pixels)
        - 'center_x': X coordinate of the peak
        - 'center_y': Y coordinate of the peak
        - 'profile_x': 1D profile along X through the center
        - 'profile_y': 1D profile along Y through the center
    """
    image = np.asarray(image, dtype=float)
    
    # Apply Gaussian smoothing to reduce noise
    if smooth_sigma > 0:
        smoothed = gaussian_filter(image, sigma=smooth_sigma)
    else:
        smoothed = image
    
    # Find the peak position
    max_idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    center_y, center_x = max_idx
    
    # Extract 1D profiles through the peak
    profile_x = smoothed[center_y, :]
    profile_y = smoothed[:, center_x]
    
    # Calculate FWHM for each direction
    fwhm_x = calculate_fwhm_1d(profile_x)
    fwhm_y = calculate_fwhm_1d(profile_y)
    
    return {
        'fwhm_x': fwhm_x,
        'fwhm_y': fwhm_y,
        'center_x': center_x,
        'center_y': center_y,
        'profile_x': profile_x.tolist(),
        'profile_y': profile_y.tolist()
    }


def process_image_data(image_data, smooth_sigma=1.0, background=None):
    """
    Process raw image data and calculate FWHM.
    
    Parameters
    ----------
    image_data : 2D or 3D array-like
        The image data. If 3D (RGB/RGBA), it will be converted to grayscale.
    smooth_sigma : float, optional
        Gaussian smoothing sigma (default: 1.0)
    background : 2D or 3D array-like, optional
        Background image to subtract from the main image (default: None).
        Must have the same dimensions as image_data.
        
    Returns
    -------
    dict
        FWHM results from calculate_fwhm_2d
    """
    image = np.asarray(image_data, dtype=float)
    
    # Convert RGB/RGBA to grayscale if necessary
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            # RGBA - use only RGB channels
            image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        elif image.shape[2] == 3:
            # RGB
            image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            # Assume single channel
            image = image[:, :, 0]
    
    # Process background if provided
    if background is not None:
        bg = np.asarray(background, dtype=float)
        
        # Convert background to grayscale if necessary
        if len(bg.shape) == 3:
            if bg.shape[2] == 4:
                bg = 0.299 * bg[:, :, 0] + 0.587 * bg[:, :, 1] + 0.114 * bg[:, :, 2]
            elif bg.shape[2] == 3:
                bg = 0.299 * bg[:, :, 0] + 0.587 * bg[:, :, 1] + 0.114 * bg[:, :, 2]
            else:
                bg = bg[:, :, 0]
        
        # Resize background if dimensions don't match
        if bg.shape != image.shape:
            # Use simple resize by cropping or padding
            min_h = min(image.shape[0], bg.shape[0])
            min_w = min(image.shape[1], bg.shape[1])
            bg_resized = np.zeros_like(image)
            bg_resized[:min_h, :min_w] = bg[:min_h, :min_w]
            bg = bg_resized
        
        # Subtract background and clip to non-negative values
        image = np.clip(image - bg, 0, None)
    
    return calculate_fwhm_2d(image, smooth_sigma)
