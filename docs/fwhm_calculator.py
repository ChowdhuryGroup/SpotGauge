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


def calculate_fwhm_2d(image, smooth_sigma=1.0, lineout_width=1):
    """
    Calculate the FWHM of a 2D focal spot image in both X and Y directions.
    
    Parameters
    ----------
    image : 2D array-like
        The focal spot image
    smooth_sigma : float, optional
        Gaussian smoothing sigma to reduce noise (default: 1.0)
    lineout_width : int, optional
        Width of the lineout in pixels (default: 1). If greater than 1,
        the profile is averaged over multiple adjacent rows/columns.
        
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
    lineout_width = max(1, int(lineout_width))
    
    # Apply Gaussian smoothing to reduce noise
    if smooth_sigma > 0:
        smoothed = gaussian_filter(image, sigma=smooth_sigma)
    else:
        smoothed = image
    
    # Find the peak position
    max_idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    center_y, center_x = max_idx
    
    # Extract 1D profiles through the peak with lineout width
    half_width = lineout_width // 2
    height, width = smoothed.shape
    
    # X profile: average over lineout_width rows centered at center_y
    y_start = max(0, center_y - half_width)
    y_end = min(height, center_y + half_width + 1)
    profile_x = np.mean(smoothed[y_start:y_end, :], axis=0)
    
    # Y profile: average over lineout_width columns centered at center_x
    x_start = max(0, center_x - half_width)
    x_end = min(width, center_x + half_width + 1)
    profile_y = np.mean(smoothed[:, x_start:x_end], axis=1)
    
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


def apply_jet_colormap(data):
    """
    Apply jet colormap to normalized data.
    
    Parameters
    ----------
    data : 2D array-like
        Normalized 2D data (0-1 range)
        
    Returns
    -------
    list
        RGB values as a list of lists (height x width x 3)
    """
    data = np.asarray(data, dtype=float)
    
    # Jet colormap interpolation
    # Blue -> Cyan -> Green -> Yellow -> Red
    r = np.clip(1.5 - np.abs(4.0 * data - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * data - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * data - 1.0), 0, 1)
    
    # Combine into RGB array (0-255)
    rgb = np.stack([r, g, b], axis=-1) * 255
    return rgb.astype(np.uint8).tolist()


def process_image_data(image_data, smooth_sigma=1.0, background=None, lineout_width=1, crop_size=None):
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
        If dimensions differ from image_data, the background is center-aligned
        and cropped/padded to match.
    lineout_width : int, optional
        Width of the lineout in pixels (default: 1)
    crop_size : int, optional
        Size of the cropped focal spot region (default: None, auto-calculated based on FWHM)
        
    Returns
    -------
    dict
        FWHM results from calculate_fwhm_2d, plus:
        - 'cropped_jet': Cropped focal spot with jet colormap (RGB list)
        - 'crop_bounds': Dictionary with x_start, x_end, y_start, y_end
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
        
        # Center-align and resize background if dimensions don't match
        if bg.shape != image.shape:
            bg_resized = np.zeros_like(image)
            img_h, img_w = image.shape
            bg_h, bg_w = bg.shape
            
            # Calculate center-aligned offsets
            # For source (background): where to start reading
            src_y = max(0, (bg_h - img_h) // 2)
            src_x = max(0, (bg_w - img_w) // 2)
            # For destination (resized): where to start writing
            dst_y = max(0, (img_h - bg_h) // 2)
            dst_x = max(0, (img_w - bg_w) // 2)
            
            # Calculate the overlap region size
            copy_h = min(img_h - dst_y, bg_h - src_y)
            copy_w = min(img_w - dst_x, bg_w - src_x)
            
            bg_resized[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = \
                bg[src_y:src_y + copy_h, src_x:src_x + copy_w]
            bg = bg_resized
        
        # Subtract background and clip to non-negative values
        image = np.clip(image - bg, 0, None)
    
    result = calculate_fwhm_2d(image, smooth_sigma, lineout_width)
    
    # Create cropped focal spot with jet colormap
    center_x = result['center_x']
    center_y = result['center_y']
    fwhm_x = result['fwhm_x']
    fwhm_y = result['fwhm_y']
    
    # Auto-calculate crop size based on FWHM (3x the larger FWHM)
    if crop_size is None:
        crop_size = int(max(fwhm_x, fwhm_y) * 3)
    crop_size = max(crop_size, 10)  # Minimum size
    
    height, width = image.shape
    half_size = crop_size // 2
    
    # Calculate crop bounds
    x_start = max(0, center_x - half_size)
    x_end = min(width, center_x + half_size)
    y_start = max(0, center_y - half_size)
    y_end = min(height, center_y + half_size)
    
    # Extract cropped region
    cropped = image[y_start:y_end, x_start:x_end]
    
    # Normalize to 0-1 for colormap
    if cropped.max() > cropped.min():
        normalized = (cropped - cropped.min()) / (cropped.max() - cropped.min())
    else:
        normalized = np.zeros_like(cropped)
    
    # Apply jet colormap
    cropped_jet = apply_jet_colormap(normalized)
    
    result['cropped_jet'] = cropped_jet
    result['crop_bounds'] = {
        'x_start': int(x_start),
        'x_end': int(x_end),
        'y_start': int(y_start),
        'y_end': int(y_end)
    }
    
    return result
