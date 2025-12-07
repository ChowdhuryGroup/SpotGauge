"""
FWHM (Full Width at Half Maximum) Calculator for Focal Spot Analysis

This module provides functions to calculate the FWHM of a focal spot image.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def calculate_width_at_threshold(profile, threshold_fraction):
    """
    Calculate the width of a 1D intensity profile at a given threshold fraction.
    
    Parameters
    ----------
    profile : array-like
        1D array of intensity values
    threshold_fraction : float
        The fraction of maximum intensity to measure width at (e.g., 0.5 for FWHM)
        
    Returns
    -------
    float
        The width in pixels at the specified threshold
    """
    profile = np.asarray(profile, dtype=float)
    
    print(f"[DEBUG] calculate_width_at_threshold: profile length={len(profile)}, threshold_fraction={threshold_fraction}")
    
    # Find the maximum value and its position
    max_val = np.max(profile)
    max_idx = np.argmax(profile)
    
    print(f"[DEBUG] calculate_width_at_threshold: max_val={max_val}, max_idx={max_idx}")
    
    # Validate that we have a meaningful signal
    if max_val <= 0 or np.isnan(max_val) or np.isinf(max_val):
        print(f"[ERROR] Invalid maximum value in profile: max_val={max_val}")
        return 0.0
    
    # Calculate threshold value
    threshold = max_val * threshold_fraction
    
    print(f"[DEBUG] calculate_width_at_threshold: threshold={threshold}")
    
    # Find the indices where the profile crosses threshold
    # Left side
    left_idx = max_idx
    while left_idx > 0 and profile[left_idx] > threshold:
        left_idx -= 1
    
    # Interpolate for more accurate position
    if left_idx > 0 and profile[left_idx] != profile[left_idx + 1]:
        left_pos = left_idx + (threshold - profile[left_idx]) / (profile[left_idx + 1] - profile[left_idx])
    else:
        left_pos = left_idx
    
    # Right side
    right_idx = max_idx
    while right_idx < len(profile) - 1 and profile[right_idx] > threshold:
        right_idx += 1
    
    # Interpolate for more accurate position
    if 0 < right_idx < len(profile) and profile[right_idx] != profile[right_idx - 1]:
        right_pos = right_idx - 1 + (threshold - profile[right_idx - 1]) / (profile[right_idx] - profile[right_idx - 1])
    else:
        right_pos = right_idx
    
    width = right_pos - left_pos
    
    print(f"[DEBUG] calculate_width_at_threshold: left_pos={left_pos}, right_pos={right_pos}, width={width}")
    
    # Validate result
    if np.isnan(width) or np.isinf(width) or width < 0:
        print(f"[ERROR] Invalid width calculated: width={width}")
        return 0.0
    
    return width


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
    return calculate_width_at_threshold(profile, 0.5)


def estimate_lineout_background(profile, edge_fraction=0.1):
    """
    Estimate background level from the edges of a lineout profile.
    
    Parameters
    ----------
    profile : array-like
        1D array of intensity values
    edge_fraction : float, optional
        Fraction of profile length to use from each edge (default: 0.1)
        
    Returns
    -------
    float
        Estimated background level (minimum of edge averages)
    """
    profile = np.asarray(profile, dtype=float)
    n = len(profile)
    edge_size = max(1, int(n * edge_fraction))
    
    # Get average of left and right edges
    left_edge = np.mean(profile[:edge_size])
    right_edge = np.mean(profile[-edge_size:])
    
    # Use minimum of the two edges as background estimate
    # This is conservative and avoids overestimating background
    return min(left_edge, right_edge)


def calculate_fwhm_2d(image, smooth_sigma=1.0, lineout_width=1, subtract_lineout_bg=False):
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
    subtract_lineout_bg : bool, optional
        If True, estimate and subtract background from lineout profiles (default: False)
        
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
    
    # Subtract lineout background if requested
    bg_x = None
    bg_y = None
    if subtract_lineout_bg:
        bg_x = estimate_lineout_background(profile_x)
        bg_y = estimate_lineout_background(profile_y)
        print(f"[DEBUG] Background estimation: bg_x={bg_x}, bg_y={bg_y}")
        profile_x = np.clip(profile_x - bg_x, 0, None)
        profile_y = np.clip(profile_y - bg_y, 0, None)
    
    print(f"[DEBUG] Profile X stats: min={np.min(profile_x)}, max={np.max(profile_x)}, mean={np.mean(profile_x)}")
    print(f"[DEBUG] Profile Y stats: min={np.min(profile_y)}, max={np.max(profile_y)}, mean={np.mean(profile_y)}")
    
    # Calculate FWHM for each direction
    print("[DEBUG] Calculating FWHM X...")
    fwhm_x = calculate_fwhm_1d(profile_x)
    print(f"[DEBUG] FWHM X calculated: {fwhm_x}")
    
    print("[DEBUG] Calculating FWHM Y...")
    fwhm_y = calculate_fwhm_1d(profile_y)
    print(f"[DEBUG] FWHM Y calculated: {fwhm_y}")
    
    # Calculate 1/e^2 radius - a standard laser beam measurement.
    # The 1/e^2 radius is where the intensity drops to 1/e^2 ≈ 13.5% of peak.
    # For a Gaussian beam exp(-2r²/w²), this occurs at r = w (the beam radius).
    e2_threshold = 1.0 / (np.e ** 2)  # ≈ 0.1353
    print(f"[DEBUG] Calculating 1/e² widths with threshold={e2_threshold}...")
    width_e2_x = calculate_width_at_threshold(profile_x, e2_threshold)
    width_e2_y = calculate_width_at_threshold(profile_y, e2_threshold)
    radius_e2_x = width_e2_x / 2.0
    radius_e2_y = width_e2_y / 2.0
    print(f"[DEBUG] 1/e² radius X: {radius_e2_x}, Y: {radius_e2_y}")
    
    # Validate all calculated values before creating result
    print("[DEBUG] Validating calculated values...")
    
    # Check for NaN or Inf values
    values_to_check = {
        'fwhm_x': fwhm_x,
        'fwhm_y': fwhm_y,
        'radius_e2_x': radius_e2_x,
        'radius_e2_y': radius_e2_y
    }
    
    for name, value in values_to_check.items():
        if np.isnan(value) or np.isinf(value):
            print(f"[ERROR] Invalid value detected: {name}={value}")
            # Set to 0 as a safe fallback
            values_to_check[name] = 0.0
            print(f"[WARNING] Setting {name} to 0.0 as fallback")
    
    result = {
        'fwhm_x': float(values_to_check['fwhm_x']),
        'fwhm_y': float(values_to_check['fwhm_y']),
        'radius_e2_x': float(values_to_check['radius_e2_x']),
        'radius_e2_y': float(values_to_check['radius_e2_y']),
        'center_x': int(center_x),
        'center_y': int(center_y),
        'profile_x': profile_x.tolist(),
        'profile_y': profile_y.tolist()
    }
    
    print(f"[DEBUG] Result dictionary created with keys: {list(result.keys())}")
    print(f"[DEBUG] FWHM values in result: fwhm_x={result['fwhm_x']}, fwhm_y={result['fwhm_y']}")
    
    # Include background values if subtraction was performed
    if bg_x is not None:
        result['bg_x'] = float(bg_x)
        result['bg_y'] = float(bg_y)
    
    return result


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


def process_image_data(image_data, smooth_sigma=1.0, background=None, lineout_width=1, crop_size=None, subtract_lineout_bg=False):
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
    subtract_lineout_bg : bool, optional
        If True and no background image is provided, estimate and subtract background 
        from lineout profiles (default: False)
        
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
    
    # Apply lineout background subtraction only if requested AND no background image was provided
    apply_lineout_bg_subtraction = subtract_lineout_bg and (background is None)
    
    result = calculate_fwhm_2d(image, smooth_sigma, lineout_width, subtract_lineout_bg=apply_lineout_bg_subtraction)
    
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
