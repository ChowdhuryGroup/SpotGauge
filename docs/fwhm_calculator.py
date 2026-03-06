"""
FWHM (Full Width at Half Maximum) Calculator for Focal Spot Analysis

This module provides functions to calculate the FWHM of a focal spot image.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, affine_transform, label as ndimage_label


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
        native_val = value.item()
        return type(native_val)(native_val)
    else:
        # Already a Python native type
        return value


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
    
    print(f"[DEBUG] calculate_width_at_threshold: left_pos={left_pos}, right_pos={right_pos}, width={width}, width_type={type(width)}")
    
    # Validate result
    if np.isnan(width) or np.isinf(width) or width < 0:
        print(f"[ERROR] Invalid width calculated: width={width}")
        return 0.0
    
    # Ensure we return a Python float, not numpy scalar (important for Pyodide)
    return float(width)


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


def rotate_about_center(data, theta, cx, cy, order=3):
    """
    Rotate image data by angle theta (radians) about pixel coordinate (cx, cy).

    Parameters
    ----------
    data : 2D array-like
        The image to rotate
    theta : float
        Rotation angle in radians
    cx : float
        X coordinate (column index) of the rotation center
    cy : float
        Y coordinate (row index) of the rotation center
    order : int, optional
        Spline interpolation order (default: 3)

    Returns
    -------
    numpy.ndarray
        Rotated image with same shape as input
    """
    data = np.asarray(data, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    # Forward rotation matrix in (x, y) = (col, row) space
    R = np.array([[c, -s],
                  [s,  c]])
    # scipy.ndimage.affine_transform maps output coords -> input coords,
    # so use the inverse (transpose) rotation.
    A = R.T
    center = np.array([cx, cy])
    offset = center - A @ center
    # affine_transform expects matrix in (row, col) = (y, x) order
    A_rc = np.array([[A[1, 1], A[1, 0]],
                     [A[0, 1], A[0, 0]]])
    offset_rc = np.array([offset[1], offset[0]])
    return affine_transform(data, A_rc, offset=offset_rc, order=order,
                            mode='constant', cval=0.0)


def compute_orientation_and_centroid(image, threshold_fraction=1e-3):
    """
    Compute the orientation angle and weighted centroid of the brightest region.

    Uses intensity-weighted second central moments of the connected region
    containing the peak to find the principal axis orientation.

    Parameters
    ----------
    image : 2D array-like
        The focal spot image
    threshold_fraction : float, optional
        Fraction of peak intensity used to threshold the region (default: 1e-3)

    Returns
    -------
    tuple of (float, float, float)
        ``(theta, cx, cy)`` where *theta* is the rotation angle in radians
        to pass to :func:`rotate_about_center` in order to align the major
        axis of the focal spot with the image X (column) axis; *cx* is the
        intensity-weighted centroid column; *cy* is the intensity-weighted
        centroid row.
    """
    image = np.asarray(image, dtype=float)
    max_val = np.max(image)
    if max_val <= 0:
        h, w = image.shape
        return 0.0, float(w // 2), float(h // 2)

    # Threshold to find the bright region
    mask = image > threshold_fraction * max_val

    # Label connected components and isolate the one containing the peak
    labeled, _ = ndimage_label(mask)
    py, px = np.unravel_index(np.argmax(image), image.shape)
    peak_label = labeled[py, px]
    component_mask = (labeled == peak_label) if peak_label > 0 else mask

    weights = image * component_mask
    M00 = np.sum(weights)
    if M00 <= 0:
        return 0.0, float(px), float(py)

    # Intensity-weighted centroid
    y_coords, x_coords = np.indices(image.shape)
    cx = float(np.sum(x_coords * weights) / M00)
    cy = float(np.sum(y_coords * weights) / M00)

    # Normalized second central moments
    dx = x_coords - cx
    dy = y_coords - cy
    mu20_y = float(np.sum(dy ** 2 * weights) / M00)   # variance in row direction
    mu02_x = float(np.sum(dx ** 2 * weights) / M00)   # variance in col direction
    mu11   = float(np.sum(dx * dy * weights) / M00)    # cross term

    # Rotation angle to align the major axis with the image X (column) axis.
    # Derived from the covariance matrix in (col, row) space:
    #   [[mu02_x, mu11], [mu11, mu20_y]]
    # The angle of the major eigenvector from the x-axis is:
    #   angle_major = 0.5 * arctan2(2*mu11, mu02_x - mu20_y)
    # The rotation to align it with x is theta = -angle_major.
    #
    # When |mu11| is negligible (axis-aligned spot), handle explicitly to
    # avoid the sign of floating-point zero flipping the arctan2 quadrant.
    if abs(mu11) < 1e-10 * max(mu02_x, mu20_y, 1.0):
        # No cross-term: spot is already axis-aligned.
        # If wider in x (cols): no rotation needed.
        # If wider in y (rows): rotate 90° to align major axis with x.
        theta = 0.0 if mu02_x >= mu20_y else float(np.pi / 2)
    else:
        theta = 0.5 * float(np.arctan2(-2.0 * mu11, mu02_x - mu20_y))
    return theta, cx, cy


def generate_visualization_png(data_rot, profile_x, profile_y,
                                fwhm_x, fwhm_y, center_x, center_y,
                                lineout_width=0, rotation_angle_deg=0.0):
    """
    Generate a 3-panel matplotlib figure (focal spot + X lineout + Y lineout)
    and return it as a base64-encoded PNG string.

    Parameters
    ----------
    data_rot : 2D array-like
        The (possibly rotated) focal spot image for display
    profile_x : array-like
        1D intensity profile along the X (column) direction through the peak
    profile_y : array-like
        1D intensity profile along the Y (row) direction through the peak
    fwhm_x : float
        FWHM in the X direction (pixels)
    fwhm_y : float
        FWHM in the Y direction (pixels)
    center_x : float
        Column index of the peak
    center_y : float
        Row index of the peak
    lineout_width : int, optional
        Half-width of the averaging region used for lineouts (default: 0)
    rotation_angle_deg : float, optional
        Rotation angle in degrees applied to align principal axes (default: 0.0)

    Returns
    -------
    str or None
        Base64-encoded PNG string, or *None* if matplotlib is unavailable
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        import base64
    except ImportError:
        return None

    profile_x = np.asarray(profile_x, dtype=float)
    profile_y = np.asarray(profile_y, dtype=float)

    # Normalize lineouts to [0, 1]
    max_x = np.max(profile_x) if np.max(profile_x) > 0 else 1.0
    max_y = np.max(profile_y) if np.max(profile_y) > 0 else 1.0
    profile_x_norm = profile_x / max_x
    profile_y_norm = profile_y / max_y

    fwhm_mult = 2.0
    x_peak = int(np.argmax(profile_x_norm))
    y_peak = int(np.argmax(profile_y_norm))

    # Crop limits for the lineout plots
    x_lo = max(0.0, x_peak - fwhm_mult * fwhm_x)
    x_hi = min(float(len(profile_x) - 1), x_peak + fwhm_mult * fwhm_x)
    y_lo = max(0.0, y_peak - fwhm_mult * fwhm_y)
    y_hi = min(float(len(profile_y) - 1), y_peak + fwhm_mult * fwhm_y)

    # Normalize the image to [0, 1] for display
    data_rot = np.asarray(data_rot, dtype=float)
    data_max = np.max(data_rot)
    data_display = data_rot / data_max if data_max > 0 else data_rot

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # --- Panel 1: rotated focal spot image with inferno colormap ---
    im = axes[0].imshow(data_display, cmap='inferno', vmin=0, vmax=1,
                        origin='upper', interpolation='nearest')
    axes[0].set_xlim(x_lo, x_hi)   # column limits (horizontal)
    axes[0].set_ylim(y_lo, y_hi)   # row limits (vertical)
    axes[0].axhline(y=center_y, color='cyan', linewidth=1,
                    alpha=0.8, linestyle='--')
    axes[0].axvline(x=center_x, color='cyan', linewidth=1,
                    alpha=0.8, linestyle='--')
    title = (f'Focal Spot (θ={rotation_angle_deg:.1f}°)'
             if abs(rotation_angle_deg) > 0.5 else 'Focal Spot')
    axes[0].set_title(title)
    axes[0].set_xlabel('Column (px)')
    axes[0].set_ylabel('Row (px)')
    cbar = plt.colorbar(im, ax=axes[0])
    cbar.set_label('Normalized Intensity')

    # --- Panel 2: X profile (along columns) ---
    lw_label = f'{2 * lineout_width + 1}' if lineout_width > 0 else '1'
    x_pixels = np.arange(len(profile_x))
    axes[1].plot(x_pixels, profile_x_norm, color='#3498db', linewidth=1.5)
    axes[1].axhline(y=0.5, color='red', linestyle='--', linewidth=1,
                    label='Half-max')
    axes[1].axvline(x=x_peak - fwhm_x / 2, color='green',
                    linestyle='--', linewidth=1, label='FWHM')
    axes[1].axvline(x=x_peak + fwhm_x / 2, color='green',
                    linestyle='--', linewidth=1)
    axes[1].set_xlim(x_lo, x_hi)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel('Pixel Index')
    axes[1].set_ylabel('Normalized Intensity')
    axes[1].set_title(
        f'X Lineout (avg {lw_label} px, FWHM={fwhm_x:.1f} px)')
    axes[1].legend(fontsize=8)

    # --- Panel 3: Y profile (along rows) ---
    y_pixels = np.arange(len(profile_y))
    axes[2].plot(y_pixels, profile_y_norm, color='#3498db', linewidth=1.5)
    axes[2].axhline(y=0.5, color='red', linestyle='--', linewidth=1,
                    label='Half-max')
    axes[2].axvline(x=y_peak - fwhm_y / 2, color='green',
                    linestyle='--', linewidth=1, label='FWHM')
    axes[2].axvline(x=y_peak + fwhm_y / 2, color='green',
                    linestyle='--', linewidth=1)
    axes[2].set_xlim(y_lo, y_hi)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xlabel('Pixel Index')
    axes[2].set_ylabel('Normalized Intensity')
    axes[2].set_title(
        f'Y Lineout (avg {lw_label} px, FWHM={fwhm_y:.1f} px)')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


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
    print(f"[DEBUG] fwhm_x type: {type(fwhm_x)}, value: {fwhm_x}")
    print(f"[DEBUG] fwhm_y type: {type(fwhm_y)}, value: {fwhm_y}")
    print(f"[DEBUG] radius_e2_x type: {type(radius_e2_x)}, value: {radius_e2_x}")
    print(f"[DEBUG] radius_e2_y type: {type(radius_e2_y)}, value: {radius_e2_y}")
    
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
    
    # Explicitly convert to Python native types (important for Pyodide)
    try:
        fwhm_x_val = float(to_python_scalar(values_to_check['fwhm_x']))
        fwhm_y_val = float(to_python_scalar(values_to_check['fwhm_y']))
        radius_e2_x_val = float(to_python_scalar(values_to_check['radius_e2_x']))
        radius_e2_y_val = float(to_python_scalar(values_to_check['radius_e2_y']))
        center_x_val = int(to_python_scalar(center_x))
        center_y_val = int(to_python_scalar(center_y))
    except Exception as e:
        print(f"[ERROR] Type conversion error: {e}")
        # Fallback to basic conversion
        fwhm_x_val = float(values_to_check['fwhm_x'])
        fwhm_y_val = float(values_to_check['fwhm_y'])
        radius_e2_x_val = float(values_to_check['radius_e2_x'])
        radius_e2_y_val = float(values_to_check['radius_e2_y'])
        center_x_val = int(center_x)
        center_y_val = int(center_y)
    
    print(f"[DEBUG] Converted values - fwhm_x: {fwhm_x_val} (type: {type(fwhm_x_val)})")
    print(f"[DEBUG] Converted values - fwhm_y: {fwhm_y_val} (type: {type(fwhm_y_val)})")
    
    result = {
        'fwhm_x': fwhm_x_val,
        'fwhm_y': fwhm_y_val,
        'radius_e2_x': radius_e2_x_val,
        'radius_e2_y': radius_e2_y_val,
        'center_x': center_x_val,
        'center_y': center_y_val,
        'profile_x': profile_x.tolist(),
        'profile_y': profile_y.tolist()
    }
    
    print(f"[DEBUG] Result dictionary created with keys: {list(result.keys())}")
    print(f"[DEBUG] FWHM values in result: fwhm_x={result['fwhm_x']}, fwhm_y={result['fwhm_y']}")
    print(f"[DEBUG] Result value types: fwhm_x={type(result['fwhm_x'])}, fwhm_y={type(result['fwhm_y'])}")
    
    # Include background values if subtraction was performed
    if bg_x is not None:
        result['bg_x'] = float(to_python_scalar(bg_x))
        result['bg_y'] = float(to_python_scalar(bg_y))
    
    return result


def process_image_data(image_data, smooth_sigma=1.0, background=None, lineout_width=1, crop_size=None, subtract_lineout_bg=False, auto_rotate=False):
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
        Unused; kept for backward compatibility.
    subtract_lineout_bg : bool, optional
        If True and no background image is provided, estimate and subtract background 
        from lineout profiles (default: False)
    auto_rotate : bool, optional
        If True, automatically rotate the image to align the principal axes of the
        focal spot with the image X/Y axes before computing FWHM (default: False).
        
    Returns
    -------
    dict
        FWHM results from calculate_fwhm_2d, plus:
        - 'visualization_png': base64-encoded PNG of a 3-panel matplotlib figure
          (focal spot with inferno colormap + normalized colorbar, X lineout,
          Y lineout), or None if matplotlib is unavailable
        - 'rotation_angle_deg': rotation angle applied (degrees); 0 when
          auto_rotate is False
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
    
    # Auto-rotate to align the principal axes with the image X/Y axes
    rotation_angle_deg = 0.0
    image_for_display = image
    if auto_rotate and np.max(image) > 0:
        theta, cx_rot, cy_rot = compute_orientation_and_centroid(image)
        rotation_angle_deg = float(np.degrees(theta))
        print(f"[DEBUG] Auto-rotate: theta={rotation_angle_deg:.2f}°, "
              f"centroid=({cx_rot:.1f}, {cy_rot:.1f})")
        image_for_display = rotate_about_center(image, theta, cx_rot, cy_rot)
    else:
        image_for_display = image

    # Apply lineout background subtraction only if requested AND no background image was provided
    apply_lineout_bg_subtraction = subtract_lineout_bg and (background is None)
    
    result = calculate_fwhm_2d(image_for_display, smooth_sigma, lineout_width,
                                subtract_lineout_bg=apply_lineout_bg_subtraction)

    # Generate matplotlib visualization (inferno colormap + normalized colorbar)
    lineout_half_width = max(0, (lineout_width - 1) // 2)
    visualization_png = generate_visualization_png(
        data_rot=image_for_display,
        profile_x=result['profile_x'],
        profile_y=result['profile_y'],
        fwhm_x=result['fwhm_x'],
        fwhm_y=result['fwhm_y'],
        center_x=result['center_x'],
        center_y=result['center_y'],
        lineout_width=lineout_half_width,
        rotation_angle_deg=rotation_angle_deg,
    )

    result['visualization_png'] = visualization_png
    result['rotation_angle_deg'] = float(rotation_angle_deg)
    
    return result
