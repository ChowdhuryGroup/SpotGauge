"""
Tests for auto-rotation, orientation detection, and matplotlib visualization.

Run with: python test_auto_rotation.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'docs'))

from fwhm_calculator import (
    rotate_about_center,
    compute_orientation_and_centroid,
    generate_visualization_png,
    process_image_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tilted_gaussian(shape=(120, 120), center=(60, 60),
                          sigma_major=18.0, sigma_minor=10.0,
                          angle_deg=30.0):
    """Return a 2D Gaussian tilted by *angle_deg* degrees from the x-axis."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    cy, cx = center
    dx = x - cx
    dy = y - cy
    # Rotate coordinates to the Gaussian's frame
    xr = c * dx + s * dy
    yr = -s * dx + c * dy
    return np.exp(-(xr ** 2) / (2 * sigma_major ** 2)
                  - (yr ** 2) / (2 * sigma_minor ** 2))


# ---------------------------------------------------------------------------
# rotate_about_center tests
# ---------------------------------------------------------------------------

def test_rotate_identity():
    """Rotating by 0 should return the same image."""
    np.random.seed(0)
    data = np.random.rand(50, 50)
    result = rotate_about_center(data, 0.0, 25.0, 25.0)
    assert np.allclose(data, result, atol=1e-6), \
        "Zero rotation should leave image unchanged"
    print("rotate_about_center identity test: ✓ PASSED\n")


def test_rotate_360():
    """Four 90-degree rotations should return the image to its original state."""
    sigma = 8.0
    y, x = np.mgrid[0:60, 0:60]
    data = np.exp(-((x - 30) ** 2 + (y - 20) ** 2) / (2 * sigma ** 2))

    cx, cy = 30.0, 30.0
    rotated = data.copy()
    for _ in range(4):
        rotated = rotate_about_center(rotated, np.pi / 2, cx, cy)

    # Allow some tolerance for spline interpolation artifacts
    peak_orig = np.unravel_index(np.argmax(data), data.shape)
    peak_rot = np.unravel_index(np.argmax(rotated), rotated.shape)
    dist = np.sqrt((peak_orig[0] - peak_rot[0]) ** 2 +
                   (peak_orig[1] - peak_rot[1]) ** 2)
    assert dist < 2.0, f"Peak moved too much after 4×90° rotations: dist={dist:.2f}"
    print("rotate_about_center 4×90° test: ✓ PASSED\n")


def test_rotate_center_preserved():
    """Rotating about the center of a symmetric Gaussian leaves peak in place."""
    sigma = 10.0
    y, x = np.mgrid[0:80, 0:80]
    data = np.exp(-((x - 40) ** 2 + (y - 40) ** 2) / (2 * sigma ** 2))
    cx, cy = 40.0, 40.0

    for angle_deg in [15, 30, 45, 60, 90]:
        rotated = rotate_about_center(data, np.radians(angle_deg), cx, cy)
        peak = np.unravel_index(np.argmax(rotated), rotated.shape)
        dist = np.sqrt((peak[0] - cy) ** 2 + (peak[1] - cx) ** 2)
        assert dist < 1.5, \
            f"Peak of symmetric Gaussian moved at angle={angle_deg}°: dist={dist:.2f}"

    print("rotate_about_center center-preservation test: ✓ PASSED\n")


# ---------------------------------------------------------------------------
# compute_orientation_and_centroid tests
# ---------------------------------------------------------------------------

def test_orientation_axis_aligned_horizontal():
    """Axis-aligned horizontal ellipse needs no rotation (theta ≈ 0)."""
    sigma_x, sigma_y = 20.0, 8.0
    y, x = np.mgrid[0:100, 0:100]
    data = np.exp(-((x - 50) ** 2) / (2 * sigma_x ** 2)
                  - ((y - 50) ** 2) / (2 * sigma_y ** 2))
    theta, cx, cy = compute_orientation_and_centroid(data)

    # Major axis is already along x: no rotation needed → theta ≈ 0
    assert abs(theta) < 0.15, \
        f"Horizontal ellipse should need no rotation: theta={np.degrees(theta):.1f}°"
    assert abs(cx - 50) < 1.0, f"Centroid cx wrong: {cx}"
    assert abs(cy - 50) < 1.0, f"Centroid cy wrong: {cy}"
    print("compute_orientation_and_centroid horizontal ellipse test: ✓ PASSED\n")


def test_orientation_axis_aligned_vertical():
    """Axis-aligned vertical ellipse needs 90° rotation to align with x (theta ≈ π/2)."""
    sigma_x, sigma_y = 8.0, 20.0
    y, x = np.mgrid[0:100, 0:100]
    data = np.exp(-((x - 50) ** 2) / (2 * sigma_x ** 2)
                  - ((y - 50) ** 2) / (2 * sigma_y ** 2))
    theta, cx, cy = compute_orientation_and_centroid(data)

    # Major axis is along y (rows): rotate 90° CW to align with x → theta ≈ π/2
    assert abs(theta - np.pi / 2) < 0.15, \
        f"Vertical ellipse should need π/2 rotation: theta={np.degrees(theta):.1f}°"
    assert abs(cx - 50) < 1.0
    assert abs(cy - 50) < 1.0
    print("compute_orientation_and_centroid vertical ellipse test: ✓ PASSED\n")


def test_orientation_zero_image():
    """All-zero image should return (0, center_col, center_row) without error."""
    data = np.zeros((60, 60))
    theta, cx, cy = compute_orientation_and_centroid(data)
    assert theta == 0.0
    assert cx == 30.0 and cy == 30.0
    print("compute_orientation_and_centroid zero image test: ✓ PASSED\n")


# ---------------------------------------------------------------------------
# Auto-rotation integration tests
# ---------------------------------------------------------------------------

def test_auto_rotate_aligns_axes():
    """After auto-rotation, FWHM values should match the Gaussian sigma values."""
    sigma_major = 18.0
    sigma_minor = 8.0
    expected_fwhm_major = 2.355 * sigma_major   # ≈ 42.4 px
    expected_fwhm_minor = 2.355 * sigma_minor    # ≈ 18.8 px

    # Create an elliptical Gaussian tilted 30° from horizontal
    data = make_tilted_gaussian(shape=(150, 150), center=(75, 75),
                                sigma_major=sigma_major, sigma_minor=sigma_minor,
                                angle_deg=30.0)

    result = process_image_data(data, smooth_sigma=0, auto_rotate=True)

    fwhm_major = max(result['fwhm_x'], result['fwhm_y'])
    fwhm_minor = min(result['fwhm_x'], result['fwhm_y'])

    err_major = abs(fwhm_major - expected_fwhm_major)
    err_minor = abs(fwhm_minor - expected_fwhm_minor)

    print(f"Auto-rotate aligns axes test:")
    print(f"  Expected major FWHM: {expected_fwhm_major:.2f} px, "
          f"Measured: {fwhm_major:.2f} px, Error: {err_major:.2f}")
    print(f"  Expected minor FWHM: {expected_fwhm_minor:.2f} px, "
          f"Measured: {fwhm_minor:.2f} px, Error: {err_minor:.2f}")
    print(f"  Rotation angle: {result['rotation_angle_deg']:.1f}°")

    assert err_major < 3.0, f"Major FWHM error too large: {err_major:.2f}"
    assert err_minor < 3.0, f"Minor FWHM error too large: {err_minor:.2f}"
    print("  ✓ PASSED\n")


def test_no_auto_rotate_unchanged():
    """process_image_data with auto_rotate=False should give same FWHM as before."""
    sigma_x, sigma_y = 15.0, 10.0
    y, x = np.mgrid[0:100, 0:100]
    data = np.exp(-((x - 50) ** 2) / (2 * sigma_x ** 2)
                  - ((y - 50) ** 2) / (2 * sigma_y ** 2))

    result = process_image_data(data, smooth_sigma=0, auto_rotate=False)

    assert abs(result['fwhm_x'] - 2.355 * sigma_x) < 1.0
    assert abs(result['fwhm_y'] - 2.355 * sigma_y) < 1.0
    assert result['rotation_angle_deg'] == 0.0
    print("process_image_data auto_rotate=False test: ✓ PASSED\n")


def test_rotation_angle_returned():
    """rotation_angle_deg is present and nonzero for a tilted Gaussian."""
    data = make_tilted_gaussian(shape=(120, 120), center=(60, 60),
                                sigma_major=18.0, sigma_minor=8.0,
                                angle_deg=45.0)
    result = process_image_data(data, smooth_sigma=0, auto_rotate=True)

    assert 'rotation_angle_deg' in result, "rotation_angle_deg missing from result"
    assert abs(result['rotation_angle_deg']) > 1.0, \
        f"Rotation angle unexpectedly small: {result['rotation_angle_deg']}"
    print(f"rotation_angle_deg returned test: "
          f"angle={result['rotation_angle_deg']:.1f}° ✓ PASSED\n")


# ---------------------------------------------------------------------------
# generate_visualization_png tests
# ---------------------------------------------------------------------------

def test_visualization_png_returns_string():
    """generate_visualization_png should return a non-empty base64 string."""
    sigma = 12.0
    y, x = np.mgrid[0:80, 0:80]
    data = np.exp(-((x - 40) ** 2 + (y - 40) ** 2) / (2 * sigma ** 2))
    profile_x = data[40, :]
    profile_y = data[:, 40]

    png = generate_visualization_png(
        data_rot=data,
        profile_x=profile_x,
        profile_y=profile_y,
        fwhm_x=2.355 * sigma,
        fwhm_y=2.355 * sigma,
        center_x=40,
        center_y=40,
        lineout_width=0,
        rotation_angle_deg=0.0,
    )

    assert png is not None, "generate_visualization_png returned None"
    assert isinstance(png, str), "PNG result should be a string"
    assert len(png) > 100, "PNG base64 string too short"
    # Verify it's valid base64
    import base64
    decoded = base64.b64decode(png)
    assert decoded[:8] == b'\x89PNG\r\n\x1a\n', "Result is not a valid PNG"
    print("generate_visualization_png string test: ✓ PASSED\n")


def test_process_image_data_has_visualization_png():
    """process_image_data result should include visualization_png key."""
    sigma = 10.0
    y, x = np.mgrid[0:60, 0:60]
    data = np.exp(-((x - 30) ** 2 + (y - 30) ** 2) / (2 * sigma ** 2))

    result = process_image_data(data, smooth_sigma=0)

    assert 'visualization_png' in result, "'visualization_png' key missing from result"
    assert 'rotation_angle_deg' in result, "'rotation_angle_deg' key missing from result"
    # visualization_png is either a string or None (None only if matplotlib unavailable)
    if result['visualization_png'] is not None:
        assert isinstance(result['visualization_png'], str)
        assert len(result['visualization_png']) > 0
    print("process_image_data visualization_png key test: ✓ PASSED\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Auto-Rotation and Visualization Tests")
    print("=" * 60 + "\n")

    test_rotate_identity()
    test_rotate_360()
    test_rotate_center_preserved()

    test_orientation_axis_aligned_horizontal()
    test_orientation_axis_aligned_vertical()
    test_orientation_zero_image()

    test_auto_rotate_aligns_axes()
    test_no_auto_rotate_unchanged()
    test_rotation_angle_returned()

    test_visualization_png_returns_string()
    test_process_image_data_has_visualization_png()

    print("=" * 60)
    print("All auto-rotation tests passed! ✓")
    print("=" * 60)
