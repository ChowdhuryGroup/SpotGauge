"""
Test script for FWHM calculator.

Run with: python test_fwhm.py
"""

import numpy as np
import sys
import os

# Add the docs directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'docs'))

from fwhm_calculator import calculate_fwhm_1d, calculate_fwhm_2d, process_image_data


def test_gaussian_1d():
    """Test FWHM calculation on a 1D Gaussian with known FWHM."""
    # Create a Gaussian with known parameters
    # FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
    sigma = 10.0
    expected_fwhm = 2.355 * sigma  # ~23.55
    
    x = np.arange(100)
    center = 50
    gaussian = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    
    fwhm = calculate_fwhm_1d(gaussian)
    
    # Allow for some numerical error
    error = abs(fwhm - expected_fwhm)
    print(f"1D Gaussian test:")
    print(f"  Expected FWHM: {expected_fwhm:.2f}")
    print(f"  Calculated FWHM: {fwhm:.2f}")
    print(f"  Error: {error:.2f} ({100*error/expected_fwhm:.1f}%)")
    
    assert error < 1.0, f"FWHM error too large: {error}"
    print("  ✓ PASSED\n")


def test_gaussian_2d():
    """Test FWHM calculation on a 2D Gaussian with known FWHM."""
    # Create a 2D Gaussian with different widths in X and Y
    sigma_x = 15.0
    sigma_y = 10.0
    expected_fwhm_x = 2.355 * sigma_x  # ~35.3
    expected_fwhm_y = 2.355 * sigma_y  # ~23.5
    
    y, x = np.mgrid[0:100, 0:100]
    center_x, center_y = 50, 50
    gaussian_2d = np.exp(-((x - center_x) ** 2) / (2 * sigma_x ** 2) 
                        - ((y - center_y) ** 2) / (2 * sigma_y ** 2))
    
    # Test without smoothing (sigma=0) since input is already smooth
    result = calculate_fwhm_2d(gaussian_2d, smooth_sigma=0)
    
    error_x = abs(result['fwhm_x'] - expected_fwhm_x)
    error_y = abs(result['fwhm_y'] - expected_fwhm_y)
    
    print(f"2D Gaussian test:")
    print(f"  Expected FWHM X: {expected_fwhm_x:.2f}, Calculated: {result['fwhm_x']:.2f}, Error: {error_x:.2f}")
    print(f"  Expected FWHM Y: {expected_fwhm_y:.2f}, Calculated: {result['fwhm_y']:.2f}, Error: {error_y:.2f}")
    print(f"  Center: ({result['center_x']}, {result['center_y']})")
    
    assert error_x < 1.0, f"FWHM X error too large: {error_x}"
    assert error_y < 1.0, f"FWHM Y error too large: {error_y}"
    assert np.isclose(result['center_x'], center_x, atol=1), f"Center X mismatch: {result['center_x']} vs {center_x}"
    assert np.isclose(result['center_y'], center_y, atol=1), f"Center Y mismatch: {result['center_y']} vs {center_y}"
    print("  ✓ PASSED\n")


def test_process_rgb_image():
    """Test processing of RGB image data."""
    # Create a synthetic RGB image with a Gaussian spot
    sigma = 12.0
    expected_fwhm = 2.355 * sigma  # ~28.3
    
    y, x = np.mgrid[0:80, 0:80]
    center = 40
    gaussian_2d = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    
    # Create RGB image (same value in all channels)
    rgb_image = np.stack([gaussian_2d * 255, gaussian_2d * 200, gaussian_2d * 150], axis=-1)
    
    result = process_image_data(rgb_image, smooth_sigma=0)
    
    print(f"RGB Image test:")
    print(f"  Expected FWHM: ~{expected_fwhm:.2f}")
    print(f"  Calculated FWHM X: {result['fwhm_x']:.2f}")
    print(f"  Calculated FWHM Y: {result['fwhm_y']:.2f}")
    
    # FWHM should be close to expected (symmetric Gaussian)
    assert abs(result['fwhm_x'] - expected_fwhm) < 2.0
    assert abs(result['fwhm_y'] - expected_fwhm) < 2.0
    print("  ✓ PASSED\n")


def test_process_rgba_image():
    """Test processing of RGBA image data."""
    sigma = 10.0
    expected_fwhm = 2.355 * sigma  # ~23.5
    
    y, x = np.mgrid[0:60, 0:60]
    center = 30
    gaussian_2d = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    
    # Create RGBA image
    rgba_image = np.stack([
        gaussian_2d * 255, 
        gaussian_2d * 200, 
        gaussian_2d * 150, 
        np.ones_like(gaussian_2d) * 255  # Alpha channel
    ], axis=-1)
    
    result = process_image_data(rgba_image, smooth_sigma=0)
    
    print(f"RGBA Image test:")
    print(f"  Expected FWHM: ~{expected_fwhm:.2f}")
    print(f"  Calculated FWHM X: {result['fwhm_x']:.2f}")
    print(f"  Calculated FWHM Y: {result['fwhm_y']:.2f}")
    
    assert abs(result['fwhm_x'] - expected_fwhm) < 2.0
    assert abs(result['fwhm_y'] - expected_fwhm) < 2.0
    print("  ✓ PASSED\n")


def test_noisy_gaussian():
    """Test FWHM calculation with noise and smoothing."""
    sigma = 15.0
    expected_fwhm = 2.355 * sigma
    
    y, x = np.mgrid[0:100, 0:100]
    center = 50
    gaussian_2d = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    
    # Add noise
    np.random.seed(42)
    noisy = gaussian_2d + np.random.normal(0, 0.05, gaussian_2d.shape)
    
    # Test with smoothing
    result = calculate_fwhm_2d(noisy, smooth_sigma=2.0)
    
    print(f"Noisy Gaussian test (with smoothing):")
    print(f"  Expected FWHM: ~{expected_fwhm:.2f}")
    print(f"  Calculated FWHM X: {result['fwhm_x']:.2f}")
    print(f"  Calculated FWHM Y: {result['fwhm_y']:.2f}")
    
    # Allow more tolerance due to noise
    assert abs(result['fwhm_x'] - expected_fwhm) < 3.0
    assert abs(result['fwhm_y'] - expected_fwhm) < 3.0
    print("  ✓ PASSED\n")


def test_background_subtraction():
    """Test FWHM calculation with background subtraction."""
    # Create a 2D Gaussian focal spot with a constant background
    sigma = 12.0
    expected_fwhm = 2.355 * sigma  # ~28.26
    background_level = 50.0
    
    y, x = np.mgrid[0:80, 0:80]
    center = 40
    gaussian_2d = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    
    # Create image with background (RGB)
    image_with_bg = np.stack([
        gaussian_2d * 200 + background_level,
        gaussian_2d * 200 + background_level,
        gaussian_2d * 200 + background_level
    ], axis=-1)
    
    # Create background image (constant)
    background = np.ones((80, 80, 3)) * background_level
    
    # Test with background subtraction
    result = process_image_data(image_with_bg, smooth_sigma=0, background=background)
    
    print(f"Background Subtraction test:")
    print(f"  Expected FWHM: ~{expected_fwhm:.2f}")
    print(f"  Calculated FWHM X: {result['fwhm_x']:.2f}")
    print(f"  Calculated FWHM Y: {result['fwhm_y']:.2f}")
    
    # FWHM should be close to expected
    assert abs(result['fwhm_x'] - expected_fwhm) < 2.0, f"FWHM X error: {abs(result['fwhm_x'] - expected_fwhm)}"
    assert abs(result['fwhm_y'] - expected_fwhm) < 2.0, f"FWHM Y error: {abs(result['fwhm_y'] - expected_fwhm)}"
    print("  ✓ PASSED\n")


def test_background_subtraction_rgba():
    """Test FWHM calculation with RGBA background subtraction."""
    sigma = 10.0
    expected_fwhm = 2.355 * sigma  # ~23.55
    background_level = 30.0
    
    y, x = np.mgrid[0:60, 0:60]
    center = 30
    gaussian_2d = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    
    # Create RGBA image with background
    image_with_bg = np.stack([
        gaussian_2d * 200 + background_level,
        gaussian_2d * 200 + background_level,
        gaussian_2d * 200 + background_level,
        np.ones_like(gaussian_2d) * 255  # Alpha channel
    ], axis=-1)
    
    # Create RGBA background image
    background = np.stack([
        np.ones((60, 60)) * background_level,
        np.ones((60, 60)) * background_level,
        np.ones((60, 60)) * background_level,
        np.ones((60, 60)) * 255
    ], axis=-1)
    
    result = process_image_data(image_with_bg, smooth_sigma=0, background=background)
    
    print(f"RGBA Background Subtraction test:")
    print(f"  Expected FWHM: ~{expected_fwhm:.2f}")
    print(f"  Calculated FWHM X: {result['fwhm_x']:.2f}")
    print(f"  Calculated FWHM Y: {result['fwhm_y']:.2f}")
    
    assert abs(result['fwhm_x'] - expected_fwhm) < 2.0
    assert abs(result['fwhm_y'] - expected_fwhm) < 2.0
    print("  ✓ PASSED\n")


def test_background_dimension_mismatch():
    """Test FWHM calculation with background of different dimensions."""
    sigma = 12.0
    expected_fwhm = 2.355 * sigma  # ~28.26
    background_level = 40.0
    
    # Create 80x80 image with center at 40,40
    y, x = np.mgrid[0:80, 0:80]
    center = 40
    gaussian_2d = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    
    # Create image with background (grayscale 2D)
    image_with_bg = gaussian_2d * 200 + background_level
    
    # Create a smaller 60x60 background image (should be center-aligned)
    background = np.ones((60, 60)) * background_level
    
    result = process_image_data(image_with_bg, smooth_sigma=0, background=background)
    
    print(f"Background Dimension Mismatch test:")
    print(f"  Image size: 80x80, Background size: 60x60")
    print(f"  Expected FWHM: ~{expected_fwhm:.2f}")
    print(f"  Calculated FWHM X: {result['fwhm_x']:.2f}")
    print(f"  Calculated FWHM Y: {result['fwhm_y']:.2f}")
    
    # FWHM should still be reasonable (center is covered by background)
    assert abs(result['fwhm_x'] - expected_fwhm) < 3.0, f"FWHM X error: {abs(result['fwhm_x'] - expected_fwhm)}"
    assert abs(result['fwhm_y'] - expected_fwhm) < 3.0, f"FWHM Y error: {abs(result['fwhm_y'] - expected_fwhm)}"
    print("  ✓ PASSED\n")


def test_radius_e2():
    """Test 1/e^2 radius calculation on a 2D Gaussian with known parameters."""
    # For a Gaussian exp(-x^2/(2*sigma^2)), intensity = 1/e^2 when:
    # exp(-x^2/(2*sigma^2)) = exp(-2) → x^2 = 4*sigma^2 → x = 2*sigma
    # So the 1/e^2 radius = 2*sigma
    sigma_x = 15.0
    sigma_y = 10.0
    expected_radius_e2_x = 2 * sigma_x  # = 30
    expected_radius_e2_y = 2 * sigma_y  # = 20
    
    y, x = np.mgrid[0:100, 0:100]
    center_x, center_y = 50, 50
    gaussian_2d = np.exp(-((x - center_x) ** 2) / (2 * sigma_x ** 2) 
                        - ((y - center_y) ** 2) / (2 * sigma_y ** 2))
    
    # Test without smoothing (sigma=0) since input is already smooth
    result = calculate_fwhm_2d(gaussian_2d, smooth_sigma=0)
    
    error_x = abs(result['radius_e2_x'] - expected_radius_e2_x)
    error_y = abs(result['radius_e2_y'] - expected_radius_e2_y)
    
    print(f"1/e^2 Radius test:")
    print(f"  Expected 1/e^2 Radius X: {expected_radius_e2_x:.2f}, Calculated: {result['radius_e2_x']:.2f}, Error: {error_x:.2f}")
    print(f"  Expected 1/e^2 Radius Y: {expected_radius_e2_y:.2f}, Calculated: {result['radius_e2_y']:.2f}, Error: {error_y:.2f}")
    
    assert error_x < 1.0, f"1/e^2 Radius X error too large: {error_x}"
    assert error_y < 1.0, f"1/e^2 Radius Y error too large: {error_y}"
    print("  ✓ PASSED\n")


if __name__ == '__main__':
    print("=" * 50)
    print("FWHM Calculator Tests")
    print("=" * 50 + "\n")
    
    test_gaussian_1d()
    test_gaussian_2d()
    test_radius_e2()
    test_process_rgb_image()
    test_process_rgba_image()
    test_noisy_gaussian()
    test_background_subtraction()
    test_background_subtraction_rgba()
    test_background_dimension_mismatch()
    
    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
