"""
Test edge cases for FWHM calculator that could cause undefined values.

Run with: python test_fwhm_edge_cases.py
"""

import numpy as np
import sys
import os

# Add the docs directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'docs'))

from fwhm_calculator import calculate_fwhm_1d, calculate_fwhm_2d, process_image_data


def test_all_zeros():
    """Test FWHM calculation on an all-zeros image."""
    print("All zeros image test:")
    image = np.zeros((100, 100))
    
    result = process_image_data(image)
    
    # Should return 0.0 for FWHM instead of NaN or undefined
    print(f"  FWHM X: {result['fwhm_x']}")
    print(f"  FWHM Y: {result['fwhm_y']}")
    
    assert not np.isnan(result['fwhm_x']), "FWHM X should not be NaN"
    assert not np.isnan(result['fwhm_y']), "FWHM Y should not be NaN"
    assert not np.isinf(result['fwhm_x']), "FWHM X should not be Inf"
    assert not np.isinf(result['fwhm_y']), "FWHM Y should not be Inf"
    assert result['fwhm_x'] == 0.0, "FWHM X should be 0.0 for all zeros"
    assert result['fwhm_y'] == 0.0, "FWHM Y should be 0.0 for all zeros"
    
    print("  ✓ PASSED\n")


def test_flat_image():
    """Test FWHM calculation on a flat (constant) image."""
    print("Flat image test:")
    image = np.ones((100, 100)) * 100
    
    result = process_image_data(image)
    
    # Should return 0.0 for FWHM instead of NaN or undefined
    print(f"  FWHM X: {result['fwhm_x']}")
    print(f"  FWHM Y: {result['fwhm_y']}")
    
    assert not np.isnan(result['fwhm_x']), "FWHM X should not be NaN"
    assert not np.isnan(result['fwhm_y']), "FWHM Y should not be NaN"
    assert not np.isinf(result['fwhm_x']), "FWHM X should not be Inf"
    assert not np.isinf(result['fwhm_y']), "FWHM Y should not be Inf"
    assert result['fwhm_x'] == 0.0, "FWHM X should be 0.0 for flat image"
    assert result['fwhm_y'] == 0.0, "FWHM Y should be 0.0 for flat image"
    
    print("  ✓ PASSED\n")


def test_single_pixel_peak():
    """Test FWHM calculation on an image with a single bright pixel."""
    print("Single pixel peak test:")
    image = np.zeros((100, 100))
    image[50, 50] = 1000  # Single bright pixel
    
    result = process_image_data(image)
    
    # Should have very small FWHM but not NaN or undefined
    print(f"  FWHM X: {result['fwhm_x']}")
    print(f"  FWHM Y: {result['fwhm_y']}")
    
    assert not np.isnan(result['fwhm_x']), "FWHM X should not be NaN"
    assert not np.isnan(result['fwhm_y']), "FWHM Y should not be NaN"
    assert not np.isinf(result['fwhm_x']), "FWHM X should not be Inf"
    assert not np.isinf(result['fwhm_y']), "FWHM Y should not be Inf"
    
    # FWHM should be very small (smoothing will spread it slightly)
    assert result['fwhm_x'] >= 0, "FWHM X should be non-negative"
    assert result['fwhm_y'] >= 0, "FWHM Y should be non-negative"
    
    print("  ✓ PASSED\n")


def test_negative_values():
    """Test FWHM calculation on an image with negative values (after background subtraction)."""
    print("Negative values test:")
    image = np.random.randn(100, 100) * 10  # Random noise centered at 0
    # Add a Gaussian peak
    y, x = np.mgrid[0:100, 0:100]
    gaussian = 200 * np.exp(-((x - 50)**2 + (y - 50)**2) / (2 * 10**2))
    image = image + gaussian
    
    result = process_image_data(image)
    
    # Should handle negative values gracefully
    print(f"  FWHM X: {result['fwhm_x']}")
    print(f"  FWHM Y: {result['fwhm_y']}")
    
    assert not np.isnan(result['fwhm_x']), "FWHM X should not be NaN"
    assert not np.isnan(result['fwhm_y']), "FWHM Y should not be NaN"
    assert not np.isinf(result['fwhm_x']), "FWHM X should not be Inf"
    assert not np.isinf(result['fwhm_y']), "FWHM Y should not be Inf"
    assert result['fwhm_x'] > 0, "FWHM X should be positive"
    assert result['fwhm_y'] > 0, "FWHM Y should be positive"
    
    print("  ✓ PASSED\n")


def test_very_small_image():
    """Test FWHM calculation on a very small image."""
    print("Very small image test:")
    image = np.array([[0, 0, 0],
                      [0, 100, 0],
                      [0, 0, 0]], dtype=float)
    
    result = process_image_data(image)
    
    # Should handle small images
    print(f"  FWHM X: {result['fwhm_x']}")
    print(f"  FWHM Y: {result['fwhm_y']}")
    
    assert not np.isnan(result['fwhm_x']), "FWHM X should not be NaN"
    assert not np.isnan(result['fwhm_y']), "FWHM Y should not be NaN"
    assert not np.isinf(result['fwhm_x']), "FWHM X should not be Inf"
    assert not np.isinf(result['fwhm_y']), "FWHM Y should not be Inf"
    
    print("  ✓ PASSED\n")


if __name__ == '__main__':
    print("="*50)
    print("Running edge case tests for FWHM calculator")
    print("="*50)
    print()
    
    test_all_zeros()
    test_flat_image()
    test_single_pixel_peak()
    test_negative_values()
    test_very_small_image()
    
    print("="*50)
    print("All edge case tests passed! ✓")
    print("="*50)
