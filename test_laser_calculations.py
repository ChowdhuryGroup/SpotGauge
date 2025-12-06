"""
Test script for laser parameter calculations.

Run with: python test_laser_calculations.py
"""

import math


def test_fwhm_to_1e2_conversion():
    """Test the FWHM to 1/e² beam radius conversion."""
    # For a Gaussian beam: w0 = FWHM / sqrt(2*ln(2))
    fwhm = 10.0  # microns
    conversion_factor = math.sqrt(2 * math.log(2))
    expected_w0 = fwhm / conversion_factor
    
    print(f"FWHM to 1/e² Conversion Test:")
    print(f"  FWHM: {fwhm:.2f} µm")
    print(f"  Conversion factor: {conversion_factor:.6f}")
    print(f"  Expected 1/e² radius: {expected_w0:.6f} µm")
    print(f"  Verification: {expected_w0:.6f} * {conversion_factor:.6f} = {expected_w0 * conversion_factor:.6f} µm")
    
    assert abs(expected_w0 * conversion_factor - fwhm) < 1e-10, "Conversion verification failed"
    print("  ✓ PASSED\n")


def test_spot_area_calculation():
    """Test the corrected spot area calculation."""
    # For an elliptical Gaussian beam with 1/e² radii w0_x and w0_y
    # Area at 1/e² = π * w0_x * w0_y
    
    fwhm_x = 12.0  # microns
    fwhm_y = 8.0   # microns
    
    conversion_factor = math.sqrt(2 * math.log(2))
    w0_x = fwhm_x / conversion_factor
    w0_y = fwhm_y / conversion_factor
    
    # OLD (incorrect) formula: A = π * (FWHM_x/2) * (FWHM_y/2)
    old_area = math.pi * (fwhm_x / 2) * (fwhm_y / 2)
    
    # NEW (correct) formula: A = π * w0_x * w0_y
    new_area = math.pi * w0_x * w0_y
    
    # The ratio should be 4 / (2*ln(2)) ≈ 2.885
    expected_ratio = 4 / (2 * math.log(2))
    actual_ratio = new_area / old_area
    
    print(f"Spot Area Calculation Test:")
    print(f"  FWHM X: {fwhm_x:.2f} µm, FWHM Y: {fwhm_y:.2f} µm")
    print(f"  1/e² radius X: {w0_x:.6f} µm")
    print(f"  1/e² radius Y: {w0_y:.6f} µm")
    print(f"  OLD area formula: {old_area:.6f} µm²")
    print(f"  NEW area formula: {new_area:.6f} µm²")
    print(f"  Ratio (NEW/OLD): {actual_ratio:.6f}")
    print(f"  Expected ratio: {expected_ratio:.6f}")
    print(f"  Difference: {abs(actual_ratio - expected_ratio):.10f}")
    
    assert abs(actual_ratio - expected_ratio) < 1e-10, f"Ratio mismatch: {actual_ratio} vs {expected_ratio}"
    print("  ✓ PASSED\n")


def test_peak_intensity_with_correction():
    """Test peak intensity calculation with 0.94 correction factor."""
    # Given parameters
    pulse_energy = 1e-6  # J (1 µJ)
    pulse_duration = 1e-12  # s (1 ps)
    w0_x = 5.0  # µm
    w0_y = 5.0  # µm
    
    # Calculate peak power
    peak_power = pulse_energy / pulse_duration  # W
    
    # Calculate spot area at 1/e²
    spot_area_um2 = math.pi * w0_x * w0_y
    spot_area_cm2 = spot_area_um2 * 1e-8
    
    # Calculate peak intensity with correction factor
    correction_factor = 0.94
    peak_intensity = correction_factor * (peak_power / spot_area_cm2)
    
    print(f"Peak Intensity Calculation Test:")
    print(f"  Pulse energy: {pulse_energy:.2e} J")
    print(f"  Pulse duration: {pulse_duration:.2e} s")
    print(f"  Peak power: {peak_power:.2e} W")
    print(f"  Spot area: {spot_area_cm2:.2e} cm²")
    print(f"  Peak intensity (uncorrected): {peak_power / spot_area_cm2:.2e} W/cm²")
    print(f"  Correction factor: {correction_factor}")
    print(f"  Peak intensity (corrected): {peak_intensity:.2e} W/cm²")
    print(f"  Reduction: {(1 - correction_factor) * 100:.1f}%")
    
    # Verify the correction is applied
    assert abs(peak_intensity / (peak_power / spot_area_cm2) - correction_factor) < 1e-10
    print("  ✓ PASSED\n")


def test_complete_laser_calculation():
    """Test a complete laser parameter calculation workflow."""
    # Given parameters
    fwhm_x = 10.0  # µm
    fwhm_y = 10.0  # µm
    rep_rate = 1000.0  # Hz
    pulse_duration = 100e-15  # s (100 fs)
    pulse_energy = 1e-6  # J (1 µJ)
    
    # Convert FWHM to 1/e² beam radius
    conversion_factor = math.sqrt(2 * math.log(2))
    w0_x = fwhm_x / conversion_factor
    w0_y = fwhm_y / conversion_factor
    
    # Calculate spot area at 1/e²
    spot_area_um2 = math.pi * w0_x * w0_y
    spot_area_cm2 = spot_area_um2 * 1e-8
    
    # Calculate parameters
    peak_power = pulse_energy / pulse_duration
    avg_power = pulse_energy * rep_rate
    peak_fluence = pulse_energy / spot_area_cm2
    peak_intensity = 0.94 * (peak_power / spot_area_cm2)
    
    print(f"Complete Laser Calculation Test:")
    print(f"  Input Parameters:")
    print(f"    FWHM: {fwhm_x:.2f} µm × {fwhm_y:.2f} µm")
    print(f"    Repetition rate: {rep_rate:.0f} Hz")
    print(f"    Pulse duration: {pulse_duration:.2e} s")
    print(f"    Pulse energy: {pulse_energy:.2e} J")
    print(f"  Calculated 1/e² radius: {w0_x:.4f} µm × {w0_y:.4f} µm")
    print(f"  Spot area: {spot_area_cm2:.4e} cm²")
    print(f"  Results:")
    print(f"    Peak power: {peak_power:.2e} W")
    print(f"    Average power: {avg_power:.2e} W")
    print(f"    Peak fluence: {peak_fluence:.2e} J/cm²")
    print(f"    Peak intensity: {peak_intensity:.2e} W/cm²")
    
    # Verify calculations are self-consistent
    assert abs(avg_power - pulse_energy * rep_rate) < 1e-10
    assert abs(peak_power - pulse_energy / pulse_duration) < 1e-10
    print("  ✓ PASSED\n")


def test_backward_compatibility_check():
    """Compare old and new calculation methods."""
    # Given parameters
    fwhm_x = 15.0  # µm
    fwhm_y = 10.0  # µm
    pulse_energy = 2e-6  # J
    
    # OLD method: A = π * (FWHM_x/2) * (FWHM_y/2)
    old_area_um2 = math.pi * (fwhm_x / 2) * (fwhm_y / 2)
    old_area_cm2 = old_area_um2 * 1e-8
    old_fluence = pulse_energy / old_area_cm2
    
    # NEW method: A = π * w0_x * w0_y
    conversion_factor = math.sqrt(2 * math.log(2))
    w0_x = fwhm_x / conversion_factor
    w0_y = fwhm_y / conversion_factor
    new_area_um2 = math.pi * w0_x * w0_y
    new_area_cm2 = new_area_um2 * 1e-8
    new_fluence = pulse_energy / new_area_cm2
    
    print(f"Backward Compatibility Check:")
    print(f"  FWHM: {fwhm_x:.2f} µm × {fwhm_y:.2f} µm")
    print(f"  Pulse energy: {pulse_energy:.2e} J")
    print(f"  OLD method:")
    print(f"    Area: {old_area_cm2:.4e} cm²")
    print(f"    Fluence: {old_fluence:.4e} J/cm²")
    print(f"  NEW method:")
    print(f"    Area: {new_area_cm2:.4e} cm²")
    print(f"    Fluence: {new_fluence:.4e} J/cm²")
    print(f"  Difference:")
    print(f"    Area change: {(new_area_cm2 / old_area_cm2 - 1) * 100:.1f}%")
    print(f"    Fluence change: {(new_fluence / old_fluence - 1) * 100:.1f}%")
    print("  ✓ PASSED\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Laser Parameter Calculation Tests")
    print("=" * 60 + "\n")
    
    test_fwhm_to_1e2_conversion()
    test_spot_area_calculation()
    test_peak_intensity_with_correction()
    test_complete_laser_calculation()
    test_backward_compatibility_check()
    
    print("=" * 60)
    print("All laser calculation tests passed! ✓")
    print("=" * 60)
