# SpotGauge Debugging Guide

This guide explains how to use the debugging features added to SpotGauge to troubleshoot FWHM calculation issues.

## Problem Addressed

Previously, users encountered the error:
```
Error calculating FWHM: undefined is not an object (evaluating 'fwhmX.toFixed')
```

This error occurred when FWHM values were invalid (NaN, undefined, or Infinity) but there was no visibility into why the calculation failed.

## Debugging Features

### 1. Browser Console Logging

When using SpotGauge on GitHub Pages, all debugging information is now visible in the browser console.

**How to access:**
1. Open your browser's Developer Tools (F12 or Ctrl+Shift+I on Windows/Linux, Cmd+Option+I on Mac)
2. Navigate to the **Console** tab
3. Upload and analyze an image
4. View detailed debug output

### 2. Debug Output Categories

The console output is organized with clear prefixes:

- **`[PYTHON]`** - Output from Python code running in Pyodide
- **`[DEBUG]`** - Step-by-step calculation progress
- **`[ERROR]`** - Error conditions detected
- **`[WARNING]`** - Non-fatal issues with fallback applied

### 3. What Information is Logged

#### Python-side (fwhm_calculator.py):

**Profile Analysis:**
```
[PYTHON] [DEBUG] Profile X stats: min=0.773, max=200.0, mean=75.133
[PYTHON] [DEBUG] Profile Y stats: min=0.773, max=200.0, mean=75.133
```

**Width Calculation:**
```
[PYTHON] [DEBUG] calculate_width_at_threshold: profile length=100, threshold_fraction=0.5
[PYTHON] [DEBUG] calculate_width_at_threshold: max_val=200.0, max_idx=50
[PYTHON] [DEBUG] calculate_width_at_threshold: threshold=100.0
[PYTHON] [DEBUG] calculate_width_at_threshold: left_pos=25.869, right_pos=54.130, width=28.261
```

**FWHM Results:**
```
[PYTHON] [DEBUG] FWHM X calculated: 28.261
[PYTHON] [DEBUG] FWHM Y calculated: 28.261
```

**Validation:**
```
[PYTHON] [DEBUG] Validating calculated values...
[PYTHON] [DEBUG] fwhm_x type: <class 'numpy.float64'>, value: 28.261
[PYTHON] [DEBUG] Converted values - fwhm_x: 28.261 (type: <class 'float'>)
```

#### JavaScript-side (index.html):

**Value Extraction:**
```
[DEBUG] Extracted values from result:
  fwhmX: 28.26 (type: number)
  fwhmY: 28.26 (type: number)
  radiusE2X: 24.0 (type: number)
  radiusE2Y: 24.0 (type: number)
```

**Validation:**
```
[DEBUG] Validating numeric values...
[DEBUG] fwhmX validated: 28.26
[DEBUG] fwhmY validated: 28.26
[DEBUG] All numeric values validated successfully
```

### 4. Error Detection and Handling

The system now detects and handles multiple error conditions:

#### Invalid Profile (All Zeros):
```
[PYTHON] [DEBUG] Profile X stats: min=0.0, max=0.0, mean=0.0
[PYTHON] [ERROR] Invalid maximum value in profile: max_val=0.0
[PYTHON] [WARNING] Setting fwhm_x to 0.0 as fallback
```

#### NaN or Infinity Values:
```
[PYTHON] [ERROR] Invalid value detected: fwhm_x=nan
[PYTHON] [WARNING] Setting fwhm_x to 0.0 as fallback
```

#### JavaScript Validation:
```
[ERROR] fwhmX is NaN
Error calculating FWHM: fwhmX is NaN. The image may not have sufficient signal for FWHM calculation.
```

## Common Issues and Solutions

### Issue: "fwhmX is undefined"
**Cause:** Data not properly transferred from Python to JavaScript  
**Check:** Look for `[PYTHON] [ERROR]` messages showing validation failures  
**Solution:** Ensure image has sufficient signal/contrast

### Issue: "fwhmX is NaN"
**Cause:** Image has no peak or is completely flat  
**Check:** Profile statistics showing min ≈ max  
**Solution:** Use a different image or adjust preprocessing

### Issue: Very large FWHM values
**Cause:** Image is nearly flat or threshold not being crossed  
**Check:** Profile stats and threshold values in debug output  
**Solution:** Verify image contains a focal spot

### Issue: FWHM is 0.0
**Cause:** Image is all zeros or has invalid maximum  
**Check:** `[ERROR] Invalid maximum value` messages  
**Solution:** Check image file is valid and not corrupted

## Testing Edge Cases

The repository includes edge case tests in `test_fwhm_edge_cases.py`:

```bash
python test_fwhm_edge_cases.py
```

Tests cover:
- All-zero images
- Flat (constant) images
- Single pixel peaks
- Images with negative values
- Very small images

## Tips for Debugging

1. **Always check the console first** - Most issues are explained in the debug output

2. **Look for the data flow** - Follow the sequence:
   - Image loading → Profile extraction → FWHM calculation → Validation → Display

3. **Check type information** - The debug output shows both values and types, helping identify type conversion issues

4. **Use the browser's filter** - In the Console tab, filter by `[DEBUG]`, `[ERROR]`, or `[PYTHON]` to focus on specific output

5. **Compare working vs non-working images** - Run a known good image and compare the debug output

## Additional Resources

- Main tests: `test_fwhm.py`
- Edge case tests: `test_fwhm_edge_cases.py`
- Python implementation: `docs/fwhm_calculator.py`
- Web interface: `docs/index.html`

## Summary of Improvements

1. ✅ Added comprehensive debug logging throughout calculation pipeline
2. ✅ Configured Pyodide to redirect Python print() to browser console
3. ✅ Added validation for NaN/Inf/undefined values at every stage
4. ✅ Provide specific error messages indicating likely causes
5. ✅ Graceful fallback to 0.0 for invalid calculations
6. ✅ Type information in debug output for troubleshooting type conversion issues
7. ✅ Explicit Python native type conversion for Pyodide compatibility
