# SpotGauge

Toolkit to analyze focal spots - Calculate FWHM (Full Width at Half Maximum) and laser parameters directly in your browser.

## 🚀 Live Demo

Visit the live application: [SpotGauge FWHM Calculator](https://chowdhurygroup.github.io/SpotGauge/)

## 📖 Features

### FWHM Analysis
- **Upload focal spot images** - Supports PNG, JPG, TIFF, and BMP formats
- **Calculate FWHM** - Automatically calculates Full Width at Half Maximum in both X and Y directions
- **Interactive visualization** - View intensity profiles with half-maximum reference lines
- **Adjustable smoothing** - Control Gaussian smoothing to reduce noise
- **1/e² radius calculation** - Calculate beam radius at 1/e² intensity level

### Laser Calculations
- **Peak fluence** - Calculate energy density at the focal spot (J/cm²)
- **Peak intensity** - Calculate peak power density (W/cm²)
- **Peak power** - Calculate instantaneous power (W)
- **Flexible input modes** - Enter either pulse energy or average power
- **Unit conversions** - Automatic SI prefix formatting for readability

### General
- **Browser-based** - Runs entirely in your browser using Pyodide (Python in WebAssembly)
- **No uploads required** - All processing happens locally in your browser

## 🔬 How It Works

### FWHM Analysis Tab
1. **Upload** - Drag and drop or click to upload a focal spot image
2. **Configure** - Adjust the Gaussian smoothing parameter (σ) and pixel scale if needed
3. **Analyze** - Click "Analyze Focal Spot" to calculate FWHM
4. **Results** - View FWHM values, 1/e² radius, and intensity profiles

### Laser Calculations Tab
1. **Run FWHM Analysis** - First analyze your focal spot in the FWHM Analysis tab
2. **Enter Parameters** - Input repetition rate, pulse duration, and either pulse energy or average power
3. **Calculate** - Click "Calculate Parameters" to compute laser characteristics
4. **Results** - View peak fluence, peak intensity, peak power, and other derived parameters

## 🧮 Calculations

### FWHM Calculation

The FWHM (Full Width at Half Maximum) is calculated by:

1. Converting the image to grayscale if necessary
2. Applying optional Gaussian smoothing to reduce noise
3. Finding the peak intensity position
4. Extracting 1D profiles through the peak in X and Y directions
5. Determining where each profile crosses half the maximum value
6. Interpolating for sub-pixel accuracy

### Laser Parameter Calculations

Based on the measured FWHM, the following parameters are calculated:

- **Spot Area**: A = π × (FWHM_x/2) × (FWHM_y/2) for elliptical spots
- **Pulse Energy**: E_pulse = P_avg / f_rep (if using average power mode)
- **Peak Power**: P_peak = E_pulse / τ_pulse
- **Average Power**: P_avg = E_pulse × f_rep
- **Peak Fluence**: F_peak = E_pulse / A (J/cm²)
- **Peak Intensity**: I_peak = P_peak / A (W/cm²)

Where:
- f_rep = repetition rate (Hz)
- τ_pulse = pulse duration (s)
- E_pulse = pulse energy (J)
- P_avg = average power (W)

## 🛠️ Local Development

To run locally, simply serve the `docs/` directory with any HTTP server:

```bash
# Using Python
cd docs
python -m http.server 8000

# Using Node.js
npx serve docs
```

Then open `http://localhost:8000` in your browser.

## 📁 Project Structure

```
SpotGauge/
├── README.md
└── docs/
    ├── index.html           # Main web application
    └── fwhm_calculator.py   # Python FWHM calculation module
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source.
