# SpotGauge

Toolkit to analyze focal spots - Calculate FWHM (Full Width at Half Maximum) directly in your browser.

## 🚀 Live Demo

Visit the live application: [SpotGauge FWHM Calculator](https://chowdhurygroup.github.io/SpotGauge/)

## 📖 Features

- **Upload focal spot images** - Supports PNG, JPG, TIFF, and BMP formats
- **Calculate FWHM** - Automatically calculates Full Width at Half Maximum in both X and Y directions
- **Interactive visualization** - View intensity profiles with half-maximum reference lines
- **Adjustable smoothing** - Control Gaussian smoothing to reduce noise
- **Browser-based** - Runs entirely in your browser using Pyodide (Python in WebAssembly)

## 🔬 How It Works

1. **Upload** - Drag and drop or click to upload a focal spot image
2. **Configure** - Adjust the Gaussian smoothing parameter (σ) if needed
3. **Analyze** - Click "Analyze Focal Spot" to calculate FWHM
4. **Results** - View FWHM values and intensity profiles

## 🧮 FWHM Calculation

The FWHM (Full Width at Half Maximum) is calculated by:

1. Converting the image to grayscale if necessary
2. Applying optional Gaussian smoothing to reduce noise
3. Finding the peak intensity position
4. Extracting 1D profiles through the peak in X and Y directions
5. Determining where each profile crosses half the maximum value
6. Interpolating for sub-pixel accuracy

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
