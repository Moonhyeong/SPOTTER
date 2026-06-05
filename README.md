# SPOTTER

**S**ubcellular **P**uncta **O**bserver & **T**ally **T**ool for **E**numeration in **R**esearch — a reproducible, high-throughput tool for automated fluorescence puncta quantification.

SPOTTER is a Windows GUI application for automated detection and counting of fluorescence puncta (e.g., LC3, DFCP1 autophagy markers) in microscopy images. It standardizes color thresholding, marker-controlled watershed declumping, and size-based particle filtering, with CPU-parallel batch processing.

## Features
- **Single-Cell** and **Multi-Cell** analysis modes
- **ROI-based parameter tuning** (automatic threshold and size suggestion from a representative region)
- **Marker-controlled watershed** (peak detection + watershed) for declumping clustered puncta
- **Step-wise verbose outputs** for transparent quality control of each pipeline stage
- **Batch processing** with an Excel (`.xlsx`) report (Results + Run_Settings sheets)

## Requirements
- Python 3.14 (Windows)
- Dependencies: `opencv-python`, `numpy`, `scikit-image`, `scipy`, `dearpygui`, `pandas`, `openpyxl`, `loguru`

## Run from source
```bash
pip install -r requirements.txt
python spotter_main.py
```

## Build a standalone Windows executable
```bash
pyinstaller SPOTTER_V1.spec --noconfirm
# output: dist/SPOTTER_V1.exe
```

## Usage
The user manual and the pre-built Windows executable (`SPOTTER_V1.exe`) are available under [Releases](../../releases).

## Citation
If you use SPOTTER in your research, please cite:

> Lim Y, Choi H-B, Lee TS, Seo M-H. *SPOTTER: a reproducible, high-throughput tool for automated fluorescence puncta quantification.* (manuscript, 2026).

## License
Released under the MIT License — see [LICENSE](LICENSE).
