# SPOTTER User Manual

**SPOTTER 1.0 (Multi-Cell Engine)**
*Subcellular Puncta Observer and Tally Tool for Enumeration in Research*
Advanced Multiprocessing Version by KIST

---

## 1. Introduction

SPOTTER (Subcellular Puncta Observer and Tally Tool for Enumeration in Research) is a Windows desktop application that automatically detects and counts intracellular punctate structures (puncta) in fluorescence microscopy images.

### Key features
- Automatic puncta detection in four channels (Red, Green, Blue, Yellow)
- Precise separation of touching puncta using marker-controlled watershed
- Fast parallel processing of large image batches via a multiprocessing engine
- Full support for Unicode (including non-ASCII) file paths and interface
- **Single-Cell Mode**: counts puncta across the entire image
- **Multi-Cell Mode**: automatically segments individual cells and counts puncta per cell
- **ROI-based parameter tuning**: estimates threshold/size parameters inside SPOTTER, without ImageJ

### System requirements
| Item | Requirement |
|---|---|
| Operating system | Windows 10 or later (64-bit) |
| Disk | At least 200 MB free space |
| Image formats | PNG, JPG, JPEG, TIFF, TIF |
| Installation | Not required (single-file `SPOTTER_V1.exe`) |

---

## 2. Quick Start

Following these steps lets you complete a first analysis within ~3 minutes.

1. Double-click `SPOTTER_V1.exe` to launch.
2. Click **Open** next to **Input Directory** and select the folder containing your images.
3. Check the channel(s) to analyze (Red, Green, Blue, Yellow).
4. Select an analysis mode: **Single Cell Mode** or **Multi Cell Mode**.
5. (Optional) Use ROI-based parameter estimation in the **Parameter Tuning** section.
6. Click **Start**.
7. When the progress bar reaches 100%, a completion pop-up appears.
8. Check the Excel result file and images in the output folder.

> **TIP** The Output Directory defaults to an `output/` subfolder inside the Input folder. You can change it manually if needed.

---

## 3. GUI Reference

### 3-1. Common settings
| Item | Default | Description |
|---|---|---|
| Input Directory | – | Folder containing the images to analyze |
| Color Channels | Red | Fluorescence channel(s) to analyze (multiple allowed) |
| Color Bottom Threshold | 50 | Minimum brightness (0–255); only pixels at or above this become puncta candidates |
| Color Top Threshold | 220 | Maximum brightness (0–255); upper bound to avoid cross-channel bleed |
| Enable Watershed | ON | Whether to use the watershed algorithm to separate touching puncta |
| Watershed Threshold | 0.18 | Watershed sensitivity (0–0.99); higher values split more |
| Min Segment Size | 3 | Minimum puncta area (pixels); smaller objects are ignored as noise |
| Max Segment Size | 200 | Maximum puncta area (pixels); larger objects are excluded |
| Save All Verboses | ON | Save intermediate processing images to a verbose folder |
| Output Directory | Auto | Folder for results; defaults to `output/` under the Input folder |

### 3-2. Analysis modes
| Mode | Description |
|---|---|
| Single Cell Mode | Treats the whole image as one cell; counts total puncta per channel per image. |
| Multi Cell Mode | Automatically segments individual cells, labels them A–J, and counts puncta separately for each cell. |

### 3-3. Multi-Cell Mode settings
Shown when Multi Cell Mode is selected.

| Item | Default | Description |
|---|---|---|
| CLAHE Clip Limit | 3.0 | Contrast-enhancement strength (1.0–10.0); higher makes cell boundaries more prominent |
| Morphology Kernel Size | 7 | Morphological kernel size (3–21, odd); higher fills small gaps and removes noise |
| Min Cell Area | 5,000 | Minimum cell area (pixels); smaller regions are not recognized as cells |
| Max Cell Area | 200,000 | Maximum cell area (pixels); larger regions are not recognized as cells |
| Cell Watershed Min Distance | 50 | Minimum distance between cell centroids (pixels); smaller values split nearby cells |

---

## 4. ROI-based Parameter Tuning

This feature helps you quickly find optimal parameters using a single representative image before batch analysis. Threshold and size parameters are estimated and applied inside SPOTTER, without a separate ImageJ workflow.

> **NOTE** ROI tuning is session-only. It is not written to the settings file and resets when the program closes. Applied values, however, are saved when you click Start.

### 4-1. Workflow
1. **Load Sample Image** – select one representative image.
2. **Pick ROI** – draw a rectangle over the region of interest in the OpenCV window.
3. **Suggest Threshold** – analyzes the brightness distribution within the ROI and recommends thresholds.
4. **Estimate Cell Size** – analyzes the cell-body size distribution within the ROI.
5. **Estimate Puncta Size** – analyzes the puncta size distribution within the ROI.
6. **Apply Suggested Range** – applies the recommended values to the GUI settings.
7. **Preview Detected Objects** – previews detection results with the current settings.
8. **Start** – runs batch analysis of all images with the applied parameters.

### 4-2. Buttons
| Button | Action |
|---|---|
| Load Sample Image | Opens a file dialog to select one image (PNG/JPG/TIFF; Unicode paths supported). |
| Pick ROI | Opens an OpenCV window; drag a rectangle, confirm with Enter/Space, cancel with ESC. Analysis without an ROI (whole image) is also possible. |
| Estimate Cell Size | Detects cell bodies in the ROI and reports the size distribution (count, min/median/max, P10/P90, recommended min/max). Apply sets Min/Max Cell Area. |
| Estimate Puncta Size | Detects puncta in the ROI and reports the same statistics. Apply sets Min/Max Segment Size. |
| Suggest Threshold | Analyzes channel brightness in the ROI and recommends thresholds (Otsu + percentile). Apply sets Color Bottom/Top Threshold. |
| Apply Suggested Range | Applies all estimates (cell size, puncta size, threshold) to the GUI at once. |
| Preview Detected Objects | Shows detected puncta contours overlaid in an OpenCV window. Press any key to close. |
| Inspect Object Size | Click an object in the OpenCV window to display its pixel area. Press ESC to exit. |

### 4-3. Estimation dialog
After Estimate Cell Size or Estimate Puncta Size, the following are shown:
- Number of detected objects (count)
- Min / median / max area
- 10th / 90th percentile (P10 / P90)
- Recommended minimum = P10 × 0.8
- Recommended maximum = P90 × 1.2

Clicking **[Apply Suggested]** populates the corresponding setting widgets automatically.

> **TIP** If no ROI is selected, estimation is performed on the whole image. Selecting an ROI where cells are clearly visible yields more accurate results.
>
> **TIP** Large images (≥ 2048×2048) are automatically downscaled to 1200 px for display, and ROI coordinates are converted back to the original resolution.

---

## 5. Analysis Pipeline

### 5-1. Single Cell Mode
1. Load and normalize the image (0–255 range).
2. Generate a binary mask per selected channel using `cv2.inRange()`.
3. (If Watershed is ON) Separate touching puncta via Distance Transform → Peak Detection → Watershed.
4. Extract contours and filter by area (Min–Max Segment Size).
5. Count valid puncta.

### 5-2. Multi Cell Mode

**Phase 1 — Cell-body segmentation**
1. Grayscale conversion → CLAHE → Gaussian blur.
2. Otsu threshold binarization (automatically falls back to adaptive threshold if it fails).
3. Morphological close/open to fill holes and remove noise.
4. Distance Transform + Watershed to separate touching cells.
5. Apply the area filter and label up to 10 cells A–J.

**Phase 2 — Puncta detection**
Runs the same per-channel puncta detection pipeline as Single Cell Mode.

**Phase 3 — Puncta assignment**
Each punctum's centroid is looked up in the cell mask and assigned to that cell. Puncta not belonging to any cell are classified as Unassigned.

**Confidence**
- solidity (contour area / convex-hull area) ≥ 0.7 : OK
- solidity < 0.7 : LOW (review recommended)

---

## 6. Output

### 6-1. Excel result file

**Single Cell Mode**
| Column | Description |
|---|---|
| image_name | Image file name |
| color | Channel (Red / Green / Blue / Yellow) |
| number_of_cells | Total puncta detected in that channel |

**Multi Cell Mode** (one row per cell)
| Column | Description |
|---|---|
| image_name | Image file name |
| cell_label | Cell label (A–J) or Unassigned |
| cell_area | Cell area (pixels) |
| color | Channel |
| puncta_count | Puncta assigned to the cell |
| puncta_density | Puncta density (per million pixels) |
| confidence | Segmentation confidence (solidity, 0–1) |
| confidence_flag | OK (≥ 0.7) or LOW (< 0.7) |

A second worksheet, **Run_Settings**, records all parameter values used for the run (for reproducibility).

### 6-2. Output folder structure
| File / Folder | Description |
|---|---|
| `SPOTTER_Result_[timestamp].xlsx` | Excel result file |
| `[color]_result_[image].png` | Puncta image with numbered labels |
| `[color]_multicell_result_[image].png` | Cell boundaries + puncta assignment overlay (Multi Cell) |
| `verbose_[ts]/0_preprocessed/` | CLAHE-preprocessed result (Multi Cell) |
| `verbose_[ts]/1_separated/` | Channel-separated image |
| `verbose_[ts]/2_watershed/` | Watershed visualization |
| `verbose_[ts]/3_contour/` | Contour overlay |
| `verbose_[ts]/4_area/` | Area-annotated image |
| `verbose_[ts]/5_cell_mask/` | Binary cell-body mask (Multi Cell) |
| `verbose_[ts]/6_cell_boundaries/` | Cell outlines + A/B/C labels + confidence (Multi Cell) |
| `verbose_[ts]/7_puncta_assignment/` | Per-cell color-coded puncta overlay (Multi Cell) |

---

## 7. Parameter Tuning Guide

### 7-1. Puncta detection
| Symptom | Cause | Action |
|---|---|---|
| Too few puncta detected | Bottom Threshold too high | Lower it (or use Suggest Threshold) |
| Noise detected as puncta | Threshold too low or Min Size too small | Raise Bottom Threshold, raise Min Segment Size |
| Touching puncta not separated | Watershed off or Threshold too low | Enable Watershed, raise Watershed Threshold |
| One punctum split into several | Watershed Threshold too high | Lower Watershed Threshold (e.g., 0.1) |

### 7-2. Cell separation (Multi Cell Mode)
| Symptom | Cause | Action |
|---|---|---|
| No cells detected | Weak cell-body signal or Min Cell Area too large | Raise CLAHE Clip Limit, reduce Min Cell Area (or use Estimate Cell Size) |
| Two cells merged into one | Cell Watershed Min Distance too large | Reduce it (e.g., 50 → 30) |
| One cell split into several | Cell Watershed Min Distance too small | Increase it (e.g., 50 → 80) |
| Many Unassigned puncta | Cell mask smaller than actual | Raise CLAHE Clip Limit, lower Bottom Threshold |
| confidence_flag is LOW | Irregular cell shape or incomplete separation | Inspect the `6_cell_boundaries` image and adjust parameters |

> **WARNING** ROI tuning greatly reduces trial-and-error. Use the Estimate functions to get approximate values first, then fine-tune with Preview.

---

## 8. Saving and Resetting Settings

Clicking **Start** automatically saves the current settings to `spotter_settings.json`. The last-used settings are restored automatically on the next launch.

- Settings file location: `spotter_settings.json`, in the same folder as `SPOTTER_V1.exe`
- **Reset** button: restores all settings to factory defaults
- ROI tuning results (estimates, ROI coordinates) are session-only and not saved
- Applied parameter values are saved automatically on Start

---

## 9. Known Limitations

- Cell separation in Multi Cell Mode depends on the cell-body fluorescence signal. Separation may degrade for images with a very weak cell-body signal.
- Up to 10 cells (A–J) are recognized per image; if more than 10 are present, only the 10 largest by area are kept.
- ROI selection and Preview open in separate OpenCV windows. The SPOTTER main window pauses meanwhile and resumes when the OpenCV window is closed.
- Closing the program window during processing aborts the ongoing analysis.

---

## 10. Contact

For problems or feature requests, please contact the development team.
