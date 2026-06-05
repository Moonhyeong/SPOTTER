import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import pathlib
import pandas as pd
import datetime
import os
import sys
import multiprocessing
import json

from loguru import logger

# Log to file for debugging (always enabled, rotates at 5MB)
logger.add("spotter_debug.log", rotation="5 MB", encoding="utf-8", level="DEBUG")

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def show_info(title, message):
    if not dpg.does_alias_exist(title):
        with dpg.window(label=title, tag=title, modal=True, no_close=False):
            dpg.add_text(message)
            dpg.add_button(label="OK", width=75, callback=lambda: dpg.delete_item(title))
    dpg.split_frame()
    width = dpg.get_viewport_client_width()
    height = dpg.get_viewport_client_height()
    dpg.set_item_pos(title, [width // 2 - 100, height // 2 - 50])

# ─── Multi-Cell Segmentation Functions ───────────────────────────────────────

CELL_LABELS = list("ABCDEFGHIJ")  # up to 10 cells

# Distinct colors for cell visualization (BGR)
CELL_COLORS = [
    (0, 0, 255),     # A: Red
    (0, 255, 0),     # B: Green
    (255, 0, 0),     # C: Blue
    (0, 255, 255),   # D: Yellow
    (255, 0, 255),   # E: Magenta
    (255, 255, 0),   # F: Cyan
    (0, 128, 255),   # G: Orange
    (128, 0, 255),   # H: Purple
    (0, 255, 128),   # I: Spring Green
    (255, 128, 0),   # J: Sky Blue
]


def preprocess_for_cell_segmentation(image, clahe_clip_limit):
    """
    Auto-preprocess to make cell body boundaries visible.
    Converts to grayscale, applies CLAHE, then Gaussian blur to suppress puncta peaks.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return blurred


def segment_cell_bodies(preprocessed, min_cell_area, max_cell_area,
                        morph_kernel_size, cell_ws_min_dist):
    """
    Segment individual cell bodies from preprocessed grayscale image.
    Uses Otsu threshold + morphology + watershed.
    Returns cell_label_map and cell_info list.
    """
    import skimage.feature
    import skimage.segmentation

    # 1. Otsu threshold
    thresh_val, binary = cv2.threshold(preprocessed, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Check if Otsu result is reasonable (5%-90% of image)
    fill_ratio = np.count_nonzero(binary) / binary.size
    if fill_ratio < 0.05 or fill_ratio > 0.90:
        # Fallback to adaptive threshold
        binary = cv2.adaptiveThreshold(preprocessed, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 51, -5)

    # 2. Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (morph_kernel_size, morph_kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. Remove small objects
    contours_all, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(cleaned)
    for cnt in contours_all:
        if cv2.contourArea(cnt) >= min_cell_area:
            cv2.drawContours(clean_mask, [cnt], 0, 255, -1)

    # 4. Watershed to split touching cells
    dist_transform = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)

    if dist_transform.max() > 0:
        local_max = skimage.feature.peak_local_max(
            dist_transform,
            min_distance=cell_ws_min_dist,
            threshold_abs=0.2 * dist_transform.max(),
            labels=clean_mask
        )

        markers = np.zeros_like(dist_transform, dtype=np.int32)
        for i, pt in enumerate(local_max):
            markers[pt[0], pt[1]] = i + 1

        if markers.max() > 0:
            markers = cv2.dilate(markers.astype(np.uint8),
                                 np.ones((3, 3), np.uint8),
                                 iterations=1).astype(np.int32)
            labels = skimage.segmentation.watershed(-dist_transform, markers,
                                                     mask=clean_mask > 0)
        else:
            # No peaks found, use connected components
            _, labels = cv2.connectedComponents(clean_mask)
    else:
        _, labels = cv2.connectedComponents(clean_mask)

    # 5. Build cell info from labels
    cell_candidates = []
    for label_idx in np.unique(labels):
        if label_idx == 0:
            continue
        cell_mask = (labels == label_idx).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_cell_area or area > max_cell_area:
            continue

        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

        # Confidence: solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        cell_candidates.append({
            'label_idx': label_idx,
            'area': area,
            'centroid': (cx, cy),
            'contour': cnt,
            'confidence': solidity
        })

    # Sort by area descending, keep top 10
    cell_candidates.sort(key=lambda c: c['area'], reverse=True)
    cell_candidates = cell_candidates[:10]

    # Sort by centroid x for deterministic labeling (left to right)
    cell_candidates.sort(key=lambda c: c['centroid'][0])

    # Assign letters and rebuild label map
    h, w = labels.shape
    cell_label_map = np.zeros((h, w), dtype=np.int32)
    cell_info = []
    for i, cand in enumerate(cell_candidates):
        letter = CELL_LABELS[i]
        label_num = i + 1
        # Fill label map from contour
        mask_single = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_single, [cand['contour']], 0, 255, -1)
        cell_label_map[mask_single > 0] = label_num

        cell_info.append({
            'label': label_num,
            'letter': letter,
            'area': cand['area'],
            'centroid': cand['centroid'],
            'contour': cand['contour'],
            'confidence': cand['confidence']
        })

    return cell_label_map, cell_info, clean_mask


def assign_puncta_to_cells(valid_contours, cell_label_map, cell_info):
    """
    Assign each punctum to a cell based on its centroid position in cell_label_map.
    Returns per_cell_counts dict and unassigned list.
    """
    # Build letter lookup
    label_to_letter = {ci['label']: ci['letter'] for ci in cell_info}

    per_cell_counts = {ci['letter']: 0 for ci in cell_info}
    assignment_dict = {ci['letter']: [] for ci in cell_info}
    unassigned_indices = []

    for idx, cnt in enumerate(valid_contours):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Bounds check
        h, w = cell_label_map.shape
        if 0 <= cy < h and 0 <= cx < w:
            cell_val = cell_label_map[cy, cx]
        else:
            cell_val = 0

        if cell_val > 0 and cell_val in label_to_letter:
            letter = label_to_letter[cell_val]
            per_cell_counts[letter] += 1
            assignment_dict[letter].append(idx)
        else:
            unassigned_indices.append(idx)

    return per_cell_counts, assignment_dict, unassigned_indices


# ─── ROI-Based Parameter Tuning Functions ────────────────────────────────────

# Session-only state (not persisted to settings file)
_tuning_image_path = None    # pathlib.Path
_tuning_image = None         # np.ndarray (BGR)
_tuning_roi = None           # (x, y, w, h) or None
_tuning_cell_result = None   # dict from estimate_cell_sizes
_tuning_puncta_result = None # dict from estimate_puncta_sizes
_tuning_threshold_result = None  # dict from suggest_threshold


def load_sample_image(image_path):
    """Load a single image for ROI-based parameter tuning.
    Reuses the same Korean-safe loading pattern as process_image()."""
    global _tuning_image, _tuning_image_path, _tuning_roi
    global _tuning_cell_result, _tuning_puncta_result, _tuning_threshold_result
    try:
        with open(image_path, 'rb') as f:
            bytes_array = bytearray(f.read())
        numpy_array = np.asarray(bytes_array, dtype=np.uint8)
        image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        if image is None:
            return None
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        _tuning_image = image
        _tuning_image_path = pathlib.Path(image_path)
        _tuning_roi = None  # reset ROI on new image
        _tuning_cell_result = None
        _tuning_puncta_result = None
        _tuning_threshold_result = None
        logger.info(f"Sample image loaded: {image_path} ({image.shape})")
        return image
    except Exception as e:
        logger.error(f"Failed to load sample image: {e}")
        return None


def select_roi_interactive(image):
    """Open an OpenCV window for the user to draw a rectangular ROI.
    Returns (x, y, w, h) in original image coordinates, or None if cancelled."""
    h, w = image.shape[:2]
    max_display = 1200
    if max(h, w) > max_display:
        scale = max_display / max(h, w)
        display = cv2.resize(image, None, fx=scale, fy=scale)
    else:
        scale = 1.0
        display = image.copy()

    window_name = ("SPOTTER - Pick ROI (drag rectangle, ENTER or SPACE to confirm, "
                    "top-right X button to cancel)")
    roi = cv2.selectROI(window_name, display, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    rx, ry, rw, rh = roi
    if rw == 0 or rh == 0:
        return None

    # Scale back to original coordinates
    if scale != 1.0:
        rx = int(rx / scale)
        ry = int(ry / scale)
        rw = int(rw / scale)
        rh = int(rh / scale)

    # Clamp to image bounds
    rx = max(0, min(rx, w - 1))
    ry = max(0, min(ry, h - 1))
    rw = min(rw, w - rx)
    rh = min(rh, h - ry)

    logger.info(f"ROI selected: ({rx}, {ry}, {rw}, {rh})")
    return (rx, ry, rw, rh)


def crop_to_roi(image, roi):
    """Crop image to ROI region. If roi is None, return full image."""
    if roi is None:
        return image.copy()
    x, y, w, h = roi
    return image[y:y+h, x:x+w].copy()


def estimate_cell_sizes(image, settings_dict, roi=None):
    """Estimate cell body sizes by reusing existing segmentation pipeline.
    Returns result_dict with count, areas, statistics, and recommended min/max."""
    cropped = crop_to_roi(image, roi)

    preprocessed = preprocess_for_cell_segmentation(cropped, settings_dict['clahe_clip'])
    # Use very permissive area bounds to discover all objects
    _, cell_info, _ = segment_cell_bodies(
        preprocessed,
        min_cell_area=100,       # very small minimum to catch everything
        max_cell_area=99999999,  # very large maximum
        morph_kernel_size=settings_dict['morph_kernel'],
        cell_ws_min_dist=settings_dict['cell_ws_min_dist']
    )

    if not cell_info:
        return {"count": 0, "areas": [], "min": 0, "median": 0, "max": 0,
                "p10": 0, "p90": 0, "recommended_min": 0, "recommended_max": 0}

    areas = sorted([ci['area'] for ci in cell_info])
    return {
        "count": len(areas),
        "areas": areas,
        "min": float(np.min(areas)),
        "median": float(np.median(areas)),
        "max": float(np.max(areas)),
        "p10": float(np.percentile(areas, 10)),
        "p90": float(np.percentile(areas, 90)),
        "recommended_min": int(np.percentile(areas, 10) * 0.8),
        "recommended_max": int(np.percentile(areas, 90) * 1.2),
    }


def _get_puncta_contours(image, settings_dict):
    """Extract puncta contours using the same logic as _process_image_core().
    Returns list of (contour, area) tuples across all selected channels."""
    import skimage.feature
    import skimage.segmentation

    lower_thres = settings_dict['lower_thres']
    upper_thres = settings_dict['upper_thres']
    enable_watershed = settings_dict['enable_watershed']
    watershed_thres = settings_dict['watershed_thres']

    color_ranges_dict = {
        'Red Channel': (np.array([0, 0, lower_thres]), np.array([120, 120, upper_thres])),
        'Green Channel': (np.array([0, lower_thres, 0]), np.array([120, upper_thres, 120])),
        'Blue Channel': (np.array([lower_thres, 0, 0]), np.array([upper_thres, 120, 120])),
        'Yellow Channel': (np.array([0, lower_thres, lower_thres]), np.array([120, upper_thres, upper_thres]))
    }

    all_contours = []
    for channel in settings_dict.get('channels', ['Red Channel']):
        if channel not in color_ranges_dict:
            continue
        lower, upper = color_ranges_dict[channel]
        color_mask = cv2.inRange(image, lower, upper)

        if enable_watershed and np.any(color_mask):
            dist_transform = cv2.distanceTransform(color_mask, cv2.DIST_L2, 3)
            if dist_transform.max() > 0:
                local_max = skimage.feature.peak_local_max(
                    dist_transform,
                    min_distance=int(watershed_thres * 10) + 1,
                    threshold_abs=watershed_thres * dist_transform.max(),
                    labels=color_mask
                )
                markers = np.zeros_like(dist_transform, dtype=np.int32)
                for i, pt in enumerate(local_max):
                    markers[pt[0], pt[1]] = i + 1
                if markers.max() > 0:
                    markers = cv2.dilate(markers.astype(np.uint8),
                                         np.ones((3, 3), np.uint8),
                                         iterations=1).astype(np.int32)
                    labels = skimage.segmentation.watershed(
                        -dist_transform, markers, mask=color_mask > 0)
                    for label_idx in np.unique(labels):
                        if label_idx == 0:
                            continue
                        cell_mask = (labels == label_idx).astype(np.uint8) * 255
                        cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in cnts:
                            area = cv2.contourArea(cnt)
                            if area > 0:
                                all_contours.append((cnt, area))
                else:
                    cnts, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in cnts:
                        area = cv2.contourArea(cnt)
                        if area > 0:
                            all_contours.append((cnt, area))
            else:
                cnts, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    area = cv2.contourArea(cnt)
                    if area > 0:
                        all_contours.append((cnt, area))
        else:
            cnts, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area > 0:
                    all_contours.append((cnt, area))

    return all_contours


def estimate_puncta_sizes(image, settings_dict, roi=None):
    """Estimate puncta sizes by reusing existing color mask + watershed pipeline.
    Returns result_dict with count, areas, statistics, and recommended min/max."""
    cropped = crop_to_roi(image, roi)
    contour_areas = _get_puncta_contours(cropped, settings_dict)

    if not contour_areas:
        return {"count": 0, "areas": [], "min": 0, "median": 0, "max": 0,
                "p10": 0, "p90": 0, "recommended_min": 0, "recommended_max": 0}

    areas = sorted([a for _, a in contour_areas])
    return {
        "count": len(areas),
        "areas": areas,
        "min": float(np.min(areas)),
        "median": float(np.median(areas)),
        "max": float(np.max(areas)),
        "p10": float(np.percentile(areas, 10)),
        "p90": float(np.percentile(areas, 90)),
        "recommended_min": max(1, int(np.percentile(areas, 10) * 0.8)),
        "recommended_max": int(np.percentile(areas, 90) * 1.2),
    }


def _validate_threshold_range(lower, upper):
    """Ensure (lower, upper) is a valid threshold pair.
    Guarantees: 0 <= lower < upper <= 255. Swaps if inverted, separates if equal.
    Returns (lower, upper) as ints."""
    try:
        lower = int(lower)
        upper = int(upper)
    except (TypeError, ValueError):
        lower, upper = 50, 220

    # Swap if inverted
    if lower > upper:
        lower, upper = upper, lower

    # Clamp to [0, 255]
    lower = max(0, min(lower, 255))
    upper = max(0, min(upper, 255))

    # Ensure strict inequality (lower < upper)
    if lower >= upper:
        if upper < 255:
            upper = upper + 1
        else:
            lower = max(0, upper - 1)

    return lower, upper


def suggest_threshold_from_roi(image, roi, channel_name):
    """Suggest color threshold values based on ROI pixel intensity distribution.
    Uses Otsu thresholding and percentile analysis.

    ROOT CAUSE FIX (Green Channel bug):
    Lower bound was derived from Otsu (full-pixel distribution) while upper bound
    came from the 95th percentile of NON-ZERO pixels only. For channels with
    cytoplasmic background fluorescence (typical for GFP/green), the distribution
    is near-unimodal and Otsu can land high in the curve while P95 of nonzero
    pixels stays lower, producing lower > upper. Red/Blue puncta markers usually
    have a strong bimodal distribution so Otsu < P95 holds naturally; Yellow uses
    an R+G average that smooths the distribution. The fix is twofold:
      (1) compute upper from the SAME nonzero pool used for Otsu fallback,
          using max(P95, Otsu) so upper >= Otsu's lower estimate,
      (2) call _validate_threshold_range() before returning so the GUI always
          receives lower < upper regardless of distribution shape.
    """
    cropped = crop_to_roi(image, roi)

    # BGR channel mapping
    channel_map = {
        'Red Channel': 2,     # BGR index for Red
        'Green Channel': 1,
        'Blue Channel': 0,
        'Yellow Channel': None  # average of R and G
    }

    idx = channel_map.get(channel_name)
    if idx is not None:
        channel_data = cropped[:, :, idx]
    else:
        # Yellow = average of R and G
        channel_data = ((cropped[:, :, 2].astype(float) +
                         cropped[:, :, 1].astype(float)) / 2).astype(np.uint8)

    # Otsu threshold
    try:
        otsu_val, _ = cv2.threshold(channel_data, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except cv2.error as e:
        logger.warning(f"Otsu threshold failed for {channel_name}: {e}")
        otsu_val = 128.0

    # Non-zero pixel statistics
    nonzero = channel_data[channel_data > 0]
    if len(nonzero) > 0:
        p5 = float(np.percentile(nonzero, 5))
        p95 = float(np.percentile(nonzero, 95))
    else:
        p5, p95 = 0.0, 255.0

    raw_lower = max(10, int(otsu_val * 0.8))
    # Use max(P95, Otsu+margin) so upper is always >= Otsu-derived lower.
    # This is the root-cause fix for the Green Channel inversion bug.
    raw_upper = max(int(p95), int(otsu_val) + 10)
    raw_upper = min(250, raw_upper)

    # Defensive logging when the raw values would have been inverted
    if raw_lower >= raw_upper:
        logger.warning(
            f"Suggested threshold range inversion for {channel_name}: "
            f"raw_lower={raw_lower}, raw_upper={raw_upper}, "
            f"otsu={otsu_val:.1f}, p95={p95:.1f}. "
            f"Validating and swapping if needed."
        )

    suggested_lower, suggested_upper = _validate_threshold_range(raw_lower, raw_upper)

    logger.info(
        f"Threshold suggestion for {channel_name}: "
        f"lower={suggested_lower}, upper={suggested_upper} "
        f"(otsu={otsu_val:.1f}, p5={p5:.1f}, p95={p95:.1f})"
    )

    return {
        "channel": channel_name,
        "otsu_threshold": int(otsu_val),
        "suggested_lower": suggested_lower,
        "suggested_upper": suggested_upper,
        "mean_intensity": float(np.mean(channel_data)),
        "std_intensity": float(np.std(channel_data)),
    }


def _compute_adaptive_font_scale(image_shape, base_scale=0.5, min_scale=0.25, max_scale=0.8):
    """Return a cv2 font scale that scales with the shorter image dimension.
    For a 500px side, returns base_scale. Clamped to [min_scale, max_scale]."""
    h, w = image_shape[:2]
    short_side = min(h, w)
    raw = base_scale * (short_side / 500.0)
    return float(max(min_scale, min(max_scale, raw)))


def _safe_put_text(img, text, anchor, font, scale, color, thickness, placement="above"):
    """Draw text near `anchor` (x, y) while keeping it inside the image.
    placement: 'above' (label above anchor), 'below', or 'center'.
    Returns the actual baseline-left origin used."""
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    H, W = img.shape[:2]
    ax, ay = int(anchor[0]), int(anchor[1])

    if placement == "above":
        ox, oy = ax - tw // 2, ay - 6
    elif placement == "below":
        ox, oy = ax - tw // 2, ay + th + 6
    else:  # center
        ox, oy = ax - tw // 2, ay + th // 2

    # Clamp horizontally
    ox = max(2, min(ox, max(2, W - tw - 2)))
    # Clamp vertically (origin is baseline-left)
    oy = max(th + 2, min(oy, max(th + 2, H - baseline - 2)))

    cv2.putText(img, text, (ox, oy), font, scale, color, thickness, cv2.LINE_AA)
    return (ox, oy)


def _detect_channel_mismatch(image, channel_name, roi=None, ratio_threshold=2.0):
    """Heuristic: return (mismatch: bool, info: str).
    True if another channel's mean intensity in the (optionally cropped) image
    is at least ratio_threshold times brighter than the selected channel's mean."""
    region = crop_to_roi(image, roi)
    if region.size == 0:
        return False, ""
    # BGR means
    means = {
        'Blue Channel':   float(np.mean(region[:, :, 0])),
        'Green Channel':  float(np.mean(region[:, :, 1])),
        'Red Channel':    float(np.mean(region[:, :, 2])),
    }
    means['Yellow Channel'] = (means['Red Channel'] + means['Green Channel']) / 2.0

    selected = means.get(channel_name, 0.0)
    if selected < 1.0:  # essentially dark
        selected = 1.0
    brightest_other_name = None
    brightest_other = 0.0
    for name, val in means.items():
        if name == channel_name:
            continue
        if val > brightest_other:
            brightest_other = val
            brightest_other_name = name

    if brightest_other / selected >= ratio_threshold:
        info = (f"Selected: {channel_name} (mean {selected:.1f})\n"
                f"Brightest other: {brightest_other_name} (mean {brightest_other:.1f})\n"
                f"Ratio: {brightest_other / selected:.2f}x")
        return True, info
    return False, ""


def _show_zoomable_image(image, window_name, info_text="", enable_roi_shortcut=False):
    """Zoomable/pannable cv2 viewer.
    Mouse wheel or +/- : zoom in/out
    Arrow keys         : pan
    0                  : reset view
    R (if enabled)     : trigger ROI selection request
    ESC / X button     : close
    Always resamples from the ORIGINAL image to avoid blur accumulation.
    Preserves aspect ratio (single scale factor).
    Returns 'roi_request' if R pressed with enable_roi_shortcut, else None.
    """
    if image is None:
        return None
    orig = image
    H, W = orig.shape[:2]

    # Initial auto-fit scale using existing helper contract
    _, fit_scale = _ensure_min_display_size(orig)
    state = {
        "zoom": float(fit_scale),
        "cx": W / 2.0,
        "cy": H / 2.0,
        "vw": max(600, int(round(W * fit_scale))),
        "vh": max(400, int(round(H * fit_scale))),
        "dirty": True,
    }
    MIN_ZOOM = max(0.05, fit_scale * 0.25)
    MAX_ZOOM = 8.0

    def render():
        z = state["zoom"]
        vw, vh = state["vw"], state["vh"]
        src_w = max(1, int(round(vw / z)))
        src_h = max(1, int(round(vh / z)))
        x0 = int(round(state["cx"] - src_w / 2))
        y0 = int(round(state["cy"] - src_h / 2))
        x0 = max(0, min(x0, max(0, W - src_w)))
        y0 = max(0, min(y0, max(0, H - src_h)))
        x1 = min(W, x0 + src_w)
        y1 = min(H, y0 + src_h)
        crop = orig[y0:y1, x0:x1]
        if crop.size == 0:
            crop = orig.copy()
        interp = cv2.INTER_LINEAR if z >= 1.0 else cv2.INTER_AREA
        disp = cv2.resize(crop, (vw, vh), interpolation=interp)
        if info_text:
            fs = _compute_adaptive_font_scale(disp.shape, base_scale=0.5)
            _safe_put_text(disp, info_text, (vw // 2, 18),
                           cv2.FONT_HERSHEY_SIMPLEX, fs,
                           (255, 255, 255), 1, placement="center")
        return disp

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            # On Windows, the wheel delta is in the high-order 16 bits of flags
            delta = (flags >> 16)
            if delta & 0x8000:
                delta -= 0x10000
            factor = 1.25 if delta > 0 else 0.8
            new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, state["zoom"] * factor))
            vw, vh = state["vw"], state["vh"]
            src_w = vw / state["zoom"]
            src_h = vh / state["zoom"]
            img_x = state["cx"] - src_w / 2 + (mx / vw) * src_w
            img_y = state["cy"] - src_h / 2 + (my / vh) * src_h
            state["zoom"] = new_zoom
            new_src_w = vw / new_zoom
            new_src_h = vh / new_zoom
            state["cx"] = img_x + (0.5 - mx / vw) * new_src_w
            state["cy"] = img_y + (0.5 - my / vh) * new_src_h
            state["dirty"] = True

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(window_name, state["vw"], state["vh"])
    cv2.setMouseCallback(window_name, on_mouse)

    result = None
    while True:
        if state["dirty"]:
            try:
                disp = render()
                cv2.imshow(window_name, disp)
            except cv2.error:
                break
            state["dirty"] = False
        try:
            key = cv2.waitKeyEx(30)
        except cv2.error:
            break
        if key == -1:
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            continue
        k = key & 0xFF
        if k == 27:  # ESC
            break
        if enable_roi_shortcut and k in (ord('r'), ord('R')):
            result = "roi_request"
            break
        if k in (ord('+'), ord('=')):
            state["zoom"] = min(MAX_ZOOM, state["zoom"] * 1.25)
            state["dirty"] = True
        elif k == ord('-'):
            state["zoom"] = max(MIN_ZOOM, state["zoom"] * 0.8)
            state["dirty"] = True
        elif k == ord('0'):
            state["zoom"] = fit_scale
            state["cx"] = W / 2.0
            state["cy"] = H / 2.0
            state["dirty"] = True
        else:
            pan_step = max(10, int(50 / max(state["zoom"], 0.1)))
            # Arrow key codes: Windows (waitKeyEx), Linux/Mac low-byte
            if key in (2490368, 0xF700, 82, 65362):
                state["cy"] -= pan_step; state["dirty"] = True
            elif key in (2621440, 0xF701, 84, 65364):
                state["cy"] += pan_step; state["dirty"] = True
            elif key in (2424832, 0xF702, 81, 65361):
                state["cx"] -= pan_step; state["dirty"] = True
            elif key in (2555904, 0xF703, 83, 65363):
                state["cx"] += pan_step; state["dirty"] = True

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass
    return result


def _ensure_min_display_size(image, min_width=600, min_height=400):
    """Ensure image is at least min_width x min_height for readable display.
    Also cap at max 1200px. Returns (display_image, scale)."""
    h, w = image.shape[:2]
    # Scale up if too small
    scale = 1.0
    if w < min_width or h < min_height:
        scale = max(min_width / w, min_height / h)
    # Scale down if too large
    max_display = 1200
    if max(h * scale, w * scale) > max_display:
        scale = max_display / max(h, w)
    if abs(scale - 1.0) > 0.01:
        display = cv2.resize(image, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
    else:
        display = image.copy()
    return display, scale


def _safe_cv2_window_loop(window_name, display):
    """Show an OpenCV window and wait until ESC or X button close.
    Handles both ESC key and window close button safely."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    dh, dw = display.shape[:2]
    cv2.resizeWindow(window_name, dw, dh)
    cv2.imshow(window_name, display)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 27 or key != 255:  # ESC or any key
            break
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def preview_detected_objects(image, settings_dict, roi=None, mode="puncta"):
    """Show detected objects in an OpenCV preview window.
    mode: 'threshold', 'puncta', or 'cells'."""
    cropped = crop_to_roi(image, roi)
    overlay = cropped.copy()

    if mode == "threshold":
        lower_thres = settings_dict['lower_thres']
        upper_thres = settings_dict['upper_thres']
        color_ranges_dict = {
            'Red Channel': (np.array([0, 0, lower_thres]), np.array([120, 120, upper_thres])),
            'Green Channel': (np.array([0, lower_thres, 0]), np.array([120, upper_thres, 120])),
            'Blue Channel': (np.array([lower_thres, 0, 0]), np.array([upper_thres, 120, 120])),
            'Yellow Channel': (np.array([0, lower_thres, lower_thres]), np.array([120, upper_thres, upper_thres]))
        }
        combined_mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
        for ch in settings_dict.get('channels', ['Red Channel']):
            if ch in color_ranges_dict:
                lower, upper = color_ranges_dict[ch]
                mask = cv2.inRange(cropped, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)

        tint = np.zeros_like(overlay)
        tint[:, :, 2] = 255
        overlay[combined_mask > 0] = cv2.addWeighted(
            overlay[combined_mask > 0], 0.5,
            tint[combined_mask > 0], 0.5, 0)

    elif mode == "puncta":
        contour_areas = _get_puncta_contours(cropped, settings_dict)
        contours = [ca[0] for ca in contour_areas]
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
        font_scale = _compute_adaptive_font_scale(overlay.shape, base_scale=0.45)
        for cnt, area in contour_areas:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                _safe_put_text(overlay, str(int(area)), (cx, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                               (0, 255, 255), 1, placement="center")

    elif mode == "cells":
        preprocessed = preprocess_for_cell_segmentation(cropped, settings_dict['clahe_clip'])
        _, cell_info, _ = segment_cell_bodies(
            preprocessed, 100, 99999999,
            settings_dict['morph_kernel'], settings_dict['cell_ws_min_dist'])
        font_scale = _compute_adaptive_font_scale(overlay.shape, base_scale=0.6)
        thickness = max(1, int(round(font_scale * 2)))
        for i, ci in enumerate(cell_info):
            color_bgr = CELL_COLORS[i % len(CELL_COLORS)]
            cv2.drawContours(overlay, [ci['contour']], 0, color_bgr, 2)
            cx, cy = ci['centroid']
            label = f"{ci['letter']}:{int(ci['area'])}px"
            _safe_put_text(overlay, label, (cx, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                           color_bgr, thickness, placement="above")

    display, _ = _ensure_min_display_size(overlay)
    window_name = f"SPOTTER Preview ({mode}) - Press any key or top-right X button to close"
    _safe_cv2_window_loop(window_name, display)


def inspect_object_at_click(image, settings_dict, roi=None):
    """Show detected objects and let user click to see individual pixel area.
    Press ESC or close window to exit."""
    cropped = crop_to_roi(image, roi)
    contour_areas = _get_puncta_contours(cropped, settings_dict)
    contours = [ca[0] for ca in contour_areas]
    areas = [ca[1] for ca in contour_areas]

    if not contours:
        return

    base_overlay = cropped.copy()
    cv2.drawContours(base_overlay, contours, -1, (0, 255, 255), 1)

    display_base, scale = _ensure_min_display_size(base_overlay)
    display = display_base.copy()

    window_name = "SPOTTER - Click object to inspect (ESC or top-right X button to close)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    dh, dw = display.shape[:2]
    cv2.resizeWindow(window_name, dw, dh)

    def on_mouse(event, mx, my, flags, param):
        nonlocal display
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ox = int(mx / scale)
        oy = int(my / scale)

        for i, cnt in enumerate(contours):
            if cv2.pointPolygonTest(cnt, (ox, oy), False) >= 0:
                temp = base_overlay.copy()
                cv2.drawContours(temp, [cnt], 0, (0, 0, 255), 2)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = ox, oy
                text = f"Area: {int(areas[i])} px"
                font_scale = _compute_adaptive_font_scale(temp.shape, base_scale=0.7)
                thickness = max(1, int(round(font_scale * 2)))
                _safe_put_text(temp, text, (cx, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                               (0, 0, 255), thickness, placement="above")
                display, _ = _ensure_min_display_size(temp)
                cv2.imshow(window_name, display)
                break

    cv2.imshow(window_name, display)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC
            break
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


# ─── Main Processing Function ────────────────────────────────────────────────

def process_image(args):
    # Unpack args
    (image_path, output_dir, lower_thres, upper_thres, watershed_thres,
     min_area, max_area, save_all_verboses, color_ranges_selected, ts_now,
     enable_watershed, multicell_mode, clahe_clip_limit, morph_kernel_size,
     min_cell_area, max_cell_area, cell_ws_min_dist) = args

    img_name = str(image_path.name)

    # Enable reading images from paths with Korean/foreign characters using numpy + cv2.imdecode
    try:
        with open(image_path, 'rb') as f:
            bytes_array = bytearray(f.read())
        numpy_array = np.asarray(bytes_array, dtype=np.uint8)
        image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        if image is None:
            return []
    except Exception as e:
        logger.error(f"Failed to read image {img_name}: {e}")
        return []

    # Normalize image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    try:
        return _process_image_core(image, img_name, output_dir, lower_thres, upper_thres,
                                   watershed_thres, min_area, max_area, save_all_verboses,
                                   color_ranges_selected, ts_now, enable_watershed,
                                   multicell_mode, clahe_clip_limit, morph_kernel_size,
                                   min_cell_area, max_cell_area, cell_ws_min_dist)
    except Exception as e:
        logger.error(f"Error processing {img_name}: {e}")
        return []


def _process_image_core(image, img_name, output_dir, lower_thres, upper_thres,
                        watershed_thres, min_area, max_area, save_all_verboses,
                        color_ranges_selected, ts_now, enable_watershed,
                        multicell_mode, clahe_clip_limit, morph_kernel_size,
                        min_cell_area, max_cell_area, cell_ws_min_dist):
    """Core processing logic, separated for clean error handling."""

    # ── Multi-Cell Segmentation (before color loop) ──
    cell_label_map = None
    cell_info = []
    preprocessed_img = None
    clean_mask_img = None

    if multicell_mode:
        preprocessed_img = preprocess_for_cell_segmentation(image, clahe_clip_limit)
        cell_label_map, cell_info, clean_mask_img = segment_cell_bodies(
            preprocessed_img, min_cell_area, max_cell_area,
            morph_kernel_size, cell_ws_min_dist
        )
        # Fallback: if no cells detected, treat entire image as one cell
        if len(cell_info) == 0:
            h, w = image.shape[:2]
            cell_label_map = np.ones((h, w), dtype=np.int32)
            full_contour = np.array([[[0, 0]], [[w-1, 0]], [[w-1, h-1]], [[0, h-1]]])
            cell_info = [{
                'label': 1, 'letter': 'A',
                'area': h * w,
                'centroid': (w // 2, h // 2),
                'contour': full_contour,
                'confidence': 0.0
            }]

    color_ranges_dict = {
        'Red Channel': (np.array([0, 0, lower_thres]), np.array([120, 120, upper_thres])),
        'Green Channel': (np.array([0, lower_thres, 0]), np.array([120, upper_thres, 120])),
        'Blue Channel': (np.array([lower_thres, 0, 0]), np.array([upper_thres, 120, 120])),
        'Yellow Channel': (np.array([0, lower_thres, lower_thres]), np.array([120, upper_thres, upper_thres]))
    }

    results = []

    def save_korean_path(img_path, img_data):
        try:
            is_success, buffer = cv2.imencode('.png', img_data)
            if is_success:
                with open(img_path, 'wb') as f:
                    f.write(buffer)
        except Exception as e:
            logger.warning(f"Failed to save image {img_path}: {e}")

    # Save multi-cell verbose images (once per image, not per color)
    if multicell_mode and save_all_verboses:
        verbose_dir = output_dir / f"verbose_{ts_now}"

        # 0_preprocessed
        dir_0 = verbose_dir / "0_preprocessed"
        dir_0.mkdir(parents=True, exist_ok=True)
        if preprocessed_img is not None:
            save_korean_path(str(dir_0 / f"preprocessed_{img_name}"), preprocessed_img)

        # 5_cell_mask
        dir_5 = verbose_dir / "5_cell_mask"
        dir_5.mkdir(parents=True, exist_ok=True)
        if clean_mask_img is not None:
            save_korean_path(str(dir_5 / f"cell_mask_{img_name}"), clean_mask_img)

        # 6_cell_boundaries (labeled cell outlines on original)
        dir_6 = verbose_dir / "6_cell_boundaries"
        dir_6.mkdir(parents=True, exist_ok=True)
        boundary_img = image.copy()
        for ci in cell_info:
            color_bgr = CELL_COLORS[ci['label'] - 1] if ci['label'] <= len(CELL_COLORS) else (255, 255, 255)
            cv2.drawContours(boundary_img, [ci['contour']], 0, color_bgr, 2)
            cx, cy = ci['centroid']
            conf_str = f"{ci['letter']} ({ci['confidence']:.2f})"
            cv2.putText(boundary_img, conf_str, (cx - 30, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
        save_korean_path(str(dir_6 / f"cell_boundaries_{img_name}"), boundary_img)

    for color in color_ranges_selected:
        lower, upper = color_ranges_dict[color]

        # 1. Binarize by Color Range
        color_mask = cv2.inRange(image, lower, upper)

        if enable_watershed:
            # 2. Distance Transform
            dist_transform = cv2.distanceTransform(color_mask, cv2.DIST_L2, 3)

            # 3. ImageJ-style Binary Watershed (Declumping)
            import skimage.feature
            import skimage.segmentation
            local_max = skimage.feature.peak_local_max(dist_transform,
                                                       min_distance=int(watershed_thres * 10) + 1,
                                                       threshold_abs=watershed_thres * dist_transform.max(),
                                                       labels=color_mask)

            # Create markers from local maxima
            markers = np.zeros_like(dist_transform, dtype=np.int32)
            for i, pt in enumerate(local_max):
                markers[pt[0], pt[1]] = i + 1

            markers = cv2.dilate(markers.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1).astype(np.int32)

            # Apply marker-controlled watershed
            labels = skimage.segmentation.watershed(-dist_transform, markers, mask=color_mask > 0)

            # Extract contours from the separated segments
            valid_contours = []
            for label_idx in np.unique(labels):
                if label_idx == 0:
                    continue
                cell_mask = (labels == label_idx).astype(np.uint8) * 255
                contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if min_area <= area <= max_area:
                        valid_contours.append(cnt)
        else:
            labels = None
            all_contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = []
            for cnt in all_contours:
                area = cv2.contourArea(cnt)
                if min_area <= area <= max_area:
                    valid_contours.append(cnt)

        number_of_cells = len(valid_contours)

        # ── Separated image (single-channel threshold) ──
        separated_img = np.zeros_like(image)
        if color == 'Red Channel':
            ch = image[:, :, 2]
            _, ch_mask = cv2.threshold(ch, int(lower_thres), 255, cv2.THRESH_BINARY)
            separated_img[:, :, 2] = cv2.bitwise_and(ch, ch_mask)
        elif color == 'Green Channel':
            ch = image[:, :, 1]
            _, ch_mask = cv2.threshold(ch, int(lower_thres), 255, cv2.THRESH_BINARY)
            separated_img[:, :, 1] = cv2.bitwise_and(ch, ch_mask)
        elif color == 'Blue Channel':
            ch = image[:, :, 0]
            _, ch_mask = cv2.threshold(ch, int(lower_thres), 255, cv2.THRESH_BINARY)
            separated_img[:, :, 0] = cv2.bitwise_and(ch, ch_mask)
        elif color == 'Yellow Channel':
            ch_r = image[:, :, 2]
            ch_g = image[:, :, 1]
            _, mask_r = cv2.threshold(ch_r, int(lower_thres), 255, cv2.THRESH_BINARY)
            _, mask_g = cv2.threshold(ch_g, int(lower_thres), 255, cv2.THRESH_BINARY)
            ch_mask = cv2.bitwise_and(mask_r, mask_g)
            separated_img[:, :, 2] = cv2.bitwise_and(ch_r, ch_mask)
            separated_img[:, :, 1] = cv2.bitwise_and(ch_g, ch_mask)

        # ── Watershed visualization ──
        watershed_img = np.zeros_like(image)
        if labels is not None:
            for label_idx in np.unique(labels):
                if label_idx == 0:
                    continue
                clr = (
                    int((label_idx * 67) % 256),
                    int((label_idx * 131) % 256),
                    int((label_idx * 199) % 256),
                )
                watershed_img[labels == label_idx] = clr

        # ── Contour overlay ──
        contour_img = image.copy()
        cv2.drawContours(contour_img, valid_contours, -1, (255, 255, 255), 1)

        # ── Area annotation ──
        area_img = np.zeros_like(image)

        # ── Result image ──
        result_img = image.copy()

        for i, cnt in enumerate(valid_contours):
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            cv2.drawContours(area_img, [cnt], 0, (255, 255, 255), 1)
            cv2.putText(area_img, str(cv2.contourArea(cnt)), (cx - 50, cy + 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(result_img, str(i + 1), (cx - 30, cy + 5), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1)

        # ── Collect results ──
        color_prefix = color.split()[0].lower()

        if multicell_mode and cell_label_map is not None and len(cell_info) > 0:
            # Assign puncta to cells
            per_cell_counts, assignment_dict, unassigned_indices = assign_puncta_to_cells(
                valid_contours, cell_label_map, cell_info
            )

            for ci in cell_info:
                letter = ci['letter']
                count = per_cell_counts.get(letter, 0)
                density = count / ci['area'] if ci['area'] > 0 else 0
                conf_flag = "LOW" if ci['confidence'] < 0.7 else "OK"
                results.append({
                    "image_name": img_name,
                    "cell_label": letter,
                    "cell_area": int(ci['area']),
                    "color": color.split()[0],
                    "puncta_count": count,
                    "puncta_density": round(density * 1e6, 2),
                    "confidence": round(ci['confidence'], 3),
                    "confidence_flag": conf_flag
                })

            if len(unassigned_indices) > 0:
                results.append({
                    "image_name": img_name,
                    "cell_label": "Unassigned",
                    "cell_area": 0,
                    "color": color.split()[0],
                    "puncta_count": len(unassigned_indices),
                    "puncta_density": 0,
                    "confidence": 0,
                    "confidence_flag": "N/A"
                })

            # ── Multi-cell result image: cell boundaries + puncta assignment overlay ──
            multicell_result_img = image.copy()
            # Draw cell boundaries
            for ci in cell_info:
                color_bgr = CELL_COLORS[ci['label'] - 1] if ci['label'] <= len(CELL_COLORS) else (255, 255, 255)
                cv2.drawContours(multicell_result_img, [ci['contour']], 0, color_bgr, 2)
                cx_c, cy_c = ci['centroid']
                label_text = f"{ci['letter']}:{per_cell_counts.get(ci['letter'], 0)}"
                cv2.putText(multicell_result_img, label_text, (cx_c - 30, cy_c - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

            # Draw puncta colored by cell assignment
            for ci in cell_info:
                color_bgr = CELL_COLORS[ci['label'] - 1] if ci['label'] <= len(CELL_COLORS) else (255, 255, 255)
                for pidx in assignment_dict.get(ci['letter'], []):
                    cv2.drawContours(multicell_result_img, [valid_contours[pidx]], 0, color_bgr, -1)

            # Draw unassigned puncta in white
            for pidx in unassigned_indices:
                cv2.drawContours(multicell_result_img, [valid_contours[pidx]], 0, (255, 255, 255), -1)

            # Save multicell result image to output root
            save_korean_path(str(output_dir / f"{color_prefix}_multicell_result_{img_name}"), multicell_result_img)

            # Save puncta assignment overlay to verbose
            if save_all_verboses:
                verbose_dir = output_dir / f"verbose_{ts_now}"
                dir_7 = verbose_dir / "7_puncta_assignment"
                dir_7.mkdir(parents=True, exist_ok=True)
                save_korean_path(str(dir_7 / f"{color_prefix}_puncta_assignment_{img_name}"), multicell_result_img)

        else:
            # Single-cell mode (original behavior)
            results.append({
                "image_name": img_name,
                "color": color.split()[0],
                "number_of_cells": number_of_cells
            })

        # ── Save verbose images (shared by both modes) ──
        if save_all_verboses:
            verbose_dir = output_dir / f"verbose_{ts_now}"
            dir_1 = verbose_dir / "1_separated"
            dir_2 = verbose_dir / "2_watershed"
            dir_3 = verbose_dir / "3_contour"
            dir_4 = verbose_dir / "4_area"
            for d in [dir_1, dir_2, dir_3, dir_4]:
                d.mkdir(parents=True, exist_ok=True)
            save_korean_path(str(dir_1 / f"{color_prefix}_separated_{img_name}"), separated_img)
            save_korean_path(str(dir_2 / f"{color_prefix}_watershed_{img_name}"), watershed_img)
            save_korean_path(str(dir_3 / f"{color_prefix}_contour_{img_name}"), contour_img)
            save_korean_path(str(dir_4 / f"{color_prefix}_area_{img_name}"), area_img)

        # Result image always saved in the output directory (outside verbose)
        save_korean_path(str(output_dir / f"{color_prefix}_result_{img_name}"), result_img)

    return results

def get_images_from_path():
    path = pathlib.Path(dpg.get_value("val_input_path"))
    if not path.exists() or not path.is_dir():
        return []

    images = []
    extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}
    for f in path.iterdir():
        if f.suffix.lower() in extensions:
            images.append(f)
    return images

def save_settings():
    settings = {
        "red": dpg.get_value("val_red_channel"),
        "green": dpg.get_value("val_green_channel"),
        "blue": dpg.get_value("val_blue_channel"),
        "yellow": dpg.get_value("val_yellow_channel"),
        "c_bottom": dpg.get_value("val_color_bottom_threshold"),
        "c_top": dpg.get_value("val_color_top_threshold"),
        "watershed": dpg.get_value("val_watershed_threshold"),
        "min_area": dpg.get_value("val_min_area"),
        "max_area": dpg.get_value("val_max_area"),
        "save_verbose": dpg.get_value("val_save_all_verboses"),
        "enable_watershed": dpg.get_value("val_enable_watershed"),
        "cell_mode": dpg.get_value("val_cell_mode"),
        "clahe_clip": dpg.get_value("val_clahe_clip"),
        "morph_kernel": dpg.get_value("val_morph_kernel"),
        "min_cell_area": dpg.get_value("val_min_cell_area"),
        "max_cell_area": dpg.get_value("val_max_cell_area"),
        "cell_ws_min_dist": dpg.get_value("val_cell_ws_min_dist"),
    }

    with open("spotter_settings.json", "w") as f:
        json.dump(settings, f)

def load_settings():
    if os.path.exists("spotter_settings.json"):
        try:
            with open("spotter_settings.json", "r") as f:
                settings = json.load(f)
            dpg.set_value("val_red_channel", settings.get("red", True))
            dpg.set_value("val_green_channel", settings.get("green", False))
            dpg.set_value("val_blue_channel", settings.get("blue", False))
            dpg.set_value("val_yellow_channel", settings.get("yellow", False))
            dpg.set_value("val_color_bottom_threshold", settings.get("c_bottom", 50.0))
            dpg.set_value("val_color_top_threshold", settings.get("c_top", 220.0))
            dpg.set_value("val_watershed_threshold", settings.get("watershed", 0.18))
            dpg.set_value("val_min_area", settings.get("min_area", 3))
            dpg.set_value("val_max_area", settings.get("max_area", 200))
            dpg.set_value("val_save_all_verboses", settings.get("save_verbose", True))
            ew = settings.get("enable_watershed", True)
            dpg.set_value("val_enable_watershed", ew)
            dpg.configure_item("val_watershed_threshold", enabled=ew)
            # Multi-cell settings
            cell_mode = settings.get("cell_mode", "Single Cell Mode")
            dpg.set_value("val_cell_mode", cell_mode)
            dpg.set_value("val_clahe_clip", settings.get("clahe_clip", 3.0))
            dpg.set_value("val_morph_kernel", settings.get("morph_kernel", 7))
            dpg.set_value("val_min_cell_area", settings.get("min_cell_area", 5000))
            dpg.set_value("val_max_cell_area", settings.get("max_cell_area", 200000))
            dpg.set_value("val_cell_ws_min_dist", settings.get("cell_ws_min_dist", 50))
            dpg.configure_item("multicell_settings_group",
                               show=(cell_mode == "Multi Cell Mode"))
        except Exception:
            pass

def _reset_values():
    dpg.set_value("val_red_channel", True)
    dpg.set_value("val_green_channel", False)
    dpg.set_value("val_blue_channel", False)
    dpg.set_value("val_yellow_channel", False)
    dpg.set_value("val_color_bottom_threshold", 50.0)
    dpg.set_value("val_color_top_threshold", 220.0)
    dpg.set_value("val_watershed_threshold", 0.18)
    dpg.set_value("val_min_area", 3)
    dpg.set_value("val_max_area", 200)
    dpg.set_value("val_save_all_verboses", True)
    dpg.set_value("val_enable_watershed", True)
    dpg.configure_item("val_watershed_threshold", enabled=True)
    # Multi-cell defaults
    dpg.set_value("val_cell_mode", "Single Cell Mode")
    dpg.set_value("val_clahe_clip", 3.0)
    dpg.set_value("val_morph_kernel", 7)
    dpg.set_value("val_min_cell_area", 5000)
    dpg.set_value("val_max_cell_area", 200000)
    dpg.set_value("val_cell_ws_min_dist", 50)
    dpg.configure_item("multicell_settings_group", show=False)

def _on_watershed_toggle(sender, value):
    dpg.configure_item("val_watershed_threshold", enabled=value)

def _on_mode_changed(sender, value):
    show = (value == "Multi Cell Mode")
    dpg.configure_item("multicell_settings_group", show=show)

def _on_input_dir_updated(sender, app_data):
    path = app_data["file_path_name"]
    dpg.set_value("val_input_path", path)
    logger.info(f"Input directory selected: {path}")

    images = get_images_from_path()
    logger.info(f"Found {len(images)} images in: {path}")
    dpg.set_value("val_data_count_str", f"{len(images)} Images Found.")

    # Auto set output path
    out_dir = pathlib.Path(path) / "output"
    dpg.set_value("val_output_path", str(out_dir))

def _on_output_dir_updated(sender, app_data):
    path = app_data["file_path_name"]
    out_dir = pathlib.Path(path) / "output"
    dpg.set_value("val_output_path", str(out_dir))


# ─── ROI Parameter Tuning Callbacks ──────────────────────────────────────────

def _gather_current_settings():
    """Collect current GUI widget values into a settings dict."""
    channels = []
    if dpg.get_value("val_red_channel"): channels.append("Red Channel")
    if dpg.get_value("val_green_channel"): channels.append("Green Channel")
    if dpg.get_value("val_blue_channel"): channels.append("Blue Channel")
    if dpg.get_value("val_yellow_channel"): channels.append("Yellow Channel")
    return {
        'lower_thres': int(dpg.get_value("val_color_bottom_threshold")),
        'upper_thres': int(dpg.get_value("val_color_top_threshold")),
        'watershed_thres': dpg.get_value("val_watershed_threshold"),
        'min_area': dpg.get_value("val_min_area"),
        'max_area': dpg.get_value("val_max_area"),
        'enable_watershed': dpg.get_value("val_enable_watershed"),
        'clahe_clip': dpg.get_value("val_clahe_clip"),
        'morph_kernel': dpg.get_value("val_morph_kernel"),
        'cell_ws_min_dist': dpg.get_value("val_cell_ws_min_dist"),
        'channels': channels,
    }


def _show_estimation_results(title, result_dict, mode):
    """Show estimation results in a DearPyGui modal dialog with Apply/Close buttons."""
    tag = f"estimation_dialog_{mode}"
    if dpg.does_alias_exist(tag):
        dpg.delete_item(tag)

    with dpg.window(label=title, tag=tag, modal=True, no_close=False,
                     width=420, height=350):
        if result_dict['count'] == 0:
            dpg.add_text("No objects detected.", color=(255, 100, 100))
            dpg.add_text("Try adjusting thresholds or ROI position.")
        else:
            dpg.add_text(f"Detected: {result_dict['count']} objects")
            dpg.add_separator()
            dpg.add_text(f"Min area:    {result_dict['min']:.0f} px")
            dpg.add_text(f"Median area: {result_dict['median']:.0f} px")
            dpg.add_text(f"Max area:    {result_dict['max']:.0f} px")
            dpg.add_text(f"P10:         {result_dict['p10']:.0f} px")
            dpg.add_text(f"P90:         {result_dict['p90']:.0f} px")
            dpg.add_separator()
            dpg.add_text(f"Recommended Min: {result_dict['recommended_min']} px",
                         color=(100, 255, 100))
            dpg.add_text(f"Recommended Max: {result_dict['recommended_max']} px",
                         color=(100, 255, 100))
            dpg.add_separator()

            def _apply_and_close():
                _apply_result(result_dict, mode)
                dpg.delete_item(tag)

            with dpg.group(horizontal=True):
                dpg.add_button(label="Apply Suggested", width=150,
                               callback=lambda: _apply_and_close())
                dpg.add_button(label="Close", width=80,
                               callback=lambda: dpg.delete_item(tag))

    dpg.split_frame()
    vw = dpg.get_viewport_client_width()
    vh = dpg.get_viewport_client_height()
    dpg.set_item_pos(tag, [vw // 2 - 210, vh // 2 - 175])


def _apply_result(result_dict, mode):
    """Apply recommended values from estimation result to GUI widgets."""
    if result_dict['count'] == 0:
        return
    if mode == "cell":
        dpg.set_value("val_min_cell_area", result_dict['recommended_min'])
        dpg.set_value("val_max_cell_area", result_dict['recommended_max'])
        logger.info(f"Applied cell size: min={result_dict['recommended_min']}, max={result_dict['recommended_max']}")
    elif mode == "puncta":
        dpg.set_value("val_min_area", result_dict['recommended_min'])
        dpg.set_value("val_max_area", result_dict['recommended_max'])
        logger.info(f"Applied puncta size: min={result_dict['recommended_min']}, max={result_dict['recommended_max']}")


def _on_load_sample(sender, app_data):
    """Callback for sample image file dialog.
    Shows image in a zoomable viewer. If user presses 'R', immediately transition
    to Pick ROI on the same image (no extra window reopening)."""
    global _tuning_image, _tuning_image_path
    file_path = app_data.get("file_path_name", "")
    if not file_path:
        return
    img = load_sample_image(file_path)
    if img is None:
        show_info("ERROR", "Failed to load image. Check file format.")
        return

    name = pathlib.Path(file_path).name
    h, w = img.shape[:2]
    dpg.set_value("tuning_status_text",
                  f"Loaded: {name} ({w}x{h}) | No ROI selected")

    # Show zoomable viewer; if user presses R, transition directly to Pick ROI
    window_name = (f"SPOTTER - {name} ({w}x{h}) | "
                   "Wheel=zoom, Arrows=pan, 0=reset, R=Pick ROI, "
                   "ESC or top-right X button to close")
    info_text = "Wheel=zoom | Arrows=pan | 0=reset | R=Pick ROI | ESC=close"
    result = _show_zoomable_image(img, window_name,
                                    info_text=info_text,
                                    enable_roi_shortcut=True)
    if result == "roi_request":
        # Transition directly to ROI selection without requiring button click
        _on_pick_roi()


def _on_pick_roi():
    """Open OpenCV ROI selection window with improved user feedback."""
    global _tuning_roi
    if _tuning_image is None:
        show_info("ERROR", "Please load a sample image first.")
        return
    roi = select_roi_interactive(_tuning_image)
    _tuning_roi = roi
    name = _tuning_image_path.name if _tuning_image_path else "image"
    if roi:
        x, y, w, h = roi
        # Show ROI with clear labels (x, y = top-left corner, w x h = size)
        dpg.set_value("tuning_status_text",
                      f"Loaded: {name} | ROI: X={x}, Y={y}, W={w}, H={h}")

        # Show the ROI on the image briefly so user can confirm
        confirm_img = _tuning_image.copy()
        cv2.rectangle(confirm_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"ROI: {w}x{h}px at ({x},{y})"
        font_scale = _compute_adaptive_font_scale(confirm_img.shape, base_scale=0.7)
        thickness = max(1, int(round(font_scale * 2)))
        _safe_put_text(confirm_img, label,
                       (x + w // 2, max(y - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       (0, 255, 0), thickness, placement="above")
        display, _ = _ensure_min_display_size(confirm_img)
        window_name = ("SPOTTER - ROI Selected "
                        "(press any key or top-right X button to continue)")
        _safe_cv2_window_loop(window_name, display)

        show_info("ROI SET", f"ROI successfully set!\n\n"
                  f"Position: X={x}, Y={y} (top-left corner)\n"
                  f"Size: {w} x {h} pixels")
    else:
        dpg.set_value("tuning_status_text",
                      f"Loaded: {name} | Full image (no ROI)")
        show_info("INFO", "No ROI selected.\nFull image will be used for analysis.")


def _on_estimate_cell_size():
    """Estimate cell body sizes from ROI with visual feedback."""
    global _tuning_cell_result
    if _tuning_image is None:
        show_info("ERROR", "Please load a sample image first.")
        return
    try:
        settings = _gather_current_settings()
        result = estimate_cell_sizes(_tuning_image, settings, _tuning_roi)
        _tuning_cell_result = result

        # Show visual feedback: detected cell outlines on the image
        if result['count'] > 0:
            preview_detected_objects(_tuning_image, settings, _tuning_roi, mode="cells")

        _show_estimation_results("Cell Size Estimation", result, "cell")
    except Exception as e:
        logger.error(f"Cell size estimation error: {e}")
        show_info("ERROR", f"Estimation failed:\n{e}")


def _on_estimate_puncta_size():
    """Estimate puncta sizes from ROI with visual feedback."""
    global _tuning_puncta_result
    if _tuning_image is None:
        show_info("ERROR", "Please load a sample image first.")
        return
    settings = _gather_current_settings()
    if not settings['channels']:
        show_info("ERROR", "Select at least one color channel.")
        return
    try:
        result = estimate_puncta_sizes(_tuning_image, settings, _tuning_roi)
        _tuning_puncta_result = result

        # Show visual feedback: detected puncta outlines on the image
        if result['count'] > 0:
            preview_detected_objects(_tuning_image, settings, _tuning_roi, mode="puncta")

        _show_estimation_results("Puncta Size Estimation", result, "puncta")
    except Exception as e:
        logger.error(f"Puncta size estimation error: {e}")
        show_info("ERROR", f"Estimation failed:\n{e}")


def _interactive_threshold_preview(image, roi, channel_name, initial_lower, initial_upper):
    """Show an interactive threshold preview with trackbars.
    User can adjust lower/upper threshold and see the mask overlay in real-time.
    Returns (lower, upper) on Enter/Space, or None on ESC."""
    cropped = crop_to_roi(image, roi)

    # BGR channel mapping
    channel_map = {
        'Red Channel': 2, 'Green Channel': 1,
        'Blue Channel': 0, 'Yellow Channel': None
    }

    def build_color_ranges(lower_t, upper_t):
        return {
            'Red Channel': (np.array([0, 0, lower_t]), np.array([120, 120, upper_t])),
            'Green Channel': (np.array([0, lower_t, 0]), np.array([120, upper_t, 120])),
            'Blue Channel': (np.array([lower_t, 0, 0]), np.array([upper_t, 120, 120])),
            'Yellow Channel': (np.array([0, lower_t, lower_t]), np.array([120, upper_t, upper_t]))
        }

    def update_preview(lower_t, upper_t):
        ranges = build_color_ranges(lower_t, upper_t)
        if channel_name in ranges:
            lower_arr, upper_arr = ranges[channel_name]
            mask = cv2.inRange(cropped, lower_arr, upper_arr)
        else:
            mask = np.zeros(cropped.shape[:2], dtype=np.uint8)

        overlay = cropped.copy()
        tint = np.zeros_like(overlay)
        tint[:, :, 1] = 255  # Green tint for detected areas
        overlay[mask > 0] = cv2.addWeighted(
            overlay[mask > 0], 0.5, tint[mask > 0], 0.5, 0)

        # Count detected pixels
        pixel_count = np.count_nonzero(mask)
        info = f"Channel: {channel_name} | Lower: {lower_t} | Upper: {upper_t} | Detected pixels: {pixel_count}"
        # Put info bar at top
        bar = np.zeros((30, overlay.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        result = np.vstack([bar, overlay])
        return result

    window_name = (f"Threshold Preview ({channel_name}) - Sliders adjust, "
                    "ENTER to apply, ESC or top-right X button to cancel")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

    cv2.createTrackbar("Lower", window_name, int(initial_lower), 255, lambda x: None)
    cv2.createTrackbar("Upper", window_name, int(initial_upper), 255, lambda x: None)

    # Initial display
    preview = update_preview(int(initial_lower), int(initial_upper))
    display, _ = _ensure_min_display_size(preview)
    dh, dw = display.shape[:2]
    cv2.resizeWindow(window_name, dw, dh)
    cv2.imshow(window_name, display)

    final_lower = int(initial_lower)
    final_upper = int(initial_upper)

    while True:
        key = cv2.waitKey(50) & 0xFF

        # Read current trackbar values
        cur_lower = cv2.getTrackbarPos("Lower", window_name)
        cur_upper = cv2.getTrackbarPos("Upper", window_name)

        # Enforce Lower <= Upper to prevent crashes
        if cur_lower > cur_upper:
            if cur_lower != final_lower and cur_upper == final_upper:
                # Lower was dragged above Upper -> clamp Lower to Upper
                cur_lower = cur_upper
                cv2.setTrackbarPos("Lower", window_name, cur_lower)
            elif cur_upper != final_upper and cur_lower == final_lower:
                # Upper was dragged below Lower -> clamp Upper to Lower
                cur_upper = cur_lower
                cv2.setTrackbarPos("Upper", window_name, cur_upper)
            else:
                # Both changed or neither matches: push Upper up to match Lower
                cur_upper = cur_lower
                cv2.setTrackbarPos("Upper", window_name, cur_upper)

        if cur_lower != final_lower or cur_upper != final_upper:
            final_lower = cur_lower
            final_upper = cur_upper
            try:
                preview = update_preview(final_lower, final_upper)
                display, _ = _ensure_min_display_size(preview)
                cv2.imshow(window_name, display)
            except Exception as e:
                logger.warning(f"Preview render failed at L={final_lower} U={final_upper}: {e}")

        if key == 13 or key == 32:  # Enter or Space = apply
            break
        if key == 27:  # ESC = cancel
            final_lower = None
            break
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                final_lower = None
                break
        except cv2.error:
            final_lower = None
            break

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass

    if final_lower is None:
        return None
    return (final_lower, final_upper)


def _on_suggest_threshold():
    """Suggest threshold values based on ROI intensity with interactive preview."""
    global _tuning_threshold_result
    if _tuning_image is None:
        show_info("ERROR", "Please load a sample image first.")
        return
    settings = _gather_current_settings()
    if not settings['channels']:
        show_info("ERROR", "Select at least one color channel.")
        return

    try:
        channel = settings['channels'][0]

        # Guard: detect channel mismatch (e.g., Red Channel selected but image is green)
        try:
            mismatch, info = _detect_channel_mismatch(_tuning_image, channel, _tuning_roi)
        except Exception as mismatch_err:
            logger.warning(f"Channel mismatch check failed: {mismatch_err}")
            mismatch, info = False, ""

        if mismatch:
            show_info(
                "Channel mismatch?",
                "The selected channel looks much dimmer than another channel in this image.\n\n"
                f"{info}\n\n"
                "Please verify you picked the correct channel color for this image, "
                "then run Suggest Threshold again."
            )
            return

        result = suggest_threshold_from_roi(_tuning_image, _tuning_roi, channel)

        # Second safety net: validate again after suggest_threshold_from_roi.
        # If a swap was needed here, suggest_threshold_from_roi already logged it,
        # but we surface a non-fatal warning to the user so they know the auto
        # values were corrected.
        orig_lower = result['suggested_lower']
        orig_upper = result['suggested_upper']
        v_lower, v_upper = _validate_threshold_range(orig_lower, orig_upper)
        result['suggested_lower'] = v_lower
        result['suggested_upper'] = v_upper
        if (v_lower, v_upper) != (orig_lower, orig_upper):
            logger.warning(
                f"Threshold values were corrected for {channel}: "
                f"({orig_lower}, {orig_upper}) -> ({v_lower}, {v_upper})"
            )
            show_info(
                "Threshold auto-corrected",
                f"The auto-suggested threshold range for {channel} was invalid\n"
                f"({orig_lower} > {orig_upper}) and has been corrected to\n"
                f"Lower={v_lower}, Upper={v_upper}.\n\n"
                "This typically happens when the channel has near-uniform\n"
                "background fluorescence (e.g. cytoplasmic GFP). You may want\n"
                "to adjust the values manually in the preview window."
            )

        _tuning_threshold_result = result

        # Show interactive threshold preview first
        try:
            adjusted = _interactive_threshold_preview(
                _tuning_image, _tuning_roi, channel,
                result['suggested_lower'], result['suggested_upper']
            )
        except Exception as preview_err:
            logger.error(f"Threshold preview failed: {preview_err}")
            adjusted = None

        # If user adjusted values in the preview, update result
        if adjusted is not None:
            v_lower, v_upper = _validate_threshold_range(adjusted[0], adjusted[1])
            result['suggested_lower'] = v_lower
            result['suggested_upper'] = v_upper
            _tuning_threshold_result = result

        tag = "threshold_suggest_dialog"
        if dpg.does_alias_exist(tag):
            dpg.delete_item(tag)

        with dpg.window(label="Threshold Suggestion", tag=tag, modal=True,
                         no_close=False, width=420, height=340):
            dpg.add_text(f"Channel: {result['channel']}")
            dpg.add_separator()
            dpg.add_text(f"Mean intensity:    {result['mean_intensity']:.1f}")
            dpg.add_text(f"Std intensity:     {result['std_intensity']:.1f}")
            dpg.add_text(f"Otsu threshold:    {result['otsu_threshold']}")
            dpg.add_separator()
            if adjusted is not None:
                dpg.add_text(f"Final Lower:   {result['suggested_lower']}  (adjusted in preview)",
                             color=(100, 255, 100))
                dpg.add_text(f"Final Upper:   {result['suggested_upper']}  (adjusted in preview)",
                             color=(100, 255, 100))
            else:
                dpg.add_text(f"Suggested Lower:   {result['suggested_lower']}",
                             color=(100, 255, 100))
                dpg.add_text(f"Suggested Upper:   {result['suggested_upper']}",
                             color=(100, 255, 100))
                dpg.add_text("(Preview was cancelled - showing auto-suggested values)",
                             color=(180, 180, 100))
            dpg.add_separator()

            def _apply_thresh_and_close():
                # Final validation before pushing to GUI widgets - belt and suspenders.
                final_lower, final_upper = _validate_threshold_range(
                    result['suggested_lower'], result['suggested_upper'])
                try:
                    dpg.set_value("val_color_bottom_threshold", float(final_lower))
                    dpg.set_value("val_color_top_threshold", float(final_upper))
                    logger.info(f"Applied threshold: lower={final_lower}, upper={final_upper}")
                except Exception as apply_err:
                    logger.error(f"Failed to apply threshold to GUI: {apply_err}")
                    show_info("ERROR", f"Could not apply threshold values:\n{apply_err}")
                finally:
                    try:
                        dpg.delete_item(tag)
                    except Exception:
                        pass

            with dpg.group(horizontal=True):
                dpg.add_button(label="Apply", width=150,
                               callback=lambda: _apply_thresh_and_close())
                dpg.add_button(label="Close", width=80,
                               callback=lambda: dpg.delete_item(tag))

        dpg.split_frame()
        vw = dpg.get_viewport_client_width()
        vh = dpg.get_viewport_client_height()
        dpg.set_item_pos(tag, [vw // 2 - 210, vh // 2 - 170])

    except Exception as e:
        logger.error(f"Threshold suggestion error: {e}")
        show_info("ERROR", f"Suggestion failed:\n{e}")


def _on_apply_all_suggested():
    """Apply all available suggested values to GUI widgets."""
    applied = []
    if _tuning_cell_result and _tuning_cell_result['count'] > 0:
        _apply_result(_tuning_cell_result, "cell")
        applied.append("cell size")
    if _tuning_puncta_result and _tuning_puncta_result['count'] > 0:
        _apply_result(_tuning_puncta_result, "puncta")
        applied.append("puncta size")
    if _tuning_threshold_result:
        dpg.set_value("val_color_bottom_threshold",
                      float(_tuning_threshold_result['suggested_lower']))
        dpg.set_value("val_color_top_threshold",
                      float(_tuning_threshold_result['suggested_upper']))
        applied.append("threshold")

    if applied:
        show_info("APPLIED", f"Applied suggested values for: {', '.join(applied)}")
    else:
        show_info("INFO", "No estimation results to apply.\nRun estimation first.")


def _on_preview_detected():
    """Preview detected objects in an OpenCV window."""
    if _tuning_image is None:
        show_info("ERROR", "Please load a sample image first.")
        return
    settings = _gather_current_settings()
    if not settings['channels']:
        show_info("ERROR", "Select at least one color channel.")
        return
    try:
        preview_detected_objects(_tuning_image, settings, _tuning_roi, mode="puncta")
    except Exception as e:
        logger.error(f"Preview error: {e}")
        show_info("ERROR", f"Preview failed:\n{e}")


def _on_inspect_object():
    """Open interactive object inspection window."""
    if _tuning_image is None:
        show_info("ERROR", "Please load a sample image first.")
        return
    settings = _gather_current_settings()
    if not settings['channels']:
        show_info("ERROR", "Select at least one color channel.")
        return
    try:
        inspect_object_at_click(_tuning_image, settings, _tuning_roi)
    except Exception as e:
        logger.error(f"Inspect error: {e}")
        show_info("ERROR", f"Inspection failed:\n{e}")


def _on_start():
    images = get_images_from_path()
    if not images:
        show_info("ERROR", "No images found.")
        return

    save_settings()

    output_path = pathlib.Path(dpg.get_value("val_output_path"))
    output_path.mkdir(parents=True, exist_ok=True)

    cb = dpg.get_value("val_color_bottom_threshold")
    ct = dpg.get_value("val_color_top_threshold")
    wt = dpg.get_value("val_watershed_threshold")
    min_area = dpg.get_value("val_min_area")
    max_area = dpg.get_value("val_max_area")
    save_verbose = dpg.get_value("val_save_all_verboses")
    enable_watershed = dpg.get_value("val_enable_watershed")

    # Multi-cell parameters
    multicell_mode = (dpg.get_value("val_cell_mode") == "Multi Cell Mode")
    clahe_clip = dpg.get_value("val_clahe_clip")
    morph_kernel = dpg.get_value("val_morph_kernel")
    min_cell_area = dpg.get_value("val_min_cell_area")
    max_cell_area = dpg.get_value("val_max_cell_area")
    cell_ws_min_dist = dpg.get_value("val_cell_ws_min_dist")

    # Selected Colors
    color_ranges_selected = []
    if dpg.get_value("val_red_channel"): color_ranges_selected.append("Red Channel")
    if dpg.get_value("val_green_channel"): color_ranges_selected.append("Green Channel")
    if dpg.get_value("val_blue_channel"): color_ranges_selected.append("Blue Channel")
    if dpg.get_value("val_yellow_channel"): color_ranges_selected.append("Yellow Channel")

    if not color_ranges_selected:
        show_info("ERROR", "Select at least one color channel.")
        return

    dpg.configure_item("progress_bar_id", show=True)
    dpg.set_value("progress_bar_id", 0.0)

    ts_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare Multiprocessing
    pool_args = []
    for img_path in images:
        pool_args.append((
            img_path, output_path, int(cb), int(ct), wt,
            min_area, max_area, save_verbose, color_ranges_selected, ts_now,
            enable_watershed, multicell_mode, clahe_clip, morph_kernel,
            min_cell_area, max_cell_area, cell_ws_min_dist
        ))

    all_results = []

    # Multiprocessing Pool
    logger.info("Starting Processing with Multiprocessing Engine")
    total = len(pool_args)

    try:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for i, res in enumerate(pool.imap_unordered(process_image, pool_args)):
                all_results.extend(res)
                progress = (i + 1) / total
                dpg.set_value("progress_bar_id", progress)
                dpg.split_frame()
    except Exception as e:
        logger.error(f"Processing error: {e}")
        show_info("ERROR", f"Processing failed:\n{e}")
        dpg.configure_item("progress_bar_id", show=False)
        return

    # Build run settings rows from the exact GUI values captured above.
    # All values were already read at the moment Start was pressed
    # (lines above this block), so no race with later GUI changes.
    settings_rows = [
        ("SPOTTER Version", "v2.0 (Multi-Cell Engine)"),
        ("Run Timestamp", ts_now),
        ("Input Directory", str(dpg.get_value("val_input_path"))),
        ("Output Directory", str(output_path)),
        ("Number of Images Processed", len(images)),
        ("", ""),
        ("Analysis Mode", "Multi Cell Mode" if multicell_mode else "Single Cell Mode"),
        ("", ""),
        ("Red Channel",    bool(dpg.get_value("val_red_channel"))),
        ("Green Channel",  bool(dpg.get_value("val_green_channel"))),
        ("Blue Channel",   bool(dpg.get_value("val_blue_channel"))),
        ("Yellow Channel", bool(dpg.get_value("val_yellow_channel"))),
        ("", ""),
        ("Color Bottom Threshold", int(cb)),
        ("Color Top Threshold",    int(ct)),
        ("Enable Watershed",       bool(enable_watershed)),
        ("Watershed Threshold",    float(wt)),
        ("Min Segment Size",       int(min_area)),
        ("Max Segment Size",       int(max_area)),
        ("", ""),
        ("CLAHE Clip Limit",            float(clahe_clip)),
        ("Morphology Kernel Size",      int(morph_kernel)),
        ("Min Cell Area",               int(min_cell_area)),
        ("Max Cell Area",               int(max_cell_area)),
        ("Cell Watershed Min Distance", int(cell_ws_min_dist)),
        ("", ""),
        ("Save All Verboses", bool(save_verbose)),
    ]
    settings_df = pd.DataFrame(settings_rows, columns=["Parameter", "Value"])

    # Generate Excel Report (results + run settings sheet)
    if all_results:
        try:
            df = pd.DataFrame(all_results)
            excel_path = output_path / f"SPOTTER_Result_{ts_now}.xlsx"
            # Use ExcelWriter so we can add a second worksheet without
            # touching the existing result sheet contents/columns.
            with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Results", index=False)
                settings_df.to_excel(writer, sheet_name="Run_Settings", index=False)
            logger.info(f"Report saved to {excel_path}")
        except Exception as e:
            logger.error(f"Excel save error: {e}")
            show_info("ERROR", f"Failed to save Excel:\n{e}")
            dpg.configure_item("progress_bar_id", show=False)
            return
    else:
        # No analysis results, but still preserve the run settings for traceability
        try:
            settings_only_path = output_path / f"SPOTTER_RunSettings_{ts_now}.xlsx"
            with pd.ExcelWriter(str(settings_only_path), engine="openpyxl") as writer:
                settings_df.to_excel(writer, sheet_name="Run_Settings", index=False)
            logger.info(f"Run settings (no results) saved to {settings_only_path}")
        except Exception as e:
            logger.warning(f"Could not save run settings file: {e}")

    show_info("SUCCESS", "Processing Completed Successfully!")
    dpg.configure_item("progress_bar_id", show=False)

def main():
    logger.info("Starting SPOTTER")
    dpg.create_context()
    dpg.create_viewport(title="SPOTTER", width=800, height=900)
    dpg.setup_dearpygui()

    # Load fonts with Korean character support
    default_font = None
    font_candidates = [
        (resource_path('malgun.ttf'), True),
        (r'C:\Windows\Fonts\malgun.ttf', True),
        (resource_path('arial.ttf'), False),
    ]
    for font_file, supports_korean in font_candidates:
        if os.path.exists(font_file):
            with dpg.font_registry():
                default_font = dpg.add_font(font_file, 16)
                if supports_korean:
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Korean, parent=default_font)
            break

    # File dialogs
    with dpg.file_dialog(directory_selector=True, show=False, callback=_on_input_dir_updated, tag="input_dir_dialog", width=600, height=400):
        pass
    with dpg.file_dialog(directory_selector=True, show=False, callback=_on_output_dir_updated, tag="output_dir_dialog", width=600, height=400):
        pass

    # Sample image file dialog for ROI tuning
    with dpg.file_dialog(directory_selector=False, show=False, callback=_on_load_sample,
                          tag="sample_file_dialog", width=600, height=400):
        # "All supported image files" is the first (default) filter
        dpg.add_file_extension(
            "Source files (*.png *.jpg *.jpeg *.tif *.tiff){.png,.jpg,.jpeg,.tif,.tiff}",
            color=(0, 255, 0, 255))
        dpg.add_file_extension(".png", color=(0, 255, 0, 255))
        dpg.add_file_extension(".jpg", color=(0, 255, 0, 255))
        dpg.add_file_extension(".jpeg", color=(0, 255, 0, 255))
        dpg.add_file_extension(".tif", color=(0, 255, 255, 255))
        dpg.add_file_extension(".tiff", color=(0, 255, 255, 255))
        dpg.add_file_extension(".*", color=(200, 200, 200, 255))

    with dpg.window(tag="primary_window"):
        dpg.add_text("SPOTTER v2.0 (Multi-Cell Engine)")
        dpg.add_text("Advanced Multiprocessing Version by KIST")
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_text("Input Directory: ")
            dpg.add_input_text(tag="val_input_path", width=500, readonly=True)
            dpg.add_button(label="Open", callback=lambda: dpg.show_item("input_dir_dialog"))

        with dpg.group(horizontal=True):
            dpg.add_text("0 Images Found.", tag="val_data_count_str")

        dpg.add_separator()
        dpg.add_text("Settings")

        # ── Mode Selector ──
        dpg.add_radio_button(["Single Cell Mode", "Multi Cell Mode"],
                              tag="val_cell_mode", horizontal=True,
                              callback=_on_mode_changed, default_value="Single Cell Mode")
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="Red Channel", tag="val_red_channel")
            dpg.add_checkbox(label="Green Channel", tag="val_green_channel")
            dpg.add_checkbox(label="Blue Channel", tag="val_blue_channel")
            dpg.add_checkbox(label="Yellow Channel", tag="val_yellow_channel")

        dpg.add_input_float(label="Color Bottom Threshold", min_value=0.0, max_value=255.0, tag="val_color_bottom_threshold")
        dpg.add_input_float(label="Color Top Threshold", min_value=0.0, max_value=255.0, tag="val_color_top_threshold")
        dpg.add_checkbox(label="Enable Watershed", tag="val_enable_watershed", callback=_on_watershed_toggle)
        dpg.add_input_float(label="Watershed Threshold", min_value=0.0, max_value=0.99, min_clamped=True, max_clamped=True, tag="val_watershed_threshold")
        dpg.add_input_int(label="Min Segment Size", min_value=0, max_value=1000, tag="val_min_area")
        dpg.add_input_int(label="Max Segment Size", min_value=0, max_value=1000, tag="val_max_area")

        # ── Multi-Cell Settings Group (show/hide) ──
        with dpg.group(tag="multicell_settings_group", show=False):
            dpg.add_separator()
            dpg.add_text("Cell Body Segmentation Settings")
            dpg.add_input_float(label="CLAHE Clip Limit", min_value=1.0, max_value=10.0,
                                min_clamped=True, max_clamped=True,
                                tag="val_clahe_clip", default_value=3.0)
            dpg.add_input_int(label="Morphology Kernel Size", min_value=3, max_value=21,
                              min_clamped=True, max_clamped=True,
                              tag="val_morph_kernel", default_value=7)
            dpg.add_input_int(label="Min Cell Area", min_value=100, max_value=1000000,
                              min_clamped=True, max_clamped=True,
                              tag="val_min_cell_area", default_value=5000)
            dpg.add_input_int(label="Max Cell Area", min_value=1000, max_value=5000000,
                              min_clamped=True, max_clamped=True,
                              tag="val_max_cell_area", default_value=200000)
            dpg.add_input_int(label="Cell Watershed Min Distance", min_value=10, max_value=500,
                              min_clamped=True, max_clamped=True,
                              tag="val_cell_ws_min_dist", default_value=50)
            dpg.add_text("* Low confidence cells (solidity < 0.7) will be flagged in results.",
                         color=(180, 180, 100))

        dpg.add_checkbox(label="Save All Verboses", tag="val_save_all_verboses")
        dpg.add_button(label="Reset", callback=_reset_values)

        dpg.add_separator()

        # ── Parameter Tuning (ROI-Based) ──
        dpg.add_text("Parameter Tuning (ROI-Based)", color=(180, 220, 255))

        with dpg.group(horizontal=True):
            dpg.add_button(label="Load Sample Image", width=160,
                           callback=lambda: dpg.show_item("sample_file_dialog"))
            dpg.add_button(label="Pick ROI", width=100, callback=_on_pick_roi)

        dpg.add_text("No image loaded.", tag="tuning_status_text",
                     color=(180, 180, 100))

        with dpg.group(horizontal=True):
            dpg.add_button(label="Estimate Cell Size", width=160,
                           callback=_on_estimate_cell_size)
            dpg.add_button(label="Estimate Puncta Size", width=160,
                           callback=_on_estimate_puncta_size)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Suggest Threshold", width=160,
                           callback=_on_suggest_threshold)
            dpg.add_button(label="Apply Suggested Range", width=160,
                           callback=_on_apply_all_suggested)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Preview Detected Objects", width=180,
                           callback=_on_preview_detected)
            dpg.add_button(label="Inspect Object Size", width=160,
                           callback=_on_inspect_object)

        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_text("Output Directory: ")
            dpg.add_input_text(tag="val_output_path", width=500)
            dpg.add_button(label="Open", callback=lambda: dpg.show_item("output_dir_dialog"))

        dpg.add_button(label="Start", callback=_on_start, width=100)
        dpg.add_progress_bar(label="Progress", tag="progress_bar_id", show=False, width=-1)

    _reset_values()
    load_settings()

    if default_font: dpg.bind_font(default_font)

    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as e:
        logger.exception("Fatal error in main")
