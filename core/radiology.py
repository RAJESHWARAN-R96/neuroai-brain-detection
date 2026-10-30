"""
Clinical Radiology and Adaptive MRI Preprocessing Engine.
Provides robust artifact removal for web screenshots/downloads,
unilateral ring-enhancement analysis, necrotic core detection,
and deep feature reconciliation for brain tumor diagnosis.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

def is_web_screenshot_or_artifact(img_np: np.ndarray) -> bool:
    """
    Detects whether an MRI image contains non-standard web borders,
    white screenshot margins, or non-black background padding.
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
    corners = [float(gray[0, 0]), float(gray[0, -1]), float(gray[-1, 0]), float(gray[-1, -1])]
    edges = [float(gray[0, :].mean()), float(gray[-1, :].mean()), float(gray[:, 0].mean()), float(gray[:, -1].mean())]
    return (np.mean(corners) > 25.0) or (np.mean(edges) > 35.0)

def sanitize_web_mri(img_np: np.ndarray, target_size: Tuple[int, int] = (224, 224), brain_ratio: float = 0.68) -> np.ndarray:
    """
    Sanitizes web MRI screenshots:
    - Inverts/floodfills white borders
    - Locates cranial vault contour
    - Sets exterior background to pitch black (0, 0, 0)
    - Centers cranial vault on a square canvas at standard clinical scale (~68%)
    """
    if len(img_np.shape) == 2:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img_np.copy()
        
    h, w = img_rgb.shape[:2]
    work = img_rgb.copy()
    work[:4, :] = 0
    work[-4:, :] = 0
    work[:, :4] = 0
    work[:, -4:] = 0
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    
    # Floodfill exterior white
    flood = gray.copy()
    fmask = np.zeros((h + 2, w + 2), np.uint8)
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        cv2.floodFill(flood, fmask, seed, 0, loDiff=35, upDiff=35)
        
    _, thresh = cv2.threshold(cv2.GaussianBlur(flood, (5, 5), 0), 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        brain_only = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        x, y, bw, bh = cv2.boundingRect(hull)
        pad = 2
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
        cropped_brain = brain_only[y1:y2, x1:x2]
    else:
        cropped_brain = img_rgb
        
    cb_h, cb_w = cropped_brain.shape[:2]
    max_dim = max(cb_h, cb_w)
    canvas_dim = int(max_dim / brain_ratio)
    canvas = np.zeros((canvas_dim, canvas_dim, 3), dtype=np.uint8)
    y_off = (canvas_dim - cb_h) // 2
    x_off = (canvas_dim - cb_w) // 2
    canvas[y_off:y_off + cb_h, x_off:x_off + cb_w] = cropped_brain
    return cv2.resize(canvas, target_size, interpolation=cv2.INTER_AREA if canvas_dim > target_size[0] else cv2.INTER_CUBIC)

def detect_unilateral_ring_glioma(img_np: np.ndarray) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Detects classic Glioblastoma / High-Grade Glioma hallmarks:
    - Enhancing intraparenchymal ring lesion
    - Internal necrotic / cystic core
    - Unilateral cerebral hemisphere location (outside the ventricle midline)
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
    h, w = gray.shape
    brain_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(brain_mask, (w // 2, h // 2), (int(w * 0.38), int(h * 0.38)), 0, 0, 360, 255, -1)
    
    parenchyma = gray[brain_mask > 0]
    if len(parenchyma) == 0:
        return False, None
        
    p_mean, p_std = float(parenchyma.mean()), float(parenchyma.std())
    thresh_val = max(145, int(p_mean + 1.2 * p_std))
    
    _, binary = cv2.threshold(cv2.bitwise_and(gray, gray, mask=brain_mask), thresh_val, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None
        
    largest_c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_c)
    if area < 80:
        return False, None
        
    x, y, bw, bh = cv2.boundingRect(largest_c)
    cx, cy = x + bw // 2, y + bh // 2
    
    # Must be unilateral (outside the central ventricular midline strip)
    if abs(cx - w // 2) < (w * 0.10):
        return False, None
        
    patch = gray[y:y + bh, x:x + bw]
    if patch.size > 0:
        inner = patch[int(bh * 0.25):int(bh * 0.75), int(bw * 0.25):int(bw * 0.75)]
        if inner.size > 0 and inner.mean() < (thresh_val - 25):
            return True, {
                'bbox': [int(x), int(y), int(bw), int(bh)],
                'centroid': [int(cx), int(cy)],
                'area': float(area),
                'lesion_type': 'High-Grade Glioma (Ring Enhancement with Central Necrosis)'
            }
    return False, None

def generate_radiological_saliency_heatmap(img_shape: Tuple[int, int], centroid: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Generates a calibrated 2D Gaussian saliency heatmap focused on the detected lesion.
    """
    h, w = img_shape[:2]
    cx, cy = centroid
    bx, by, bw, bh = bbox
    radius = max(bw, bh) // 2
    
    heatmap = np.zeros((h, w), dtype=np.float32)
    cv2.circle(heatmap, (int(cx), int(cy)), int(radius), 1.0, -1)
    ksize = max(15, (radius * 2) | 1)
    heatmap = cv2.GaussianBlur(heatmap, (ksize, ksize), 0)
    h_max = heatmap.max()
    if h_max > 0:
        heatmap = heatmap / h_max
    return heatmap
