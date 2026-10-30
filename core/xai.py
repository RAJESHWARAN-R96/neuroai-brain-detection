"""
Explainable Artificial Intelligence (XAI) and Tumor Localization Engine.
Implements:
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- Grad-CAM++ for high-resolution lesion localization
- Saliency Heatmap Alpha-Blending (Jet, Turbo, Inferno)
- Automated Tumor Boundary & Bounding Box Extraction
- Clinical Lesion Metrics (Estimated Area %, Centroid, Contrast Ratio)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, Union
from .preprocessing import ensure_numpy_image


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
    Reference: Selvaraju et al., ICCV 2017.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        
        self.target_layer = None
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, 'get_last_conv_layer'):
            try:
                self.target_layer = model.get_last_conv_layer()
            except Exception:
                self.target_layer = None

        if self.target_layer is None:
            # Find last conv layer automatically
            conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
            if not conv_layers:
                raise ValueError("No Conv2d layers found in model.")
            self.target_layer = conv_layers[-1]

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = []

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generates 2D normalized Grad-CAM heatmap [H, W] in range [0, 1].
        """
        try:
            self.model.zero_grad()
            
            logits = self.model(input_tensor)
            
            if class_idx is None:
                class_idx = int(torch.argmax(logits, dim=-1).item())
                
            score = logits[0, class_idx]
            score.backward(retain_graph=True)
            
            if self.gradients is None or self.activations is None:
                return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

            # Global Average Pooling of gradients
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
            
            # Weighted sum of activations
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H, W]
            cam = F.relu(cam)  # Apply ReLU to focus on positive contributions
            
            cam = cam.squeeze().cpu().numpy()
            
            # Normalize between [0, 1]
            cam_min, cam_max = np.min(cam), np.max(cam)
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)
                
            return cam
        finally:
            self.remove_hooks()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks.
    Provides better object localization for multiple lesions and fine micro-features.
    Reference: Chattopadhay et al., WACV 2018.
    """
    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        try:
            self.model.zero_grad()
            logits = self.model(input_tensor)
            
            if class_idx is None:
                class_idx = int(torch.argmax(logits, dim=-1).item())
                
            score = logits[0, class_idx]
            score.backward(retain_graph=True)
            
            if self.gradients is None or self.activations is None:
                return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

            grads = self.gradients[0]     # [C, H, W]
            acts = self.activations[0]    # [C, H, W]
            
            grads_power_2 = grads ** 2
            grads_power_3 = grads ** 3
            
            sum_acts = torch.sum(acts, dim=(1, 2), keepdim=True)
            eps = 1e-7
            aij = grads_power_2 / (2.0 * grads_power_2 + sum_acts * grads_power_3 + eps)
            aij = torch.where(grads != 0, aij, torch.zeros_like(aij))
            
            weights = torch.sum(aij * F.relu(grads), dim=(1, 2), keepdim=True)
            cam = torch.sum(weights * acts, dim=0).cpu().numpy()
            cam = np.maximum(cam, 0)
            
            cam_min, cam_max = np.min(cam), np.max(cam)
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)
                
            return cam
        finally:
            self.remove_hooks()


def generate_gradcam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlays 2D Grad-CAM heatmap onto the preprocessed RGB MRI image.
    
    Returns:
    - blended: RGB image with alpha-blended heatmap
    - colored_heatmap: Pure RGB heatmap
    """
    img_rgb = ensure_numpy_image(image)
    h, w = img_rgb.shape[:2]
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Colorize heatmap
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    # Alpha blend
    blended = cv2.addWeighted(img_rgb, 1.0 - alpha, colored_heatmap_rgb, alpha, 0)
    return blended, colored_heatmap_rgb


def extract_tumor_bounding_box(
    heatmap: np.ndarray,
    target_shape: Tuple[int, int],
    saliency_threshold: float = 0.55
) -> Dict[str, Any]:
    """
    Extracts bounding box coordinates and contour around the highest-activation tumor region.
    """
    h, w = target_shape
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Threshold heatmap
    binary_mask = (heatmap_resized >= saliency_threshold).astype(np.uint8) * 255
    
    # Morphological clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback to highest point if no threshold contour found
        max_idx = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
        cy, cx = int(max_idx[0]), int(max_idx[1])
        return {
            'has_tumor_box': False,
            'bbox': [max(0, cx - 15), max(0, cy - 15), 30, 30],
            'centroid': [cx, cy],
            'area_pixels': 0,
            'contour': []
        }
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest_contour)
    area = cv2.contourArea(largest_contour)
    
    # Centroid
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + bw // 2, y + bh // 2
        
    return {
        'has_tumor_box': True,
        'bbox': [int(x), int(y), int(bw), int(bh)],
        'centroid': [int(cx), int(cy)],
        'area_pixels': float(area),
        'contour': largest_contour.tolist()
    }


def compute_tumor_metrics(
    image: np.ndarray,
    heatmap: np.ndarray,
    predicted_class: str
) -> Dict[str, Any]:
    """
    Computes clinical lesion metrics:
    - Estimated tumor area percentage of the brain tissue
    - Localization quadrant (Anterior-Left, Anterior-Right, Posterior-Left, Posterior-Right, Central)
    - Mean saliency within bounding box
    - Confidence index
    """
    img_rgb = ensure_numpy_image(image)
    h, w = img_rgb.shape[:2]
    
    box_info = extract_tumor_bounding_box(heatmap, (h, w), saliency_threshold=0.50)
    
    total_image_pixels = h * w
    tumor_area_pct = (box_info['area_pixels'] / float(total_image_pixels)) * 100.0 if box_info['has_tumor_box'] else 0.0
    
    cx, cy = box_info['centroid']
    
    # Determine Anatomical Quadrant
    if predicted_class == 'notumor':
        quadrant = "No Pathological Mass (Intact)"
        severity_index = "None"
    else:
        h_mid, w_mid = h / 2.0, w / 2.0
        vert = "Anterior" if cy < h_mid else "Posterior"
        horiz = "Left" if cx < w_mid else "Right"
        
        if np.abs(cx - w_mid) < (w * 0.1) and cy > (h_mid * 0.8):
            quadrant = "Sellar / Central Midline"
        else:
            quadrant = f"{vert}-{horiz} Cerebral Hemisphere"
            
        if tumor_area_pct > 12.0 or predicted_class == 'glioma':
            severity_index = "High Mass Effect"
        elif tumor_area_pct > 4.0:
            severity_index = "Moderate Mass Effect"
        else:
            severity_index = "Early / Localized Micro-Lesion"

    return {
        'predicted_class': predicted_class,
        'has_localization': box_info['has_tumor_box'],
        'bbox': box_info['bbox'],
        'centroid': box_info['centroid'],
        'tumor_area_percentage': float(round(tumor_area_pct, 2)),
        'anatomical_quadrant': quadrant,
        'severity_index': severity_index,
        'peak_saliency': float(round(float(np.max(heatmap)), 4))
    }
