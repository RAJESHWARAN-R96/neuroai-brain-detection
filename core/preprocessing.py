"""
Medical MRI Preprocessing Pipeline for Brain Tumor Detection.
Includes:
- Skull Stripping & Extreme Point Contour Cropping
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Edge-Preserving Bilateral Denoising
- Intensity Normalization and Medical Data Augmentation
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from typing import Tuple, Union, Optional, Dict, Any


def ensure_numpy_image(image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
    """
    Converts input (filepath, PIL Image, or NumPy array) to uint8 RGB NumPy array.
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Could not load image from path: {image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    elif isinstance(image, Image.Image):
        return np.array(image.convert('RGB'))
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def crop_brain_contour(image: np.ndarray, threshold_val: int = 15) -> np.ndarray:
    """
    Finds the largest contour in MRI (the brain tissue boundary) and crops away
    extraneous background borders/skull padding using extreme contour points.
    
    Args:
        image: RGB NumPy array of MRI scan.
        threshold_val: Binary threshold intensity.
        
    Returns:
        Cropped RGB NumPy array.
    """
    img = ensure_numpy_image(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold the image, then perform a series of erosions + dilations
    # to remove any small regions of noise
    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours in thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
    
    # Find the largest contour which represents the brain
    c = max(contours, key=cv2.contourArea)
    
    # If contour is too small (e.g. noise artifact), return original
    if cv2.contourArea(c) < (img.shape[0] * img.shape[1] * 0.05):
        return img
    
    # Determine the most extreme points along the contour
    ext_left = tuple(c[c[:, :, 0].argmin()][0])
    ext_right = tuple(c[c[:, :, 0].argmax()][0])
    ext_top = tuple(c[c[:, :, 1].argmin()][0])
    ext_bot = tuple(c[c[:, :, 1].argmax()][0])
    
    # Crop the image with safety padding
    pad = 4
    x1 = max(0, ext_left[0] - pad)
    x2 = min(img.shape[1], ext_right[0] + pad)
    y1 = max(0, ext_top[1] - pad)
    y2 = min(img.shape[0], ext_bot[1] + pad)
    
    if (x2 - x1) > 10 and (y2 - y1) > 10:
        return img[y1:y2, x1:x2]
    return img


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel of LAB color space
    to accentuate micro-lesions, tumor boundaries, and internal tissue textures without over-amplifying noise.
    
    Args:
        image: RGB NumPy array.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.
        
    Returns:
        Contrast-enhanced RGB NumPy array.
    """
    img = ensure_numpy_image(image)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    
    lab_enhanced = cv2.merge((l_clahe, a, b))
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return rgb_enhanced


def apply_denoising(image: np.ndarray, method: str = 'bilateral', d: int = 5, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    """
    Applies edge-preserving filtering to reduce MRI acquisition noise while preserving tumor boundaries.
    
    Args:
        image: RGB NumPy array.
        method: 'bilateral', 'gaussian', or 'median'.
        d: Diameter of each pixel neighborhood for bilateral filter.
        sigma_color: Filter sigma in color space.
        sigma_space: Filter sigma in coordinate space.
        
    Returns:
        Denoised RGB NumPy array.
    """
    img = ensure_numpy_image(image)
    if method == 'bilateral':
        return cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    elif method == 'gaussian':
        return cv2.GaussianBlur(img, (5, 5), 0)
    elif method == 'median':
        return cv2.medianBlur(img, 3)
    return img


def normalize_intensity(image: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalizes pixel intensity distribution across MRI scan.
    
    Args:
        image: RGB NumPy array.
        method: 'minmax' [0.0, 1.0] or 'zscore' (mean 0, std 1).
        
    Returns:
        Normalized float32 array.
    """
    img = image.astype(np.float32)
    if method == 'minmax':
        min_val = img.min()
        max_val = img.max()
        if max_val - min_val > 1e-6:
            return (img - min_val) / (max_val - min_val)
        return img / 255.0
    elif method == 'zscore':
        mean = img.mean()
        std = img.std()
        if std > 1e-6:
            return (img - mean) / std
        return img - mean
    return img


def preprocess_mri_pipeline(
    image: Union[np.ndarray, Image.Image, str],
    target_size: Tuple[int, int] = (224, 224),
    apply_crop: bool = True,
    apply_clahe_enhancement: bool = True,
    apply_denoise: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: Tuple[int, int] = (8, 8),
    denoise_method: str = 'bilateral',
    return_tensor: bool = True
) -> Dict[str, Any]:
    """
    Full end-to-end medical preprocessing pipeline for brain MRI scans.
    
    Returns a dictionary with:
    - 'original': raw input image (uint8 RGB)
    - 'cropped': after brain contour cropping (uint8 RGB)
    - 'enhanced': after CLAHE and denoising (uint8 RGB)
    - 'resized': enhanced image resized to target_size (uint8 RGB)
    - 'normalized': float32 normalized image [0, 1]
    - 'tensor': PyTorch Tensor [1, 3, H, W] normalized with ImageNet statistics (if return_tensor=True)
    """
    raw_img = ensure_numpy_image(image)
    
    # 1. Brain Contour Cropping (Skull Stripping)
    if apply_crop:
        cropped_img = crop_brain_contour(raw_img)
    else:
        cropped_img = raw_img.copy()
        
    # 2. Denoising
    if apply_denoise:
        denoised_img = apply_denoising(cropped_img, method=denoise_method)
    else:
        denoised_img = cropped_img.copy()
        
    # 3. CLAHE Contrast Enhancement
    if apply_clahe_enhancement:
        enhanced_img = apply_clahe(denoised_img, clip_limit=clahe_clip_limit, tile_grid_size=clahe_grid_size)
    else:
        enhanced_img = denoised_img.copy()
        
    # 4. Resize to target dimension (e.g. 224x224) using high-quality INTER_AREA / INTER_CUBIC
    resized_img = cv2.resize(enhanced_img, target_size, interpolation=cv2.INTER_CUBIC)
    
    # 5. Normalization [0, 1]
    normalized_img = normalize_intensity(resized_img, method='minmax')
    
    result = {
        'original': raw_img,
        'cropped': cropped_img,
        'enhanced': enhanced_img,
        'resized': resized_img,
        'normalized': normalized_img
    }
    
    if return_tensor:
        # Standard PyTorch normalization with ImageNet mean/std for transfer learning
        tensor_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = tensor_transform(resized_img).unsqueeze(0)  # Shape: [1, 3, H, W]
        result['tensor'] = tensor
        
    return result


class MedicalMRIPreprocessor:
    """
    Configurable Medical MRI Preprocessor class that can be integrated into PyTorch datasets or live inference.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        apply_crop: bool = True,
        apply_clahe: bool = True,
        apply_denoise: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        is_training: bool = False
    ):
        self.target_size = target_size
        self.apply_crop = apply_crop
        self.apply_clahe = apply_clahe
        self.apply_denoise = apply_denoise
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.is_training = is_training
        
        # PyTorch normalization
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Augmentations for training
        self.train_augmentations = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            self.normalize
        ])
        
        self.val_transforms = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            self.normalize
        ])

    def __call__(self, image: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
        processed = preprocess_mri_pipeline(
            image,
            target_size=self.target_size,
            apply_crop=self.apply_crop,
            apply_clahe_enhancement=self.apply_clahe,
            apply_denoise=self.apply_denoise,
            clahe_clip_limit=self.clahe_clip_limit,
            clahe_grid_size=self.clahe_grid_size,
            return_tensor=False
        )
        
        resized_img = processed['resized']
        if self.is_training:
            return self.train_augmentations(resized_img)
        else:
            return self.val_transforms(resized_img)
