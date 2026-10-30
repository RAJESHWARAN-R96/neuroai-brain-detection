"""
Brain Tumor MRI Dataset Loader and Synthetic MRI Benchmark Generator.
Handles:
- Loading official Kaggle Brain Tumor MRI Dataset splits (Training / Testing)
- Label mapping and stratification
- Realistic benchmark MRI slice generation for offline zero-download testing
- DataLoader factory functions with medical augmentation
"""

import os
import glob
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .preprocessing import MedicalMRIPreprocessor, ensure_numpy_image

# 4 Standard Classes
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_LABELS = {name: idx for idx, name in enumerate(CLASS_NAMES)}
LABEL_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

CLASS_DESCRIPTIONS = {
    'glioma': {
        'title': 'Glioma',
        'type': 'Malignant / Invasive',
        'origin': 'Glial cells (Astrocytes, Oligodendrocytes)',
        'description': 'A primary brain tumor originating in the supportive glial cells of the brain and spinal cord. Typically exhibits irregular, infiltrating margins with surrounding edema.',
        'urgency': 'High',
        'color': '#ef4444'  # Red
    },
    'meningioma': {
        'title': 'Meningioma',
        'type': 'Typically Benign / Extra-Axial',
        'origin': 'Meninges (Membranes covering brain and spinal cord)',
        'description': 'A slow-growing tumor arising from the arachnoid cells of the meninges. Characterized by well-demarcated margins, dural tail sign, and homogenous contrast enhancement.',
        'urgency': 'Moderate',
        'color': '#f59e0b'  # Amber
    },
    'pituitary': {
        'title': 'Pituitary Tumor',
        'type': 'Benign Adenoma / Sellar',
        'origin': 'Pituitary gland (Base of brain)',
        'description': 'An abnormal growth in the pituitary gland at the base of the skull (sella turcica). Can cause hormonal imbalances and visual field deficits due to optic chiasm compression.',
        'urgency': 'Moderate',
        'color': '#3b82f6'  # Blue
    },
    'notumor': {
        'title': 'Healthy / No Tumor',
        'type': 'Normal Brain Tissue',
        'origin': 'Intact cerebral parenchyma',
        'description': 'No pathological mass effect, abnormal contrast enhancement, or structural lesion detected. Symmetrical cerebral hemispheres and normal ventricular size.',
        'urgency': 'None',
        'color': '#10b981'  # Green
    }
}


def find_default_dataset_path() -> Optional[str]:
    """
    Auto-discovers the dataset location, checking Kaggle cache and local data/ directories.
    """
    kaggle_cache = os.path.expanduser(r"~\.cache\kagglehub\datasets\masoudnickparvar\brain-tumor-mri-dataset\versions\2")
    if os.path.isdir(kaggle_cache) and os.path.isdir(os.path.join(kaggle_cache, "Training")):
        return kaggle_cache
    
    local_data = os.path.abspath("data")
    if os.path.isdir(local_data) and os.path.isdir(os.path.join(local_data, "Training")):
        return local_data
    if os.path.isdir(local_data) and os.path.isdir(os.path.join(local_data, "training")):
        return local_data
        
    return None


class BrainMRIDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor MRI Images.
    Supports directories structured as:
    root_dir/
      ├── glioma/
      ├── meningioma/
      ├── notumor/ (or no_tumor/)
      └── pituitary/
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        target_size: Tuple[int, int] = (224, 224),
        apply_crop: bool = True,
        apply_clahe: bool = True,
        apply_denoise: bool = True,
        is_training: bool = False
    ):
        self.samples = samples
        self.preprocessor = MedicalMRIPreprocessor(
            target_size=target_size,
            apply_crop=apply_crop,
            apply_clahe=apply_clahe,
            apply_denoise=apply_denoise,
            is_training=is_training
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]
        image = ensure_numpy_image(img_path)
        tensor = self.preprocessor(image)
        return tensor, label, img_path


def load_dataset_samples(dataset_dir: str, split_name: Optional[str] = None) -> List[Tuple[str, int]]:
    """
    Scans directory for MRI image files and associates them with class labels.
    """
    samples = []
    base_dir = dataset_dir
    if split_name:
        # Check Training vs training
        possible_split_dirs = [
            os.path.join(dataset_dir, split_name),
            os.path.join(dataset_dir, split_name.capitalize()),
            os.path.join(dataset_dir, split_name.lower())
        ]
        for p in possible_split_dirs:
            if os.path.isdir(p):
                base_dir = p
                break

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    
    for class_name in CLASS_NAMES:
        class_idx = CLASS_LABELS[class_name]
        
        # Check possible folder variations (e.g., 'notumor', 'no_tumor', 'glioma_tumor')
        candidates = [
            class_name,
            class_name + '_tumor',
            class_name.replace('notumor', 'no_tumor'),
            class_name.capitalize(),
            (class_name + '_tumor').capitalize()
        ]
        
        found = False
        for c in candidates:
            class_path = os.path.join(base_dir, c)
            if os.path.isdir(class_path):
                found = True
                for root, _, files in os.walk(class_path):
                    for f in files:
                        if f.lower().endswith(valid_extensions):
                            samples.append((os.path.join(root, f), class_idx))
                break
                
        if not found:
            # Look directly for glob pattern in base_dir
            for f in glob.glob(os.path.join(base_dir, f"*{class_name}*")):
                if os.path.isfile(f) and f.lower().endswith(valid_extensions):
                    samples.append((f, class_idx))
                    
    return samples


def get_data_loaders(
    dataset_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    target_size: Tuple[int, int] = (224, 224),
    val_split: float = 0.15,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Creates train, validation, and test DataLoaders.
    """
    if dataset_dir is None:
        dataset_dir = find_default_dataset_path()
        if dataset_dir is None:
            raise FileNotFoundError("Could not find dataset path. Please specify dataset_dir.")

    # Check for separate Training and Testing folders
    has_train_test = (
        (os.path.isdir(os.path.join(dataset_dir, "Training")) or os.path.isdir(os.path.join(dataset_dir, "training"))) and
        (os.path.isdir(os.path.join(dataset_dir, "Testing")) or os.path.isdir(os.path.join(dataset_dir, "testing")))
    )

    if has_train_test:
        train_val_samples = load_dataset_samples(dataset_dir, "Training")
        test_samples = load_dataset_samples(dataset_dir, "Testing")
        
        # Split train_val into train and validation
        random.seed(seed)
        random.shuffle(train_val_samples)
        
        val_size = int(len(train_val_samples) * val_split)
        val_samples = train_val_samples[:val_size]
        train_samples = train_val_samples[val_size:]
    else:
        all_samples = load_dataset_samples(dataset_dir)
        random.seed(seed)
        random.shuffle(all_samples)
        
        total = len(all_samples)
        test_size = int(total * 0.15)
        val_size = int(total * val_split)
        
        test_samples = all_samples[:test_size]
        val_samples = all_samples[test_size:test_size + val_size]
        train_samples = all_samples[test_size + val_size:]

    train_dataset = BrainMRIDataset(train_samples, target_size=target_size, is_training=True)
    val_dataset = BrainMRIDataset(val_samples, target_size=target_size, is_training=False)
    test_dataset = BrainMRIDataset(test_samples, target_size=target_size, is_training=False)

    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False),
        'train_samples': train_samples,
        'val_samples': val_samples,
        'test_samples': test_samples
    }
    
    return loaders


def create_synthetic_mri_slice(class_name: str, size: Tuple[int, int] = (256, 256), seed: Optional[int] = None) -> np.ndarray:
    """
    Generates a realistic anatomical brain MRI slice with:
    - Skull boundary and scalp tissue
    - Cerebral cortex hemispheres and longitudinal fissure
    - Lateral ventricles (butterfly shape)
    - Pathologically realistic tumor lesions corresponding to class_name:
      - Glioma: Infiltrating, irregular, hyper-intense core with fuzzy perimeter & edema.
      - Meningioma: Dural-attached, well-circumscribed, round/oval dense mass with dural tail.
      - Pituitary: Sellar mass situated precisely at base of brain near sella turcica.
      - Healthy/NoTumor: Normal symmetrical anatomy with no mass effect.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    h, w = size
    img = np.zeros((h, w), dtype=np.float32)
    
    center_x, center_y = w // 2, h // 2
    r_x, r_y = int(w * 0.40), int(h * 0.44)
    
    # 1. Outer Skull / Scalp
    y, x = np.ogrid[:h, :w]
    dist_from_center = ((x - center_x) ** 2) / (r_x ** 2) + ((y - center_y) ** 2) / (r_y ** 2)
    
    skull_mask = (dist_from_center <= 1.0)
    skull_bone = (dist_from_center <= 1.0) & (dist_from_center >= 0.92)
    img[skull_mask] = 40.0 + np.random.normal(0, 3, img[skull_mask].shape)
    img[skull_bone] = 180.0 + np.random.normal(0, 10, img[skull_bone].shape)
    
    # 2. Brain Parenchyma (Cortex & White Matter)
    brain_mask = (dist_from_center < 0.88)
    img[brain_mask] = 110.0 + np.random.normal(0, 8, img[brain_mask].shape)
    
    # Longitudinal cerebral fissure (midline)
    midline = (np.abs(x - center_x) <= 1) & brain_mask
    img[midline] = 30.0 + np.random.normal(0, 4, img[midline].shape)
    
    # 3. Lateral Ventricles (Butterfly-shaped CSF spaces)
    v_lx, v_rx = center_x - int(w * 0.08), center_x + int(w * 0.08)
    v_y = center_y - int(h * 0.02)
    v_rx_radius = int(w * 0.04)
    v_ry_radius = int(h * 0.12)
    
    v_left_dist = ((x - v_lx) ** 2) / (v_rx_radius ** 2) + ((y - v_y) ** 2) / (v_ry_radius ** 2)
    v_right_dist = ((x - v_rx) ** 2) / (v_rx_radius ** 2) + ((y - v_y) ** 2) / (v_ry_radius ** 2)
    
    ventricle_mask = ((v_left_dist <= 1.0) | (v_right_dist <= 1.0)) & brain_mask
    img[ventricle_mask] = 25.0 + np.random.normal(0, 5, img[ventricle_mask].shape)
    
    # 4. Pathological Lesion Insertion
    if class_name == 'glioma':
        # Intra-axial irregular invasive mass in hemisphere
        tx = center_x + random.choice([-1, 1]) * random.randint(int(w * 0.12), int(w * 0.22))
        ty = center_y + random.randint(-int(h * 0.15), int(h * 0.15))
        rad = random.randint(int(w * 0.08), int(w * 0.14))
        
        # Edema region
        t_dist = np.sqrt((x - tx) ** 2 + (y - ty) ** 2)
        edema_mask = (t_dist <= rad * 1.5) & brain_mask
        img[edema_mask] = np.clip(img[edema_mask] + 45.0 + np.random.normal(0, 10, img[edema_mask].shape), 0, 255)
        
        # Hyperintense necrotic / active tumor core with irregular perimeter
        core_mask = (t_dist <= rad) & brain_mask
        img[core_mask] = np.clip(img[core_mask] + 95.0 + np.random.normal(0, 15, img[core_mask].shape), 0, 255)
        
    elif class_name == 'meningioma':
        # Extra-axial circumscribed mass attached to periphery / dura
        angle = random.uniform(0.2, 2.8) if random.random() > 0.5 else random.uniform(3.4, 6.0)
        tx = int(center_x + r_x * 0.72 * np.cos(angle))
        ty = int(center_y + r_y * 0.72 * np.sin(angle))
        rad = random.randint(int(w * 0.07), int(w * 0.11))
        
        t_dist = ((x - tx) ** 2) / (rad ** 2) + ((y - ty) ** 2) / ((rad * 0.85) ** 2)
        tumor_mask = (t_dist <= 1.0) & skull_mask
        img[tumor_mask] = np.clip(210.0 + np.random.normal(0, 10, img[tumor_mask].shape), 0, 255)
        
    elif class_name == 'pituitary':
        # Sellar mass at the base of the brain near midline
        tx = center_x + random.randint(-4, 4)
        ty = center_y + int(r_y * 0.52)
        rad_x = random.randint(int(w * 0.06), int(w * 0.09))
        rad_y = random.randint(int(h * 0.05), int(h * 0.08))
        
        t_dist = ((x - tx) ** 2) / (rad_x ** 2) + ((y - ty) ** 2) / (rad_y ** 2)
        tumor_mask = (t_dist <= 1.0) & skull_mask
        img[tumor_mask] = np.clip(225.0 + np.random.normal(0, 8, img[tumor_mask].shape), 0, 255)
        
    elif class_name == 'notumor':
        # Normal scan - no mass effect, symmetrical parenchyma
        pass

    # Final post-processing (Gaussian smoothing + noise)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    # Convert to 3-channel RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return rgb


def create_sample_benchmark_dataset(
    output_dir: str = "data/sample_mris",
    samples_per_class: int = 5,
    use_kaggle_if_available: bool = True
) -> Dict[str, List[str]]:
    """
    Populates sample MRI repository with realistic samples from Kaggle dataset
    (or synthetic generation if Kaggle dataset is not present).
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_paths = {c: [] for c in CLASS_NAMES}
    
    kaggle_dir = find_default_dataset_path()
    
    for c in CLASS_NAMES:
        class_dir = os.path.join(output_dir, c)
        os.makedirs(class_dir, exist_ok=True)
        
        # Check if Kaggle has real images to copy
        copied = False
        if use_kaggle_if_available and kaggle_dir:
            test_samples = load_dataset_samples(kaggle_dir, "Testing")
            class_test_samples = [s[0] for s in test_samples if s[1] == CLASS_LABELS[c]]
            
            if len(class_test_samples) >= samples_per_class:
                for idx, src_path in enumerate(class_test_samples[:samples_per_class]):
                    dest_path = os.path.join(class_dir, f"{c}_sample_{idx + 1}.jpg")
                    shutil.copy2(src_path, dest_path)
                    sample_paths[c].append(dest_path)
                copied = True
                
        if not copied:
            # Generate high-fidelity synthetic benchmark slices
            for idx in range(samples_per_class):
                slice_img = create_synthetic_mri_slice(c, size=(256, 256), seed=idx * 100 + CLASS_LABELS[c])
                dest_path = os.path.join(class_dir, f"{c}_sample_{idx + 1}.jpg")
                cv2.imwrite(dest_path, cv2.cvtColor(slice_img, cv2.COLOR_RGB2BGR))
                sample_paths[c].append(dest_path)
                
    return sample_paths
