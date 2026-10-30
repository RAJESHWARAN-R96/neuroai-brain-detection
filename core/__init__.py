"""
Core module for Brain Tumor MRI Detection and Classification using Metaheuristic Optimization.
"""

from .preprocessing import (
    crop_brain_contour,
    apply_clahe,
    apply_denoising,
    normalize_intensity,
    preprocess_mri_pipeline,
    MedicalMRIPreprocessor
)
from .models import (
    EfficientNetBackbone,
    ResNetBackbone,
    DenseNetBackbone,
    AttentionBrainCNN,
    MetaheuristicEnsembleModel,
    build_model
)
from .xai import (
    GradCAM,
    GradCAMPlusPlus,
    generate_gradcam_overlay,
    extract_tumor_bounding_box,
    compute_tumor_metrics
)
from .dataset import (
    BrainMRIDataset,
    get_data_loaders,
    create_sample_benchmark_dataset,
    CLASS_NAMES,
    CLASS_LABELS
)

__all__ = [
    "crop_brain_contour",
    "apply_clahe",
    "apply_denoising",
    "normalize_intensity",
    "preprocess_mri_pipeline",
    "MedicalMRIPreprocessor",
    "EfficientNetBackbone",
    "ResNetBackbone",
    "DenseNetBackbone",
    "AttentionBrainCNN",
    "MetaheuristicEnsembleModel",
    "build_model",
    "GradCAM",
    "GradCAMPlusPlus",
    "generate_gradcam_overlay",
    "extract_tumor_bounding_box",
    "compute_tumor_metrics",
    "BrainMRIDataset",
    "get_data_loaders",
    "create_sample_benchmark_dataset",
    "CLASS_NAMES",
    "CLASS_LABELS"
]
