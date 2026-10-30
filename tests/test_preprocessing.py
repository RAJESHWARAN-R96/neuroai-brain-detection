import sys
import os
import unittest
import numpy as np
import cv2
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.preprocessing import (
    crop_brain_contour, apply_clahe, apply_denoising, normalize_intensity, preprocess_mri_pipeline
)
from core.dataset import create_synthetic_mri_slice

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create synthetic test MRI image (256x256x3)
        self.test_img = create_synthetic_mri_slice('glioma', size=(256, 256), seed=42)

    def test_crop_brain_contour(self):
        # Add black border around image
        padded = cv2.copyMakeBorder(self.test_img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        cropped = crop_brain_contour(padded)
        self.assertIsNotNone(cropped)
        self.assertTrue(cropped.shape[0] < padded.shape[0] or cropped.shape[1] < padded.shape[1])

    def test_apply_clahe(self):
        enhanced = apply_clahe(self.test_img, clip_limit=2.0)
        self.assertEqual(enhanced.shape, self.test_img.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_apply_denoising(self):
        denoised = apply_denoising(self.test_img, method='bilateral')
        self.assertEqual(denoised.shape, self.test_img.shape)

    def test_normalize_intensity(self):
        norm = normalize_intensity(self.test_img, method='minmax')
        self.assertTrue(norm.min() >= 0.0)
        self.assertTrue(norm.max() <= 1.0)

    def test_pipeline(self):
        res = preprocess_mri_pipeline(self.test_img, target_size=(224, 224), return_tensor=True)
        self.assertIn('original', res)
        self.assertIn('cropped', res)
        self.assertIn('enhanced', res)
        self.assertIn('resized', res)
        self.assertIn('tensor', res)
        self.assertEqual(res['tensor'].shape, (1, 3, 224, 224))
        self.assertIsInstance(res['tensor'], torch.Tensor)

if __name__ == '__main__':
    unittest.main()
