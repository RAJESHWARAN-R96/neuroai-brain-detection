import sys
import os
import unittest
import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.report_generator import generate_pdf_report
from core.dataset import create_synthetic_mri_slice

class TestReportGenerator(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp_report_test')
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        # Create test images
        img = create_synthetic_mri_slice('glioma', size=(224, 224), seed=42)
        self.orig_path = os.path.join(self.tmp_dir, 'test_orig.jpg')
        self.prep_path = os.path.join(self.tmp_dir, 'test_prep.jpg')
        self.grad_path = os.path.join(self.tmp_dir, 'test_grad.jpg')
        self.pdf_path = os.path.join(self.tmp_dir, 'test_report.pdf')
        
        cv2.imwrite(self.orig_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(self.prep_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(self.grad_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def tearDown(self):
        # Cleanup
        for f in [self.orig_path, self.prep_path, self.grad_path, self.pdf_path]:
            if os.path.isfile(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
        if os.path.isdir(self.tmp_dir):
            try:
                os.rmdir(self.tmp_dir)
            except Exception:
                pass

    def test_pdf_generation(self):
        patient_info = {'patient_id': 'TEST-1234', 'patient_name': 'Test Subject'}
        diagnosis_results = {
            'predicted_class': 'glioma',
            'confidence': 96.5,
            'probabilities': {'glioma': 96.5, 'meningioma': 2.0, 'pituitary': 1.0, 'notumor': 0.5},
            'tumor_metrics': {'tumor_area_percentage': 8.45, 'anatomical_quadrant': 'Anterior-Right Cerebral Hemisphere', 'severity_index': 'High Mass Effect'}
        }
        
        out = generate_pdf_report(
            patient_info=patient_info,
            diagnosis_results=diagnosis_results,
            original_img_path=self.orig_path,
            preprocessed_img_path=self.prep_path,
            gradcam_img_path=self.grad_path,
            output_pdf_path=self.pdf_path
        )
        
        self.assertTrue(os.path.isfile(self.pdf_path))
        self.assertTrue(os.path.getsize(self.pdf_path) > 1000)

if __name__ == '__main__':
    unittest.main()
