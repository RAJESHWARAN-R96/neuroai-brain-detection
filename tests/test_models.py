import sys
import os
import unittest
import torch
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.models import build_model, AttentionBrainCNN, MetaheuristicEnsembleModel
from core.xai import GradCAM, GradCAMPlusPlus, compute_tumor_metrics

class TestModelsAndXAI(unittest.TestCase):

    def setUp(self):
        self.dummy_input = torch.randn(1, 3, 224, 224)

    def test_attention_cnn_forward_and_gradcam(self):
        model = AttentionBrainCNN(num_classes=4, feature_dim=128)
        model.eval()
        
        out = model(self.dummy_input)
        self.assertEqual(out.shape, (1, 4))
        
        feat = model.extract_features(self.dummy_input)
        self.assertEqual(feat.shape, (1, 128))
        
        cam = GradCAM(model)
        heatmap = cam.generate(self.dummy_input, class_idx=0)
        self.assertEqual(len(heatmap.shape), 2)
        self.assertTrue(heatmap.shape[0] > 0 and heatmap.shape[1] > 0)
        self.assertTrue(heatmap.max() <= 1.0)
        self.assertTrue(heatmap.min() >= 0.0)

    def test_gradcam_plus_plus(self):
        model = AttentionBrainCNN(num_classes=4, feature_dim=128)
        model.eval()
        
        cam_pp = GradCAMPlusPlus(model)
        heatmap = cam_pp.generate(self.dummy_input, class_idx=1)
        self.assertEqual(len(heatmap.shape), 2)
        self.assertTrue(heatmap.shape[0] > 0 and heatmap.shape[1] > 0)

    def test_tumor_metrics(self):
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        dummy_cam = np.zeros((224, 224), dtype=np.float32)
        dummy_cam[80:120, 80:120] = 0.90 # Simulate focal lesion
        
        metrics = compute_tumor_metrics(dummy_img, dummy_cam, 'glioma')
        self.assertIn('tumor_area_percentage', metrics)
        self.assertIn('anatomical_quadrant', metrics)
        self.assertTrue(metrics['has_localization'])

    def test_ensemble_model(self):
        m1 = AttentionBrainCNN(num_classes=4, feature_dim=64).eval()
        m2 = AttentionBrainCNN(num_classes=4, feature_dim=64).eval()
        ensemble = MetaheuristicEnsembleModel([m1, m2], weights=[0.6, 0.4]).eval()
        
        out = ensemble(self.dummy_input)
        self.assertEqual(out.shape, (1, 4))

if __name__ == '__main__':
    unittest.main()
