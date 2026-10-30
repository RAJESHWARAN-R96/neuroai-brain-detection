"""
Fast High-Accuracy Pre-trained Feature Extractor and Fine-Tuning Engine.
Calibrates and saves trained weights for EfficientNet, ResNet, DenseNet, AttentionBrainCNN,
and optimizes Metaheuristic GWO Ensemble weights on the Kaggle Brain Tumor MRI dataset.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.models import build_model, MetaheuristicEnsembleModel
from core.dataset import get_data_loaders, find_default_dataset_path, CLASS_NAMES, CLASS_LABELS
from core.trainer import get_device
from core.metaheuristics import MetaheuristicEnsembleOptimizer

def extract_features_from_loader(model, loader, device, max_samples=600):
    model.eval()
    features_list = []
    labels_list = []
    count = 0
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            if hasattr(model, 'extract_features'):
                feat = model.extract_features(images)
            else:
                feat = model(images)
            features_list.append(feat.cpu())
            labels_list.append(labels)
            count += labels.size(0)
            print(f"    Extracted {count}/{max_samples} samples...", end='\r', flush=True)
            if count >= max_samples:
                break
                
    print(f"    [+] Extracted {count} samples successfully.          ", flush=True)
    X = torch.cat(features_list, dim=0)
    y = torch.cat(labels_list, dim=0)
    return X, y

def train_head_and_save(model_name: str, train_loader, val_loader, device, save_path: str):
    print(f"\n[*] Training & Calibrating {model_name} on MRI Dataset...", flush=True)
    model = build_model(model_name, num_classes=4, pretrained=True).to(device)
    model.eval()
    
    print("  -> Extracting training features...", flush=True)
    X_train, y_train = extract_features_from_loader(model, train_loader, device, max_samples=800)
    print("  -> Extracting validation features...", flush=True)
    X_val, y_val = extract_features_from_loader(model, val_loader, device, max_samples=250)
    
    feat_dim = X_train.shape[1]
    
    head = model.classifier.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    best_acc = 0.0
    for epoch in range(25):
        head.train()
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = head(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
        head.eval()
        with torch.no_grad():
            val_out = head(X_val.to(device))
            val_preds = torch.argmax(val_out, dim=1).cpu()
            val_acc = (torch.sum(val_preds == y_val).item() / len(y_val)) * 100.0
            if val_acc > best_acc:
                best_acc = val_acc
                
    print(f"  [+] {model_name} Best Validation Accuracy: {best_acc:.2f}%", flush=True)
    
    # Save complete model state dict
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    # Evaluate final probabilities
    head.eval()
    with torch.no_grad():
        val_logits = head(X_val.to(device))
        val_probs = F.softmax(val_logits, dim=-1).cpu().numpy()
        
    return model, val_probs, y_val.numpy()

def main():
    device = get_device()
    print("=" * 70, flush=True)
    print("      FAST CALIBRATION AND OPTIMIZATION OF BRAIN MRI MODELS     ", flush=True)
    print("=" * 70, flush=True)
    print(f"[*] Device: {device}", flush=True)
    
    data_dir = find_default_dataset_path()
    print(f"[*] Dataset: {data_dir}", flush=True)
    
    loaders = get_data_loaders(data_dir, batch_size=32, num_workers=0)
    
    weights_dir = os.path.join(BASE_DIR, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    model_names = ['efficientnet_b0', 'densenet121', 'resnet50', 'attention_cnn']
    trained_models = {}
    val_probs_dict = {}
    y_val_ground_truth = None
    
    for mname in model_names:
        save_path = os.path.join(weights_dir, f"{mname}_best.pth")
        m, probs, y_v = train_head_and_save(mname, loaders['train'], loaders['val'], device, save_path)
        trained_models[mname] = m
        val_probs_dict[mname] = probs
        y_val_ground_truth = y_v
        
    print("\n[*] Optimizing Metaheuristic Ensemble Blending Weights with GWO...", flush=True)
    prob_list = [val_probs_dict[m] for m in model_names]
    gwo = MetaheuristicEnsembleOptimizer(algorithm='gwo', pop_size=20, max_iter=25, seed=42)
    res = gwo.fit(prob_list, y_val_ground_truth, model_names=model_names)
    
    print(f"\n[+] Metaheuristic GWO Ensemble Accuracy: {res['optimized_accuracy']}% (Gain: +{res['accuracy_gain']}%)", flush=True)
    print(f"[+] Optimal Model Weights: {res['model_weights']}", flush=True)
    
    with open(os.path.join(weights_dir, 'ensemble_config.json'), 'w') as f:
        json.dump(res, f, indent=2)
        
    print("\n[+] Done! Checkpoints and configuration saved in weights/.", flush=True)
    print("=" * 70, flush=True)

if __name__ == '__main__':
    main()
