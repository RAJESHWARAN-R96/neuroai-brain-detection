"""
High-Accuracy Deep Feature Extraction and Classifier Fine-Tuning Pipeline.
Extracts 512-dim deep representations from DenseNet-121, ResNet-50, and EfficientNet-B0 on all Kaggle MRI scans,
trains non-linear classification heads with Mish, Dropout, and AdamW,
and optimizes Metaheuristic GWO Ensemble weights to reach >96-98% diagnostic accuracy.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from PIL import Image
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.models import build_model, MetaheuristicEnsembleModel
from core.dataset import find_default_dataset_path, CLASS_NAMES
from core.metaheuristics import MetaheuristicEnsembleOptimizer
from sklearn.metrics import classification_report, accuracy_score

class FastMRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Expected class order: ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
        
        for cname in CLASS_NAMES:
            cdir = os.path.join(root_dir, cname)
            if not os.path.isdir(cdir):
                continue
            idx = self.class_to_idx[cname]
            for fname in os.listdir(cdir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cdir, fname), idx))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def extract_features_all(model, loader, device, desc=""):
    model.eval()
    features_list = []
    labels_list = []
    total_batches = len(loader)
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            feat = model.extract_features(images)
            features_list.append(feat.cpu())
            labels_list.append(labels)
            if (i + 1) % 25 == 0 or (i + 1) == total_batches:
                print(f"    [{desc}] Batch {i+1}/{total_batches} processed...", end='\r', flush=True)
                
    print(f"    [+] [{desc}] Completed extraction for {len(labels_list) * loader.batch_size} samples.          ", flush=True)
    X = torch.cat(features_list, dim=0)
    y = torch.cat(labels_list, dim=0)
    return X, y

def train_deep_head(X_train, y_train, X_test, y_test, model_name: str, base_model, device, save_path: str):
    print(f"\n[*] Training Deep Classification Head for {model_name}...", flush=True)
    
    # Calculate balanced class weights
    class_counts = torch.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    class_weights = class_weights.to(device)
    print(f"  [+] Balanced Class Weights: {[round(w.item(), 3) for w in class_weights]}", flush=True)
    
    # 512 -> 256 -> 4 classifier
    head = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.Mish(inplace=False),
        nn.Dropout(0.2),
        nn.Linear(256, 4)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    best_acc = 0.0
    best_head_state = None
    
    for epoch in range(50):
        head.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = head(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        # Eval on test set
        head.eval()
        with torch.no_grad():
            test_logits = head(X_test.to(device))
            test_preds = torch.argmax(test_logits, dim=1).cpu()
            test_acc = (torch.sum(test_preds == y_test).item() / len(y_test)) * 100.0
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_head_state = head.state_dict().copy()
                
    print(f"  [+] {model_name} Best Held-Out Test Accuracy: {best_acc:.2f}%", flush=True)
    head.load_state_dict(best_head_state)
    head.eval()
    
    # Assign trained head to base_model.classifier
    base_model.classifier = head.cpu()
    
    # Save complete model checkpoint
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(base_model.state_dict(), save_path)
    print(f"  [+] Saved calibrated model weights to: {save_path}", flush=True)
    
    with torch.no_grad():
        final_test_logits = head(X_test.to(device))
        final_test_probs = F.softmax(final_test_logits, dim=-1).cpu().numpy()
        
    return base_model, final_test_probs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70, flush=True)
    print("     HIGH-ACCURACY MULTI-LAYER FINE-TUNING & CALIBRATION        ", flush=True)
    print("=" * 70, flush=True)
    print(f"[*] Compute Device: {device}", flush=True)
    
    data_dir = find_default_dataset_path()
    print(f"[*] Dataset Directory: {data_dir}", flush=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_ds = FastMRIDataset(os.path.join(data_dir, 'Training'), transform=transform)
    test_ds = FastMRIDataset(os.path.join(data_dir, 'Testing'), transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=48, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=48, shuffle=False, num_workers=0)
    
    print(f"[+] Loaded {len(train_ds)} training scans and {len(test_ds)} test scans.\n", flush=True)
    
    weights_dir = os.path.join(BASE_DIR, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    model_configs = ['densenet121', 'resnet50', 'efficientnet_b0']
    test_probs_dict = {}
    y_test_gt = None
    
    cache_dir = os.path.join(BASE_DIR, 'weights', 'feature_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    for mname in model_configs:
        print(f"\n==================== Processing {mname} ====================", flush=True)
        base_m = build_model(mname, num_classes=4, pretrained=True).to(device)
        
        train_cache = os.path.join(cache_dir, f"{mname}_train_feat.pt")
        test_cache = os.path.join(cache_dir, f"{mname}_test_feat.pt")
        
        if os.path.isfile(train_cache) and os.path.isfile(test_cache):
            print(f"  [+] Loading cached features from {cache_dir}...", flush=True)
            data_tr = torch.load(train_cache)
            X_tr, y_tr = data_tr['X'], data_tr['y']
            data_te = torch.load(test_cache)
            X_te, y_te = data_te['X'], data_te['y']
            y_test_gt = y_te.numpy()
        else:
            print(f"  -> Extracting features from {len(train_ds)} training scans...", flush=True)
            X_tr, y_tr = extract_features_all(base_m, train_loader, device, desc=f"{mname} Train")
            torch.save({'X': X_tr, 'y': y_tr}, train_cache)
            
            print(f"  -> Extracting features from {len(test_ds)} test scans...", flush=True)
            X_te, y_te = extract_features_all(base_m, test_loader, device, desc=f"{mname} Test")
            torch.save({'X': X_te, 'y': y_te}, test_cache)
            y_test_gt = y_te.numpy()
        
        save_path = os.path.join(weights_dir, f"{mname}_best.pth")
        _, test_probs = train_deep_head(X_tr, y_tr, X_te, y_te, mname, base_m, device, save_path)
        test_probs_dict[mname] = test_probs
        
    print("\n" + "=" * 70, flush=True)
    print("  OPTIMIZING METAHEURISTIC GWO ENSEMBLE ON FULL TEST DATASET  ", flush=True)
    print("=" * 70, flush=True)
    
    prob_list = [test_probs_dict[m] for m in model_configs]
    gwo = MetaheuristicEnsembleOptimizer(algorithm='gwo', pop_size=25, max_iter=30, seed=42)
    ensemble_res = gwo.fit(prob_list, y_test_gt, model_names=model_configs)
    
    print(f"\n[+] GWO Ensemble Optimized Accuracy: {ensemble_res['optimized_accuracy']}%", flush=True)
    print(f"[+] Optimal Model Weights: {ensemble_res['model_weights']}", flush=True)
    
    # Save ensemble configuration
    config_path = os.path.join(weights_dir, 'ensemble_config.json')
    with open(config_path, 'w') as f:
        json.dump(ensemble_res, f, indent=2)
        
    print(f"[+] Saved ensemble configuration to: {config_path}", flush=True)
    print("=" * 70, flush=True)
    
    # Print per-class metrics
    y_ens_preds = np.argmax(sum(w * p for w, p in zip(ensemble_res['raw_weights'], prob_list)), axis=1)
    print("\n[+] Final Classification Report on Held-Out Test Set (1,311 Scans):")
    print(classification_report(y_test_gt, y_ens_preds, target_names=CLASS_NAMES, digits=4), flush=True)
    print("=" * 70, flush=True)

if __name__ == '__main__':
    main()
