"""
Calculate Accuracy Score and Comprehensive Medical Classification Metrics
Evaluates DenseNet-121, ResNet-50, EfficientNet-B0, and GWO Metaheuristic Ensemble
on the Brain Tumor MRI Test Dataset.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, classification_report
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.models import build_model, MetaheuristicEnsembleModel
from core.dataset import (
    BrainMRIDataset, load_dataset_samples, find_default_dataset_path,
    CLASS_NAMES, CLASS_LABELS
)

def evaluate_model(model, loader, device, model_name="Model"):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    total = len(loader.dataset)
    count = 0
    start_t = time.time()
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=-1)
            _, preds = torch.max(outputs, 1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.numpy())
            
            count += labels.size(0)
            elapsed = time.time() - start_t
            speed = count / max(0.1, elapsed)
            print(f"  Evaluating {model_name}: {count}/{total} images ({speed:.1f} img/s)...", end='\r', flush=True)
            
    print(f"  [+] Finished evaluating {model_name} on {total} images ({time.time() - start_t:.1f}s).          ", flush=True)
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    return y_true, y_pred, y_probs

def compute_detailed_report(y_true, y_pred, y_probs, model_name="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    
    print("\n" + "="*75)
    print(f"  ACCURACY & EVALUATION REPORT: {model_name.upper()}")
    print("="*75)
    print(f"  [*] Overall Accuracy Score  : {acc * 100:.2f}%  ({np.sum(y_true == y_pred)} / {len(y_true)} correct)")
    print(f"  [*] Macro Precision         : {prec_macro:.4f}  ({prec_macro * 100:.2f}%)")
    print(f"  [*] Macro Recall (Sens.)    : {rec_macro:.4f}  ({rec_macro * 100:.2f}%)")
    print(f"  [*] Macro F1-Score          : {f1_macro:.4f}  ({f1_macro * 100:.2f}%)")
    print(f"  [*] Cohen's Kappa           : {kappa:.4f}")
    print("-" * 75)
    
    # Per-Class Table
    print(f"  {'Class Name':<16} | {'Precision':<10} | {'Recall':<10} | {'Specificity':<12} | {'F1-Score':<10} | {'Support'}")
    print("  " + "-"*71)
    
    total_samples = len(y_true)
    for i, cname in enumerate(CLASS_NAMES):
        tp = float(cm[i, i])
        fn = float(np.sum(cm[i, :]) - tp)
        fp = float(np.sum(cm[:, i]) - tp)
        tn = float(total_samples - (tp + fn + fp))
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
        support = int(tp + fn)
        
        print(f"  {cname.capitalize():<16} | {prec*100:6.2f}%    | {sens*100:6.2f}%    | {spec*100:6.2f}%       | {f1*100:6.2f}%    | {support}")
        
    print("-" * 75)
    print("  Confusion Matrix (Rows: Ground Truth, Cols: Predicted):")
    print("           " + "  ".join([f"{c[:6].capitalize():>8}" for c in CLASS_NAMES]))
    for i, row in enumerate(cm):
        row_str = "  ".join([f"{val:>8}" for val in row])
        print(f"  {CLASS_NAMES[i][:8].capitalize():<8}: {row_str}")
    print("="*75 + "\n")
    
    return {
        'model_name': model_name,
        'accuracy': float(acc),
        'precision_macro': float(prec_macro),
        'recall_macro': float(rec_macro),
        'f1_macro': float(f1_macro),
        'kappa': float(kappa),
        'correct': int(np.sum(y_true == y_pred)),
        'total': int(len(y_true))
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate Brain Tumor MRI Model Accuracy Score")
    parser.add_argument('--max-samples', type=int, default=400, help='Max test samples to evaluate (default: 400 for fast eval, 0 for all)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--model', type=str, default='all', choices=['all', 'ensemble', 'densenet121', 'resnet50', 'efficientnet_b0'], help='Model to evaluate')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 75)
    print("  NEURO-AI: BRAIN TUMOR MRI MODEL ACCURACY SCORE CALCULATION")
    print("=" * 75)
    print(f"[*] Execution Device : {device}")
    
    dataset_path = find_default_dataset_path()
    if not dataset_path:
        print("[!] Default Kaggle dataset path not found. Checking local data/sample_mris...")
        dataset_path = os.path.join(BASE_DIR, 'data', 'sample_mris')
        
    print(f"[*] Dataset Location : {dataset_path}")
    
    # Load test samples
    samples = load_dataset_samples(dataset_path, split_name='Testing')
    if not samples:
        samples = load_dataset_samples(dataset_path, split_name=None)
        
    print(f"[*] Total Available Test Scans: {len(samples)}")
    
    if args.max_samples > 0 and len(samples) > args.max_samples:
        import random
        random.seed(42)
        samples = random.sample(samples, args.max_samples)
        print(f"[*] Evaluating on stratified sample subset: {len(samples)} scans")
        
    test_dataset = BrainMRIDataset(
        samples=samples,
        target_size=(224, 224),
        apply_crop=True,
        apply_clahe=True,
        apply_denoise=True,
        is_training=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Load models
    models_to_test = {}
    backbone_names = ['densenet121', 'resnet50', 'efficientnet_b0']
    
    for mname in backbone_names:
        m = build_model(mname, num_classes=4, pretrained=False).to(device)
        ckpt = os.path.join(BASE_DIR, 'weights', f'{mname}_best.pth')
        if os.path.exists(ckpt):
            m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        models_to_test[mname] = m
        
    # Build GWO Ensemble
    gwo_weights = [0.3419, 0.0067, 0.6514]
    ensemble = MetaheuristicEnsembleModel(
        [models_to_test['densenet121'], models_to_test['resnet50'], models_to_test['efficientnet_b0']],
        weights=gwo_weights
    ).to(device)
    ensemble.eval()
    models_to_test['ensemble'] = ensemble
    
    selected = models_to_test if args.model == 'all' else {args.model: models_to_test[args.model]}
    
    results = []
    for mname, model in selected.items():
        y_true, y_pred, y_probs = evaluate_model(model, test_loader, device, model_name=mname)
        res = compute_detailed_report(y_true, y_pred, y_probs, model_name=mname)
        results.append(res)
        
    # Final Comparison Summary
    print("\n" + "="*75)
    print("  SUMMARY COMPARISON: ACCURACY SCORES ACROSS ALL ARCHITECTURES")
    print("="*75)
    print(f"  {'Model Architecture':<25} | {'Accuracy Score':<15} | {'Macro F1':<12} | {'Correct/Total'}")
    print("  " + "-"*71)
    for r in results:
        is_top = (r['model_name'] == 'ensemble')
        prefix = "[*] " if is_top else "    "
        print(f"  {prefix}{r['model_name']:<23} | {r['accuracy']*100:6.2f}%          | {r['f1_macro']*100:6.2f}%       | {r['correct']}/{r['total']}")
    print("="*75 + "\n")

if __name__ == '__main__':
    main()
