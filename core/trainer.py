"""
Training and Evaluation Pipeline for Brain Tumor MRI Deep Learning Models.
Includes:
- Training and Validation loops with Early Stopping & Learning Rate Scheduling
- Multi-Class Metric Evaluation (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, Cohen's Kappa)
- Confusion Matrix Generation
- Deep Embedding Extraction for Metaheuristic Optimization
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, confusion_matrix
)
from .dataset import CLASS_NAMES


def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    use_amp = scaler is not None and device.type == 'cuda'
    
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
    epoch_loss = total_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates model on validation/test DataLoader.
    Returns: loss, accuracy, y_true, y_pred, y_probs
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            probs = F.softmax(outputs, dim=-1)
            _, preds = torch.max(outputs, 1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            total += labels.size(0)
            
    val_loss = total_loss / max(1, total)
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    val_acc = float(accuracy_score(y_true, y_pred))
    
    return val_loss, val_acc, y_true, y_pred, y_probs


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str] = CLASS_NAMES
) -> Dict[str, Any]:
    """
    Calculates publication-standard medical metrics across all classes.
    """
    num_classes = len(class_names)
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    
    # ROC-AUC (Multi-class One-vs-Rest)
    try:
        if num_classes == 2:
            roc_auc = float(roc_auc_score(y_true, y_probs[:, 1]))
        else:
            roc_auc = float(roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro'))
    except Exception:
        roc_auc = 0.0

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # Per-class Sensitivity, Specificity, Precision, F1
    per_class_metrics = {}
    total_samples = len(y_true)
    
    for i, cname in enumerate(class_names):
        tp = float(cm[i, i])
        fn = float(np.sum(cm[i, :]) - tp)
        fp = float(np.sum(cm[:, i]) - tp)
        tn = float(total_samples - (tp + fn + fp))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        class_f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        
        per_class_metrics[cname] = {
            'sensitivity_recall': round(sensitivity * 100.0, 2),
            'specificity': round(specificity * 100.0, 2),
            'precision': round(precision * 100.0, 2),
            'f1_score': round(class_f1 * 100.0, 2),
            'support': int(np.sum(cm[i, :]))
        }
        
    return {
        'accuracy': round(acc * 100.0, 2),
        'precision_macro': round(prec_macro * 100.0, 2),
        'recall_macro': round(rec_macro * 100.0, 2),
        'f1_macro': round(f1_macro * 100.0, 2),
        'f1_weighted': round(f1_weighted * 100.0, 2),
        'roc_auc': round(roc_auc * 100.0, 2),
        'cohen_kappa': round(kappa, 4),
        'confusion_matrix': cm.tolist(),
        'per_class': per_class_metrics,
        'class_names': class_names
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    early_stop_patience: int = 4,
    callback: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Trains PyTorch model with Cosine Annealing learning rate schedule and early stopping.
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc, y_true, y_pred, y_probs = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc * 100.0)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc * 100.0)
        
        if callback is not None:
            callback(epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break

    # Compute final metrics on validation set
    _, _, y_true, y_pred, y_probs = evaluate(model, val_loader, criterion, device)
    metrics = compute_comprehensive_metrics(y_true, y_pred, y_probs)
    metrics['history'] = history
    metrics['training_time_sec'] = round(time.time() - start_time, 2)
    return metrics


def extract_all_features(model: nn.Module, loader: DataLoader, device: Optional[torch.device] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Passes DataLoader samples through model.extract_features() to get feature matrix X [N, D] and labels y [N].
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()
    
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            if hasattr(model, 'extract_features'):
                feat = model.extract_features(images)
            else:
                feat = model(images)
            features_list.append(feat.cpu().numpy())
            labels_list.append(labels.numpy())
            
    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X, y
