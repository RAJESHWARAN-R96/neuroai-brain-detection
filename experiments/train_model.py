"""
Automated Model Training CLI Script on the Kaggle Brain Tumor MRI Dataset.
"""

import os
import sys
import argparse
import torch

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.models import build_model
from core.dataset import get_data_loaders, find_default_dataset_path
from core.trainer import train_model, evaluate, get_device

def main():
    parser = argparse.ArgumentParser(description="Train Deep Learning Model for Brain Tumor MRI Detection")
    parser.add_argument('--model', type=str, default='attention_cnn', choices=['attention_cnn', 'efficientnet_b0', 'resnet50', 'densenet121'])
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dataset_dir', type=str, default=None, help='Path to Kaggle dataset')
    parser.add_argument('--save_name', type=str, default=None, help='Model checkpoint name')
    args = parser.parse_args()

    device = get_device()
    print(f"[*] Training on device: {device}")
    
    data_path = args.dataset_dir or find_default_dataset_path()
    print(f"[*] Loading dataset from: {data_path}")
    
    loaders = get_data_loaders(data_path, batch_size=args.batch_size, num_workers=0)
    print(f"[+] Loaded {len(loaders['train_samples'])} training, {len(loaders['val_samples'])} validation, {len(loaders['test_samples'])} testing samples.")

    print(f"[*] Initializing model: {args.model}...")
    model = build_model(args.model, num_classes=4, pretrained=True)
    
    save_filename = args.save_name or f"{args.model}_best.pth"
    save_path = os.path.join(BASE_DIR, 'weights', save_filename)
    
    print(f"[*] Starting training for {args.epochs} epochs...")
    metrics = train_model(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        save_path=save_path
    )
    
    print("\n" + "=" * 60)
    print("                 TRAINING COMPLETE RESULTS                  ")
    print("=" * 60)
    print(f"Validation Accuracy:   {metrics['accuracy']}%")
    print(f"Validation Macro-F1:   {metrics['f1_macro']}%")
    print(f"Validation ROC-AUC:    {metrics['roc_auc']}%")
    print(f"Cohen's Kappa Score:   {metrics['cohen_kappa']}")
    print(f"Training Time:         {metrics['training_time_sec']}s")
    print(f"Saved Checkpoint to:   {save_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()
