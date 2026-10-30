"""
Metaheuristic Algorithm Benchmarking CLI Experiment.
Compares GWO, PSO, WOA, GA, HHO, and ChOA across Ensemble Blending and Feature Selection tasks.
"""

import os
import sys
import argparse
import time
import numpy as np

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.metaheuristics import (
    MetaheuristicEnsembleOptimizer,
    MetaheuristicFeatureSelector,
    OPTIMIZERS
)

def run_ensemble_blending_benchmark(pop_size=20, max_iter=25):
    print("\n" + "=" * 80)
    print("      BENCHMARK 1: MULTI-MODEL ENSEMBLE WEIGHT BLENDING OPTIMIZATION       ")
    print("=" * 80)

    np.random.seed(42)
    n_val = 300
    y_true = np.random.choice([0, 1, 2, 3], size=n_val)
    
    # 4 distinct deep backbone probability distributions
    p1 = np.eye(4)[y_true] * 0.85 + np.random.uniform(0, 0.15, (n_val, 4))
    p1 /= np.sum(p1, axis=1, keepdims=True)
    
    p2 = np.eye(4)[y_true] * 0.82 + np.random.uniform(0, 0.18, (n_val, 4))
    p2 /= np.sum(p2, axis=1, keepdims=True)
    
    p3 = np.eye(4)[y_true] * 0.84 + np.random.uniform(0, 0.16, (n_val, 4))
    p3 /= np.sum(p3, axis=1, keepdims=True)
    
    p4 = np.eye(4)[y_true] * 0.86 + np.random.uniform(0, 0.14, (n_val, 4))
    p4 /= np.sum(p4, axis=1, keepdims=True)

    algorithms = ['gwo', 'pso', 'woa', 'ga', 'hho', 'choa']
    results = []

    for alg in algorithms:
        print(f"[*] Running {alg.upper()} Optimizer (Pop: {pop_size}, Iter: {max_iter})...")
        opt = MetaheuristicEnsembleOptimizer(algorithm=alg, pop_size=pop_size, max_iter=max_iter, seed=42)
        res = opt.fit(
            model_prob_list=[p1, p2, p3, p4],
            y_true=y_true,
            model_names=['EfficientNet', 'ResNet-50', 'DenseNet-121', 'AttentionCNN']
        )
        results.append(res)

    print("\n" + "-" * 80)
    print(f"{'Algorithm':<10} | {'Optimized Acc (%)':<18} | {'Macro-F1 (%)':<15} | {'Log Loss':<10} | {'Time (s)':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['algorithm']:<10} | {r['optimized_accuracy']:<18} | {r['optimized_f1']:<15} | {r['optimized_loss']:<10} | {r['execution_time_sec']:<8}")
    print("-" * 80)


def run_feature_selection_benchmark(pop_size=20, max_iter=25):
    print("\n" + "=" * 80)
    print("      BENCHMARK 2: BINARY DEEP FEATURE SELECTION (DIMENSIONALITY REDUCTION)  ")
    print("=" * 80)

    np.random.seed(42)
    n_samples = 250
    n_features = 128
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1, 2, 3], size=n_samples)
    for i in range(4):
        mask = (y == i)
        X[mask, :int(n_features * 0.25)] += (i + 1) * 0.8

    algorithms = ['bgwo', 'bpso', 'bwoa', 'bga']
    results = []

    for alg in algorithms:
        print(f"[*] Running Binary {alg.upper()} Optimizer...")
        selector = MetaheuristicFeatureSelector(algorithm=alg, alpha=0.90, pop_size=pop_size, max_iter=max_iter, seed=42)
        res = selector.fit(X, y)
        results.append(res)

    print("\n" + "-" * 85)
    print(f"{'Algorithm':<10} | {'Baseline Acc (%)':<17} | {'Opt. Acc (%)':<14} | {'Features Selected':<20} | {'Reduction (%)':<15}")
    print("-" * 85)
    for r in results:
        feat_str = f"{r['selected_features']} / {r['total_features']}"
        print(f"{r['algorithm']:<10} | {r['baseline_accuracy']:<17} | {r['optimized_accuracy']:<14} | {feat_str:<20} | {r['reduction_percentage']}%")
    print("-" * 85)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark Nature-Inspired Metaheuristic Optimizers")
    parser.add_argument('--population', type=int, default=20, help='Swarm population size')
    parser.add_argument('--iterations', type=int, default=25, help='Number of search iterations')
    args = parser.parse_args()

    run_ensemble_blending_benchmark(pop_size=args.population, max_iter=args.iterations)
    run_feature_selection_benchmark(pop_size=args.population, max_iter=args.iterations)
