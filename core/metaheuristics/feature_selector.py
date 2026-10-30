"""
Metaheuristic Deep Feature Selection Module.
Optimizes high-dimensional deep feature embeddings extracted from CNN backbones
using Binary Metaheuristic Algorithms (BGWO, BPSO, BWOA, BGA).
"""

import numpy as np
import time
import warnings
from typing import Dict, Any, List, Optional, Tuple, Callable
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore')

from .base import OptimizationResult
from .gwo import BinaryGWO
from .pso import BinaryPSO
from .woa import BinaryWOA
from .ga import BinaryGA


class MetaheuristicFeatureSelector:
    """
    Performs feature selection on deep CNN embeddings using binary metaheuristics.
    Multi-objective fitness: Fitness = alpha * Accuracy + (1 - alpha) * (1 - (n_selected / n_total))
    """
    def __init__(
        self,
        algorithm: str = 'bgwo',
        alpha: float = 0.90,  # Weight on accuracy vs sparsity
        pop_size: int = 20,
        max_iter: int = 25,
        cv_folds: int = 3,
        seed: int = 42
    ):
        self.algorithm_name = algorithm.lower()
        self.alpha = alpha
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.cv_folds = cv_folds
        self.seed = seed
        
        self.selected_mask: Optional[np.ndarray] = None
        self.selected_indices: Optional[List[int]] = None
        self.optimization_result: Optional[OptimizationResult] = None
        self.metrics_summary: Dict[str, Any] = {}

    def _get_binary_optimizer(self, objective_func: Callable[[np.ndarray], float], dim: int, callback=None):
        kwargs = {
            'objective_func': objective_func,
            'dim': dim,
            'pop_size': self.pop_size,
            'max_iter': self.max_iter,
            'maximize': True,
            'seed': self.seed,
            'callback': callback
        }
        
        if self.algorithm_name in ('bgwo', 'gwo'):
            return BinaryGWO(**kwargs)
        elif self.algorithm_name in ('bpso', 'pso'):
            return BinaryPSO(**kwargs)
        elif self.algorithm_name in ('bwoa', 'woa'):
            return BinaryWOA(**kwargs)
        elif self.algorithm_name in ('bga', 'ga'):
            return BinaryGA(**kwargs)
        else:
            return BinaryGWO(**kwargs)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        callback: Optional[Callable[[int, float, np.ndarray, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes feature selection optimization on feature matrix X [N, D] and labels y [N].
        """
        n_samples, n_features = X.shape
        start_time = time.time()
        
        # 1. Evaluate baseline accuracy with ALL features
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        
        def evaluate_subset(mask: np.ndarray) -> Tuple[float, float]:
            selected = np.where(mask > 0.5)[0]
            if len(selected) == 0:
                return 0.0, 0.0
                
            X_sub = X[:, selected]
            accuracies = []
            f1_scores = []
            
            from sklearn.linear_model import RidgeClassifier
            for train_idx, val_idx in skf.split(X_sub, y):
                clf = RidgeClassifier(alpha=1.0, random_state=self.seed)
                clf.fit(X_sub[train_idx], y[train_idx])
                preds = clf.predict(X_sub[val_idx])
                accuracies.append(accuracy_score(y[val_idx], preds))
                f1_scores.append(f1_score(y[val_idx], preds, average='macro', zero_division=0))
                
            mean_acc = float(np.mean(accuracies))
            mean_f1 = float(np.mean(f1_scores))
            return mean_acc, mean_f1

        # Baseline evaluation
        all_mask = np.ones(n_features)
        baseline_acc, baseline_f1 = evaluate_subset(all_mask)
        
        # 2. Objective Function for Metaheuristic Optimizer
        def objective(mask: np.ndarray) -> float:
            selected_count = np.sum(mask > 0.5)
            if selected_count == 0:
                return 0.0
            acc, _ = evaluate_subset(mask)
            sparsity = 1.0 - (selected_count / float(n_features))
            fitness = self.alpha * acc + (1.0 - self.alpha) * sparsity
            return float(fitness)

        # 3. Run Binary Metaheuristic
        optimizer = self._get_binary_optimizer(objective, dim=n_features, callback=callback)
        opt_res = optimizer.optimize()
        self.optimization_result = opt_res
        
        # 4. Extract Best Selected Features
        best_mask = (opt_res.best_position > 0.5).astype(int)
        if np.sum(best_mask) == 0:
            best_mask[0] = 1
            
        self.selected_mask = best_mask
        self.selected_indices = np.where(best_mask == 1)[0].tolist()
        
        opt_acc, opt_f1 = evaluate_subset(best_mask)
        num_selected = len(self.selected_indices)
        reduction_pct = ((n_features - num_selected) / float(n_features)) * 100.0
        
        self.metrics_summary = {
            'algorithm': self.algorithm_name.upper(),
            'total_features': int(n_features),
            'selected_features': int(num_selected),
            'reduction_percentage': float(round(reduction_pct, 2)),
            'baseline_accuracy': float(round(baseline_acc * 100.0, 2)),
            'optimized_accuracy': float(round(opt_acc * 100.0, 2)),
            'accuracy_gain': float(round((opt_acc - baseline_acc) * 100.0, 2)),
            'baseline_f1': float(round(baseline_f1 * 100.0, 2)),
            'optimized_f1': float(round(opt_f1 * 100.0, 2)),
            'convergence_curve': opt_res.convergence_curve,
            'diversity_curve': opt_res.diversity_curve,
            'execution_time_sec': float(round(time.time() - start_time, 2)),
            'selected_indices': self.selected_indices
        }
        
        return self.metrics_summary

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projects high-dimensional feature matrix X onto the optimal selected feature subspace."""
        if self.selected_indices is None:
            raise ValueError("Feature selector is not fitted yet. Call fit() first.")
        return X[:, self.selected_indices]
