"""
Metaheuristic Multi-Model Ensemble Blending Weight Optimizer.
Optimizes continuous weight vectors [w1, w2, ..., wK] to fuse predictions from
EfficientNet, ResNet, DenseNet, and CBAM Attention CNN using GWO, PSO, WOA, HHO, GA, and ChOA.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Callable
from sklearn.metrics import log_loss, f1_score, accuracy_score

from .base import OptimizationResult
from .gwo import GreyWolfOptimizer
from .pso import ParticleSwarmOptimizer
from .woa import WhaleOptimizer
from .ga import GeneticAlgorithm
from .hho import HarrisHawksOptimizer
from .choa import ChimpOptimizer


class MetaheuristicEnsembleOptimizer:
    """
    Optimizes ensemble blending weights w = [w_1, ..., w_K] such that sum(w_i) = 1.0.
    """
    def __init__(
        self,
        algorithm: str = 'gwo',
        pop_size: int = 25,
        max_iter: int = 30,
        seed: int = 42
    ):
        self.algorithm_name = algorithm.lower()
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.seed = seed
        
        self.best_weights: Optional[np.ndarray] = None
        self.optimization_result: Optional[OptimizationResult] = None
        self.metrics_summary: Dict[str, Any] = {}

    def _get_optimizer(self, objective_func: Callable[[np.ndarray], float], dim: int, callback=None):
        kwargs = {
            'objective_func': objective_func,
            'dim': dim,
            'lb': 0.0,
            'ub': 1.0,
            'pop_size': self.pop_size,
            'max_iter': self.max_iter,
            'maximize': True,
            'seed': self.seed,
            'callback': callback
        }
        
        if self.algorithm_name in ('gwo', 'grey_wolf'):
            return GreyWolfOptimizer(**kwargs)
        elif self.algorithm_name in ('pso', 'particle_swarm'):
            return ParticleSwarmOptimizer(**kwargs)
        elif self.algorithm_name in ('woa', 'whale'):
            return WhaleOptimizer(**kwargs)
        elif self.algorithm_name in ('ga', 'genetic'):
            return GeneticAlgorithm(**kwargs)
        elif self.algorithm_name in ('hho', 'harris_hawks'):
            return HarrisHawksOptimizer(**kwargs)
        elif self.algorithm_name in ('choa', 'chimp'):
            return ChimpOptimizer(**kwargs)
        else:
            return GreyWolfOptimizer(**kwargs)

    @staticmethod
    def _normalize_weights(w: np.ndarray) -> np.ndarray:
        """Projects continuous vector w to probability simplex (sum(w) == 1, w_i >= 0)."""
        w = np.maximum(0.0, w)
        s = np.sum(w)
        if s > 1e-8:
            return w / s
        return np.full_like(w, 1.0 / len(w))

    def fit(
        self,
        model_prob_list: List[np.ndarray],  # List of K arrays, each [N, NumClasses]
        y_true: np.ndarray,                # Ground truth labels [N]
        model_names: Optional[List[str]] = None,
        callback: Optional[Callable[[int, float, np.ndarray, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Runs metaheuristic optimization over model probability predictions.
        """
        k = len(model_prob_list)
        if k == 0:
            raise ValueError("model_prob_list cannot be empty.")
            
        n_samples = len(y_true)
        if model_names is None or len(model_names) != k:
            model_names = [f"Model_{i+1}" for i in range(k)]
            
        start_time = time.time()
        
        # 1. Baseline metrics: Individual models and Equal-Weight Ensemble
        individual_metrics = []
        for i, probs in enumerate(model_prob_list):
            preds = np.argmax(probs, axis=-1)
            acc = float(accuracy_score(y_true, preds))
            f1 = float(f1_score(y_true, preds, average='macro'))
            loss = float(log_loss(y_true, probs))
            individual_metrics.append({
                'name': model_names[i],
                'accuracy': round(acc * 100.0, 2),
                'f1_score': round(f1 * 100.0, 2),
                'log_loss': round(loss, 4)
            })
            
        # Equal average ensemble
        equal_w = np.full(k, 1.0 / k)
        equal_probs = np.zeros_like(model_prob_list[0])
        for i in range(k):
            equal_probs += equal_w[i] * model_prob_list[i]
        equal_preds = np.argmax(equal_probs, axis=-1)
        equal_acc = float(accuracy_score(y_true, equal_preds))
        equal_f1 = float(f1_score(y_true, equal_preds, average='macro'))
        equal_loss = float(log_loss(y_true, equal_probs))
        
        # 2. Objective Function for Metaheuristic Optimizer
        def objective(weights: np.ndarray) -> float:
            w_norm = self._normalize_weights(weights)
            blended_probs = np.zeros_like(model_prob_list[0])
            for i in range(k):
                blended_probs += w_norm[i] * model_prob_list[i]
                
            # Numerical clamp for stable log-loss
            eps = 1e-7
            blended_probs = np.clip(blended_probs, eps, 1.0 - eps)
            blended_probs = blended_probs / np.sum(blended_probs, axis=1, keepdims=True)
            
            blended_preds = np.argmax(blended_probs, axis=-1)
            f1 = f1_score(y_true, blended_preds, average='macro')
            loss = log_loss(y_true, blended_probs)
            
            # Multi-objective fitness combining Macro-F1 and Log-Loss
            fitness = f1 - 0.05 * loss
            return float(fitness)

        # 3. Run Metaheuristic Optimization
        optimizer = self._get_optimizer(objective, dim=k, callback=callback)
        opt_res = optimizer.optimize()
        self.optimization_result = opt_res
        
        best_w = self._normalize_weights(opt_res.best_position)
        self.best_weights = best_w
        
        # Compute final optimized ensemble metrics
        opt_probs = np.zeros_like(model_prob_list[0])
        for i in range(k):
            opt_probs += best_w[i] * model_prob_list[i]
        opt_probs = opt_probs / np.sum(opt_probs, axis=1, keepdims=True)
        opt_preds = np.argmax(opt_probs, axis=-1)
        
        opt_acc = float(accuracy_score(y_true, opt_preds))
        opt_f1 = float(f1_score(y_true, opt_preds, average='macro'))
        opt_loss = float(log_loss(y_true, opt_probs))
        
        weights_dict = {model_names[i]: float(round(best_w[i], 4)) for i in range(k)}
        
        self.metrics_summary = {
            'algorithm': self.algorithm_name.upper(),
            'model_weights': weights_dict,
            'raw_weights': best_w.tolist(),
            'individual_models': individual_metrics,
            'equal_ensemble_accuracy': float(round(equal_acc * 100.0, 2)),
            'equal_ensemble_f1': float(round(equal_f1 * 100.0, 2)),
            'equal_ensemble_loss': float(round(equal_loss, 4)),
            'optimized_accuracy': float(round(opt_acc * 100.0, 2)),
            'optimized_f1': float(round(opt_f1 * 100.0, 2)),
            'optimized_loss': float(round(opt_loss, 4)),
            'accuracy_gain': float(round((opt_acc - equal_acc) * 100.0, 2)),
            'convergence_curve': opt_res.convergence_curve,
            'diversity_curve': opt_res.diversity_curve,
            'execution_time_sec': float(round(time.time() - start_time, 2))
        }
        
        return self.metrics_summary

    def predict(self, model_prob_list: List[np.ndarray]) -> np.ndarray:
        """Combines probabilities using the optimized weights and returns predicted class labels."""
        if self.best_weights is None:
            raise ValueError("Ensemble optimizer is not fitted yet.")
        blended = np.zeros_like(model_prob_list[0])
        for i, probs in enumerate(model_prob_list):
            blended += self.best_weights[i] * probs
        return np.argmax(blended, axis=-1)
