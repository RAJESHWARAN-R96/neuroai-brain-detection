"""
Metaheuristic Hyperparameter Optimization (HPO) for Deep Learning Brain MRI Classifiers.
Optimizes Learning Rate, Weight Decay, and Dropout Rate using GWO, PSO, WOA, HHO.
"""

import numpy as np
import time
from typing import Dict, Any, Optional, Callable

from .base import OptimizationResult
from .gwo import GreyWolfOptimizer
from .pso import ParticleSwarmOptimizer
from .woa import WhaleOptimizer
from .hho import HarrisHawksOptimizer


class MetaheuristicHyperparameterTuner:
    """
    Continuous Metaheuristic Tuner for Deep Learning Hyperparameters:
    - Learning rate: [1e-5, 1e-2] (searched in log10 space [-5, -2])
    - Weight decay:  [1e-6, 1e-2] (searched in log10 space [-6, -2])
    - Dropout rate:  [0.10, 0.60] (linear space)
    """
    def __init__(
        self,
        algorithm: str = 'woa',
        pop_size: int = 15,
        max_iter: int = 15,
        seed: int = 42
    ):
        self.algorithm_name = algorithm.lower()
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.seed = seed
        
        self.best_params: Dict[str, float] = {}
        self.optimization_result: Optional[OptimizationResult] = None
        self.metrics_summary: Dict[str, Any] = {}

    def _decode_params(self, vector: np.ndarray) -> Dict[str, float]:
        """Maps search space vector [x0, x1, x2] into real hyperparameters."""
        lr_log = vector[0]  # [-5, -2]
        wd_log = vector[1]  # [-6, -2]
        dropout = vector[2] # [0.1, 0.6]
        
        return {
            'learning_rate': float(10.0 ** lr_log),
            'weight_decay': float(10.0 ** wd_log),
            'dropout_rate': float(round(dropout, 3))
        }

    def tune(
        self,
        eval_fn: Callable[[Dict[str, float]], float],
        callback: Optional[Callable[[int, float, np.ndarray, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Runs hyperparameter optimization with objective function eval_fn.
        eval_fn takes a dictionary of decoded parameters and returns validation accuracy/F1.
        """
        start_time = time.time()
        
        # Search space bounds:
        # Dim 0: log10(lr) in [-5, -2]
        # Dim 1: log10(wd) in [-6, -2]
        # Dim 2: dropout in [0.1, 0.6]
        lb = np.array([-5.0, -6.0, 0.10])
        ub = np.array([-2.0, -2.0, 0.60])
        
        def objective(vector: np.ndarray) -> float:
            params = self._decode_params(vector)
            val_score = eval_fn(params)
            return float(val_score)

        kwargs = {
            'objective_func': objective,
            'dim': 3,
            'lb': lb,
            'ub': ub,
            'pop_size': self.pop_size,
            'max_iter': self.max_iter,
            'maximize': True,
            'seed': self.seed,
            'callback': callback
        }

        if self.algorithm_name in ('woa', 'whale'):
            optimizer = WhaleOptimizer(**kwargs)
        elif self.algorithm_name in ('gwo', 'grey_wolf'):
            optimizer = GreyWolfOptimizer(**kwargs)
        elif self.algorithm_name in ('pso', 'particle_swarm'):
            optimizer = ParticleSwarmOptimizer(**kwargs)
        elif self.algorithm_name in ('hho', 'harris_hawks'):
            optimizer = HarrisHawksOptimizer(**kwargs)
        else:
            optimizer = WhaleOptimizer(**kwargs)

        opt_res = optimizer.optimize()
        self.optimization_result = opt_res
        self.best_params = self._decode_params(opt_res.best_position)
        
        self.metrics_summary = {
            'algorithm': self.algorithm_name.upper(),
            'best_hyperparameters': {
                'learning_rate': f"{self.best_params['learning_rate']:.6f}",
                'weight_decay': f"{self.best_params['weight_decay']:.6f}",
                'dropout_rate': self.best_params['dropout_rate']
            },
            'best_score': float(round(opt_res.best_fitness * 100.0, 2)),
            'convergence_curve': opt_res.convergence_curve,
            'execution_time_sec': float(round(time.time() - start_time, 2))
        }
        
        return self.metrics_summary
