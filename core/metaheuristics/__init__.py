"""
Nature-Inspired Metaheuristic Optimization Engine for Deep Learning.
Implements:
- Grey Wolf Optimizer (GWO) & Binary GWO
- Particle Swarm Optimization (PSO) & Binary PSO
- Whale Optimization Algorithm (WOA) & Binary WOA
- Genetic Algorithm (GA) & Binary GA
- Harris Hawks Optimization (HHO)
- Chimp Optimization Algorithm (ChOA)
- Metaheuristic Deep Feature Selection
- Metaheuristic Ensemble Blending Weight Optimization
- Metaheuristic Hyperparameter Tuning
"""

from .base import BaseMetaheuristic, Solution, OptimizationResult
from .gwo import GreyWolfOptimizer, BinaryGWO
from .pso import ParticleSwarmOptimizer, BinaryPSO
from .woa import WhaleOptimizer, BinaryWOA
from .ga import GeneticAlgorithm, BinaryGA
from .hho import HarrisHawksOptimizer
from .choa import ChimpOptimizer
from .feature_selector import MetaheuristicFeatureSelector
from .ensemble_optimizer import MetaheuristicEnsembleOptimizer
from .hyperparameter_tuner import MetaheuristicHyperparameterTuner

OPTIMIZERS = {
    'gwo': GreyWolfOptimizer,
    'pso': ParticleSwarmOptimizer,
    'woa': WhaleOptimizer,
    'ga': GeneticAlgorithm,
    'hho': HarrisHawksOptimizer,
    'choa': ChimpOptimizer
}

BINARY_OPTIMIZERS = {
    'bgwo': BinaryGWO,
    'bpso': BinaryPSO,
    'bwoa': BinaryWOA,
    'bga': BinaryGA
}

def get_optimizer(name: str, **kwargs):
    """
    Factory to retrieve an optimizer by name.
    """
    name_lower = name.lower()
    if name_lower in OPTIMIZERS:
        return OPTIMIZERS[name_lower](**kwargs)
    elif name_lower in BINARY_OPTIMIZERS:
        return BINARY_OPTIMIZERS[name_lower](**kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Supported: {list(OPTIMIZERS.keys()) + list(BINARY_OPTIMIZERS.keys())}")

__all__ = [
    "BaseMetaheuristic",
    "Solution",
    "OptimizationResult",
    "GreyWolfOptimizer",
    "BinaryGWO",
    "ParticleSwarmOptimizer",
    "BinaryPSO",
    "WhaleOptimizer",
    "BinaryWOA",
    "GeneticAlgorithm",
    "BinaryGA",
    "HarrisHawksOptimizer",
    "ChimpOptimizer",
    "MetaheuristicFeatureSelector",
    "MetaheuristicEnsembleOptimizer",
    "MetaheuristicHyperparameterTuner",
    "OPTIMIZERS",
    "BINARY_OPTIMIZERS",
    "get_optimizer"
]
