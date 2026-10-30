"""
Abstract Base Class and Common Utilities for Nature-Inspired Metaheuristic Optimizers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
import numpy as np
from typing import Callable, List, Dict, Any, Optional, Tuple, Union


@dataclass
class Solution:
    """Represents an individual agent/candidate solution in the swarm."""
    position: np.ndarray
    fitness: float = -np.inf
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Encapsulates the complete trajectory and final output of an optimization run."""
    algorithm_name: str
    best_position: np.ndarray
    best_fitness: float
    convergence_curve: List[float]
    diversity_curve: List[float]
    execution_time_sec: float
    iterations_run: int
    population_history: Optional[List[List[List[float]]]] = None  # Samples for 2D/3D visualization
    final_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'algorithm_name': self.algorithm_name,
            'best_position': self.best_position.tolist() if isinstance(self.best_position, np.ndarray) else self.best_position,
            'best_fitness': float(self.best_fitness),
            'convergence_curve': [float(x) for x in self.convergence_curve],
            'diversity_curve': [float(x) for x in self.diversity_curve],
            'execution_time_sec': float(self.execution_time_sec),
            'iterations_run': int(self.iterations_run),
            'final_metrics': self.final_metrics,
            'population_history': self.population_history
        }


class BaseMetaheuristic(ABC):
    """
    Abstract Base Class for Swarm Intelligence and Evolutionary Metaheuristics.
    """
    def __init__(
        self,
        objective_func: Optional[Callable[[np.ndarray], float]] = None,
        dim: int = 10,
        lb: Union[float, np.ndarray] = 0.0,
        ub: Union[float, np.ndarray] = 1.0,
        pop_size: int = 25,
        max_iter: int = 30,
        maximize: bool = True,
        seed: Optional[int] = 42,
        callback: Optional[Callable[[int, float, np.ndarray, Dict[str, Any]], None]] = None
    ):
        self.objective_func = objective_func
        self.dim = dim
        self.lb = np.array(lb) if isinstance(lb, (list, np.ndarray)) else np.full(dim, lb)
        self.ub = np.array(ub) if isinstance(ub, (list, np.ndarray)) else np.full(dim, ub)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.maximize = maximize
        self.seed = seed
        self.callback = callback
        
        if seed is not None:
            np.random.seed(seed)
            
        self.population: List[Solution] = []
        self.best_solution: Optional[Solution] = None
        self.convergence_curve: List[float] = []
        self.diversity_curve: List[float] = []
        self.population_history: List[List[List[float]]] = []

    def initialize_population(self):
        """Initializes continuous population uniformly within lower and upper bounds."""
        self.population = []
        for _ in range(self.pop_size):
            pos = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
            self.population.append(Solution(position=pos))
            
        self.evaluate_population()
        self.update_best()

    def evaluate_population(self):
        """Evaluates objective function for all individuals in the population."""
        if self.objective_func is None:
            raise ValueError("Objective function not set.")
            
        for ind in self.population:
            val = self.objective_func(ind.position)
            # Standardize: higher fitness is always better internally
            ind.fitness = float(val) if self.maximize else -float(val)

    def update_best(self):
        """Finds and updates global best solution."""
        for ind in self.population:
            if self.best_solution is None or ind.fitness > self.best_solution.fitness:
                self.best_solution = Solution(
                    position=ind.position.copy(),
                    fitness=ind.fitness,
                    metrics=ind.metrics.copy()
                )

    def enforce_bounds(self, position: np.ndarray) -> np.ndarray:
        """Clamps positions to within defined [lb, ub] range."""
        return np.clip(position, self.lb, self.ub)

    def compute_diversity(self) -> float:
        """Computes population diversity (average Euclidean distance from population centroid)."""
        if not self.population:
            return 0.0
        positions = np.array([ind.position for ind in self.population])
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        return float(np.mean(distances))

    @staticmethod
    def sigmoid_transfer(x: np.ndarray) -> np.ndarray:
        """S-shaped sigmoid transfer function for Binary Metaheuristics."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))

    @staticmethod
    def v_shape_transfer(x: np.ndarray) -> np.ndarray:
        """V-shaped transfer function: T(x) = |x / sqrt(1 + x^2)|."""
        return np.abs(x / np.sqrt(1.0 + x**2))

    @abstractmethod
    def iterate(self, iteration: int):
        """Perform one iteration of the metaheuristic update equations."""
        pass

    def optimize(self) -> OptimizationResult:
        """Runs the complete optimization loop and returns OptimizationResult."""
        start_time = time.time()
        self.initialize_population()
        
        self.convergence_curve = []
        self.diversity_curve = []
        self.population_history = []
        
        for it in range(self.max_iter):
            self.iterate(it)
            self.evaluate_population()
            self.update_best()
            
            curr_best = self.best_solution.fitness if self.maximize else -self.best_solution.fitness
            diversity = self.compute_diversity()
            
            self.convergence_curve.append(curr_best)
            self.diversity_curve.append(diversity)
            
            # Save 2D slice for visualizer (e.g. first 2 dimensions)
            if self.dim >= 2:
                sample_pts = [[float(p.position[0]), float(p.position[1])] for p in self.population[:15]]
            else:
                sample_pts = [[float(p.position[0]), float(p.fitness)] for p in self.population[:15]]
            self.population_history.append(sample_pts)
            
            if self.callback is not None:
                self.callback(it + 1, curr_best, self.best_solution.position, {
                    'diversity': diversity,
                    'iteration': it + 1,
                    'max_iter': self.max_iter
                })
                
        exec_time = time.time() - start_time
        final_best = self.best_solution.fitness if self.maximize else -self.best_solution.fitness
        
        return OptimizationResult(
            algorithm_name=self.__class__.__name__,
            best_position=self.best_solution.position.copy(),
            best_fitness=final_best,
            convergence_curve=self.convergence_curve,
            diversity_curve=self.diversity_curve,
            execution_time_sec=exec_time,
            iterations_run=self.max_iter,
            population_history=self.population_history,
            final_metrics={'best_fitness': final_best, 'execution_time_sec': exec_time}
        )
