"""
Grey Wolf Optimizer (GWO) & Binary Grey Wolf Optimizer (BGWO).
Reference: Mirjalili et al., "Grey Wolf Optimizer", Advances in Engineering Software, 2014.
"""

import numpy as np
from typing import Optional, Callable
from .base import BaseMetaheuristic, Solution


class GreyWolfOptimizer(BaseMetaheuristic):
    """
    Continuous Grey Wolf Optimizer (GWO).
    Models Alpha (alpha), Beta (beta), and Delta (delta) leadership hierarchy for hunting prey.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha_wolf: Optional[Solution] = None
        self.beta_wolf: Optional[Solution] = None
        self.delta_wolf: Optional[Solution] = None

    def initialize_population(self):
        super().initialize_population()
        self._update_hierarchy()

    def _update_hierarchy(self):
        # Sort population by descending fitness (higher is better)
        sorted_pop = sorted(self.population, key=lambda s: s.fitness, reverse=True)
        self.alpha_wolf = Solution(position=sorted_pop[0].position.copy(), fitness=sorted_pop[0].fitness)
        self.beta_wolf = Solution(position=sorted_pop[1].position.copy(), fitness=sorted_pop[1].fitness)
        self.delta_wolf = Solution(position=sorted_pop[2].position.copy(), fitness=sorted_pop[2].fitness)

    def iterate(self, iteration: int):
        # Linearly decreasing parameter 'a' from 2 to 0
        a = 2.0 - iteration * (2.0 / self.max_iter)
        
        for wolf in self.population:
            # Update position influenced by Alpha, Beta, Delta
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * r2
            D_alpha = np.abs(C1 * self.alpha_wolf.position - wolf.position)
            X1 = self.alpha_wolf.position - A1 * D_alpha
            
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2
            D_beta = np.abs(C2 * self.beta_wolf.position - wolf.position)
            X2 = self.beta_wolf.position - A2 * D_beta
            
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2
            D_delta = np.abs(C3 * self.delta_wolf.position - wolf.position)
            X3 = self.delta_wolf.position - A3 * D_delta
            
            new_pos = (X1 + X2 + X3) / 3.0
            wolf.position = self.enforce_bounds(new_pos)
            
        self.evaluate_population()
        self._update_hierarchy()


class BinaryGWO(GreyWolfOptimizer):
    """
    Binary Grey Wolf Optimizer (BGWO) for Feature Selection.
    Uses S-shaped (Sigmoid) transfer function to map continuous wolf positions into binary feature masks {0, 1}.
    """
    def __init__(self, **kwargs):
        kwargs['lb'] = 0.0
        kwargs['ub'] = 1.0
        super().__init__(**kwargs)

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            # Binary vector with at least 1 active feature
            pos = (np.random.rand(self.dim) > 0.5).astype(float)
            if np.sum(pos) == 0:
                pos[np.random.randint(0, self.dim)] = 1.0
            self.population.append(Solution(position=pos))
        self.evaluate_population()
        self._update_hierarchy()

    def iterate(self, iteration: int):
        a = 2.0 - iteration * (2.0 / self.max_iter)
        
        for wolf in self.population:
            # Continuous step calculation
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * r2
            D_alpha = np.abs(C1 * self.alpha_wolf.position - wolf.position)
            X1 = self.alpha_wolf.position - A1 * D_alpha
            
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2
            D_beta = np.abs(C2 * self.beta_wolf.position - wolf.position)
            X2 = self.beta_wolf.position - A2 * D_beta
            
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2
            D_delta = np.abs(C3 * self.delta_wolf.position - wolf.position)
            X3 = self.delta_wolf.position - A3 * D_delta
            
            continuous_pos = (X1 + X2 + X3) / 3.0
            
            # Sigmoid transfer into binary domain
            prob = self.sigmoid_transfer(continuous_pos)
            binary_pos = (np.random.rand(self.dim) < prob).astype(float)
            
            # Ensure at least 1 feature selected
            if np.sum(binary_pos) == 0:
                binary_pos[np.random.randint(0, self.dim)] = 1.0
                
            wolf.position = binary_pos
            
        self.evaluate_population()
        self._update_hierarchy()
