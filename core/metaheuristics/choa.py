"""
Chimp Optimization Algorithm (ChOA).
Reference: Khishe & Mosavi, "Chimp optimization algorithm", Expert Systems with Applications, 2020.
"""

import numpy as np
from typing import Optional
from .base import BaseMetaheuristic, Solution


class ChimpOptimizer(BaseMetaheuristic):
    """
    Chimp Optimization Algorithm (ChOA) simulating driving, chasing, blocking,
    and attacking behaviors with chaotic exploration maps.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attacker: Optional[Solution] = None
        self.barrier: Optional[Solution] = None
        self.chaser: Optional[Solution] = None
        self.driver: Optional[Solution] = None

    def initialize_population(self):
        super().initialize_population()
        self._update_hierarchy()

    def _update_hierarchy(self):
        sorted_pop = sorted(self.population, key=lambda s: s.fitness, reverse=True)
        self.attacker = Solution(position=sorted_pop[0].position.copy(), fitness=sorted_pop[0].fitness)
        self.barrier = Solution(position=sorted_pop[1].position.copy(), fitness=sorted_pop[1].fitness)
        self.chaser = Solution(position=sorted_pop[2].position.copy(), fitness=sorted_pop[2].fitness)
        self.driver = Solution(position=sorted_pop[3].position.copy(), fitness=sorted_pop[3].fitness)

    @staticmethod
    def _chaotic_map(iteration: int, max_iter: int) -> float:
        """Tent chaotic map to maintain population diversity."""
        x = (iteration + 1) / max_iter
        if x < 0.7:
            return x / 0.7
        else:
            return (10.0 / 3.0) * (1.0 - x)

    def iterate(self, iteration: int):
        f = 2.5 - iteration * (2.5 / self.max_iter)
        
        for chimp in self.population:
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            
            # Attacker update
            C1 = 2.0 * r1
            a1 = 2.0 * f * r2 - f
            d_att = np.abs(C1 * self.attacker.position - chimp.position)
            X1 = self.attacker.position - a1 * d_att
            
            # Barrier update
            C2 = 2.0 * r1
            a2 = 2.0 * f * r2 - f
            d_bar = np.abs(C2 * self.barrier.position - chimp.position)
            X2 = self.barrier.position - a2 * d_bar
            
            # Chaser update
            C3 = 2.0 * r1
            a3 = 2.0 * f * r2 - f
            d_cha = np.abs(C3 * self.chaser.position - chimp.position)
            X3 = self.chaser.position - a3 * d_cha
            
            # Driver update
            C4 = 2.0 * r1
            a4 = 2.0 * f * r2 - f
            d_dri = np.abs(C4 * self.driver.position - chimp.position)
            X4 = self.driver.position - a4 * d_dri
            
            # Blend
            new_pos = (X1 + X2 + X3 + X4) / 4.0
            
            # Random chaotic perturbation
            if np.random.rand() < 0.1:
                mu = self._chaotic_map(iteration, self.max_iter)
                new_pos = new_pos * mu
                
            chimp.position = self.enforce_bounds(new_pos)
            
        self.evaluate_population()
        self._update_hierarchy()
