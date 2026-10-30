"""
Whale Optimization Algorithm (WOA) & Binary Whale Optimization Algorithm (BWOA).
Reference: Mirjalili & Lewis, "The Whale Optimization Algorithm", Advances in Engineering Software, 2016.
"""

import numpy as np
from .base import BaseMetaheuristic, Solution


class WhaleOptimizer(BaseMetaheuristic):
    """
    Whale Optimization Algorithm mimicking the bubble-net hunting strategy of humpback whales:
    1. Encircling prey
    2. Bubble-net attacking method (spiral-shaped path)
    3. Search for prey (exploration)
    """
    def __init__(self, b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.b = b  # Constant for defining shape of the logarithmic spiral

    def iterate(self, iteration: int):
        # 'a' decreases linearly from 2 to 0
        a = 2.0 - iteration * (2.0 / self.max_iter)
        
        for whale in self.population:
            r = np.random.rand()
            A = 2.0 * a * np.random.rand(self.dim) - a
            C = 2.0 * np.random.rand(self.dim)
            l = np.random.uniform(-1.0, 1.0)
            p = np.random.rand()
            
            if p < 0.5:
                if np.linalg.norm(A) < 1.0:
                    # Exploitation: Encircling prey (best solution)
                    D = np.abs(C * self.best_solution.position - whale.position)
                    new_pos = self.best_solution.position - A * D
                else:
                    # Exploration: Search for random prey
                    rand_idx = np.random.randint(0, self.pop_size)
                    rand_whale = self.population[rand_idx]
                    D = np.abs(C * rand_whale.position - whale.position)
                    new_pos = rand_whale.position - A * D
            else:
                # Exploitation: Bubble-net spiral updating position
                D_prime = np.abs(self.best_solution.position - whale.position)
                new_pos = D_prime * np.exp(self.b * l) * np.cos(2.0 * np.pi * l) + self.best_solution.position
                
            whale.position = self.enforce_bounds(new_pos)
            
        self.evaluate_population()


class BinaryWOA(WhaleOptimizer):
    """
    Binary Whale Optimization Algorithm (BWOA) for Feature Selection.
    """
    def __init__(self, **kwargs):
        kwargs['lb'] = 0.0
        kwargs['ub'] = 1.0
        super().__init__(**kwargs)

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            pos = (np.random.rand(self.dim) > 0.5).astype(float)
            if np.sum(pos) == 0:
                pos[np.random.randint(0, self.dim)] = 1.0
            self.population.append(Solution(position=pos))
        self.evaluate_population()
        self.update_best()

    def iterate(self, iteration: int):
        a = 2.0 - iteration * (2.0 / self.max_iter)
        
        for whale in self.population:
            A = 2.0 * a * np.random.rand(self.dim) - a
            C = 2.0 * np.random.rand(self.dim)
            l = np.random.uniform(-1.0, 1.0)
            p = np.random.rand()
            
            if p < 0.5:
                if np.linalg.norm(A) < 1.0:
                    D = np.abs(C * self.best_solution.position - whale.position)
                    cont_pos = self.best_solution.position - A * D
                else:
                    rand_idx = np.random.randint(0, self.pop_size)
                    rand_whale = self.population[rand_idx]
                    D = np.abs(C * rand_whale.position - whale.position)
                    cont_pos = rand_whale.position - A * D
            else:
                D_prime = np.abs(self.best_solution.position - whale.position)
                cont_pos = D_prime * np.exp(self.b * l) * np.cos(2.0 * np.pi * l) + self.best_solution.position
                
            prob = self.sigmoid_transfer(cont_pos)
            binary_pos = (np.random.rand(self.dim) < prob).astype(float)
            
            if np.sum(binary_pos) == 0:
                binary_pos[np.random.randint(0, self.dim)] = 1.0
                
            whale.position = binary_pos
            
        self.evaluate_population()
