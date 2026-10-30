"""
Particle Swarm Optimization (PSO) & Binary Particle Swarm Optimization (BPSO).
Reference: Kennedy & Eberhart, "Particle Swarm Optimization", IEEE ICNN, 1995.
"""

import numpy as np
from typing import List, Optional
from .base import BaseMetaheuristic, Solution


class ParticleSwarmOptimizer(BaseMetaheuristic):
    """
    Particle Swarm Optimization with inertia weight damping, cognitive (c1), and social (c2) acceleration.
    """
    def __init__(
        self,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.velocities: List[np.ndarray] = []
        self.pbest: List[Solution] = []

    def initialize_population(self):
        super().initialize_population()
        self.velocities = []
        self.pbest = []
        v_max = 0.2 * (self.ub - self.lb)
        for ind in self.population:
            vel = np.random.uniform(-v_max, v_max, self.dim)
            self.velocities.append(vel)
            self.pbest.append(Solution(position=ind.position.copy(), fitness=ind.fitness))

    def iterate(self, iteration: int):
        # Linearly decreasing inertia weight w
        w = self.w_max - iteration * ((self.w_max - self.w_min) / self.max_iter)
        v_max = 0.2 * (self.ub - self.lb)
        
        for i, ind in enumerate(self.population):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            
            # Velocity update
            cognitive = self.c1 * r1 * (self.pbest[i].position - ind.position)
            social = self.c2 * r2 * (self.best_solution.position - ind.position)
            self.velocities[i] = w * self.velocities[i] + cognitive + social
            self.velocities[i] = np.clip(self.velocities[i], -v_max, v_max)
            
            # Position update
            new_pos = ind.position + self.velocities[i]
            ind.position = self.enforce_bounds(new_pos)
            
        self.evaluate_population()
        
        # Update Personal Best (pbest)
        for i, ind in enumerate(self.population):
            if ind.fitness > self.pbest[i].fitness:
                self.pbest[i] = Solution(position=ind.position.copy(), fitness=ind.fitness)


class BinaryPSO(ParticleSwarmOptimizer):
    """
    Binary Particle Swarm Optimization (BPSO) for Feature Selection.
    Applies Sigmoid transfer function to map continuous particle velocity into probability of bit flipping.
    """
    def __init__(self, **kwargs):
        kwargs['lb'] = 0.0
        kwargs['ub'] = 1.0
        super().__init__(**kwargs)

    def initialize_population(self):
        self.population = []
        self.velocities = []
        self.pbest = []
        
        for _ in range(self.pop_size):
            pos = (np.random.rand(self.dim) > 0.5).astype(float)
            if np.sum(pos) == 0:
                pos[np.random.randint(0, self.dim)] = 1.0
            vel = np.random.uniform(-4.0, 4.0, self.dim)
            
            self.population.append(Solution(position=pos))
            self.velocities.append(vel)
            
        self.evaluate_population()
        self.update_best()
        
        for ind in self.population:
            self.pbest.append(Solution(position=ind.position.copy(), fitness=ind.fitness))

    def iterate(self, iteration: int):
        w = self.w_max - iteration * ((self.w_max - self.w_min) / self.max_iter)
        v_max = 6.0
        
        for i, ind in enumerate(self.population):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive = self.c1 * r1 * (self.pbest[i].position - ind.position)
            social = self.c2 * r2 * (self.best_solution.position - ind.position)
            
            self.velocities[i] = w * self.velocities[i] + cognitive + social
            self.velocities[i] = np.clip(self.velocities[i], -v_max, v_max)
            
            # Sigmoid transfer to bit flip probability
            s_v = self.sigmoid_transfer(self.velocities[i])
            binary_pos = (np.random.rand(self.dim) < s_v).astype(float)
            
            if np.sum(binary_pos) == 0:
                binary_pos[np.random.randint(0, self.dim)] = 1.0
                
            ind.position = binary_pos
            
        self.evaluate_population()
        
        # Update Personal Best
        for i, ind in enumerate(self.population):
            if ind.fitness > self.pbest[i].fitness:
                self.pbest[i] = Solution(position=ind.position.copy(), fitness=ind.fitness)
