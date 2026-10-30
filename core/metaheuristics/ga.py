"""
Genetic Algorithm (GA) & Binary Genetic Algorithm (BGA).
Classical evolutionary optimization benchmark with tournament selection, crossover, mutation, and elitism.
"""

import numpy as np
from typing import List
from .base import BaseMetaheuristic, Solution


class GeneticAlgorithm(BaseMetaheuristic):
    """
    Continuous Genetic Algorithm with arithmetic crossover and Gaussian mutation.
    """
    def __init__(
        self,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        tournament_size: int = 3,
        elitism_count: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count

    def _tournament_selection(self) -> Solution:
        candidates = np.random.choice(self.population, size=self.tournament_size, replace=False)
        return max(candidates, key=lambda s: s.fitness)

    def iterate(self, iteration: int):
        new_pop: List[Solution] = []
        
        # Elitism: retain top individuals
        sorted_pop = sorted(self.population, key=lambda s: s.fitness, reverse=True)
        for i in range(min(self.elitism_count, self.pop_size)):
            new_pop.append(Solution(position=sorted_pop[i].position.copy(), fitness=sorted_pop[i].fitness))
            
        while len(new_pop) < self.pop_size:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if np.random.rand() < self.crossover_rate:
                alpha = np.random.rand(self.dim)
                child_pos = alpha * parent1.position + (1.0 - alpha) * parent2.position
            else:
                child_pos = parent1.position.copy()
                
            # Mutation (Gaussian perturbation)
            for d in range(self.dim):
                if np.random.rand() < self.mutation_rate:
                    sigma = 0.1 * (self.ub[d] - self.lb[d])
                    child_pos[d] += np.random.normal(0, sigma)
                    
            child_pos = self.enforce_bounds(child_pos)
            new_pop.append(Solution(position=child_pos))
            
        self.population = new_pop[:self.pop_size]
        self.evaluate_population()


class BinaryGA(BaseMetaheuristic):
    """
    Binary Genetic Algorithm (BGA) for Feature Selection.
    Uses two-point crossover, bit-flip mutation, and elitism.
    """
    def __init__(
        self,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        tournament_size: int = 3,
        elitism_count: int = 2,
        **kwargs
    ):
        kwargs['lb'] = 0.0
        kwargs['ub'] = 1.0
        super().__init__(**kwargs)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            pos = (np.random.rand(self.dim) > 0.5).astype(float)
            if np.sum(pos) == 0:
                pos[np.random.randint(0, self.dim)] = 1.0
            self.population.append(Solution(position=pos))
        self.evaluate_population()
        self.update_best()

    def _tournament_selection(self) -> Solution:
        candidates = np.random.choice(self.population, size=self.tournament_size, replace=False)
        return max(candidates, key=lambda s: s.fitness)

    def iterate(self, iteration: int):
        new_pop: List[Solution] = []
        sorted_pop = sorted(self.population, key=lambda s: s.fitness, reverse=True)
        for i in range(min(self.elitism_count, self.pop_size)):
            new_pop.append(Solution(position=sorted_pop[i].position.copy(), fitness=sorted_pop[i].fitness))
            
        while len(new_pop) < self.pop_size:
            p1 = self._tournament_selection()
            p2 = self._tournament_selection()
            
            # Two-point crossover
            if np.random.rand() < self.crossover_rate and self.dim > 2:
                pts = sorted(np.random.choice(range(1, self.dim), size=2, replace=False))
                child_pos = np.concatenate([p1.position[:pts[0]], p2.position[pts[0]:pts[1]], p1.position[pts[1]:]])
            else:
                child_pos = p1.position.copy()
                
            # Bit-flip mutation
            for d in range(self.dim):
                if np.random.rand() < self.mutation_rate:
                    child_pos[d] = 1.0 - child_pos[d]
                    
            if np.sum(child_pos) == 0:
                child_pos[np.random.randint(0, self.dim)] = 1.0
                
            new_pop.append(Solution(position=child_pos))
            
        self.population = new_pop[:self.pop_size]
        self.evaluate_population()
