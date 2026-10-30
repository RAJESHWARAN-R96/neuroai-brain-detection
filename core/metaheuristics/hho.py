"""
Harris Hawks Optimization (HHO).
Reference: Heidari et al., "Harris hawks optimization: Algorithm and applications",
Future Generation Computer Systems, 2019.
"""

import numpy as np
import scipy.special as special
from .base import BaseMetaheuristic, Solution


class HarrisHawksOptimizer(BaseMetaheuristic):
    """
    Harris Hawks Optimization (HHO) with exploration, soft besiege, hard besiege,
    and progressive rapid dives with Lévy flights.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _levy_flight(dim: int) -> np.ndarray:
        """Generates step from Lévy distribution."""
        beta = 1.5
        sigma = (
            special.gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0) /
            (special.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0)))
        ) ** (1.0 / beta)
        
        u = np.random.normal(0, sigma, size=dim)
        v = np.random.normal(0, 1.0, size=dim)
        step = u / (np.abs(v) ** (1.0 / beta))
        return step

    def iterate(self, iteration: int):
        # Escaping energy of prey: decreases from 2 to 0
        E0 = 2.0 * np.random.rand() - 1.0  # [-1, 1]
        E = 2.0 * E0 * (1.0 - (iteration / self.max_iter))
        
        rabbit = self.best_solution.position
        
        for i, hawk in enumerate(self.population):
            # Phase 1: Exploration (|E| >= 1)
            if np.abs(E) >= 1.0:
                q = np.random.rand()
                if q >= 0.5:
                    rand_idx = np.random.randint(0, self.pop_size)
                    rand_hawk = self.population[rand_idx].position
                    r1, r2 = np.random.rand(), np.random.rand()
                    new_pos = rand_hawk - r1 * np.abs(rand_hawk - 2.0 * r2 * hawk.position)
                else:
                    r3, r4 = np.random.rand(), np.random.rand()
                    avg_pos = np.mean([h.position for h in self.population], axis=0)
                    new_pos = (rabbit - avg_pos) - r3 * (self.lb + r4 * (self.ub - self.lb))
                    
                hawk.position = self.enforce_bounds(new_pos)
                
            # Phase 2: Exploitation (|E| < 1)
            else:
                r = np.random.rand()
                J = 2.0 * (1.0 - np.random.rand())  # Random jump strength of prey
                
                # Case 1: Soft besiege (r >= 0.5 and |E| >= 0.5)
                if r >= 0.5 and np.abs(E) >= 0.5:
                    delta_X = rabbit - hawk.position
                    new_pos = delta_X - E * np.abs(J * rabbit - hawk.position)
                    hawk.position = self.enforce_bounds(new_pos)
                    
                # Case 2: Hard besiege (r >= 0.5 and |E| < 0.5)
                elif r >= 0.5 and np.abs(E) < 0.5:
                    delta_X = rabbit - hawk.position
                    new_pos = rabbit - E * np.abs(delta_X)
                    hawk.position = self.enforce_bounds(new_pos)
                    
                # Case 3: Soft besiege with rapid dives (r < 0.5 and |E| >= 0.5)
                elif r < 0.5 and np.abs(E) >= 0.5:
                    # Strategy Y
                    Y = rabbit - E * np.abs(J * rabbit - hawk.position)
                    Y = self.enforce_bounds(Y)
                    
                    # Strategy Z with Levy flight
                    LF = self._levy_flight(self.dim)
                    S = np.random.rand(self.dim)
                    Z = Y + S * LF
                    Z = self.enforce_bounds(Z)
                    
                    val_Y = self.objective_func(Y) if self.maximize else -self.objective_func(Y)
                    val_Z = self.objective_func(Z) if self.maximize else -self.objective_func(Z)
                    
                    if val_Y > hawk.fitness:
                        hawk.position = Y
                    elif val_Z > hawk.fitness:
                        hawk.position = Z
                        
                # Case 4: Hard besiege with rapid dives (r < 0.5 and |E| < 0.5)
                elif r < 0.5 and np.abs(E) < 0.5:
                    avg_pos = np.mean([h.position for h in self.population], axis=0)
                    Y = rabbit - E * np.abs(J * rabbit - avg_pos)
                    Y = self.enforce_bounds(Y)
                    
                    LF = self._levy_flight(self.dim)
                    S = np.random.rand(self.dim)
                    Z = Y + S * LF
                    Z = self.enforce_bounds(Z)
                    
                    val_Y = self.objective_func(Y) if self.maximize else -self.objective_func(Y)
                    val_Z = self.objective_func(Z) if self.maximize else -self.objective_func(Z)
                    
                    if val_Y > hawk.fitness:
                        hawk.position = Y
                    elif val_Z > hawk.fitness:
                        hawk.position = Z

        self.evaluate_population()
