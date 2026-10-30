import sys
import os
import unittest
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.metaheuristics import (
    GreyWolfOptimizer, BinaryGWO,
    ParticleSwarmOptimizer, BinaryPSO,
    WhaleOptimizer, BinaryWOA,
    GeneticAlgorithm, BinaryGA,
    HarrisHawksOptimizer, ChimpOptimizer,
    MetaheuristicEnsembleOptimizer,
    MetaheuristicFeatureSelector
)

class TestMetaheuristics(unittest.TestCase):

    def setUp(self):
        # 5-Dimensional Sphere function: min f(x) = sum(x_i^2) -> max -sum(x_i^2)
        self.sphere_func = lambda x: -float(np.sum(x**2))

    def test_continuous_optimizers_convergence(self):
        optimizers = [
            GreyWolfOptimizer(objective_func=self.sphere_func, dim=4, lb=-5.0, ub=5.0, pop_size=15, max_iter=15, seed=42),
            ParticleSwarmOptimizer(objective_func=self.sphere_func, dim=4, lb=-5.0, ub=5.0, pop_size=15, max_iter=15, seed=42),
            WhaleOptimizer(objective_func=self.sphere_func, dim=4, lb=-5.0, ub=5.0, pop_size=15, max_iter=15, seed=42),
            GeneticAlgorithm(objective_func=self.sphere_func, dim=4, lb=-5.0, ub=5.0, pop_size=15, max_iter=15, seed=42),
            HarrisHawksOptimizer(objective_func=self.sphere_func, dim=4, lb=-5.0, ub=5.0, pop_size=15, max_iter=15, seed=42),
            ChimpOptimizer(objective_func=self.sphere_func, dim=4, lb=-5.0, ub=5.0, pop_size=15, max_iter=15, seed=42)
        ]

        for opt in optimizers:
            res = opt.optimize()
            self.assertIsNotNone(res.best_position)
            self.assertEqual(len(res.best_position), 4)
            self.assertEqual(len(res.convergence_curve), 15)
            # Fitness should improve or stay non-decreasing
            self.assertTrue(res.convergence_curve[-1] >= res.convergence_curve[0])

    def test_binary_optimizers(self):
        # Binary target: find all 1s
        bin_func = lambda x: float(np.sum(x))
        
        bin_optimizers = [
            BinaryGWO(objective_func=bin_func, dim=8, pop_size=10, max_iter=10, seed=42),
            BinaryPSO(objective_func=bin_func, dim=8, pop_size=10, max_iter=10, seed=42),
            BinaryWOA(objective_func=bin_func, dim=8, pop_size=10, max_iter=10, seed=42),
            BinaryGA(objective_func=bin_func, dim=8, pop_size=10, max_iter=10, seed=42)
        ]

        for opt in bin_optimizers:
            res = opt.optimize()
            self.assertIsNotNone(res.best_position)
            # All elements should be 0 or 1
            self.assertTrue(np.all((res.best_position == 0) | (res.best_position == 1)))

    def test_ensemble_optimizer(self):
        p1 = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])
        p2 = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
        y_true = np.array([0, 1, 0])
        
        opt = MetaheuristicEnsembleOptimizer(algorithm='gwo', pop_size=10, max_iter=10)
        res = opt.fit([p1, p2], y_true, model_names=['M1', 'M2'])
        self.assertIn('model_weights', res)
        self.assertIn('optimized_accuracy', res)

if __name__ == '__main__':
    unittest.main()
