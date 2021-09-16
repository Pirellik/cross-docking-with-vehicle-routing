import unittest
from pso import pso
from sa import sa
from aco import aco
import numpy as np
from problem_definition import ProblemDefinition

def cost_f(x):
    return np.sum(x ** 2)

class TestPSO(unittest.TestCase):
    def test_pso_2D_quadratic(self):
        lb = [-100, -100]
        ub = [100, 100]
        x, cost = pso(cost_f, lb, ub)
        self.assertLess(cost, 0.01)

    def test_pso_3D_quadratic(self):
        lb = [-100, -100, -100]
        ub = [100, 100, 100]
        x, cost = pso(cost_f, lb, ub)
        self.assertLess(cost, 0.01)

    def test_pso_5D_quadratic(self):
        lb = [-100, -100, -100, -100, -100]
        ub = [100, 100, 100, 100, 100]
        x, cost = pso(cost_f, lb, ub, swarmsize=100)
        self.assertLess(cost, 0.01)

class TestSA(unittest.TestCase):
    def test_sa_2D_quadratic(self):
        lb = [-100, -100]
        ub = [100, 100]
        x, cost = sa(cost_f, lb, ub)
        self.assertLess(cost, 0.01)

    def test_sa_3D_quadratic(self):
        lb = [-100, -100, -100]
        ub = [100, 100, 100]
        x, cost = sa(cost_f, lb, ub)
        self.assertLess(cost, 0.01)

    def test_sa_5D_quadratic(self):
        lb = [-100, -100, -100, -100, -100]
        ub = [100, 100, 100, 100, 100]
        x, cost = sa(cost_f, lb, ub)
        self.assertLess(cost, 0.01)

class TestACO(unittest.TestCase):
    def test_ACO_3x3(self):
        d = [[1, 2, 3],
             [3, 2, 4],
             [2, 1, 1]]
        x, cost = aco(d, num_ants=20, maxiter=50)
        self.assertLess(cost, 8)

    def test_ACO_5x5(self):
        d = [[1, 2, 3, 3, 3],
             [3, 2, 4, 3, 4],
             [2, 1, 1, 2, 2],
             [3, 2, 4, 3, 4],
             [2, 1, 1, 2, 2]]
        x, cost = aco(d, num_ants=20, maxiter=50)
        self.assertLess(cost, 12)

    def test_ACO_10x10(self):
        d = [[1, 2, 3, 3, 3, 1, 2, 3, 3, 3],
             [3, 2, 4, 3, 4, 2, 1, 1, 2, 2],
             [2, 1, 1, 2, 2, 2, 1, 1, 2, 2],
             [3, 2, 4, 3, 4, 1, 2, 3, 3, 3],
             [2, 1, 1, 2, 2, 3, 4, 5, 1, 1],
             [2, 1, 1, 2, 2, 2, 1, 1, 2, 2],
             [3, 2, 4, 3, 4, 1, 2, 3, 3, 3],
             [2, 1, 1, 2, 2, 3, 4, 5, 1, 1],
             [1, 2, 3, 3, 3, 1, 2, 3, 3, 3],
             [3, 2, 4, 3, 4, 2, 1, 1, 2, 2]]
        x, cost = aco(d, num_ants=50, maxiter=100)
        self.assertLess(cost, 16)

class TestCostFunction(unittest.TestCase):
    def test_simple_instance(self):
        params = {}

        params["I"] = 3 # liczba zamówień = liczba dostawców = liczba klientów
        params["N"] = 2 # liczba cięzarówek zaopatrzeniowych
        params["M"] = 2 # liczba cięzarówek dostawczych
        params["K"] = 2 # liczba doków rozładunkowych
        params["L"] = 2 # liczba doków załadunkowych

        params["T1"] = 1 # czas trwania załadunku jednej cięzarówki u dostawcy
        params["T2"] = 2 # czas trwania rozładunku jednej cięzarówki w doku rozładukowym
        params["T3"] = 3 # czas trwania załadunku jednej cięzarówki w doku załadunkowym
        params["T4"] = 4 # czas trwania rozładunku jednej cięzarówki u klienta detalicznego

        params["Q1"] = 3 # pojemnosc ciezarowki zaopatrzeniowej
        params["Q2"] = 3 # pojemnosc ciezarowki dostawczej

        # czas przejazdu cięzarówki od dostawcy i do dostawcy j
        params["a_ij"] = np.array([[0, 2, 1, 3],
                                   [2, 0, 2, 1],
                                   [1, 2, 0, 1],
                                   [3, 1, 1, 0]])

        # czas przejazdu cięzarówki od klienta i do klienta j
        params["b_ij"] = np.array([[0, 2, 1, 3],
                                   [2, 0, 2, 1],
                                   [1, 2, 0, 1],
                                   [3, 1, 1, 0]])

        # Zmienne decyzyjne
        x_n_ij = np.array([[[0, 1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]],
                           [[0, 0, 1, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [1, 0, 0, 0]]])

        y_m_ij = np.array([[[0, 1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]],
                           [[0, 0, 1, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [1, 0, 0, 0]]])

        u_nk = np.array([[1, 0],
                         [0, 1]])
        v_ml = np.array([[1, 0],
                         [0, 1]])

        inst = ProblemDefinition(params)
        cost = inst.cost_function(x_n_ij, u_nk, v_ml, y_m_ij)
        self.assertEqual(cost, 30.0)

if __name__ == '__main__':
    unittest.main()