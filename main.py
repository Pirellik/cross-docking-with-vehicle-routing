import numpy as np
from problem_definition import ProblemDefinition
from aco import aco
from pso import pso
from sa import sa
from helpers import get_random_params, convert_to_solution

I = 20
N = 15
M = 12
K = 5
L = 5

p = get_random_params(I, N, M, K, L)
inst = ProblemDefinition(p)


def cost_no_ACO(x):
    return inst.cost_function(*convert_to_solution(x, inst))


def cost_with_ACO(x):
    return inst.cost_function(*convert_to_solution(x, inst, use_ACO=True))


if __name__ == "__main__":
    from sa import sa
    from pso import pso

    lb = [0]*(2*I+N+M)
    ub = [0.999999]*(2*I+N+M)

    print(sa(cost_with_ACO, lb, ub, maxiter=20, display_progress=True))
    print(pso(cost_with_ACO, lb, ub, maxiter=10, swarmsize=16, processes=16, display_progress=True))

    lb = [0]*(4*I+N+M)
    ub = [0.999999]*(4*I+N+M)

    print(sa(cost_no_ACO, lb, ub, maxiter=20, display_progress=True))
    print(pso(cost_no_ACO, lb, ub, maxiter=10, swarmsize=16, processes=16, display_progress=True))
