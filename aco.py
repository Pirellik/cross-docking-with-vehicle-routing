import numpy as np
from random import random
from tqdm import tqdm


aco_cache = {}


def aco(distance_matrix, num_ants=10, alpha=1, beta=1, ro=0.5, maxiter=10, display_progress=False):
    distance_matrix = np.array(distance_matrix)
    h = hash(distance_matrix.tobytes())
    sol = aco_cache.get(h)
    if sol is not None:
        return sol

    pheromones = np.ones(distance_matrix.shape)

    ret_x, ret_cost = None, 10**10
    iters = range(maxiter)
    if display_progress:
        iters = tqdm(iters)
    for i in iters:
        solutions = np.zeros((num_ants, *distance_matrix.shape))
        for ant in range(num_ants):
            visited = np.zeros(distance_matrix.shape[0]-1)

            i = 0
            j = 0
            while np.sum(visited) != len(visited):
                allowed_j = np.setdiff1d(np.argwhere(visited != 1).flatten() + 1, np.array([i]))
                weights = np.array([pheromones[i, j] ** alpha * (1 / (distance_matrix[i, j]+0.01)) ** beta for j in allowed_j]).flatten()
                j = np.random.choice(allowed_j, 1, p=weights/np.sum(weights))
                solutions[ant, i, j] = 1
                visited[j - 1] = 1
                i = j
            solutions[ant, j, 0] = 1

        costs = np.array([np.sum(sol*distance_matrix) for sol in solutions])
        best_cost = np.min(costs)
        best_x = solutions[np.argmin(costs)]
        if best_cost < ret_cost:
            ret_cost = best_cost
            ret_x = best_x

        new_pheromones = np.sum(np.array([sol/(cost+0.01 ** 100) for sol, cost in zip(solutions, costs)]), axis=0)
        pheromones = (1-ro) *pheromones + new_pheromones

    aco_cache[h] = (ret_x, ret_cost)
    return ret_x, ret_cost


if __name__ == "__main__":
    print(aco([[1, 2, 3, 2, 3], [3, 2, 4, 2, 3],[2, 1, 3, 2, 3], [3, 2, 4, 2, 3], [3, 2, 4, 2, 3]], display_progress=True))