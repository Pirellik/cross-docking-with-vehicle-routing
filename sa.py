import numpy as np
from tqdm import tqdm


def sa(cost_f, lb, ub, maxiter=10000, step_coeff=0.001, display_progress=False):
    lb = np.array(lb)
    ub = np.array(ub)
    x = np.random.rand(len(lb))
    x = lb + x * (ub - lb)
    cost = cost_f(x)
    best_x = x
    best_cost = cost
    iters = range(maxiter)
    if display_progress:
        iters = tqdm(iters)
    for iter_num in iters:
        fraction = iter_num / float(maxiter)
        T = max(0.0000001, min(1, 1 - fraction))
        amplitude = (ub - lb) * step_coeff
        delta = (np.random.rand(len(amplitude)) - 0.5) * amplitude
        new_x = np.clip(x + delta, a_min=lb, a_max=ub)
        new_cost = cost_f(new_x)
        if new_cost < cost:
            probability = 1
        else:
            probability = np.exp(- (new_cost - cost) / T)
        if probability > np.random.random():
            x, cost = new_x, new_cost
            if new_cost < best_cost:
                best_x, best_cost = new_x, new_cost

    return best_x, best_cost
