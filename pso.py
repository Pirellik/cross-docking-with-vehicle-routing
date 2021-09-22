import numpy as np
from tqdm import tqdm
import multiprocess as mp
import time


def pso(func, lb, ub, swarm_size=16, w=0.5, c1=0.5, c2=0.5, maxiter=100,
        processes=4, display_progress=False, debug_output=False):

    mp_pool = mp.Pool(processes)

    lb = np.array(lb)
    ub = np.array(ub)

    vhigh = np.abs(ub - lb)
    vlow = -vhigh

    S = swarm_size
    D = len(lb)
    v = vlow + np.random.rand(S, D)*(vhigh - vlow)
    x = lb + np.random.rand(S, D) * (ub - lb)
    fx = np.array(mp_pool.map(func, x))
    p = x.copy()
    fp = fx
    i_min = np.argmin(fp)
    fg = fp[i_min]
    g = p[i_min, :].copy()
    if debug_output:
        start = time.time()
        times = np.zeros(maxiter)
        costs = np.zeros(maxiter)
    iters = range(maxiter)
    if display_progress:
        iters = tqdm(iters)
    for iter_num in iters:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))

        v = w*v + c1*rp*(p - x) + c2*rg*(g - x)
        x = np.clip(x + v, a_min=lb, a_max=ub)
        fx = np.array(mp_pool.map(func, x))

        i_update = fx < fp
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            p_min = p[i_min, :].copy()
            stepsize = np.sqrt(np.sum((g - p_min)**2))
            g = p_min.copy()
            fg = fp[i_min]
        if debug_output:
            times[iter_num] = time.time() - start
            costs[iter_num] = fg
    if debug_output:
        return g, fg, times, costs
    return g, fg