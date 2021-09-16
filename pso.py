import numpy as np
from tqdm import tqdm
import multiprocess as mp


def pso(func, lb, ub, swarmsize=16, omega=0.5, phip=0.5, phig=0.5, maxiter=100,
        minstep=1e-8, minfunc=1e-8, processes=4, display_progress=False):

    mp_pool = mp.Pool(processes)

    lb = np.array(lb)
    ub = np.array(ub)

    vhigh = np.abs(ub - lb)
    vlow = -vhigh

    S = swarmsize
    D = len(lb)
    v = vlow + np.random.rand(S, D)*(vhigh - vlow)
    x = lb + np.random.rand(S, D) * (ub - lb)
    fx = np.array(mp_pool.map(func, x))
    p = x.copy()
    fp = fx
    i_min = np.argmin(fp)
    fg = fp[i_min]
    g = p[i_min, :].copy()

    iters = range(maxiter)
    if display_progress:
        iters = tqdm(iters)
    for _ in iters:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))

        v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)
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

    return g, fg