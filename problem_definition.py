import numpy as np
from copy import copy
import math
from aco import aco


class ProblemDefinition(dict):
    def __init__(self, params):
        self.I = params["I"]
        self.N = params["N"]
        self.M = params["M"]
        self.K = params["K"]
        self.L = params["L"]
        self.T1 = params["T1"]
        self.T2 = params["T2"]
        self.T3 = params["T3"]
        self.T4 = params["T4"]
        self.Q1 = params["Q1"]
        self.Q2 = params["Q2"]
        self.a_ij = np.array(params["a_ij"])
        self.b_ij = np.array(params["b_ij"])
        self.cost_cache = {}

    def cost_function(self, x_n_ij, u_nk, v_ml, y_m_ij):
        bts = x_n_ij.tobytes()+u_nk.tobytes()+v_ml.tobytes()+y_m_ij.tobytes()
        h = hash(bts)
        cost = self.cost_cache.get(h)
        if cost is not None:
            return cost
        I    = copy(self.I )
        N    = copy(self.N )
        M    = copy(self.M )
        K    = copy(self.K )
        L    = copy(self.L )
        T1   = copy(self.T1)
        T2   = copy(self.T2)
        T3   = copy(self.T3)
        T4   = copy(self.T4)
        Q1   = copy(self.Q1)
        Q2   = copy(self.Q2)
        a_ij = copy(self.a_ij)
        b_ij = copy(self.b_ij)

        c_ij = a_ij
        c_ij[:, 1:] = c_ij[:, 1:] + T1
        t1_nk = (np.sum(np.sum(c_ij * x_n_ij, axis=1), axis=1) * u_nk.T).T

        t2_nk = np.zeros((N, K))
        for k in range(0, K):
            t2_nk[0, k] = (t1_nk[0, k] + T2 * np.sum(np.sum(x_n_ij[0, :, 1:], axis=0), axis=0)) * u_nk[0, k]

        def f(n, k):
            if n < 0:
                return 0
            if u_nk[n, k] > 0:
                return t2_nk[n, k]
            return f(n-1, k)

        def heaviside_f(x):
            if x > 0:
                return 1
            return 0

        for n in range(1, N):
            for k in range(0, K):
                f_ret = f(n-1, k)
                tmp = t1_nk[n, k] - f_ret
                t2_nk[n, k] = (f_ret + tmp * heaviside_f(tmp) + T2 * np.sum(np.sum(x_n_ij[n, :, 1:], axis=0), axis=0)) * u_nk[n, k]

        t3_n = np.sum(t2_nk, axis=1)

        t4_i = np.max(np.sum(x_n_ij[:,:, 1:], axis=1).T * t3_n, axis=1)

        t5_ml = (np.max(t4_i * np.sum(y_m_ij[:,:, 1:], axis=1), axis=1) * v_ml.T).T

        def g(m, l):
            if m < 0:
                return 0
            if v_ml[m, l] > 0:
                return t6_ml[m, l]
            return g(m-1, l)

        t6_ml = np.array([np.zeros(L)]*M)
        for l in range(0, L):
            t6_ml[0, l] = (t5_ml[0, l] + T3 * np.sum(np.sum(y_m_ij[0, :, 1:], axis=0), axis=0)) * v_ml[0, l]

        for m in range(1, M):
            for l in range(0, L):
                g_ret = g(m-1, l)
                tmp = t5_ml[m, l] - g_ret
                t6_ml[m, l] = (g_ret + tmp * heaviside_f(tmp) + T3 * np.sum(np.sum(y_m_ij[m, :, 1:], axis=0), axis=0)) * v_ml[m, l]

        t7_m = np.sum(t6_ml, axis=0)

        d_ij = b_ij
        d_ij[:, 1:] = d_ij[:, 1:] + T4
        c_max = np.max(t7_m + np.sum((np.sum(np.sum(d_ij * y_m_ij, axis=1), axis=1) * v_ml.T).T, axis=0))
        self.cost_cache[h] = c_max
        return c_max
