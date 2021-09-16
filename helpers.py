import numpy as np
from aco import aco


def get_random_params(I, N, M, K, L):
    T1 = np.random.randint(5)
    T2 = np.random.randint(5)
    T3 = np.random.randint(5)
    T4 = np.random.randint(5)
    Q1 = I
    Q2 = I
    a_ij = np.random.randint(5, size=(I+1, I+1)) + 1
    np.fill_diagonal(a_ij, 0)
    b_ij = np.random.randint(5, size=(I+1, I+1)) + 1
    np.fill_diagonal(b_ij, 0)
    return {
        "I": I,
        "N": N,
        "M": M,
        "K": K,
        "L": L,
        "T1": T1,
        "T2": T2,
        "T3": T3,
        "T4": T4,
        "Q1": Q1,
        "Q2": Q2,
        "a_ij": a_ij,
        "b_ij": b_ij,
    }


def convert_to_solution(x, problem_instance, use_ACO=False):
    I = problem_instance.I
    N = problem_instance.N
    M = problem_instance.M
    K = problem_instance.K
    L = problem_instance.L
    a_ij = problem_instance.a_ij
    b_ij = problem_instance.b_ij
    begin = 0
    end = I
    subX = x[begin:end]
    supplierAssignment = np.zeros((I, N))
    supplierAssignment[np.arange(0, I), np.floor(subX * N).astype(int)] = 1
    x_n_ij = np.zeros((N, I+1, I+1))

    if use_ACO and a_ij is not None:
        for n in range(N):
            tmp = supplierAssignment[:, n]
            nodes = np.concatenate(([0], np.argwhere(tmp > 0).flatten() + 1))
            x_ij = np.zeros((I + 1, I + 1))
            x_ij[np.ix_(nodes, nodes)], _ = aco(a_ij[np.ix_(nodes, nodes)])
            x_n_ij[n] = x_ij
    else:
        begin = end
        end = end + I
        subX = x[begin:end]
        for n in range(N):
            tmp = supplierAssignment[:, n]
            nodes = np.argwhere(tmp > 0).flatten() + 1
            nodes = sorted(nodes, key=lambda x: subX[x-1])
            nodes = np.concatenate(([0], nodes, [0])).astype(int)
            x_n_ij[n, nodes[:-1], nodes[1:]] = 1

    begin = end
    end = begin + N
    subX = x[begin:end]
    u_nk = np.zeros((N, K))
    u_nk[np.arange(0, N), np.floor(subX * K).astype(int)] = 1

    begin = end
    end = begin + M
    subX = x[begin:end]
    v_ml = np.zeros((M, L))
    v_ml[np.arange(0, M), np.floor(subX * L).astype(int)] = 1

    begin = end
    end = begin + I
    subX = x[begin:end]
    clientAssignment = np.zeros((I, M))
    clientAssignment[np.arange(0, I), np.floor(subX * M).astype(int)] = 1
    y_m_ij = np.zeros((M, I+1, I+1))

    if use_ACO and a_ij is not None:
        for m in range(M):
            tmp = clientAssignment[:, m]
            nodes = np.concatenate(([0], np.argwhere(tmp > 0).flatten() + 1))
            y_ij = np.zeros((I + 1, I + 1))
            y_ij[np.ix_(nodes, nodes)], _ = aco(b_ij[np.ix_(nodes, nodes)])
            y_m_ij[m] = y_ij
    else:
        begin = end
        end = end + I
        subX = x[begin:end]
        for m in range(M):
            tmp = clientAssignment[:, m]
            nodes = np.argwhere(tmp > 0).flatten() + 1
            nodes = sorted(nodes, key=lambda x: subX[x-1])
            nodes = np.concatenate(([0], nodes, [0])).astype(int)
            y_m_ij[m, nodes[:-1], nodes[1:]] = 1

    return x_n_ij, u_nk, v_ml, y_m_ij