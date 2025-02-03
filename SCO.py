import time
import numpy as np


def SCO(S, fobj, lb, ub, T):
    FN, dim, = S.shape[0], S.shape[1]
    ub = np.multiply(ub, np.ones((1, dim)))
    lb = np.multiply(lb, np.ones((1, dim)))
    Range = ub - lb
    P = 0
    BF = fobj(S)
    POO = 0
    m = 5
    alpha = 1000
    b = 2.4
    w = np.zeros((T))
    F = np.zeros((T))
    Best_Fitness = np.zeros((T))
    x = np.zeros((dim))
    ct = time.time()
    for t in range(len(np.arange(1, T + 1).reshape(-1))):
        w[t] = np.exp(- (b * t / T) ** b)
        if t > alpha:
            if sum(P) == 0:
                POO = 1 + POO
        K = np.random.rand()
        for j in range(len(np.arange(1, dim + 1).reshape(-1))):
            EE = w[t] * K * Range[j]
            if t < alpha:
                if np.random.rand() < 0.5:
                    # x[j] = S[j] + (w[t] * np.abs(S[j]))
                    x[j] = np.random.rand()
                else:
                    # x[j] = S[j] - (w[t] * np.abs(S[j]))
                    x[j] = np.random.rand()
            else:
                if POO == m:
                    POO = 0
                    if np.random.rand() < 0.5:
                        x[j] = S[j] + np.random.rand() * Range(j)
                    else:
                        x[j] = S[j] - np.random.rand() * Range(j)
                else:
                    if np.random.rand() < 0.5:
                        x[j] = S[j] + EE
                    else:
                        x[j] = S[j] - EE
            ## Check if a dimension of the candidate solution goes out of boundaries
                x[j] = S[j]
        ## Evaluate the fitness of the newly generated candidate sloution
        F[t] = fobj(x)
        if F[t].any() < BF.any():
            BF = F[t]
            S = x
            P = 1
        else:
            P = 0
        Best_Fitness[t] = BF[0]
        gbest = S
    ct = time.time() - ct
    return BF, Best_Fitness, gbest, ct
