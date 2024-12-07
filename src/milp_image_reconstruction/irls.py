import numpy as np

from numpy.linalg import norm
from scipy.sparse.linalg import lsqr, cg, minres

__all__ = ["irls_minres"]

def irls_minres(A, b, maxiter, xguess, lbd=1e-4, tolLower=1e-2, epsilon=1e-4):
    '''
		Solves Ax = b through x = (A.T @ A)^-1 @ A.T @ b using IRLS
		'''

    N = A.shape[1]
    W = np.zeros(shape=(N, N))

    f = np.zeros(N)
    f0 = np.sqrt(xguess ** 2 + epsilon)
    err = np.sqrt((b - A @ xguess) ** 2 + epsilon)
    f1 = xguess
    f1_log = []
    cost_fun_log = []
    converged = False
    cost_fun = np.inf

    for k in range(maxiter):
        W1 = np.diag(np.sqrt(err) ** (-1))
        W2 = np.diag(np.sqrt(f0) ** (-1))

        A1 = W1 @ A
        b1 = W1 @ b
        err1 = W1 @ err

        deltax = lsqr(
            A1.T @ A1,
            A1.T @ err1,
            atol=1e-3
        )[0]

        f1 = f1 + deltax

        # Cost-function is sum |gi - Hi * fi| + lambda |fi|
        cost_fun = np.sum(np.abs(b - A @ f1)) + lbd * np.sum(np.abs(f1))

        # Approximation of |x| ≃ sqrt(x² + delta) where delta -> 0.
        f0 = np.sqrt(f1 ** 2 + epsilon)
        err = np.sqrt((b - A @ f1) ** 2 + epsilon)

        # Logs into history:
        f1_log.append(f1)
        cost_fun_log.append(cost_fun)
        print(f"Cost-function = {cost_fun}")

        if cost_fun < tolLower:
            converged = True
            break
    x = f1
    residue = b - A @ f1
    x_log = f1_log

    return x, residue, cost_fun, converged, x_log, cost_fun_log