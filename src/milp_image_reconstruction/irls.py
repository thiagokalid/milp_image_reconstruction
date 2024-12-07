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
    f1 = None

    for k in range(maxiter):
        W1 = np.diag(np.sqrt(f0) ** (-1))
        W2 = np.diag(np.sqrt(err) ** (-1))

        A1 = W2 @ A
        b1 = W2 @ b

        f1 = minres(
            A1.T @ A1 + lbd * W1,
            A1.T @ b1
        )[0]

        ek = (norm(f1 - f0, 2) / norm(f0, 2)) ** 2
        f0 = np.sqrt(f1 ** 2 + epsilon)
        err = np.sqrt((b - A @ f1) ** 2 + epsilon)
        print("ek = ", ek)
        if ek < tolLower:
            return f0, b - A @ f0
    print("Not converged.")
    return f1, b - A @ f1