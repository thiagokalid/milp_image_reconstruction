import numpy as np
import scipy
import scipy.sparse.linalg as linalg

from acquisition import Acquisition

from scipy.optimize import milp


def passarin_method(basis_signal: np.ndarray, sampled_signal: np.ndarray, imgsize: tuple):
    A = basis_signal
    b = sampled_signal
    x = linalg.lsqr(A, b)[0]
    img = np.reshape(x, newshape=imgsize)
    return img.T


def naive_l1_method(basis_signal: np.ndarray, sampled_signal: np.ndarray, imgsize: tuple):
    M, N = basis_signal.shape
    g = sampled_signal
    H = basis_signal

    def cost_fun(f):
        return np.linalg.norm(g - H @ f, ord=1)

    result = scipy.optimize.minimize(fun=cost_fun, x0=np.zeros(N), method="SLSQP")

    img = np.reshape(result.x, newshape=imgsize)
    return img.T


def l1_method(basis_signal: np.ndarray, sampled_signal: np.ndarray, imgsize: tuple):
    N, M = basis_signal.shape
    g = sampled_signal.reshape(N, 1)
    H = basis_signal

    # c^T @ x
    c = np.ones(shape=(2*N + M, 1))
    c[:M] = 0

    #
    ei_matrix = np.identity(N)

    # A is 2N x (M + 2N)
    A = np.vstack([
        np.hstack((H, ei_matrix, np.zeros_like(ei_matrix))),
        np.hstack((-H, np.zeros_like(ei_matrix), ei_matrix))
    ])

    b_l = np.vstack((
        g,
        -g
    ))

    b_u = np.ones_like(b_l)

    constraints = scipy.optimize.LinearConstraint(A, b_l[:, 0], b_u[:, 0])
    result = scipy.optimize.milp(c=c[:, 0], constraints=constraints)
    img = np.reshape(result.x[:M], newshape=imgsize)
    return img.T
