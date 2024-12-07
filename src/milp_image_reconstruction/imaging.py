# Import of public libraries:
import numpy as np
import scipy

# Import of selected objects:
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg import cg, minres
from scipy.optimize import milp, LinearConstraint
from numpy import ndarray

# Import of custom libraries:
from ._utils import _transform_dense_to_sparse_matrix, _transform_dense_to_sparse_array
from .irls import *

__all__ = ["passarin_method", "milp_method", "irls_method"]

def passarin_method(basis_signal: ndarray, sampled_signal: ndarray, imgsize: tuple, damp=0):
    A = basis_signal
    b = sampled_signal
    x = linalg.lsqr(A, b, damp=damp)[0]
    img = np.reshape(x, newshape=imgsize)
    residue = b - A @ x
    return img.T, residue


def naive_l1_method(basis_signal: ndarray, sampled_signal: ndarray, imgsize: tuple):
    M, N = basis_signal.shape
    g = sampled_signal
    H = basis_signal

    def cost_fun(f):
        return np.linalg.norm(g - H @ f, ord=1)

    result = scipy.optimize.minimize(fun=cost_fun, x0=np.zeros(N), method="SLSQP")

    img = np.reshape(result.x, shape=imgsize)
    return img.T


def milp_method(basis_signal: ndarray, sampled_signal: ndarray, imgsize: tuple):
    N, M = basis_signal.shape
    g = sampled_signal.reshape(N, 1)
    g = _transform_dense_to_sparse_array(g)
    H = basis_signal
    H = _transform_dense_to_sparse_matrix(H)

    # c^T @ x
    c = np.ones(shape=(2 * N + M, 1))
    c[:M] = 0

    #
    ei_matrix = scipy.sparse.eye_array(N, N, format='csc')
    z_matrix = scipy.sparse.csc_array((N, N))

    # A is 2N x (M + 2N)
    A = scipy.sparse.vstack([
        scipy.sparse.hstack((H, ei_matrix, z_matrix)),
        scipy.sparse.hstack((-H, z_matrix, ei_matrix))
    ])

    b_l = np.vstack((
        g.toarray(),
        -g.toarray()
    ))

    # b_u = np.ones_like(b_l)

    constraints = LinearConstraint(A, b_l[:, 0])
    result = milp(c=c[:, 0], constraints=constraints)
    img = np.reshape(result.x[:M], newshape=imgsize)
    residue = result.x[M:]
    residue[-N:] *= -1
    return img.T, residue

def irls_method(basis_signal: ndarray, sampled_signal: ndarray, imgsize: tuple, maxiter=100, tolLower=1e-2,
                epsilon=1e-3, lbd=1e-3, method="minres"):
    A = basis_signal
    b = sampled_signal
    xguess = linalg.lsqr(A, b)[0]

    match method:
        case "minres":
            x, residue = irls_minres(A, b, maxiter=maxiter, xguess=xguess, tolLower=tolLower, epsilon=epsilon, lbd=lbd)
        case _:
            raise NotImplementedError

    img = np.reshape(x, newshape=imgsize)
    return img.T, residue
