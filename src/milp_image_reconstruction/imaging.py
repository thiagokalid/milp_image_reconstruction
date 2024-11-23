import numpy as np
import scipy
import scipy.sparse.linalg as linalg

from .acquisition import Acquisition

from scipy.optimize import milp
def transform_dense_to_sparse_array(dense_signal, epsilon = 1e-2):
    sparse_signal = np.zeros_like(dense_signal)
    non_zero = np.power(dense_signal, 2) > np.power(epsilon, 2)
    sparse_signal[non_zero] = dense_signal[non_zero]
    sparse_signal = scipy.sparse.csc_array(sparse_signal)
    return sparse_signal

def transform_dense_to_sparse_matrix(dense_signal, epsilon = 1e-2):
    sparse_signal = np.zeros_like(dense_signal)
    non_zero = np.power(dense_signal, 2) > np.power(epsilon, 2)
    sparse_signal[non_zero] = dense_signal[non_zero]
    sparse_signal = scipy.sparse.csc_array(sparse_signal)
    return sparse_signal

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
    g = transform_dense_to_sparse_array(g)
    H = basis_signal
    H = transform_dense_to_sparse_matrix(H)

    # c^T @ x
    c = np.ones(shape=(2*N + M, 1))
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

    b_u = np.ones_like(b_l)

    constraints = scipy.optimize.LinearConstraint(A, b_l[:, 0], b_u[:, 0])
    result = scipy.optimize.milp(c=c[:, 0], constraints=constraints)
    img = np.reshape(result.x[:M], newshape=imgsize)
    return img.T
