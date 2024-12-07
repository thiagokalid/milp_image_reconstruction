# Import of public libraries:
import numpy as np
import scipy

# Import of selected objects:
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg import cg, minres
from scipy.optimize import milp
from numpy import ndarray

# Import of custom libraries:
from .utils import transform_dense_to_sparse_matrix, transform_dense_to_sparse_array


def passarin_method(basis_signal: ndarray, sampled_signal: ndarray, imgsize: tuple, damp=0):
    A = basis_signal
    b = sampled_signal
    x = linalg.lsqr(A, b, damp=damp)[0]
    img = np.reshape(x, newshape=imgsize)
    residue = b - A @ x
    return img.T, b - A @ x


def naive_l1_method(basis_signal: ndarray, sampled_signal: ndarray, imgsize: tuple):
    M, N = basis_signal.shape
    g = sampled_signal
    H = basis_signal

    def cost_fun(f):
        return np.linalg.norm(g - H @ f, ord=1)

    result = scipy.optimize.minimize(fun=cost_fun, x0=np.zeros(N), method="SLSQP")

    img = np.reshape(result.x, shape=imgsize)
    return img.T


def l1_method(basis_signal: ndarray, sampled_signal: ndarray, imgsize: tuple):
    N, M = basis_signal.shape
    g = sampled_signal.reshape(N, 1)
    g = transform_dense_to_sparse_array(g)
    H = basis_signal
    H = transform_dense_to_sparse_matrix(H)

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

    b_u = np.ones_like(b_l)

    constraints = scipy.optimize.LinearConstraint(A, b_l[:, 0])
    result = scipy.optimize.milp(c=c[:, 0], constraints=constraints)
    img = np.reshape(result.x[:M], newshape=imgsize)
    residue = result.x[M:]
    residue[-N:] *= -1
    return img.T, residue


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

        f1 = linalg.lsqr(A1.T @ A1 + lbd * W1, A1.T @ b1, atol=1e-3)[0]

        ek = (np.linalg.norm(f1 - f0, 2) / np.linalg.norm(f0, 2)) ** 2
        f0 = np.sqrt(f1 ** 2 + epsilon)
        err = np.sqrt((b - A @ f1) ** 2 + epsilon)
        print("ek = ", ek)
        if ek < tolLower:
            return f0, b - A @ f0
    print("Not converged.")
    return f1, b - A @ f1


def irls_method(basis_signal: ndarray, sampled_signal: ndarray, imgsize: tuple, maxiter=100, tolLower=1e-2,
                epsilon=1e-3, lbd=1e-3):
    A = basis_signal
    b = sampled_signal
    xguess = linalg.lsqr(A, b)[0]
    x, residue = irls_minres(A, b, maxiter=maxiter, xguess=xguess, tolLower=tolLower, epsilon=epsilon, lbd=lbd)
    img = np.reshape(x, newshape=imgsize)
    return img.T, residue
