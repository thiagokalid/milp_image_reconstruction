import numpy as np
import scipy
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg import cg

from .acquisition import Acquisition

from scipy.optimize import milp


def transform_dense_to_sparse_array(dense_signal, epsilon=1e-2):
    sparse_signal = np.zeros_like(dense_signal)
    non_zero = np.power(dense_signal, 2) > np.power(epsilon, 2)
    sparse_signal[non_zero] = dense_signal[non_zero]
    sparse_signal = scipy.sparse.csc_array(sparse_signal)
    return sparse_signal


def transform_dense_to_sparse_matrix(dense_signal, epsilon=1e-2):
    sparse_signal = np.zeros_like(dense_signal)
    non_zero = np.power(dense_signal, 2) > np.power(epsilon, 2)
    sparse_signal[non_zero] = dense_signal[non_zero]
    sparse_signal = scipy.sparse.csc_array(sparse_signal)
    return sparse_signal


def passarin_method(basis_signal: np.ndarray, sampled_signal: np.ndarray, imgsize: tuple, damp=0):
    A = basis_signal
    b = sampled_signal
    x = linalg.lsqr(A, b, damp=damp)[0]
    img = np.reshape(x, shape=imgsize)
    residue = b - A @ x
    return img.T, b - A @ x


def naive_l1_method(basis_signal: np.ndarray, sampled_signal: np.ndarray, imgsize: tuple):
    M, N = basis_signal.shape
    g = sampled_signal
    H = basis_signal

    def cost_fun(f):
        return np.linalg.norm(g - H @ f, ord=1)

    result = scipy.optimize.minimize(fun=cost_fun, x0=np.zeros(N), method="SLSQP")

    img = np.reshape(result.x, shape=imgsize)
    return img.T


def l1_method(basis_signal: np.ndarray, sampled_signal: np.ndarray, imgsize: tuple):
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
    return img.T, residue


def IRLSCG(A, B, maxiter, xguess, lbd=10, tolLower=1e-2, epsilon=.01):
    '''
		Itera no maximo maxiter vezes o IRLSCG.
			Lembrando, queremos estimar a solução para A = Bx.
			A <= A
			B <= B
			xguess <= o chute inicial para x
			lbd <= lambda (deve estar de acordo com a curva L)
			tolLower <= parar o loop de iterações quando o erro é menor que este valor
			epsilon <= valor que ajuda na aproximação f(x)=x para f(x) = f1(x)=sqrt(x^2+epsilon)
			com o objetivo de tornar f(x) diferenciavel
		'''
    N = A.shape[1]
    W = np.zeros(shape=(N, N))

    f = np.zeros(N)
    f0 = np.sqrt(xguess ** 2 + epsilon)
    #a = conjgrad(H.T@H+d*W, H.T@g, f[:,0])
    minTol = np.zeros(maxiter + 1)
    f1 = None

    for k in range(maxiter):
        W = np.diag(f0 ** (-1))
        f1 = linalg.lsqr(A.T @ A + lbd * W, A.T @ B)[0]
        ek = (np.linalg.norm(f1 - f0, 2) / np.linalg.norm(f0, 2)) ** 2
        f0 = np.sqrt(f1 ** 2 + epsilon)
        if (ek < tolLower):
            return f0, B - A @ f0
    print("Not converged.")
    return f1, B - A @ f1


def irls_method(basis_signal: np.ndarray, sampled_signal: np.ndarray, imgsize: tuple, damp=0):
    A = basis_signal
    b = sampled_signal
    xguess = linalg.lsqr(A, b)[0]
    x, residue = IRLSCG(A, b, maxiter=100, xguess=xguess)
    img = np.reshape(x, shape=imgsize)
    return img.T, residue
