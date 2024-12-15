# Import of public libraries:
import numpy as np
import scipy
import time
import pylops

# Import of selected objects:
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg import cg, minres
from scipy.optimize import milp, LinearConstraint
from scipy import sparse
from numpy import ndarray


from pylops.optimization.sparsity import fista
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse import csc_array, csc_matrix

# Import of custom libraries:
from ._utils import _transform_dense_to_sparse_matrix, _transform_dense_to_sparse_array
from .irls import *
from ._imaging_result import ImagingResult

__all__ = ["passarin_method", "milp_method", "irls_method"]

def passarin_method(basis_signal: ndarray, sampled_signal: ndarray, imgsize: tuple, damp=0):
    t0 = time.time()
    A = basis_signal
    b = sampled_signal
    x, istop, itn, r1norm = linalg.lsqr(A, b, damp=damp)[:4]
    img = np.reshape(x, newshape=imgsize)
    residue = b - A @ x

    success = True if istop == 1 else False
    message = "Solves least-squares" if istop == 1 else "Approximation of least-squares"

    result = ImagingResult(
        x = x,
        img = img.T,
        cost_fun = r1norm,
        residue = residue,
        success = success,
        status = istop,
        message = message,
        elapsed_time = time.time() - t0,
        niter = itn,
        metric = r1norm**2,
        metric_name = "SSE"
    )

    return result


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
    t0 = time.time()
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

    # if True:
    #     ri_matrix = scipy.sparse.eye_array(M, M, format='csc')
    #     e_matrix = scipy.sparse.csc_array((M, N))
    #
    #     # b_u = np.ones_like(b_l)
    #     A = scipy.sparse.vstack([
    #         A,
    #         scipy.sparse.hstack((ri_matrix, e_matrix, e_matrix)),
    #     ])
    #
    #     ri = np.full(shape=(M, 1), fill_value=.1)
    #     b_l = np.vstack((
    #         b_l,
    #         ri
    #     ))

    constraints = LinearConstraint(A, b_l[:, 0])
    result = milp(c=c[:, 0], constraints=constraints)
    img = np.reshape(result.x[:M], newshape=imgsize)
    sae = np.sum(result.x[M:])
    residue = result.x[M:]
    residue[-N:] *= -1

    result = ImagingResult(
        x = result.x,
        img = img.T,
        cost_fun = result.fun,
        residue = residue,
        success = result.success,
        status = result.status,
        message = result.message,
        elapsed_time = time.time() - t0,
        metric = sae,
        metric_name = "SAE"
    )

    return result

def irls_method(basis_signal: ndarray, sampled_signal: ndarray, imgsize: tuple, maxiter=100, tolLower=1e-2,
                epsilon=1e-3, lbd=1e-3, method="minres") -> ImagingResult:
    t0 = time.time()
    A = basis_signal
    b = sampled_signal
    xguess = linalg.lsqr(A, b)[0]

    match method:
        case "minres":
            x, residue, cost_fun, converged, x_log, cost_fun_log\
                = irls_minres(A, b, maxiter=maxiter, xguess=xguess, tolLower=tolLower, epsilon=epsilon, lbd=lbd)
        case "pylops":
            Aop = pylops.MatrixMult(basis_signal, dtype="float64")
            x, x_log = pylops.irls(Aop, b, threshR=False, epsR=epsilon, epsI=lbd, kind="data", nouter=maxiter)
            residue = b - A @ x
            cost_fun = np.sum(np.abs(b - A @ x))
            converged = True
            cost_fun_log = None
        case _:
            raise NotImplementedError

    img = np.reshape(x, newshape=imgsize).T

    result = ImagingResult(
        x=x,
        img=img,
        cost_fun = cost_fun,
        converged = converged,
        elapsed_time = time.time() - t0,
        residue=residue,
        x_log = x_log,
        cost_fun_log = cost_fun_log,
        metric = np.sum(np.abs(cost_fun)),
        metric_name="SAE"
    )
    return result

def laroche_method(basis_signal: ndarray, sampled_signal: ndarray, imgsize: tuple, maxiter=100, tolLower=1e-2,
                epsilon=1e-3, lbd=1e-3, diff_ord=1, mu1=1, mu2=1) -> ImagingResult:
    t0 = time.time()
    A = basis_signal
    N, M = basis_signal.shape
    b = sampled_signal
    xguess = linalg.lsqr(A, b)[0]

    Nt = len(b)
    Dvec = np.zeros(shape=(M, 1), dtype=float)
    match diff_ord:
        case 1:
            Dmask = [1, -1] # First-order difference mask
        case 2:
            Dmask = [1, -2, 1] # Second-order difference mask
        case _:
            Dmask = [1]

    #D = scipy.linalg.convolution_matrix(Dmask, N, mode='same')
    D = scipy.sparse.csc_array((N, M))

    He = scipy.sparse.vstack((
        A,
        np.sqrt(mu2) * D
    ))

    R = He.shape[0]
    ye = np.zeros(shape=(R, 1))
    ye[:len(b), 0] = b[:]


    Aop = pylops.MatrixMult(He, dtype="float64")

    x, iter, cost_fun = fista(Aop, ye[:, 0], x0=xguess, eps=mu1)

    img = np.reshape(x, newshape=imgsize)
    residue = b - A@x

    result = ImagingResult(
        x=x,
        img=img.T,
        cost_fun = cost_fun[-1],
        metric=cost_fun[-1],
        metric_name="SSE",
        cost_fun_log = cost_fun,
        elapsed_time = time.time() - t0,
        residue=residue
    )



    return result


def watt_method(basis_signal: ndarray, sampled_signal: ndarray, imgsize: tuple, alpha_perc: float = 1) -> ImagingResult:
    A = basis_signal
    N, M = basis_signal.shape
    b = sampled_signal
    alpha = alpha_perc / 100

    U, s, Vh = scipy.sparse.linalg.svds(A, k=(A.shape[1]-1))

    maxS = np.max(s)
    s_regularized = np.zeros_like(s)
    for i in range(s.shape[0]):
        s_regularized[i] = s[i] / (s[i]**2 + (alpha * maxS)**2)

    newP = Vh.T @ sparse.diags_array(s_regularized) @ U.T

    t0 = time.time()
    x = newP @ b

    img = np.reshape(x, newshape=imgsize)
    residue = b - A @ x

    result = ImagingResult(
        x=x,
        img=img.T,
        elapsed_time=time.time() - t0,
        residue=residue,
        metric=np.sum(np.power(residue, 2)),
        metric_name="SSE",
        P=newP
    )

    return result