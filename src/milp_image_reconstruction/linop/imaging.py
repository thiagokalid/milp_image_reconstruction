import pylops
import numpy as np

from pylops import LinearOperator, fista, lsqr
from numpy import ndarray
import time

from .._imaging_result import ImagingResult


def laroche_method(A: LinearOperator, b: ndarray, imgsize: tuple, mu1=0, mu2=0) -> ImagingResult:
    # Solves Ax = b considering 'A' as LinearOperator and 'b' as dense matrix
    t0 = time.time()
    xguess = lsqr(A, b)

    He, N = __laroche_newmatrix(A, mu2)

    # Dense data array:
    ye = np.zeros(shape=(N, 1))
    ye[:len(b), 0] = b[:]

    x, iter, cost_fun = fista(
        Op=He,
        y=ye[:, 0],
        x0=xguess,
        eps=mu1,
        show=True
    )

    img = np.reshape(x, newshape=imgsize)
    residue = b - A @ x

    result = ImagingResult(
        x=x,
        img=img.T,
        cost_fun=cost_fun[-1],
        metric=cost_fun[-1],
        metric_name="SSE",
        cost_fun_log=cost_fun,
        elapsed_time=time.time() - t0,
        residue=residue
    )

    return result


def __laroche_newmatrix(A: LinearOperator, mu2: float) -> [LinearOperator, int]:
    # He = scipy.sparse.vstack((
    #     A,
    #     np.sqrt(mu2) * D
    # ))
    pass