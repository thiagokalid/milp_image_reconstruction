import pylops
import numpy as np

from pylops import LinearOperator, fista, lsqr
from numpy import ndarray
import time
import pyproximal

from numba import njit, prange

from .._imaging_result import ImagingResult


def passarin_method(A: LinearOperator, b: ndarray, imgsize: tuple, damp=0):
    t0 = time.time()

    if type(A) is np.ndarray:
        Aop = pylops.MatrixMult(A, dtype="float64")
    else:
        Aop = pylops.LinearOperator(A)

    x, istop, itn, r1norm = pylops.optimization.basic.lsqr(Aop, b, damp=damp, show=True)[:4]
    img = np.reshape(x, newshape=imgsize)
    residue = b - A @ x

    success = True if istop == 1 else False
    message = "Solves least-squares" if istop == 1 else "Approximation of least-squares"

    result = ImagingResult(
        x=x,
        img=img.T,
        cost_fun=r1norm,
        residue=residue,
        success=success,
        status=istop,
        message=message,
        elapsed_time=time.time() - t0,
        niter=itn,
        metric=r1norm ** 2,
        metric_name="SSE"
    )

    return result

def laroche_method(A: LinearOperator, b: ndarray, imgsize: tuple, mu1=0, mu2=0) -> ImagingResult:
    # Solves Ax = b considering 'A' as LinearOperator and 'b' as dense matrix
    t0 = time.time()
    N, M = A.shape

    if mu2 == 0:
        if type(A) is np.ndarray:
            He = pylops.MatrixMult(A, dtype="float64")
            ye = b
        else:
            He = pylops.LinearOperator(A)
            ye = b
    else:
        pass

    print("Begin LSQR")
    xguess, *_ = pylops.optimization.basic.lsqr(He, b, show=True, niter=10)

    print("Found initial guess through Pylops LSQR")
    x, *_ = fista(He, ye, x0=xguess, eps=mu1, show=True, niter=10, tol=1e-3)
    print("Found final guess through Pyproximal FISTA")

    img = np.reshape(x, newshape=imgsize)
    residue = b - A @ x
    cost_fun = np.sum(np.power(residue, 2))

    result = ImagingResult(
        x=x,
        img=img.T,
        cost_fun=cost_fun,
        metric=cost_fun,
        metric_name="SSE",
        elapsed_time=time.time() - t0,
        residue=residue
    )


    return result

def tfm(tof_matrix: np.ndarray, b: np.ndarray) -> ImagingResult:
    x = img = np.zeros(2)

    result = ImagingResult(
        x=x,
        img=img.T
    )

    return result

# @njit(parallel=True, cache=True)
# def kernel(f, g, rx_elements, tx_elements, wt, nb_comb, samp_dist):
#     for comb in prange(nb_comb):
#         j = samp_dist[rx_elements[comb], :] + samp_dist[tx_elements[comb], :]
#         j[j < 0] = -1
#         j[j >= g.shape[0]] = -1
#         f += g[j, tx_elements[comb], rx_elements[comb]]*wt[:, :, rx_elements[comb]]*wt[:, :, tx_elements[comb]]
#     return f
#
#     # --- INÍCIO DO ALGORITMO TFM, desenvolvido por Hector. ---
#     f = np.zeros((1, roi.h_len * roi.w_len), dtype=g.dtype)
#     combs = np.argwhere(trcomb.T)
#     tx_elements = combs[:, 0]
#     rx_elements = combs[:, 1]
#     nb_combs = combs.shape[0]
#
#     dist = cdist(data_insp.probe_params.elem_center * 1e-3, roi.get_coord() * 1e-3)
#     dist_correction = 1.0 / (c * data_insp.inspection_params.sample_time * 1e-6)
#     samp_dist = dist * dist_correction
#
#     dr = (roi.get_coord()[:, :, np.newaxis]-data_insp.probe_params.elem_center.T[np.newaxis])
#     dg = np.arctan2(dr[:, 0, :], dr[:, 2, :])#/(np.pi/2)*90
#     if data_insp.inspection_params.type_insp == 'contact':
#         k = data_insp.probe_params.central_freq*1e6/data_insp.specimen_params.cl
#     else:
#         k = data_insp.probe_params.central_freq*1e6/data_insp.inspection_params.coupling_cl
#     a = data_insp.probe_params.elem_dim*1e-3
#     wt = np.sinc(k*a/2*np.sin(dg))*np.cos(dg)
#     wt = wt[np.newaxis]
#     samp_dist[scatfilt] = g.shape[0]
#     samp_dist -= int(data_insp.inspection_params.gate_start*data_insp.inspection_params.sample_freq)//2
#     # f = kernel(f, g, rx_elements, tx_elements, nb_combs, samp_dist.astype(np.int32))
#     f = kernel(f, g, rx_elements, tx_elements, wt, nb_combs, samp_dist.astype(np.int32))