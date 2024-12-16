import numpy as np
import scipy

from numpy import ndarray
from scipy.sparse import csc_array
from concurrent.futures import ProcessPoolExecutor

__all__ = ["transform_dense_to_sparse_array", "transform_dense_to_sparse_matrix"]

def transform_dense_to_sparse_array(dense_signal: ndarray, threshold: float=1e-2) -> csc_array:
    sparse_signal = np.zeros_like(dense_signal)
    non_zero = np.power(dense_signal, 2) > np.power(threshold, 2)
    sparse_signal[non_zero] = dense_signal[non_zero]
    sparse_signal = scipy.sparse.csc_array(sparse_signal)
    return sparse_signal


def transform_dense_to_sparse_matrix(dense_signal: ndarray, threshold: float=1e-2) -> csc_array:
    sparse_signal = np.zeros_like(dense_signal)
    non_zero = np.power(dense_signal, 2) > np.power(threshold, 2)
    sparse_signal[non_zero] = dense_signal[non_zero]
    sparse_signal = scipy.sparse.csc_array(sparse_signal)
    return sparse_signal

def gausspulse(t: np.ndarray, fc: float=5e6, bw: float=.4, bwr: float=-6) -> np.ndarray:
    ref = pow(10.0, bwr / 20.0)
    a = -(np.pi * fc * bw) ** 2 / (4.0 * np.log(ref))
    return np.exp(-a * t ** 2) * np.cos(2 * np.pi * fc * t)

def _compute_reflector_contribution(k, n_elem, transducer_coords, fc, bwr, bw, reflector_grid_coords, fs, cp, tspan, n_samples):
    Nel = n_elem
    xr, zr = reflector_grid_coords
    xr, zr = xr[k], zr[k]
    Nt = 70
    th = Nt * 1 / fs * 1e6
    Ts = 1 / fs * 1e6

    local_fmc_basis = np.zeros(shape=(n_samples * Nel * Nel, 1))

    for j in range(Nel):
        x_transm, z_transm = transducer_coords
        x_transm, z_transm = x_transm[j], z_transm[j]
        for i in range(Nel):
            x_receiver, z_receiver = transducer_coords
            x_receiver, z_receiver = x_receiver[i], z_receiver[i]


            dist1 = np.sqrt((x_transm - xr) ** 2 + (z_transm - zr) ** 2)
            dist2 = np.sqrt((xr - x_receiver) ** 2 + (zr - z_receiver) ** 2)
            tof = dist1 / cp + dist2 / cp

            if (tof - th / 2) < tspan[0]:
                beg_t = tspan[0]
                end_t = tof + th / 2
            elif (tof + th / 2) >= tspan[-1]:
                beg_t = tof - th / 2
                end_t = tspan[-1]
            else:
                beg_t = tof - th / 2
                end_t = tof + th / 2

            beg_idx = 0 + (j * Nel + i) * len(tspan)
            end_idx = len(tspan) + (j * Nel + i) * len(tspan)

            shift = int(np.round(beg_t * 1e-6 * fs))

            beg_idx, end_idx = int(beg_idx), int(end_idx)

            reduced_tspan = np.arange(beg_t, end_t, Ts)
            Ntot = len(reduced_tspan)

            local_fmc_basis[beg_idx + shift: beg_idx + shift + Ntot, 0] = \
                gausspulse((reduced_tspan - tof) * 1e-6, fc=fc, bw=bw, bwr=bwr)

    return k, local_fmc_basis

def _parallel_generate_sparse_signal(n_elem: int, transducer_coords: np.ndarray, fc, bwr, bw, n_reflectors, reflector_grid_coords: np.ndarray, fs, cp, tspan, n_samples):
    Nt = 70
    Nel = n_elem
    Npx = n_reflectors
    fmc_basis = scipy.sparse.lil_matrix((
        n_samples * n_elem * n_elem,
        n_reflectors))

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                _compute_reflector_contribution,
                k,
                n_elem,
                transducer_coords,
                fc,
                bwr,
                bw,
                reflector_grid_coords,
                fs,
                cp,
                tspan,
                n_samples
            )
            for k in range(Npx)
        ]

        for i, future in enumerate(futures):
            k, local_fmc_basis = future.result()
            fmc_basis[:, k] = local_fmc_basis[:, 0]
            print(f"Progress = {i/len(futures)*100:.2f}%")

    return fmc_basis