import numpy as np
import scipy

from numpy import ndarray
from scipy.sparse import csc_array
from numba import njit
import numba as nb

__all__ = ["transform_dense_to_sparse_array", "transform_dense_to_sparse_matrix"]


def transform_dense_to_sparse_array(dense_signal: ndarray, threshold: float = 1e-2) -> csc_array:
    sparse_signal = np.zeros_like(dense_signal)
    non_zero = np.power(dense_signal, 2) > np.power(threshold, 2)
    sparse_signal[non_zero] = dense_signal[non_zero]
    sparse_signal = scipy.sparse.csc_array(sparse_signal)
    return sparse_signal


def transform_dense_to_sparse_matrix(dense_signal: ndarray, threshold: float = 1e-2) -> csc_array:
    sparse_signal = np.zeros_like(dense_signal)
    non_zero = np.power(dense_signal, 2) > np.power(threshold, 2)
    sparse_signal[non_zero] = dense_signal[non_zero]
    sparse_signal = scipy.sparse.csc_array(sparse_signal)
    return sparse_signal

@njit(fastmath=True)
def gausspulse(t: np.ndarray, fc: float = 5e6, bw: float = .4, bwr: float = -6) -> np.ndarray:
    ref = pow(10.0, bwr / 20.0)
    a = -(np.pi * fc * bw) ** 2 / (4.0 * np.log(ref))
    return np.exp(-a * t ** 2) * np.cos(2 * np.pi * fc * t)