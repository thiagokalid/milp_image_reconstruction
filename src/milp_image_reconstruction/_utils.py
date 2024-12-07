import numpy as np
import scipy

from numpy import ndarray
from scipy.sparse import csc_array

__all__ = ["_transform_dense_to_sparse_array", "_transform_dense_to_sparse_matrix"]

def _transform_dense_to_sparse_array(dense_signal: ndarray, threshold: float=1e-2) -> csc_array:
    sparse_signal = np.zeros_like(dense_signal)
    non_zero = np.power(dense_signal, 2) > np.power(threshold, 2)
    sparse_signal[non_zero] = dense_signal[non_zero]
    sparse_signal = csc_array(sparse_signal)
    return sparse_signal


def _transform_dense_to_sparse_matrix(dense_signal: ndarray, threshold: float=1e-2) -> csc_array:
    sparse_signal = np.zeros_like(dense_signal)
    non_zero = np.power(dense_signal, 2) > np.power(threshold, 2)
    sparse_signal[non_zero] = dense_signal[non_zero]
    sparse_signal = csc_array(sparse_signal)
    return sparse_signal