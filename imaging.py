import numpy as np
import scipy.sparse.linalg as linalg

from acquisition import Acquisition

from scipy.optimize import milp


def passarin_method(basis_signal: np.ndarray, sampled_signal: np.ndarray, imgsize: tuple):
    A = basis_signal
    b = sampled_signal
    x = linalg.lsqr(A, b)[0]
    img = np.reshape(x, newshape=imgsize)
    return img.T


def l1_norm_method(basis_signal: np.ndarray, sampled_signal: np.ndarray, imgsize: tuple):
    N = basis_signal.shape[1]
    H = basis_signal
    

    def cost_fun(f, r_positive, r_negative):
        # Ordem das variáveis: r, rpositivo, rnegativo
        c_f = np.ravel(A)
        c_r_positive = c_r_negative = np.ones_like(c_f)
        c = np.array([c_f, c_r_positive, c_r_negative])

        # Lower bound:
        A = np.array(
            [
                [A, -1, 0],
                [A, 0, 1]]
        )
