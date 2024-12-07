import numpy as np
import matplotlib.pyplot as plt
import time
import scipy

from scipy.optimize import milp, LinearConstraint

from src.milp_image_reconstruction.irls import irls_minres
from src.milp_image_reconstruction._utils import *

import matplotlib
matplotlib.use('TkAgg')

def milp_method(basis_signal, sampled_signal):
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
    b1 = np.ones((N, 1))

    # A is 2N x (M + 2N)
    A = scipy.sparse.vstack([
        scipy.sparse.hstack((H, ei_matrix, z_matrix)),
        scipy.sparse.hstack((-H, z_matrix, ei_matrix))
    ])

    b_l = np.vstack((
        g.toarray(),
        -g.toarray()
    ))

    constraints = LinearConstraint(A, b_l[:, 0])
    result = milp(c=c[:, 0], constraints=constraints)
    sae = np.sum(result.x[M:])

    residue = result.x[M:]
    residue[-N:] *= -1

    return x, residue, sae

if __name__ == '__main__':
    t = np.arange(0, 5, 1e-3)
    y = .1 * np.sin(t**3) + 5e-3 * np.random.randn(*t.shape)

    # Models the linear regression problem as Ax=b:
    A = np.vstack((np.sin(t**2), t**2, t, np.zeros_like(t))).T
    b = y

    lbd = 0
    x, residue, cost_fun, converged, x_log, cost_fun_log = irls_minres(A, b, np.zeros(A.shape[1]), lbd=lbd)

    x_milp, residue_milp, sae_milp = milp_method(A, b)

    x_ls, istop, itn, r1norm  = scipy.sparse.linalg.lsqr(A, b)[:4]

    plt.figure()
    plt.plot(t, y, 'o-k', alpha=.3, markersize=2)
    plt.plot(t, np.sum(x * A, axis=1), '-r', linewidth=2.5, label=f'IRLS \n SAE = {cost_fun:.2f}')
    plt.plot(t, np.sum(x_milp * A, axis=1), ':b', linewidth=2.5, label=f'LP \n SAE = {sae_milp:.2f}')
    plt.plot(t, np.sum(x_ls * A, axis=1), '--g', linewidth=2.5, label=f'LS \n SSE = {r1norm**2:.2f}')
    plt.legend()
    plt.show()
