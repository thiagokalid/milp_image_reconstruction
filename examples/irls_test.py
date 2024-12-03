import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from src.milp_image_reconstruction.imaging import IRLSCG
import scipy

np.random.seed(0)

x = np.arange(1, 100, 1)
y = x + np.random.randn(len(x)) * 5 + 5

A = np.vstack((x, np.ones_like(x))).T
b = y

epsilon = 1e-5
tol = 1e-2
lbd = 0
xguess = np.zeros(A.shape[1])
maxiter = 50

xsol, residue = IRLSCG(A, b, maxiter, xguess, lbd, tol, epsilon)

xsol_lsq = scipy.sparse.linalg.lsqr(A.T @ A, A.T @ b)[0]

plt.plot(x, y, '-or', markersize=4)
plt.plot(x, xsol[0] * x + xsol[1], 'g')
plt.plot(x, xsol_lsq[0] * x + xsol_lsq[1], 'b', alpha=.5)
plt.show()