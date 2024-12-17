from .reflector_grid import ReflectorGrid
from .transducer import Transducer
from .utils import gausspulse

import numpy as np
import scipy
from numpy import ndarray
from numba import njit, prange

__all__ = ["Acquisition"]


class Acquisition:
    def __init__(self, cp: float, fs: float, gate_start: float, gate_end: float, reflector_grid: ReflectorGrid,
                 transducer: Transducer):
        self.fmc_basis = None
        self.reflector_grid = reflector_grid
        self.transducer = transducer
        self.gate_start = gate_start
        self.gate_end = gate_end
        self.fs = fs
        self.cp = cp
        self.tspan = np.arange(self.gate_start * 1e-6, self.gate_end * 1e-6 + 1 / self.fs, 1 / self.fs) * 1e6
        self.n_samples = len(self.tspan)

        # There is no punctual reflectors
        self.xr, self.zr = [], []

        # Impulse response as functions which perform H(x) = H@x
        self.H = self.Ht = None

        # Matrix which contains all TOFs
        self.tof_matrix = None

    def generate_basis_signal(self, dense: bool = True):
        self.tof_matrix = self.__generate_tof_matrix()

        if dense:
            raise NotImplementedError
        else:
            self.H = lambda x: self.__mat_vec_mult(x, self.tof_matrix)
            self.Ht = 0

        return self.H, self.Ht

    def generate_signals(self, noise_std: float = 0) -> ndarray:
        sampled_fmc = []
        for xi, zi in zip(self.xr, self.zr):
            sampled_fmc += self.__generate_signal(xi, zi)

        if noise_std > 0:
            sampled_fmc += np.random.randn(*sampled_fmc.shape) * noise_std
        return sampled_fmc

    def add_random_reflectors(self, n_reflectors: int, method: str = "on-grid", seed=None) -> None:
        if isinstance(seed, (int, float)):
            np.random.seed(seed)
        x, z = self.reflector_grid.get_coords()
        xmin, xmax = x.min(), x.max()
        zmin, zmax = z.min(), z.max()

        match method:
            case "on-grid":
                self.xr += np.random.randint(low=xmin, high=xmax, size=n_reflectors).tolist()
                self.zr += np.random.randint(low=zmin, high=zmax, size=n_reflectors).tolist()
            case "off-grid":
                self.xr += [np.random.uniform(xmin, xmax) for _ in range(n_reflectors)]
                self.zr += [np.random.uniform(zmin, zmax) for _ in range(n_reflectors)]
            case _:
                raise ValueError("Invalid method.")

    def add_reflector(self, xr: float, zr: float) -> None:
        self.xr.append(xr)
        self.zr.append(zr)

    def __generate_signal(self, xr: float, zr: float) -> ndarray:
        fmc = np.zeros(
            shape=(self.n_samples, self.transducer.n_elem, self.transducer.n_elem))
        i, j = -1, -1
        for x_transm, z_transm in zip(*self.transducer.get_coords()):
            i += 1
            for x_receiver, z_receiver in zip(*self.transducer.get_coords()):
                j += 1
                dist1 = np.sqrt((x_transm - xr) ** 2 + (z_transm - zr) ** 2)
                dist2 = np.sqrt((xr - x_receiver) ** 2 + (zr - z_receiver) ** 2)
                tof = dist1 / self.cp + dist2 / self.cp
                fmc[:, i, j] = self.transducer.get_signal(self.tspan, tof)
            j = -1
        return fmc

    def __generate_tof_matrix(self):
        x_transd, z_transd = self.transducer.get_coords()
        x_reflector, z_reflector = self.reflector_grid.get_coords()
        coord_transd = np.vstack((x_transd, z_transd)).T
        coord_reflector = np.array([x_reflector, z_reflector]).T

        dist = scipy.spatial.distance.cdist(XA=coord_transd,
                                            XB=coord_reflector)

        tof_matrix = tof_kernel(self.transducer.n_elem, self.reflector_grid.n_reflectors, self.cp, dist)

        return tof_matrix

    def __mat_vec_mult(self, x, tof_matrix: ndarray) -> ndarray:
        Nel = self.transducer.n_elem
        Nsamp = len(self.tspan)
        return multiply_kernel(x,
                               self.tspan,
                               tof_matrix,
                               Nel, Nsamp,
                               self.transducer.fc, self.transducer.bw, self.transducer.bwr)


@njit(parallel=True, cache=True)
def tof_kernel(Nel, Npx, cp, dist):
    tof = np.zeros(shape=(Nel, Nel, Npx))
    for n in prange(Nel * Nel * Npx):
        # Calculate indices (i, j, k) from flattened index n
        k = n // (Nel * Nel)  # Reflector index
        rem = n % (Nel * Nel)
        i = rem // Nel  # Transducer i index
        j = rem % Nel  # Transducer j index

        tof[i, j, k] = dist[i, k] / cp + dist[j, k] / cp
    return tof


def multiply_kernel(x, tspan, tof_matrix, Nel, Nsamp, fc, bw, bwr):
    N = Nel * Nel * Nsamp
    y = np.zeros(N)
    t = np.arange(0, Nsamp)

    #tspan = np.arnage(0, 5e-6, 1/125e6)
    for n in range(Nel * Nel):
        i = n % Nel
        j = n // Nel

        idx = n * Nsamp + t
        tof = tof_matrix[i, j, :]

        comb = gausspulse(np.subtract.outer(tspan, tof) * 1e-6, fc=fc, bw=bw, bwr=bwr) @ x
        y[idx] += comb[:, 0]

    return y