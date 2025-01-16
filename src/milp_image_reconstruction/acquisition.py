from .reflector_grid import ReflectorGrid
from .transducer import Transducer
from .utils import gausspulse

import numpy as np
import scipy
from numpy import ndarray
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator
import cupy as cp
import cupyx
from multiprocessing import Pool

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

    def generate_basis_signal(self, linear_operator: bool = False):
        self.tof_matrix = self.__generate_tof_matrix()
        Npx = self.reflector_grid.n_reflectors
        Nel = self.transducer.n_elem
        Nt = self.n_samples
        N = Nel ** 2 * Nt

        if not linear_operator:
            self.fmc_basis = np.zeros(shape=(Nt, Nel, Nel, Npx))
            for n in prange(Nel * Nel * Npx):
                # Calculate indices (i, j, k) from flattened index n
                k = n // (Nel * Nel)  # Reflector index
                rem = n % (Nel * Nel)
                i = rem // Nel  # Transducer i index
                j = rem % Nel  # Transducer j index

                tof = float(self.tof_matrix[i, j, k])
                self.fmc_basis[:, i, j, k] = self.transducer.get_signal(self.tspan, tof)
            self.H = np.reshape(self.fmc_basis, (N, Npx), order='F')
        else:
            N = len(self.tspan) * self.transducer.n_elem**2
            Npx = self.reflector_grid.n_reflectors

            self.H = LinearOperator(shape=(N, Npx),
                                    matvec=lambda x: self._mat_vec_mult(x, self.tof_matrix),
                                    rmatvec=lambda x: self._t_mat_vec_mult(x, self.tof_matrix))

        return self.H

    def generate_signals(self, noise_std: float = 0) -> ndarray:
        Nt = len(self.tspan)
        Nel= self.transducer.n_elem
        sampled_fmc = np.zeros(shape=(Nt, Nel, Nel))
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

    def __generate_tof_matrix(self) -> np.ndarray:
        x_transd, z_transd = self.transducer.get_coords()
        x_reflector, z_reflector = self.reflector_grid.get_coords()
        coord_transd = np.vstack((x_transd, z_transd)).T
        coord_reflector = np.array([x_reflector, z_reflector]).T

        dist = scipy.spatial.distance.cdist(XA=coord_transd,
                                            XB=coord_reflector)

        tof_matrix = tof_kernel(self.transducer.n_elem, self.reflector_grid.n_reflectors, self.cp, dist)

        return tof_matrix

    def _mat_vec_mult(self, x, tof_matrix: ndarray) -> ndarray:
        Nel = self.transducer.n_elem
        Nsamp = len(self.tspan)
        return multiply_kernel(x,
                               self.tspan,
                               tof_matrix,
                               Nel, Nsamp,
                               self.transducer.fc, self.transducer.bw, self.transducer.bwr)

    def _t_mat_vec_mult(self, x, tof_matrix: ndarray) -> ndarray:
        Nel = self.transducer.n_elem
        Nsamp = len(self.tspan)
        Npx = self.reflector_grid.n_reflectors
        return t_multiply_kernel(x,
                               self.tspan,
                               tof_matrix,
                               Nel, Nsamp, Npx,
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


# Custom Gauss pulse implemented on the CPU
def gausspulse_cpu(t, fc, bw, bwr=-6):
    t = np.array(t, dtype=np.float32)
    b = bw / (2.0 * np.sqrt(np.log(2.0)))
    envelope = np.exp(-np.pi * b**2 * t**2)
    chirp = np.cos(2 * np.pi * fc * t)
    return envelope * chirp

# Worker function for multiply_kernel
def process_chunk_multiply(args):
    """Worker function for multiply_kernel chunks."""
    chunk_indices, x, tspan, tof_matrix, Nel, Nsamp, fc, bw, bwr = args
    N = len(chunk_indices) * Nsamp
    y_chunk = np.zeros(N, dtype=np.float32)
    t = np.arange(0, Nsamp, dtype=int)

    for n in chunk_indices:
        i = n % Nel
        j = n // Nel
        idx = (n - chunk_indices[0]) * Nsamp + t
        tof = tof_matrix[i, j, :].astype(np.float32)
        time_comb = (np.subtract.outer(tspan, tof)) * 1e-6
        signal_comb = np.dot(gausspulse_cpu(time_comb, fc, bw, bwr), x)
        y_chunk[idx] += np.ravel(signal_comb)

    return y_chunk

def multiply_kernel(x, tspan, tof_matrix, Nel, Nsamp, fc, bw, bwr, num_processes=4):
    N = Nel * Nel
    indices = np.arange(N)
    chunks = np.array_split(indices, num_processes)
    args = [(chunk, x, tspan, tof_matrix, Nel, Nsamp, fc, bw, bwr) for chunk in chunks]

    with Pool(num_processes) as pool:
        results = pool.map(process_chunk_multiply, args)

    return np.concatenate(results).astype(np.float32)

# Optimized t_multiply_kernel on CPU
def t_multiply_kernel(x, tspan, tof_matrix, Nel, Nsamp, Npx, fc, bw, bwr):
    y = np.zeros(Npx, dtype=np.float32)

    for n in range(Npx):
        tof = tof_matrix[:, :, n].ravel(order="C")
        M = len(tof)
        time_comb = np.tile(tspan, (M, 1)) - tof[:, np.newaxis]
        time_comb *= 1e-6
        signal_comb = gausspulse_cpu(time_comb, fc, bw, bwr)
        signal_comb_flattened = signal_comb.ravel(order="C")
        y[n] = np.sum(signal_comb_flattened @ x)

    return y
