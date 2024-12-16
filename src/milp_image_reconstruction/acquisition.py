from .reflector_grid import ReflectorGrid
from .transducer import Transducer

import numpy as np
import scipy
from numpy import ndarray
from numba import njit, prange
from multiprocessing import Pool

__all__ = ["Acquisition"]



class Acquisition:
    def __init__(self, cp: float, fs: float, gate_start: float, gate_end: float, reflector_grid: ReflectorGrid,
                 transducer: Transducer):
        self.index_matrix = None
        self.fmc_basis = None
        self.fmc_list = list()
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

    def generate_basis_signal(self, verbose=True, linear_operator: bool = False):
        if not linear_operator:
            self.fmc_basis = np.zeros(shape=(
                self.n_samples, self.transducer.n_elem, self.transducer.n_elem, self.reflector_grid.n_reflectors))

            for i, (x_transm, z_transm) in enumerate(zip(*self.transducer.get_coords())):
                for j, (x_receiver, z_receiver) in enumerate(zip(*self.transducer.get_coords())):
                    for k, (xr, zr) in enumerate(zip(*self.reflector_grid.get_coords())):
                        dist1 = np.sqrt((x_transm - xr) ** 2 + (z_transm - zr) ** 2)
                        dist2 = np.sqrt((xr - x_receiver) ** 2 + (zr - z_receiver) ** 2)
                        tof = dist1 / self.cp + dist2 / self.cp
                        self.fmc_basis[:, i, j, k] = self.transducer.get_signal(self.tspan, tof)

            if verbose:
                print(f"progress = {(i + 1) / self.transducer.n_elem * 100:.2f}")

            self.H = np.reshape(self.fmc_basis,
                                newshape=(self.n_samples, self.transducer.n_elem, self.transducer.n_elem, self.reflector_grid.n_reflectors),
                                order='F')
            return self.H, self.H.T
        else:
            Nt = 70
            self.tof_matrix = self.__generate_tof_matrix(Nt)
            print("TOF matrix generated.")

            self.H = lambda x: self.__generate_linear_operator(x, self.tof_matrix)
            #self.HT = lambda x: self.__generate_transposed_operator(x, self.tof_matrix)

            return self.H, 1

    def generate_signal(self, xr: float, zr: float) -> ndarray:
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
            self.fmc_list.append(fmc)
        return fmc

    def _generate_signals(self, xr: list, zr: list, noise_std: float) -> ndarray:
        sampled_fmc = None
        for xi, zi in zip(xr, zr):
            if sampled_fmc is None:
                sampled_fmc = self.generate_signal(xi, zi)
            else:
                sampled_fmc += self.generate_signal(xi, zi)
        if noise_std > 0:
            sampled_fmc += np.random.randn(*sampled_fmc.shape) * noise_std
        return sampled_fmc

    def generate_signals(self, xr: list = None, zr: list = None, noise_std: float = 0) -> ndarray:
        if not (xr is None and zr is None):
            return self._generate_signals(xr, zr, noise_std)

        elif xr is None and zr is None:
            if len(self.xr) == 0:
                raise ValueError("xr is empty.")
            if len(self.zr) == 0:
                raise ValueError("zr is empty.")
            return self._generate_signals(self.xr, self.zr, noise_std)
        else:
            raise ValueError("xr, zr are invalid.")

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


    def __generate_tof_matrix(self, Nt):
        th = Nt * 1 / self.fs * 1e6
        Ts = 1 / self.fs * 1e6
        tof_matrix = (np.zeros
                      (shape=(self.transducer.n_elem, self.transducer.n_elem, self.reflector_grid.n_reflectors)))

        x_transd, z_transd = self.transducer.get_coords()
        x_reflector, z_reflector = self.reflector_grid.get_coords()
        coord_transd = np.vstack((x_transd, z_transd)).T
        coord_reflector = np.array([x_reflector, z_reflector]).T

        dist = scipy.spatial.distance.cdist(XA=coord_transd,
                                            XB=coord_reflector)

        tof_matrix = tof_kernel(self.transducer.n_elem, self.reflector_grid.n_reflectors, self.cp, dist)

        return tof_matrix

    def __generate_linear_operator(self, x, tof_matrix: ndarray) -> ndarray:
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
        k = n // (Nel * Nel)               # Reflector index
        rem = n % (Nel * Nel)
        i = rem // Nel                     # Transducer i index
        j = rem % Nel                      # Transducer j index

        tof[i, j, k] = dist[i, k]/cp + dist[j, k]/cp
    return tof


def compute_chunk(args):
    x, tspan, tof_matrix, fc, bw, bwr, chunk_start, chunk_end, Nsamp, Nel = args
    N = (chunk_end - chunk_start) * Nsamp
    y_chunk = np.zeros(N)
    t = np.arange(0, Nsamp)

    for n in range(chunk_start, chunk_end):
        i = n % Nel
        j = n // Nel

        idx = (n - chunk_start) * Nsamp + t
        time = tspan
        tof = tof_matrix[i, j, :]

        comb = scipy.signal.gausspulse(
            np.subtract.outer(time, tof) * 1e-6, fc=fc, bw=bw, bwr=bwr
        ) @ x
        y_chunk[idx] += comb[:, 0]

    return y_chunk

# Main kernel function with multiprocessing
def multiply_kernel(x, tspan, tof_matrix, Nel, Nsamp, fc, bw, bwr, n_processes=12):
    N = Nel * Nel * Nsamp
    y = np.zeros(N)

    # Define chunks for multiprocessing
    chunk_size = (Nel * Nel) // n_processes
    chunks = [
        (
            x,
            tspan,
            tof_matrix,
            fc,
            bw,
            bwr,
            i * chunk_size,
            (i + 1) * chunk_size if i < n_processes - 1 else Nel * Nel,
            Nsamp,
            Nel,
        )
        for i in range(n_processes)
    ]

    # Use multiprocessing pool to compute results
    with Pool(processes=n_processes) as pool:
        results = pool.map(compute_chunk, chunks)

    # Combine results into the final output
    offset = 0
    Nr = len(results)
    r = 0
    for result in results:
        y[offset:offset + result.size] = result
        offset += result.size
        print(f"progress = {r/Nr * 100:.2f}")
        r += 1

    return y