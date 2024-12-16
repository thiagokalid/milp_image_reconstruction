from .reflector_grid import ReflectorGrid
from .transducer import Transducer
from .utils import _parallel_generate_sparse_signal

import numpy as np
from numpy import ndarray

__all__ = ["Acquisition"]


class Acquisition:
    def __init__(self, cp: float, fs: float, gate_start: float, gate_end: float, reflector_grid: ReflectorGrid, transducer: Transducer):
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


    def generate_basis_signal(self, verbose=True, sparse=False, path=None):
        if path is None:
            if not sparse:
                self.fmc_basis = np.zeros(shape=(
                self.n_samples, self.transducer.n_elem, self.transducer.n_elem, self.reflector_grid.n_reflectors))
                i, j, k = -1, -1, -1
                for i, (x_transm, z_transm) in enumerate(zip(*self.transducer.get_coords())):
                    for j, (x_receiver, z_receiver) in enumerate(zip(*self.transducer.get_coords())):
                        for k, (xr, zr) in enumerate(zip(*self.reflector_grid.get_coords())):
                            dist1 = np.sqrt((x_transm - xr) ** 2 + (z_transm - zr) ** 2)
                            dist2 = np.sqrt((xr - x_receiver) ** 2 + (zr - z_receiver) ** 2)
                            tof = dist1 / self.cp + dist2 / self.cp
                            self.fmc_basis[:, i, j, k] = self.transducer.get_signal(self.tspan, tof)

                if verbose:
                    print(f"progress = {(i + 1) / self.transducer.n_elem * 100:.2f}")
                return np.reshape(self.fmc_basis, (
                self.n_samples * self.transducer.n_elem * self.transducer.n_elem, self.reflector_grid.n_reflectors),
                                  order='F')
            else:
                self.fmc_basis = None
                self.H = _parallel_generate_sparse_signal(self.transducer.n_elem, self.transducer.get_coords(), self.transducer.fc, self.transducer.bwr, self.transducer.bw, self.reflector_grid.n_reflectors, self.reflector_grid.get_coords(), self.fs, self.cp, self.tspan, self.n_samples)
                return self.H
        elif path is str:
            self.fmc_basis = np.load(path)
            return self.fmc_basis

        else:
            raise NotImplementedError


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

    def generate_signals(self, xr: list = None, zr: list= None, noise_std:float = 0) -> ndarray:
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



    def add_random_reflectors(self, n_reflectors: int, method: str="on-grid", seed = None) -> None:
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
