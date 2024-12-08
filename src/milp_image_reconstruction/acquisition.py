from .reflector_grid import ReflectorGrid
from .transducer import Transducer

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

        # Generate the FMC from basic punctual reflector grid:
        self.__generate_basis_signal()

    def __generate_basis_signal(self):
        self.fmc_basis = np.zeros(shape=(self.n_samples, self.transducer.n_elem, self.transducer.n_elem, self.reflector_grid.n_reflectors))
        i, j, k = -1, -1, -1
        for x_transm, z_transm in zip(*self.transducer.get_coords()):
            i += 1
            for x_receiver, z_receiver in zip(*self.transducer.get_coords()):
                j += 1
                for xr, zr in zip(*self.reflector_grid.get_coords()):
                    k += 1
                    dist1 = np.sqrt((x_transm - xr) ** 2 + (z_transm - zr) ** 2)
                    dist2 = np.sqrt((xr - x_receiver) ** 2 + (zr - z_receiver) ** 2)
                    tof = dist1 / self.cp + dist2 / self.cp
                    self.fmc_basis[:, i, j, k] = self.transducer.get_signal(self.tspan, tof)
                k = -1
            j = -1
            print(f"progress = {(i + 1) / self.transducer.n_elem * 100:.2f}")
        return self.fmc_basis

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

    def generate_signals(self, xr: list, zr: list, noise_std: float=0) -> ndarray:
        sampled_fmc = None
        for xi, zi in zip(xr, zr):
            if sampled_fmc is None:
                sampled_fmc = self.generate_signal(xi, zi)
            else:
                sampled_fmc += self.generate_signal(xi, zi)
        if noise_std > 0:
            sampled_fmc += np.random.randn(*sampled_fmc.shape) * noise_std
        return sampled_fmc

    def generate_random_reflectors_signals(self, n_reflectors: int, noise_std: float=0, method: str="on-grid"):
        x, z = self.reflector_grid.get_coords()
        xmin, xmax = x.min(), x.max()
        zmin, zmax = z.min(), z.max()

        match method:
            case "on-grid":
                xr = np.random.randint(low=xmin, high=xmax, size=n_reflectors)
                zr = np.random.randint(low=zmin, high=zmax, size=n_reflectors)
            case "off-grid":
                xr = [np.random.uniform(xmin, xmax) for _ in range(n_reflectors)]
                zr = [np.random.uniform(zmin, zmax) for _ in range(n_reflectors)]
            case _:
                raise ValueError("Invalid method.")

        return self.generate_signals(xr, zr, noise_std)
