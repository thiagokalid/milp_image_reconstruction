from src.milp_image_reconstruction.acquisition import Acquisition
from src.milp_image_reconstruction.imaging import laroche_method, watt_method
from src.milp_image_reconstruction.reflector_grid import ReflectorGrid
from src.milp_image_reconstruction.transducer import Transducer
from src.milp_image_reconstruction.imaging import *

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
import time
import numpy as np

# Parâmetros de simulação:
cp = 5  # Velocidade de propagação no meio em mm/us
gate_start = 0  # Início do gate em us
gate_end = 8  # Fim do gate em us
fc = 5e6  # Frequência central do transdutor em Hz
fs = fc * 10  # Frequência de amostragem em Hz
Nelem = 64

#%% Criação dos Objetos para Simulação:

# Create punctual reflectors grid:
width = 2
height = 2
reflector_grid = ReflectorGrid(width=width, height=height, xres=20e-2, zres=20e-3)

# Create transducer:
transducer = Transducer(n_elem=Nelem, fc=fc)

# Create acquisiton object:
acq = Acquisition(cp, fs, gate_start, gate_end, reflector_grid, transducer)
H, Ht = acq.generate_basis_signal(dense=False)

# Extract useful numbers:
Npx = reflector_grid.n_reflectors
Nt = acq.n_samples
N = Nelem * Nelem * Nt

# Example of a vector:
d = np.zeros((Npx, 1))
d[0] = 1

t0 = time.time()
h = H(d)
print(f"Elapsed time: {time.time() - t0:.2f}. Number of elements on matrix: {N * Npx:.2E}")
