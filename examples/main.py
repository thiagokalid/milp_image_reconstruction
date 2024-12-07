#%% md
# ## Import das bibliotecas:
#%%
from src.milp_image_reconstruction.acquisition import Acquisition
from src.milp_image_reconstruction.reflector_grid import ReflectorGrid
from src.milp_image_reconstruction.transducer import Transducer
from src.milp_image_reconstruction.imaging import *

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
import time
import numpy as np

#%% Input de dados:

# Parâmetros de simulação:
cp = 5.9  # Velocidade de propagação no meio em mm/us
gate_start = 0  # Início do gate em us
gate_end = 10  # Fim do gate em us
fc = .25e6  # Frequência central do transdutor em Hz
fs = fc * 4  # Frequência de amostragem em Hz
Nelem = 32

#%% Criação dos Objetos para Simulação:

# Create punctual reflectors grid:
reflector_grid = ReflectorGrid(width=6, height=7, xres=.5, zres=.5)

# Create transducer:
transducer = Transducer(n_elem=Nelem, fc=fc)

# Create acquisiton object:
acq = Acquisition(cp, fs, gate_start, gate_end, reflector_grid, transducer)
# %% Aplicação do método de reconstrução de imagem:

# Localização do refletor que deseja-se reconstruir em mm:
xr = [-1.25]
zr = [3.4]

sampled_fmc = acq.generate_signals(xr, zr, noise_factor=5e-2)

# Flatten FMC:
sampled_signal = np.ravel(sampled_fmc)
signal_size = len(sampled_signal)

#
imgsize = reflector_grid.get_imgsize()

#%% MILP

print("L1 begin.")
result_milp = milp_method(
    np.reshape(acq.fmc_basis, newshape=(signal_size, reflector_grid.get_numpxs())),
    sampled_signal,
    reflector_grid.get_imgsize()
)
img_milp = result_milp.img
residue_milp = result_milp.residue
t_milp = result_milp.elapsed_time
print("L1 end.")

#%% IRLS with L1 norm

print("IRLS begin.")
epsilon = 1e-6
lbd = 100
tol = 1e-6
result_irls = irls_method(
    np.reshape(acq.fmc_basis, newshape=(signal_size, reflector_grid.get_numpxs())),
    sampled_signal,
    reflector_grid.get_imgsize(),
    lbd=lbd,
    epsilon=epsilon,
    maxiter=20,
    tolLower=tol
)
t_irls = result_irls.elapsed_time
img_irls = result_irls.img
residue_irls = result_irls.residue
print("IRLS end.")

#%% LSQR with L2 norm

print("L2 begin.")
result_l2 = passarin_method(
    np.reshape(acq.fmc_basis, newshape=(signal_size, reflector_grid.get_numpxs())),
    sampled_signal,
    reflector_grid.get_imgsize()
)
t_l2 = result_l2.elapsed_time
img_passarin = result_l2.img
residue_passarin = result_l2.residue
print("L2 end.")

#%% Display dos resultados:
min_amp = np.nanmin([img_milp, img_passarin, img_irls])
max_amp = np.nanmax([img_milp, img_passarin, img_irls])
convert_to_db = lambda img: 20 * np.log10(img - min_amp / (max_amp - min_amp) + 1e-9)

offset = (reflector_grid.xres / 2, reflector_grid.zres / 2)

img_milp_db = convert_to_db(img_milp)
img_passarin_db = convert_to_db(img_passarin)
img_irls_db = convert_to_db(img_irls)

vmin = np.nanmin([img_passarin_db, img_milp_db, img_irls_db])
vmax = np.nanmax([img_passarin_db, img_milp_db, img_irls_db])

plt.figure(figsize=(18, 10))
plt.subplot(2, 3, 1)
plt.suptitle("Image in dB")
plt.imshow(img_milp_db, extent=reflector_grid.get_extent(offset=offset), aspect='equal')
plt.plot(*reflector_grid.get_coords(), "xb", alpha=.5, label="Reflectors grid")
plt.plot(xr, zr, 'or', label='Target reflector')
plt.xlabel("x-axis in mm")
plt.ylabel("y-axis in mm")
plt.title(f"LP based Runtime = {t_milp:.2f} s")
plt.colorbar()
plt.legend(loc="upper center")

ax = plt.subplot(2, 3, 4)
plt.title(f"SAE = {result_milp.sae:.2e}")
ax.hist(residue_milp, bins=100, density=False)
plt.grid()

plt.subplot(2, 3, 2)
plt.imshow(img_irls_db, extent=reflector_grid.get_extent(offset=offset), aspect='equal')
plt.plot(*reflector_grid.get_coords(), "xb", alpha=.5, label="Reflectors grid")
plt.plot(xr, zr, 'or', label='Target reflector')
plt.xlabel("x-axis in mm")
plt.ylabel("y-axis in mm")
plt.title(f"IRLS Runtime = {t_irls:.2f} s.\n" + fr"$\lambda={lbd:.2f}$, $\epsilon={epsilon:.2E}$")
plt.colorbar()
plt.legend(loc="upper center")

ax = plt.subplot(2, 3, 5)
plt.title(f"SAE = {result_irls.sae:.2e}")
ax.hist(residue_irls, bins=100, density=False)
plt.grid()

plt.subplot(2, 3, 3)
plt.imshow(img_passarin_db, extent=reflector_grid.get_extent(offset=offset), aspect='equal')
plt.plot(*reflector_grid.get_coords(), "xb", alpha=.5, label="Reflectors grid")
plt.plot(xr, zr, 'or', label='Target reflector')
plt.xlabel("x-axis in mm")
plt.ylabel("y-axis in mm")
plt.title(f"L2 Runtime = {t_l2:.2f} s.")
plt.colorbar()
plt.legend(loc="upper center")

ax = plt.subplot(2, 3, 6)
plt.title(f"SSE = {np.sum(np.power(residue_passarin, 2)):.2e}")
ax.hist(residue_passarin, bins=100, density=False)
plt.grid()

plt.show()
plt.tight_layout()