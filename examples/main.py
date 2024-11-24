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

#%% Input de dados:

# Parâmetros de simulação:
cp = 5.9  # Velocidade de propagação no meio em mm/us
gate_start = 0  # Início do gate em us
gate_end = 10  # Fim do gate em us
fc = .25e6  # Frequência central do transdutor em Hz
fs = fc * 5  # Frequência de amostragem em Hz

#%% Criação dos Objetos para Simulação:

# Create punctual reflectors grid:
reflector_grid = ReflectorGrid(width=6, height=7, xres=.5, zres=.5)

# Create transducer:
transducer = Transducer(n_elem=32, fc=fc)

# Create acquisiton object:
acq = Acquisition(cp, fs, gate_start, gate_end, reflector_grid, transducer)
# %% Aplicação do método de reconstrução de imagem:

# Localização do refletor que deseja-se reconstruir em mm:
xr = [-1.25, 2.5, 0.75]
zr = [3.4, 6, 3]

#
sampled_fmc = None
for xi, zi in zip(xr, zr):
    if sampled_fmc is None:
        sampled_fmc = acq.generate_signal(xi, zi)
    else:
        sampled_fmc += acq.generate_signal(xi, zi)

sampled_signal = np.ravel(sampled_fmc)
signal_size = len(sampled_signal)

#
imgsize = reflector_grid.get_imgsize()

print("L1 begin.")
t0 = time.time()
img_proposed, l1_residue = l1_method(
    np.reshape(acq.fmc_basis, newshape=(signal_size, reflector_grid.get_numpxs())),
    sampled_signal,
    reflector_grid.get_imgsize()
)
t_proposed = time.time() - t0
print("L1 end.")

print("L2 begin.")
t0 = time.time()
img_reference, l2_residue = passarin_method(
    np.reshape(acq.fmc_basis, newshape=(signal_size, reflector_grid.get_numpxs())),
    sampled_signal,
    reflector_grid.get_imgsize()
)
t_reference = time.time() - t0
print("L2 end.")
#%% Display dos resultados:

#%% Display dos resultados:
convert_to_db = lambda img: 20 * np.log10(img - img.min() / (img.max() - img.min()) + 1e-9)

offset = (reflector_grid.xres/2, reflector_grid.zres/2)

img_proposed_db = convert_to_db(img_proposed)
img_reference_db = convert_to_db(img_reference)

vmin = np.min([img_reference_db])

plt.subplot(2,2,1)
plt.suptitle("Image in dB")
plt.imshow(img_proposed_db, extent=reflector_grid.get_extent(offset=offset), aspect='equal', vmin=vmin, vmax=-6.8)
plt.plot(*reflector_grid.get_coords(), "xb", alpha=.5, label="Reflectors grid")
plt.plot(xr, zr, 'or', label='Target reflector')
plt.xlabel("x-axis in mm")
plt.ylabel("y-axis in mm")
plt.title(f"Runtime = {t_proposed:.2f} s")
plt.colorbar()
plt.legend()

ax = plt.subplot(2,2,3)
ax.hist(l1_residue, bins=100, density=False)

plt.subplot(2,2,2)
plt.imshow(img_reference_db, extent=reflector_grid.get_extent(offset=offset), aspect='equal', vmin=vmin, vmax=-6.8)
plt.plot(*reflector_grid.get_coords(), "xb", alpha=.5, label="Reflectors grid")
plt.plot(xr, zr, 'or', label='Target reflector')
plt.xlabel("x-axis in mm")
plt.ylabel("y-axis in mm")
plt.title(f"Runtime = {t_reference:.2f} s")
plt.colorbar()
plt.legend()

ax = plt.subplot(2,2,4)
ax.hist(l2_residue, bins=100, density=False)

plt.show()