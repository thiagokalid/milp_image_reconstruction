#%% md
# ## Import das bibliotecas:
#%%
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

#%% Input de dados:

# Parâmetros de simulação:
cp = 5 # Velocidade de propagação no meio em mm/us
gate_start = 0  # Início do gate em us
gate_end = 8  # Fim do gate em us
fc = 5e6  # Frequência central do transdutor em Hz
fs = fc * 10  # Frequência de amostragem em Hz
Nelem = 64

#%% Criação dos Objetos para Simulação:

# Create punctual reflectors grid:
width = 2
height = 2
reflector_grid = ReflectorGrid(width=width, height=height, xres=20e-3, zres=20e-3)

# Create transducer:
transducer = Transducer(n_elem=Nelem, fc=fc)

# Create acquisiton object:
acq = Acquisition(cp, fs, gate_start, gate_end, reflector_grid, transducer)
H, _ = acq.generate_basis_signal(sparse=True, verbose=True)

# %% Aplicação do método de reconstrução de imagem:

# # Localização do refletor que deseja-se reconstruir em mm:
# acq.add_random_reflectors(2, method="off-grid", seed=1)
acq.add_reflector(.37, 1.37)
acq.add_reflector(.5, 1)

simulated_fmc = acq.generate_signals(noise_std=0.1)

# Flatten simulated_fmc to a Fortran-ordered 1D array
simulated_flatten_fmc = np.ravel(simulated_fmc, order='F')

signal_size = len(simulated_flatten_fmc)

imgsize = reflector_grid.get_imgsize()

results = []
method_names = []

# %% MILP

# print("L1 begin.")
# result_milp = milp_method(
#     np.reshape(acq.fmc_basis, H.shape, order='C'),
#     np.ravel(simulated_fmc, order='C'),
#     imgsize
# )
# results.append(result_milp)
# method_names.append("MILP based")
# print("L1 end.")

#%% IRLS with L1 norm

# print("IRLS begin.")
# epsilon = 1e-4
# lbd = 0
# tol = 1e-6
# result_irls = irls_method(
#     H,
#     simulated_flatten_fmc,
#     imgsize,
#     lbd=lbd,
#     epsilon=epsilon,
#     maxiter=20,
#     tolLower=tol,
#     method="pylops"
# )
# results.append(result_irls)
# method_names.append("L1L1 norm through IRLS")
# print("IRLS end.")

#%% LSQR with L2 norm

# print("L2 begin.")
# result_l2 = passarin_method(
#     H,
#     simulated_flatten_fmc,
#     imgsize,
#     damp=lbd
# )
# results.append(result_l2)
# method_names.append("L2L1 norm through LSQR")
# print("L2 end.")

#%% Laroche 2020:

print("Laroche begin.")
mumax = 2 * np.max(np.abs(H.T @ simulated_flatten_fmc))
mu1 = .6 * mumax
mu2 = 0
result_laroche = laroche_method(
    H,
    simulated_flatten_fmc,
    imgsize,
    mu1=mu1,
    mu2=mu2
)
results.append(result_laroche)
method_names.append("Laroche 2020")
print("Laroche end.")

#%% Watt 2024:

# H = np.reshape(acq.fmc_basis, newshape=(signal_size, reflector_grid.get_numpxs()))
# print("Watt begin.")
# alpha_perc = 35
# result_watt = watt_method(
#     H,
#     simulated_flatten_fmc,
#     imgsize,
#     alpha_perc=alpha_perc
# )
# results.append(result_watt)
# method_names.append("Watt 2024")
# print("Watt end.")

#%% Extrai os resultados individuais:
imgs = [result.img for result in results]
residues = [result.residue for result in results]
elapsed_times = [result.elapsed_time for result in results]
metrics = [result.metric for result in results]
metric_names = [result.metric_name for result in results]

# Display dos resultados:
min_amp = np.nanmin(imgs)
max_amp = np.nanmax(imgs)
convert_to_db = lambda img: 20 * np.log10((img - min_amp) / (max_amp - min_amp) + 1e-9)

offset = (reflector_grid.xres / 2, reflector_grid.zres / 2)
imgs_db = [convert_to_db(img) for img in imgs]
# imgs_db = [img for img in imgs]

vmin = np.nanmax([np.min(imgs_db), -20])
vmax = np.nanmax([imgs_db])

n_cols = len(results)
i = 1

plot_on_first = True
fig = plt.figure(figsize=(18, 10))
plt.suptitle("Image in dB")
for img_db, residue, elapsed_time, metric, metric_name, method_name in zip(imgs_db, residues, elapsed_times, metrics, metric_names, method_names):
    ax1 = plt.subplot(2, n_cols, i)
    cax = plt.imshow(img_db, extent=reflector_grid.get_extent(offset=offset), aspect='equal', vmin=vmin, vmax=vmax, interpolation='None')
    plt.plot(*reflector_grid.get_coords(), "xb", alpha=.5, label="Reflectors grid")
    plt.plot(acq.xr, acq.zr, 'or', label='Target reflector')
    plt.xlabel("x-axis in mm")
    plt.ylabel("z-axis in mm")

    if 1e-3 <= elapsed_time <= 1e0:
        time_unit = "ms"
        multiplier = 1e3
    elif 1e-6 <= elapsed_time <= 1e-3:
        time_unit = "ns"
        multiplier = 1e6
    else:
        time_unit = "s"
        multiplier = 1
    plt.title(f"{method_name}.\n Runtime = {elapsed_time * multiplier:.2f} {time_unit}")

    if i == 1 and plot_on_first:
        # Add a colorbar outside the plot (using the "ax" of the image and specifying location)
        fig.colorbar(cax, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)

        # Adjust layout to prevent overlap and allow space for the colorbar
        fig.subplots_adjust(right=0.85)  # Increase this value to move colorbar further right

        #plt.legend(loc="upper center")


    ax2 = plt.subplot(2, n_cols, n_cols + i)
    plt.title(f"{metric_name} = {metric:.2e}")
    ax2.hist(residue, bins=100, density=False)
    ax2.grid()
    i += 1


plt.show()
plt.tight_layout()
