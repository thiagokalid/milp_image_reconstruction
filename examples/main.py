#%% md
# ## Import das bibliotecas:
#%%
from acquisition import Acquisition
from reflector_grid import ReflectorGrid
from transducer import Transducer
from imaging import *

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QtAgg")

#%% Input de dados:

# Parâmetros de simulação:
cp = 1.483  # Velocidade de propagação no meio em mm/us
fs = 1e6  # Frequência de amostragem em Hz
gate_start = 0  # Início do gate em us
gate_end = 10  # Fim do gate em us
fc = .25e6  # Frequência central do transdutor em Hz

#%% Criação dos Objetos para Simulação:

# Create punctual reflectors grid:
reflector_grid = ReflectorGrid(width=6, height=7, xres=.5, zres=.5)

# Create transducer:
transducer = Transducer(n_elem=32, fc=fc)

# Create acquisiton object:
acq = Acquisition(cp, fs, gate_start, gate_end, reflector_grid, transducer)
# %% Aplicação do método de reconstrução de imagem:

# Localização do refletor que deseja-se reconstruir em mm:
xr = 1.25
zr = 3.40

#
sampled_fmc = acq.generate_signal(xr, zr)
sampled_signal = np.ravel(sampled_fmc)
signal_size = len(sampled_signal)

#
imgsize = reflector_grid.get_imgsize()
img = passarin_method(
    np.reshape(acq.fmc_basis, newshape=(signal_size, reflector_grid.get_numpxs())),
    sampled_signal,
    reflector_grid.get_imgsize()
)

#%% Display dos resultados:

img_norm = img - img.min() / (img.max() - img.min()) + 1e-9
img_db = 20 * np.log10(img_norm)

offset = (reflector_grid.xres/2, reflector_grid.zres/2)

plt.figure()
plt.imshow(img_db, extent=reflector_grid.get_extent(offset=offset), aspect='equal', vmin=-20, vmax=-6.8)
plt.plot(*reflector_grid.get_coords(), "xb", alpha=.5, label="Reflectors grid")
plt.plot(xr, zr, 'or', label='Target reflector')
plt.xlabel("x-axis in mm")
plt.ylabel("y-axis in mm")
plt.title("Image in dB")
plt.colorbar()
plt.legend()
plt.show()