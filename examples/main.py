#%% md
# ## Import das bibliotecas:
#%%
from src.milp_image_reconstruction.acquisition import Acquisition
# from src.milp_image_reconstruction.imaging import laroche_method
from src.milp_image_reconstruction.linop.imaging import passarin_method, laroche_method
from src.milp_image_reconstruction.reflector_grid import ReflectorGrid
from src.milp_image_reconstruction.transducer import Transducer

import matplotlib.pyplot as plt
import matplotlib
from src.milp_image_reconstruction._imaging_result import ImagingResult

matplotlib.use("TkAgg")
import time
import numpy as np

if __name__ == "__main__":

    #%% Input de dados:

    # Parâmetros de simulação:
    cp = 5.94 # Velocidade de propagação no meio em mm/us
    gate_start = 5.12  # Início do gate em us
    gate_end = 8  # Fim do gate em us
    fc = 5e6  # Frequência central do transdutor em Hz
    fs = 125e6  # Frequência de amostragem em Hz
    Nelem = 64

    #%% Criação dos Objetos para Simulação:

    # Create punctual reflectors grid:
    width = 3.8
    height = 1.1
    reflector_grid = ReflectorGrid(width=width, height=height, xres=10e-2, zres=10e-2, xoffset=-1.4, zoffset=18.5)

    # Create transducer:
    transducer = Transducer(n_elem=Nelem, fc=fc, pitch=.5, bw=.5)

    # Create acquisiton object:
    acq = Acquisition(cp, fs, gate_start, gate_end, reflector_grid, transducer)
    H = acq.generate_basis_signal(linear_operator=False)

    # %% Aplicação do método de reconstrução de imagem:

    # # Localização do refletor que deseja-se reconstruir em mm:
    # acq.add_random_reflectors(2, method="off-grid", seed=1)
    #acq.add_reflector(.37, 1.37)
    # acq.add_reflector(0, 1.1)

    # np.random.seed(2)
    # simulated_fmc = acq.generate_signals(noise_std=5e-2)
    simulated_fmc = np.load("/home/tekalid/Downloads/kalid/ascan_data.npy")

    # Flatten simulated_fmc to a Fortran-ordered 1D array
    simulated_flatten_fmc = np.ravel(simulated_fmc, order='F')
    simulated_flatten_fmc /= simulated_flatten_fmc.max() * 5

    # plt.plot(simulated_flatten_fmc)

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
    #
    # print("L2 begin.")
    # result_l2 = passarin_method(
    #     H,
    #     simulated_flatten_fmc,
    #     imgsize
    # )
    # results.append(result_l2)
    # method_names.append("L2L1 norm through LSQR")
    # print("L2 end.")

    #%% Laroche 2020:

    print("Laroche begin.")
    mumax = 2 * np.max(np.abs(H.T @ simulated_flatten_fmc))
    mu1 = .8 * mumax
    mu2 = 0
    result_laroche = laroche_method(
        H,
        simulated_flatten_fmc,
        imgsize,
        mu1=mu1,
        mu2=mu2
    )
    result_laroche.img = np.abs(result_laroche.img)
    result_laroche.img = (result_laroche.img - result_laroche.img.min()) / (result_laroche.img.max() - result_laroche.img.min())
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

    #%% TFM:

    tfm = np.load("/home/tekalid/Downloads/kalid/tfm.npy")
    tfm = (tfm - tfm.min()) / (tfm.max() - tfm.min())

    result_tfm = ImagingResult(
        x=result_laroche.x,
        img=tfm,
        cost_fun=0.,
        metric=0.,
        metric_name="SSE",
        elapsed_time=0.,
        residue=result_laroche.residue * 0.
    )
    results.append(result_tfm)
    method_names.append("TFM")
    print("TFM end.")

    #%% Extrai os resultados individuais:
    imgs = [np.abs(result.img) for result in results]
    residues = [result.residue for result in results]
    elapsed_times = [result.elapsed_time for result in results]
    metrics = [result.metric for result in results]
    metric_names = [result.metric_name for result in results]

    # Display dos resultados:
    min_amp = np.nanmin(imgs[0])
    max_amp = np.nanmax(imgs[0])
    convert_to_db = lambda img: 20 * np.log10(img + 1e-9)

    offset = (reflector_grid.xres / 2, reflector_grid.zres / 2)
    imgs_db = [convert_to_db(img) for img in imgs]
    # imgs_db = [img for img in imgs]

    vmin = np.nanmax([np.min(imgs_db[0]), -15])
    vmax = np.nanmax([imgs_db[0]])

    n_cols = len(results)
    i = 1

    plot_on_first = True
    fig = plt.figure(figsize=(18, 10))
    plt.suptitle("Image in dB")
    for img_db, residue, elapsed_time, metric, metric_name, method_name in zip(imgs_db, residues, elapsed_times, metrics, metric_names, method_names):
        ax1 = plt.subplot(2, n_cols, i)
        cax = plt.imshow(img_db, extent=reflector_grid.get_extent(offset=offset), aspect='equal',interpolation='None', vmin=vmin, vmax=vmax)
        if method_name != "TFM":
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
