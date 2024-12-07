import numpy as np
from scipy.signal import gausspulse
from numpy import ndarray

__all__ = ["Transducer"]

class Transducer:
    def __init__(self, pitch: float=.5, n_elem: int=64, fc: float=5e6, bw: float=.5, bwr: float=-6, pulse_type: str="gaussian"):
        self.pitch = pitch
        self.n_elem = n_elem
        self.fc = fc
        self.bw = bw  # Hz
        self.bwr = bwr  # dB
        self.pulse_type = pulse_type
        self.xt = np.arange(0, self.n_elem) * pitch
        self.xt -= np.mean(self.xt)
        self.zt = np.zeros_like(self.xt)

    def get_coords(self, i: int=-1):
        if i == -1:
            return self.xt, self.zt
        else:
            return self.xt[i], self.zt[i]

    def get_signal(self, tspan: ndarray , delta_t: float=0):
        if self.pulse_type == "gaussian":
            return gausspulse((tspan - delta_t) * 1e-6, fc=self.fc, bw=self.bw, bwr=self.bwr)
        else:
            raise NotImplementedError
