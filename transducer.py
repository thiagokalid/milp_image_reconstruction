import numpy as np
from scipy.signal import gausspulse


class Transducer:
    def __init__(self, pitch=.5, n_elem=64, fc=5e6, bw=.5, bwr=-6, pulse_type="gaussian"):
        self.pitch = pitch
        self.n_elem = n_elem
        self.fc = fc
        self.bw = bw  # Hz
        self.bwr = bwr  # dB
        self.pulse_type = pulse_type
        self.xt = np.arange(0, self.n_elem) * pitch
        self.xt -= np.mean(self.xt)
        self.zt = np.zeros_like(self.xt)

    def get_coords(self, i=None):
        if i is None:
            return self.xt, self.zt
        else:
            return self.xt[i], self.zt[i]

    def get_signal(self, tspan, delta_t=0):
        if self.pulse_type == "gaussian":
            return gausspulse((tspan - delta_t) * 1e-6, fc=self.fc, bw=self.bw, bwr=self.bwr)
        else:
            raise NotImplementedError
