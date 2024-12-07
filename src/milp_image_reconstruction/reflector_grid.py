import numpy as np

__all__ = ['ReflectorGrid']

class ReflectorGrid:
    def __init__(self, width: float, height: float, xres: float=.1, zres: float=.1):
        self.width = width
        self.height = height
        self.xres = xres
        self.zres = zres
        self.xspan, self.zspan = np.arange(-width / 2, width / 2 + xres, xres), np.arange(0, height + zres, zres)
        self.imgsize = len(self.xspan), len(self.zspan)
        self.xv, self.zv = np.meshgrid(self.xspan, self.zspan, indexing='ij')
        self.xv = np.ravel(self.xv)
        self.zv = np.ravel(self.zv)
        self.n_reflectors = len(self.xv)

    def get_coords(self, offset: [float, float]=(0, 0), i: int=-1):
        if i == -1:
            return self.xv - offset[0], self.zv - offset[1]
        else:
            return self.xv[i] - offset[0], self.zv[i] - offset[1]

    def get_extent(self, offset: [float, float]=(0, 0)) -> [float, float, float, float]:
        return [
            self.xspan[0] - offset[0], self.xspan[-1] + offset[0],
            self.zspan[-1] + offset[1], self.zspan[0] - offset[1]
        ]

    def get_imgsize(self) -> [int, int]:
        return self.imgsize

    def get_numpxs(self) -> int:
        return self.imgsize[0] * self.imgsize[1]