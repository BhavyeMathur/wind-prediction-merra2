import matplotlib.pyplot as plt
import numpy as np


class _Plotter:
    def plot(self, ax: plt.Axes, data: np.ndarray, **kwargs):
        raise NotImplementedError()


class ImagePlot2D(_Plotter):
    def __init__(self, extent: tuple[tuple[float, float], tuple[float, float]]):
        self._extent = *extent[0], *extent[1]

    def plot(self, ax, data, **kwargs):
        kwargs = dict(extent=self._extent, origin="lower", interpolation="nearest", aspect="auto") | kwargs
        return ax.imshow(data, **kwargs)


class Contour2D(_Plotter):
    def __init__(self, mesh: np.ndarray):
        self._mesh = mesh

    def plot(self, ax, data, **kwargs):
        kwargs = dict(levels=20) | kwargs
        return ax.contour(*self._mesh, data, **kwargs)


class Contourf2D(Contour2D):
    def plot(self, ax, data, **kwargs):
        kwargs = dict(levels=20) | kwargs
        return ax.contour(*self._mesh, data, **kwargs)


__all__ = ["ImagePlot2D", "Contour2D", "Contourf2D"]
