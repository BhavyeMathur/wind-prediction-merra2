import numpy as np

from era5.variables import AtmosphericVariable
from era5.util.datetime import timedelta

from .metplot import MetPlot
from .lev2D import _LonLev2D, _LatLev2D
from .time2D import _TimeLon2D, _TimeLat2D
from .latlon import _MapContour
from .plot1D import _Lon1D


def plot(variable: AtmosphericVariable | tuple[AtmosphericVariable, ...], indices: list,
         transform=lambda data: data) -> MetPlot:
    if isinstance(variable, AtmosphericVariable):
        time, lev, lat, lon = variable.get_full_index(indices)
    else:
        time, lev, lat, lon = variable[0].get_full_index(indices)

    idx = np.array(["time", "lev", "lat", "lon"])[[bool(time), bool(lev), lat is not None, lon is not None]]

    match tuple(idx):
        case ("time", "lev"):
            graph = _MapContour
        case ("time", "lat"):
            graph = _LonLev2D
        case ("time", "lon"):
            graph = _LatLev2D
        case ("lev", "lat"):
            graph = _TimeLon2D
        case ("lev", "lon"):
            graph = _TimeLat2D
        case ("time", "lev", "lat"):
            graph = _Lon1D
        case _:
            raise NotImplementedError(idx)

    if time is None:
        indices[0] = slice("TAVG-01-01 00:00", "TAVG-12-31 12:00", timedelta(days=1))

    return graph(variable, indices, transform)


__all__ = ["plot"]
