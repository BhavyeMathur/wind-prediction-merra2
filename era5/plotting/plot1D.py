from .metplot import MetPlot, PlotVersusLongitude
from .plotter import LinePlot
from era5.variables.text import format_altitude, format_latitude, format_time


class _Lon1D(PlotVersusLongitude, MetPlot):
    _axes_lims = (-180, 180), None
    _plotter = LinePlot
    _colorbar = False

    def __init__(self, variable, *args, **kwargs):
        super().__init__(variable, *args, **kwargs)
        self._yunit = variable.unit

    def _get_title_slice_substring(self) -> str:
        lev = int(self._dset['level'].values)
        return (f" at {lev} hPa {format_altitude(lev)}"
                f" on {format_time(self._dset['time'].values, self._dset.attrs['is_tavg'])}"
                f" ({format_latitude(self._lat)})" if isinstance(self._lat, float) else "")
