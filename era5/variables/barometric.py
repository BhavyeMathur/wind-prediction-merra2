from .abstract import AtmosphericVariable4D


class Temperature(AtmosphericVariable4D, name="temperature", unit="K",
                  cmap=["#d7e4fc", "#5fb1d4", "#4e9bc8", "#466ae1", "#6b1966",
                        "#952c5e", "#d12d3e", "#fa7532", "#f5d25f"]):
    def __init__(self, celsius: bool = False):
        super().__init__()
        if celsius:
            self._unit = "°C"
        else:
            self._unit = "K"
        self._celsius = celsius

    def _getitem_post(self, ds: int):
        if self._celsius:
            return ds - 273.15  # Kelvin to Celsius
        return ds

    def get_vlims(self, indices):
        time, lev, lat, lon = self.get_full_index(indices)
        if lev == 1000:
            min_, max_ = 230, 310
        elif lev == 150:
            min_, max_ = 200, 230
        else:
            return super().get_vlims(indices)

        # unit conversions
        return self._getitem_post(min_), self._getitem_post(max_)


class VerticalVelocity(AtmosphericVariable4D, name="vertical_velocity", unit="Pa/s", cmap="RdBu"):
    _diverging = True

    def get_vlims(self, indices):
        time, lev, lat, lon = self.get_full_index(indices)
        if lev == 1000:
            return -1.5, 1.5
        return super().get_vlims(indices)


__all__ = ["Temperature", "VerticalVelocity"]
