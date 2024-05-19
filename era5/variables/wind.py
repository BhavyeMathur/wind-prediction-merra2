import numpy as np

from .abstract import AtmosphericVariable4D


class UWind(AtmosphericVariable4D, name="u_component_of_wind", unit="ms⁻¹", title="East Wind"):
    def get_vlims(self, indices):
        time, lev, lat, lon = self.get_full_index(indices)
        if lev == 1000:
            return -15, 15
        elif lev == 150:
            return -40, 70
        return super().get_vlims(indices)


class VWind(AtmosphericVariable4D, name="v_component_of_wind", unit="ms⁻¹", title="North Wind"):
    def get_vlims(self, indices):
        time, lev, lat, lon = self.get_full_index(indices)
        if lev == 1000:
            return -15, 15
        elif lev == 150:
            return -20, 20
        return super().get_vlims(indices)


class WindDirection(AtmosphericVariable4D, name="wind_direction", unit="°", cmap="twilight",
                    requires=["u_component_of_wind", "v_component_of_wind"]):
    def __init__(self, radians: bool = False):
        super().__init__()
        if radians:
            self._unit = "rad"
        else:
            self._unit = "°"
        self._radians = radians

    def get_vlims(self, _):
        if self._radians:
            return -np.pi, np.pi
        return -180, 180

    def _getitem_post(self, ds):
        val = np.arctan2(ds["u_component_of_wind"], ds["v_component_of_wind"])
        if self._radians:
            return val
        return np.degrees(val)


class WindSpeed(AtmosphericVariable4D, name="wind_speed", unit="ms⁻¹",
                requires=["u_component_of_wind", "v_component_of_wind"]):
    def get_vlims(self, indices):
        time, lev, lat, lon = self.get_full_index(indices)
        if lev == 1000:
            return 0, 16
        return super().get_vlims(indices)

    def _getitem_post(self, ds):
        return np.sqrt(ds["u_component_of_wind"] ** 2 + ds["v_component_of_wind"] ** 2)


class Divergence(AtmosphericVariable4D, name="divergence", unit="s⁻¹",
                 requires=["u_component_of_wind", "v_component_of_wind"]):
    def _getitem_post(self, ds):
        dudx, _ = np.gradient(ds["u_component_of_wind"])
        _, dvdy = np.gradient(ds["v_component_of_wind"])

        return dudx + dvdy


__all__ = ["UWind", "VWind", "WindDirection", "WindSpeed", "Divergence"]
