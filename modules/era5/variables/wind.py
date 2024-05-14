import numpy as np

from .abstract import AtmosphericVariable4D


class UWind(AtmosphericVariable4D, name="u_component_of_wind", unit="m/s"):
    def get_vlims(self, level: int):
        if level == 1000:
            return -15, 15
        elif level == 150:
            return -40, 70
        raise ValueError("Unknown level for value limits")


class VWind(AtmosphericVariable4D, name="v_component_of_wind", unit="m/s"):
    def get_vlims(self, level: int):
        if level == 1000:
            return -15, 15
        elif level == 150:
            return -20, 20
        raise ValueError("Unknown level for value limits")


class VerticalVelocity(AtmosphericVariable4D, name="vertical_velocity", unit="Pa/s", cmap="RdBu"):
    def get_vlims(self, level: int):
        if level == 1000:
            return -1.5, 1.5
        raise ValueError("Unknown level for value limits")


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


class WindSpeed(AtmosphericVariable4D, name="wind_speed", unit="m/s",
                requires=["u_component_of_wind", "v_component_of_wind"]):
    def get_vlims(self, level: int):
        if level == 1000:
            return 0, 16
        raise ValueError("Unknown level for value limits")

    def _getitem_post(self, ds):
        return np.sqrt(ds["u_component_of_wind"] ** 2 + ds["v_component_of_wind"] ** 2)


__all__ = ["UWind", "VWind", "VerticalVelocity", "WindDirection", "WindSpeed"]
