from .abstract import AtmosphericVariable4D


class UWind(AtmosphericVariable4D, name="u_component_of_wind", unit="m/s"):
    @staticmethod
    def get_vlims(level: int):
        if level == 1000:
            return -15, 15
        raise ValueError("Unknown level for value limits")


class VWind(AtmosphericVariable4D, name="v_component_of_wind", unit="m/s"):
    @staticmethod
    def get_vlims(level: int):
        if level == 1000:
            return -15, 15
        raise ValueError("Unknown level for value limits")


class VerticalVelocity(AtmosphericVariable4D, name="vertical_velocity", unit="Pa/s", cmap="RdBu"):
    @staticmethod
    def get_vlims(level: int):
        if level == 1000:
            return -1.5, 1.5
        raise ValueError("Unknown level for value limits")


class WindDirection(AtmosphericVariable4D, name="wind_direction", unit="°", cmap="twilight"):
    @staticmethod
    def get_vlims(_):
        return -180, 180


class WindSpeed(AtmosphericVariable4D, name="wind_speed", unit="m/s"):
    @staticmethod
    def get_vlims(level: int):
        if level == 1000:
            return 0, 16
        raise ValueError("Unknown level for value limits")


__all__ = ["UWind", "VWind", "VerticalVelocity", "WindDirection", "WindSpeed"]