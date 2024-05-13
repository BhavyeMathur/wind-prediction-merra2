from .abstract import AtmosphericVariable


class Time(AtmosphericVariable, name="time", unit="seconds", dtype="datetime"):
    pass


class Level(AtmosphericVariable, name="level", unit="hPa", dtype="int16"):
    pass


class Latitude(AtmosphericVariable, name="latitude", unit="degrees north"):
    pass


class Longitude(AtmosphericVariable, name="longitude", unit="degrees east"):
    pass


__all__ = ["Time", "Level", "Latitude", "Longitude"]
