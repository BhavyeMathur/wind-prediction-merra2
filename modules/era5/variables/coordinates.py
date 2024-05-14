from .abstract import AtmosphericVariable


class Time(AtmosphericVariable, name="time", unit="seconds", dtype="datetime"):
    pass


class Level(AtmosphericVariable, name="level", unit="hPa", dtype="int16", axes_unit=" mb"):
    pass


class Latitude(AtmosphericVariable, name="latitude", unit="degrees north", axes_unit="°"):
    pass


class Longitude(AtmosphericVariable, name="longitude", unit="degrees east", axes_unit="°"):
    pass


__all__ = ["Time", "Level", "Latitude", "Longitude"]
