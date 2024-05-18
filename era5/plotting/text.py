from era5.util.datetime import datetime_func, DateTime
from era5.maths.barometric import height_from_pressure


@datetime_func("time")
def format_time(time, is_tavg: bool = False) -> str:
    if (isinstance(time, DateTime) and time.tavg) or is_tavg:
        return time.strftime("at %H:%M on %-d %B")
    else:
        return time.strftime("at %H:%M on %-d %B, %Y")


def format_altitude(level: int) -> str:
    return f"(≈{round(height_from_pressure(level * 100), -2):.0f} m)"


def format_latitude(lat: float) -> str:
    if lat == 0:
        return f"0°"
    if lat < 0:
        return f"{round(abs(lat), 1)}°S"
    return f"{round(lat, 1)}°N"


def format_longitude(lon: float) -> str:
    if lon == 0:
        return f"0°"
    if lon < 0:
        return f"{round(abs(lon), 1)}°W"
    return f"{round(lon, 1)}°E"
