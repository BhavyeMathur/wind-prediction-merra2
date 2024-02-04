from modules.merra2.util import get_variable_name_from_code, get_units_from_variable, get_pressure_from_level
from modules.util import parse_datetime, MONTH_NAMES
from modules.maths.barometric import height_from_pressure


def format_variable(variable: str) -> str:
    return f"{get_variable_name_from_code(variable)} ({get_units_from_variable(variable)})"


def format_time(time: str) -> str:
    minute, hour, day, month, year = parse_datetime(time)
    string = []

    if hour is not None:  # implies minute is not None
        string.append(f"at {hour:02}:{minute:02}")

    if day is not None:  # implies month is not None
        string.append(f"on {day} {MONTH_NAMES[month]}")

    if year != 0 and year is not None:
        string.append(str(year))

    return " ".join(string)


def format_level(lev: int) -> str:
    return f"(≈{round(height_from_pressure(get_pressure_from_level(lev) * 100), -2):.0f} m)"


def format_title(variable: str, time: None | str, lev: None | int) -> str:
    parts = [format_variable(variable)]

    if time:
        parts.append(format_time(time))
    if lev:
        parts.append(format_level(lev))

    return " ".join(parts)
