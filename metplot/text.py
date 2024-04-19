from modules.datetime import datetime_func, DateTime
from modules.maths.barometric import height_from_pressure


@datetime_func("time")
def format_time(time) -> str:
    if isinstance(time, DateTime) and time.tavg:
        return time.strftime("at %H:%M on %d %B")
    else:
        return time.strftime("at %H:%M on %d %B, %Y")


_VARIABLES = {"U": "East Wind",
              "u_component_of_wind": "East Wind",
              "V": "North Wind",
              "v_component_of_wind": "North Wind",
              "T": "Temperature",
              "FRLAND": "Fraction of Land",
              "FRLANDICE": "Fraction of Land Ice",
              "FROCEAN": "Fraction of Ocean",
              "FRLAKE": "Fraction of Lake",
              "PHIS": "Surface Geopotential Height",
              "SGH": "Isotropic Standard Deviation of GWD Topography"}


def get_long_variable_name(variable: str) -> str:
    if variable.isupper():
        default = variable
    else:
        default = variable.replace("_", " ").capitalize()

    return _VARIABLES.get(variable, default)


def get_units_from_variable(variable: str) -> str | None:
    if variable.startswith("FR"):
        return "%"

    if variable in {"AUTCNVRN", "COLCNVRN", "COLCNVSN", "CUCNVCI", "CUCNVCL", "CUCNVRN"} \
            or variable.startswith("DOXDT") or variable.startswith("DQ"):
        return "kg/m²/s"

    if variable in {"BKGERR"} or variable.startswith("DHDT") or variable.startswith("DKDT") \
            or variable.startswith("DPDT"):
        return "W/m²"

    if variable in {"U", "V", "Wind Speed", "u_component_of_wind", "v_component_of_wind"}:
        return "m/s"
    elif variable == "PHIS":
        return "m²/s²"
    elif variable == "SGH":
        return "m"
    elif variable in {"temperature", "T"}:
        return "°C"

    return None


def format_variable(variable: str) -> str:
    unit = get_units_from_variable(variable)
    unit = f" ({unit})" if unit else ""
    return f"{get_long_variable_name(variable)}{unit}"


def format_pressure(level: int) -> str:
    return f"(≈{round(height_from_pressure(level * 100), -2):.0f} m)"
