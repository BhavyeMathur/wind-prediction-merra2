def get_pressure_from_level(level, total_levels: int = 36) -> float:
    """
    :return: Pressure in (hPa)
    """

    if total_levels == 36:
        return get_pressure_from_level(level + 36, total_levels=72)

    elif total_levels == 42:
        return [0.1, 0.3, 0.4, 0.5, 0.7, 1, 2, 3, 4, 5, 7, 10, 20, 30, 40, 50, 70, 100,
                150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 725, 750,
                775, 800, 825, 850, 875, 900, 925, 950, 975, 1000][level]

    elif total_levels == 72:
        return [0.01, 0.02, 0.0327, 0.0476, 0.0660, 0.0893, 0.1197, 0.1595, 0.2113, 0.2785,
                0.3650, 0.4758, 0.6168, 0.7951, 1.0194, 1.3005, 1.6508, 2.0850, 2.6202,
                3.2764, 4.0766, 5.0468, 6.2168, 7.6198, 9.2929, 11.2769, 13.6434, 16.4571,
                19.7916, 23.7304, 28.3678, 33.8100, 40.1754, 47.6439, 56.3879, 66.6034,
                78.5123, 92.3657, 108.663, 127.837, 150.393, 176.930, 208.152, 244.875,
                288.083, 337.5, 375, 412, 450, 487.5, 525, 562.5, 600, 637.5, 675, 700, 725,
                750, 775, 800, 820, 835, 850, 865, 880, 895, 910, 925, 940, 955, 970, 985][level]

    else:
        raise ValueError("Unknown number of levels in MERRA-2 data")


def get_merra_stream_from_year(year: int, month: int = 1) -> int:
    if year < 1992:
        return 100
    elif year < 2001:
        return 200
    elif year < 2011:
        return 300
    elif year == 2021 and month > 5:
        return 401
    elif year < 2022:
        return 400
    else:
        return 401


def get_merra_time_index_from_time(minute: None | int, hour: None | int, dimensions: int) -> None | int:
    if minute is None or hour is None:
        return None

    if dimensions == 8:
        return int((hour + minute / 60 - 1.5) / 3)
    elif dimensions == 1:
        return int((hour + minute / 60) / 24)

    raise ValueError(f"Unsupported time format '{hour}:{minute}' for {dimensions=}")


_MERRA2_VARIABLES = {"U": "East Wind",
                     "V": "North Wind",
                     "T": "Temperature",
                     "FRLAND": "Fraction of Land",
                     "FRLANDICE": "Fraction of Land Ice",
                     "FROCEAN": "Fraction of Ocean",
                     "FRLAKE": "Fraction of Lake",
                     "PHIS": "Surface Geopotential Height",
                     "SGH": "Isotropic Standard Deviation of GWD Topography"}


def get_variable_name_from_code(variable: str) -> str:
    if variable.endswith("_est"):
        return f"Estimated {get_variable_name_from_code(variable[:-4])}"

    return _MERRA2_VARIABLES[variable]


def get_units_from_variable(variable: str) -> str:
    if variable.endswith("_est"):
        variable = variable[:-4]

    if variable.startswith("FR"):
        return "%"

    if variable in {"AUTCNVRN", "COLCNVRN", "COLCNVSN", "CUCNVCI", "CUCNVCL", "CUCNVRN"} \
            or variable.startswith("DOXDT") or variable.startswith("DQ"):
        return "kg/m²/s"

    if variable in {"BKGERR"} or variable.startswith("DHDT") or variable.startswith("DKDT") \
            or variable.startswith("DPDT"):
        return "W/m²"

    if variable == "U":
        return "m/s"
    elif variable == "V":
        return "m/s"
    elif variable == "Wind Speed":
        return "m/s"
    elif variable == "PHIS":
        return "m²/s²"
    elif variable == "SGH":
        return "m"
