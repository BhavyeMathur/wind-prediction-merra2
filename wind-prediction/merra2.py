from maths import height_from_pressure


def get_merra_stream_from_year(year: int) -> int:
    if year < 1992:
        return 100
    elif year < 2001:
        return 200
    elif year < 2011:
        return 300
    else:
        return 400


def get_merra_variables(variables):
    if isinstance(variables, str):
        return [variables]
    return variables


def get_pressure_from_level(level, total_levels: int = 72) -> float:
    if isinstance(level, float):
        pressure = get_pressure_from_level(int(level), total_levels=total_levels) * (1 - level % 1)
        pressure += get_pressure_from_level(int(level) + 1, total_levels=total_levels) * (level % 1)
        return pressure

    if total_levels == 72:
        return [0.01, 0.02, 0.0327, 0.0476, 0.0660, 0.0893, 0.1197, 0.1595, 0.2113, 0.2785,
                0.3650, 0.4758, 0.6168, 0.7951, 1.0194, 1.3005, 1.6508, 2.0850, 2.6202,
                3.2764, 4.0766, 5.0468, 6.2168, 7.6198, 9.2929, 11.2769, 13.6434, 16.4571,
                19.7916, 23.7304, 28.3678, 33.8100, 40.1754, 47.6439, 56.3879, 66.6034,
                78.5123, 92.3657, 108.663, 127.837, 150.393, 176.930, 208.152, 244.875,
                288.083, 337.5, 375, 412, 450, 487.5, 525, 562.5, 600, 637.5, 675, 700, 725,
                750, 775, 800, 820, 835, 850, 865, 880, 895, 910, 925, 940, 955, 970, 985][level]
    elif total_levels == 42:
        return [0.1, 0.3, 0.4, 0.5, 0.7, 1, 2, 3, 4, 5, 7, 10, 20, 30, 40, 50, 70, 100,
                150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 725, 750,
                775, 800, 825, 850, 875, 900, 925, 950, 975, 1000][level]
    else:
        raise ValueError("Unknown levels in MERRA-2 data")


def get_units_from_variable(variable: str) -> str:
    match variable:
        case "U":
            return "m/s"
        case "V":
            return "m/s"
        case "Wind Speed":
            return "m/s"


def get_variable_name_from_code(variable: str) -> str:
    match variable:
        case "U":
            return "East Wind"
        case "V":
            return "North Wind"
        case "T":
            return "Temperature"


def format_level(level, total_levels=72) -> str:
    pressure = get_pressure_from_level(level, total_levels=total_levels)
    return f"{pressure:.2f} hPa (~{height_from_pressure(100 * pressure):.2f} m)"


def format_latitude(latitude) -> str:
    return f"{latitude * 0.5 - 90:0>2}° lat"


def format_longitude(longitude) -> str:
    return f"{longitude * 0.625 - 180:0>2}° lon"


def _format_month(month: int) -> str:
    return ["January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December"][month]


def format_date(filename: str) -> str:
    # MERRA2_100.tavg3_3d_asm_Nv.19800101.nc4
    date = filename.split(".")[-2]
    return f"{int(date[7:])} {_format_month(int(date[4:6]) - 1)} {date[:4]}"


def format_time(time: int) -> str:
    return f"{time * 3 + 1:0>2}:30"
