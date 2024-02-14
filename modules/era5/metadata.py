def get_units(variable) -> str:
    if variable in {"u_component_of_wind", "v_component_of_wind"}:
        return "m/s"
    if variable == "temperature":
        return "K"
    if variable == "vertical_velocity":
        return "Pa/s"

    if variable == "level":
        return "hPa"
    if variable == "longitude":
        return "degrees east"
    if variable == "latitude":
        return "degrees north"


def get_coord_dtype(variable) -> str:
    if variable == "level":
        return "int16"
    return "float32"
