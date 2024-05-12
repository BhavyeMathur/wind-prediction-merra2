def get_units(variable) -> str:
    """
    Gets the units corresponding to an ERA-5 variable
    """
    if variable == "level":
        return "hPa"
    if variable == "longitude":
        return "degrees east"
    if variable == "latitude":
        return "degrees north"


def get_coord_dtype(variable) -> str:
    """
    Gets the dtype corresponding to an ERA-5 coordinate
    """

    if variable == "level":
        return "int16"
    return "float32"
