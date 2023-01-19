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
