from modules.datetime import parse_datetime_depr, MONTH_NAMES


def format_title(variable: str, time: None | str, lev: None | int) -> str | list[str]:
    parts = [format_variable(variable)]

    if time:
        parts.append(format_time(time))
    if lev:
        parts.append(format_level(lev))

    return " ".join(parts)


def format_output_time(time: None | str) -> str | list[str]:
    minute, hour, day, month, year = parse_datetime_depr(time)
    if minute and day:  # and hour and month
        return f"{year if year else 'YAVG'}{month:02}{day:02}-{hour:02}{minute:02}"
    elif minute:  # and hour
        return f"{year if year else 'YAVG'}-{hour:02}{minute:02}"
    elif day:  # and month
        return f"{year if year else 'YAVG'}{month:02}{day:02}"
    else:
        return year if year else 'YAVG'


def format_output(variable: str, time: None | str, lev: None | int) -> list[str]:
    parts = [variable]

    # FRLAND.png
    # U-YAVG0101-0430-lev36.png

    if isinstance(time, list):
        parts.append(format_output_time(time[0]))
        parts.append(format_output_time(time[-1]))
    elif time:
        parts.append(format_output_time(time))

    if lev:
        parts.append(f"lev{lev}")

    return parts
