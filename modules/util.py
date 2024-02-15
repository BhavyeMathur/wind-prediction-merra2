from typing import Generator

import datetime
from time import strftime
import math

import string


def format_bytes(size: int, si: bool = False) -> str:
    """
    Converts bytes to printable units

    Parameters
    ---------
    size: int
        Number of bytes to be converted

    si: bool
        True -> Use SI standard e.g. KB = 1000 bytes
        False -> Use JEDEC standard e.g. KB = 1024 bytes
    """

    order = 0
    while math.log10(size) > 3:
        size /= 1000 if si else 1024
        order += 1

    unit_name = size_units[order]
    return f"{size:.2f} {unit_name if si else unit_name.replace('B', 'iB')}{'' if size == 1 else 's'}"


size_units = ("bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
fmt = string.Formatter()


def get_number_of_string_format_args(s: str) -> int:
    return len([p for p in fmt.parse(s) if p[2] is not None])


def log(*args) -> None:
    print(f"[{strftime('%H:%M:%S')}] LOG:", *args)


MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]


def date_to_dayofyear(day: int, month: int) -> int:
    """
    Converts a date to day of the year (starting at 0)
    """
    assert 1 < month <= 12
    assert 1 < day <= MONTH_DAYS[month - 1]
    return sum(MONTH_DAYS[:month - 1]) + day - 1  # -1 to convert from nth day to index


def dayofyear_to_date(file: int) -> tuple[int, int]:
    """
    Converts a day of year (0 indexed) to day, month
    """
    assert 0 <= file < 365

    month = 0
    while file > MONTH_DAYS[month]:
        file -= MONTH_DAYS[month]
        month += 1

    return file, month + 1


def date_as_number(day: int, month: int, year: int) -> int:
    """
    Returns a date as a number in the format 'YYYYMMDD'
    """
    return 10000 * year + 100 * month + day


def parse_datetime(datetime: str, yavg: int = 0) -> tuple[int | None, int | None, int | None, int | None, int | None]:
    time = None
    hour = None
    minute = None
    year = None
    day = None
    month = None

    if datetime is None:
        return None, None, None, None, None

    try:
        if len(segments := datetime.split(" ")) == 2:
            time, date = segments
        elif len(segments) == 1:
            date = segments[0]
        else:
            raise ValueError(f"Invalid time format {datetime!r}")

        if len(date := date.split("-")) == 3:
            day, month, year = date
        elif len(date) == 2:
            day, month = date
        elif len(date) == 1 and len(date[0]) == 4:
            year = date[0]
        else:
            raise ValueError(f"Invalid time format {datetime!r}")

        if time is not None:
            hour, minute = time.split(":")

        return (None if minute is None else int(minute),
                None if hour is None else int(hour),
                None if day is None else int(day),
                None if month is None else int(month),
                yavg if (year is None or year == "YAVG") else int(year))

    except Exception:
        raise ValueError(f"Invalid time format {datetime!r}")


def _parse_datetime(dt: str) -> datetime.datetime:
    try:
        return datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M")
    except ValueError:
        return datetime.datetime.strptime(dt, "%m-%d %H:%M")


def datetime_range(start: str, end: str, delta: datetime.timedelta) -> Generator[datetime.datetime, None, None]:
    start = _parse_datetime(start)
    end = _parse_datetime(end)

    while start <= end:
        yield start
        start += delta


def format_datetime(month: int, day: int, hour: int, pretty: bool = False) -> str:
    if pretty:
        return f"{month:02}-{day:02} {hour:02}:00"
    return f"{month:02}{day:02}-{hour:02}00"
