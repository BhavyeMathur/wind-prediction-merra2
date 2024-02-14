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
    return f"{size:.2f} {unit_name if si else unit_name.replace('B', 'iB')}{size != 1 * 's'}"


size_units = ("bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
fmt = string.Formatter()


def get_number_of_string_format_args(s: str) -> int:
    return len([p for p in fmt.parse(s) if p[2] is not None])


def log(*args) -> None:
    print(f"[{strftime('%H:%M:%S')}] LOG:", *args)


MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]


def date_to_index(day: int, month: int) -> int:
    assert 1 < month <= 12
    assert 1 < day <= MONTH_DAYS[month - 1]
    return sum(MONTH_DAYS[:month - 1]) + day - 1  # -1 to convert from nth day to index


def index_to_date(file: int) -> tuple[int, int]:
    assert 0 <= file < 365

    month = 0
    while file > MONTH_DAYS[month]:
        file -= MONTH_DAYS[month]
        month += 1

    return file, month + 1


def date_as_number(day: int, month: int, year: int) -> int:
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


def datetime_range(start: str, end: str, delta: datetime.timedelta) -> Generator[datetime.datetime, None, None]:
    minute1, hour1, day1, month1, year1 = parse_datetime(start)
    minute2, hour2, day2, month2, year2 = parse_datetime(end)

    start = datetime.datetime(1981 if year1 == 0 else year1,
                              month1 if month1 else 1,
                              day1 if day1 else 1,
                              hour1 if hour1 else 1,
                              minute1 if minute1 else 30)

    end = datetime.datetime(1981 if year2 == 0 else year2,
                            month2 if month2 else 12,
                            day2 if day2 else 31,
                            hour2 if hour2 else 22,
                            minute2 if minute2 else 30)

    while start <= end:
        yield start
        start += delta


def months_in_year() -> range:
    return range(1, 13)


def days_in_month(month: int) -> range:
    """
    Parameters
    ----------
    month: 1 for January, 12 for December
    """
    return range(1, MONTH_DAYS[month - 1] + 1)


def days_in_year() -> Generator[tuple[int, int], None, None]:
    for month in months_in_year():
        for day in days_in_month(month):
            yield month, day


def hours_in_year() -> Generator[tuple[int, int, int], None, None]:
    for month in months_in_year():
        for day in days_in_month(month):
            for hour in range(24):
                yield month, day, hour
