from typing import Generator

from datetime import datetime, timedelta
from .util import get_function_arguments

MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]

DATETIME_TYPE = str | datetime


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


def parse_datetime(dt: DATETIME_TYPE) -> datetime:
    """
    Automatically parses a datetime string into a datetime object using the best-fit format.
    """
    if isinstance(dt, datetime):
        return dt

    try:
        return datetime.strptime(dt, "%Y-%m-%d %H:%M")
    except ValueError:
        return datetime.strptime(dt, "%m-%d %H:%M")


def datetime_func(*args: str):
    """
    A decorator used to pre-process datetime arguments to functions.
    Specify the argument names as strings that are either datetime or string objects.
    When the decorated function is called, the datetime arguments will automatically be parsed into datetime objects

    Examples:
        .. code-block:: python

            @datetime_func("datetime_object")
            def function_using_datetime_objects(datetime_object: str | datetime.datetime):
                # if caller calls function with a string, it will be parsed to a datetime.datetime
                # if caller calls function with datetime.datetime, it will be passed on as-is
    """
    def _decorator(func):
        def closure(*a, **kwargs):
            function_args = get_function_arguments(func, a, kwargs)

            for name, val in function_args.items():
                if name in args:
                    function_args[name] = parse_datetime(val)

            return func(**function_args)

        return closure

    return _decorator


@datetime_func("start", "end")
def datetime_range(start: DATETIME_TYPE, end: DATETIME_TYPE, delta: timedelta) -> Generator[datetime, None, None]:
    """
    Returns a datetime range between the start and end dates with a specified interval.
    """
    while start <= end:
        yield start
        start += delta


def format_datetime(*args, pretty: bool = False) -> str:
    """
    Formats a datetime object into a string with format 'mmdd-HHMMM'

    Args:
        pretty: change format to 'mm-dd HH:MM'
    """
    if len(args) == 1 and isinstance(args[0], datetime):
        if pretty:
            return args[0].strftime("%m-%d %H:%M")
        return args[0].strftime("%m%d-%H%M")

    month, hour, day = args
    if pretty:
        return f"{month:02}-{day:02} {hour:02}:00"
    return f"{month:02}{day:02}-{hour:02}00"


def parse_datetime_depr(dt: str, yavg: int = 0) -> tuple[int | None, int | None, int | None, int | None, int | None]:
    """
    Deprecated. Use parse_datetime instead.
    """
    time = None
    hour = None
    minute = None
    year = None
    day = None
    month = None

    if dt is None:
        return None, None, None, None, None

    try:
        if len(segments := dt.split(" ")) == 2:
            time, date = segments
        elif len(segments) == 1:
            date = segments[0]
        else:
            raise ValueError(f"Invalid time format {dt!r}")

        if len(date := date.split("-")) == 3:
            day, month, year = date
        elif len(date) == 2:
            day, month = date
        elif len(date) == 1 and len(date[0]) == 4:
            year = date[0]
        else:
            raise ValueError(f"Invalid time format {dt!r}")

        if time is not None:
            hour, minute = time.split(":")

        return (None if minute is None else int(minute),
                None if hour is None else int(hour),
                None if day is None else int(day),
                None if month is None else int(month),
                yavg if (year is None or year == "YAVG") else int(year))

    except Exception:
        raise ValueError(f"Invalid time format {dt!r}")
