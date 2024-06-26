from typing import Generator

from datetime import datetime, timedelta

import numpy as np

from era5.util import get_function_arguments

MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]

DATETIME_TYPE = str | datetime


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


class DateTime(datetime):
    def __new__(cls, month: int, day: int, hour: int, year: int | str = "tavg"):
        return datetime.__new__(cls, year=1980 if year == "tavg" else year, month=month, day=day, hour=hour)

    def __init__(self, month: int, day: int, hour: int, year: int | str = "tavg"):
        self.tavg = year == "tavg"
        super().__init__()

    def __repr__(self) -> str:
        return f"DateTime({'TAVG' if self.tavg else self.year}-{self.month:02}{self.day:02} {self.hour:02}:00:00)"

    def __str__(self) -> str:
        return f"{'TAVG' if self.tavg else self.year}-{self.month:02}-{self.day:02} {self.hour:02}:00:00"

    def __format__(self, format_spec) -> str:
        if format_spec == "date":
            return f"{'TAVG' if self.tavg else self.year}-{self.month:02}-{self.day:02}"
        return f"{'tavg' if self.tavg else self.year}-{self.month:02}{self.day:02}-{self.hour:02}00"

    def __hash__(self):
        return hash((self.year, self.month, self.day, self.hour))

    @datetime_func("other")
    def __add__(self, other):
        tavg = self.tavg or (isinstance(other, DateTime) and other.tavg)

        result = datetime(year=1981 if tavg else self.year, month=self.month, day=self.day, hour=self.hour) + other
        return DateTime(month=result.month, day=result.day, hour=result.hour, year="tavg" if tavg else result.year)

    @datetime_func("other")
    def __sub__(self, other):
        tavg = self.tavg or (isinstance(other, DateTime) and other.tavg)

        result = datetime(year=1981 if tavg else self.year, month=self.month, day=self.day, hour=self.hour) - other
        return DateTime(month=result.month, day=result.day, hour=result.hour, year="tavg" if tavg else result.year)


def date_to_dayofyear(day: int, month: int) -> int:
    """
    Converts a date to day of the year (starting at 0)
    """
    assert 1 < month <= 12
    assert 1 < day <= MONTH_DAYS[month - 1]
    return sum(MONTH_DAYS[:month - 1]) + day - 1  # -1 to convert from nth day to index


def dayofyear_to_date(i: int) -> tuple[int, int]:
    """
    Converts a day of year (0 indexed) to day, month
    """
    assert 0 <= i < 365

    month = 0
    while i >= MONTH_DAYS[month]:
        i -= MONTH_DAYS[month]
        month += 1

    return i + 1, month + 1


def date_as_number(day: int, month: int, year: int) -> int:
    """
    Returns a date as a number in the format 'YYYYMMDD'
    """
    return 10000 * year + 100 * month + day


def parse_datetime(dt: DATETIME_TYPE) -> None | DateTime | timedelta:
    """
    Automatically parses a datetime string into a datetime object using the best-fit format.
    """
    if dt is None or isinstance(dt, (timedelta, DateTime)):
        return dt

    if isinstance(dt, np.datetime64):
        dt = ((dt - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'))
        dt = datetime.utcfromtimestamp(dt)

    if isinstance(dt, datetime):
        return DateTime(month=dt.month, day=dt.day, hour=dt.hour, year=dt.year)

    if not isinstance(dt, str):
        raise ValueError(f"Unknown date format '{dt}'")

    is_tavg = "TAVG" in dt
    if is_tavg:
        dt = dt.replace("TAVG", "1980")

    fmts = ["%Y-%m-%d %H:%M", "%m-%d %H:%M", "%Y-%m-%d", "%m-%d", "%Y-%m"]

    for fmt in fmts:
        try:
            dt = datetime.strptime(dt, fmt)
            break
        except ValueError:
            continue

    else:
        raise ValueError(f"Unknown date format '{dt}'")

    return DateTime(dt.month, dt.day, dt.hour, "tavg" if is_tavg else dt.year)


@datetime_func("start", "end")
def datetime_range(start: DATETIME_TYPE, end: DATETIME_TYPE, delta: timedelta) -> Generator[datetime, None, None]:
    """
    Returns a datetime range between the start and end dates with a specified interval.
    """
    while start <= end:
        yield start

        if start + delta < start:
            return

        start += delta


def format_datetime(*args, pretty: bool = False) -> str:
    """
    Formats a datetime object into a string with format 'mmdd-HHMMM'

    Args:
        pretty: change format to 'mm-dd HH:MM'
    """
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, DateTime) and not pretty:
            return f"{arg}"
        if isinstance(arg, datetime):
            return arg.strftime("%m-%d %H:%M" if pretty else "%m%d-%H%M")

    month, day, hour = args
    if pretty:
        return f"{month:02}-{day:02} {hour:02}:00"
    return f"{month:02}{day:02}-{hour:02}00"
