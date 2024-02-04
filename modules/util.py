from time import strftime
import string


def format_bytes(size: int, unit, si: bool = False) -> str:
    """
    Converts bytes to common units such as kb, kib, KB, mb, mib, MB

    Parameters
    ---------
    size: int
        Number of bytes to be converted

    unit: str
        Desired unit of measure for output

    si: bool
        True -> Use SI standard e.g. KB = 1000 bytes
        False -> Use JEDEC standard e.g. KB = 1024 bytes

    Returns
    -------
    str:
        E.g. "7 MiB" where MiB is the original unit abbreviation supplied
    """
    if unit.lower() in {"b", "bit", "bits"}:
        return f"{size * 8} {unit}"
    unit_name = unit[0].upper()+unit[1:].replace("s", "")  # Normalised
    reference = {"Kb Kib Kibibit Kilobit": (7, 1),
                 "KB KiB Kibibyte Kilobyte": (10, 1),
                 "Mb Mib Mebibit Megabit": (17, 2),
                 "MB MiB Mebibyte Megabyte": (20, 2),
                 "Gb Gib Gibibit Gigabit": (27, 3),
                 "GB GiB Gibibyte Gigabyte": (30, 3),
                 "Tb Tib Tebibit Terabit": (37, 4),
                 "TB TiB Tebibyte Terabyte": (40, 4),
                 "Pb Pib Pebibit Petabit": (47, 5),
                 "PB PiB Pebibyte Petabyte": (50, 5),
                 "Eb Eib Exbibit Exabit": (57, 6),
                 "EB EiB Exbibyte Exabyte": (60, 6),
                 "Zb Zib Zebibit Zettabit": (67, 7),
                 "ZB ZiB Zebibyte Zettabyte": (70, 7),
                 "Yb Yib Yobibit Yottabit": (77, 8),
                 "YB YiB Yobibyte Yottabyte": (80, 8),
                 }
    key_list = '\n'.join(["     b Bit"] + [x for x in reference.keys()]) + "\n"
    if unit_name not in key_list:
        raise IndexError(f"\n\nConversion unit must be one of:\n\n{key_list}")
    units, divisors = [(k, v) for k, v in reference.items() if unit_name in k][0]
    if si:
        divisor = 1000**divisors[1]/8 if "bit" in units else 1000**divisors[1]
    else:
        divisor = float(1 << divisors[0])
    value = size / divisor
    return f"{value:,.0f} {unit_name}{(value != 1 and len(unit_name) > 3) * 's'}"


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
