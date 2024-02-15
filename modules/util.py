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
