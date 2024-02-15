from time import strftime
import inspect
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


# From Zio, https://stackoverflow.com/questions/218616/how-to-get-method-parameter-names
def get_function_arguments(func, args, kwargs, is_method=False):
    offset = 1 if is_method else 0
    specs = inspect.getfullargspec(func)
    d = {}
    non_default_args = len(specs.args) - (len(specs.defaults) if specs.defaults else 0)

    for i, parameter in enumerate(specs.args[offset:]):
        i += offset
        if i < len(args):
            d[parameter] = args[i]
        elif parameter in kwargs:
            d[parameter] = kwargs[parameter]
        else:
            d[parameter] = specs.defaults[i - non_default_args]
    return d
