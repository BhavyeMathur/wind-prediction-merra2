import sys

if sys.platform == "darwin":
    import matplotlib

    matplotlib.use("TkAgg")


from .metplot import *
from .text import *
