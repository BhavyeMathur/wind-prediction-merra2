import sys
import matplotlib as mpl

if sys.platform == "darwin":
    mpl.use("TkAgg")

mpl.rcParams["figure.dpi"] = 150

from .contour2D import *
