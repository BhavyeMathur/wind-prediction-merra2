"""
Contains all constants defined throughout the CFR Calculator
"""

from typing import Final
import math
import os
from pathlib import Path

"""CFR Constant Variables -------------------------------------------------------------------------------------------"""

# kgCO2e/kg Fuel = 3.157 from https://www.eci.ox.ac.uk/research/energy/downloads/jardine09-carboninflights.pdf
AVIATION_CONVERSION_FACTOR: Final[float] = 3.157

# RFI of 2 used as recommended by "Aviation and Climate Change: Best practice for calculation of the global warming
# potential", Niels Jungbluth, Christoph Meili
# also see Climate Forcing Metrics: https://mpimet.mpg.de/fileadmin/download/Grassl_Brockhagen.pdf
AVIATION_FORCING_MULTIPLIER: Final[float] = 2

# Correction factor as used by DEFRA
# see https://www.eci.ox.ac.uk/research/energy/downloads/jardine09-carboninflights.pdf for comparisions
VINCENTY_CORRECTION_FACTOR: Final[float] = 1.1
VINCENTY_ITERATIONS: Final[int] = 200
VINCENTY_TOLERANCE: Final[float] = 10 ** -12  # accuracy to within 0.06 mm

# Corrected from 6371 kilometers to provide more accurate results to Vincenty's Inverse
# MEAN_RADIUS: Final[float] = 6372.165
MEAN_RADIUS: Final[float] = 6371
EQUATORIAL_RADIUS: Final[float] = 6378.137  # radius at equator in kilometers (WGS-84)
FLATTENING: Final[float] = 1 / 298.257223563  # flattening of the ellipsoid (WGS-84)

UPPER_ATMOSPHERE: Final[float] = 10972  # meters, start of the Tropopause

PRESSURE_SEA_LEVEL: Final[float] = 101325  # Pascals
PRESSURE_11000_METERS: Final[float] = 22632.1  # Pascals
DENSITY_SEA_LEVEL: Final[float] = 1.225  # kg/m3
DENSITY_11000_METERS: Final[float] = 0.36392  # kg/m3

TEMPERATURE_SEA_LEVEL: Final[float] = 288.15  # Kelvin
TEMPERATURE_11000_METERS: Final[float] = 216.65  # Kelvin
TEMPERATURE_LAPSE_RATE_SEA_LEVEL: Final[float] = -0.0065  # K/m
TEMPERATURE_LAPSE_RATE_11000_METERS: Final[float] = 0.0  # K/m

UNIVERSAL_GAS_CONSTANT: Final[float] = 8.3144598  # J/(mol K)
GRAVITATIONAL_ACCELERATION: Final[float] = 9.80665  # m/s^2
AIR_MOLAR_MASS: Final[float] = 0.0289644  # kg/mol
IDEAL_GAS_LAW_COEFF: Final[float] = AIR_MOLAR_MASS / UNIVERSAL_GAS_CONSTANT

SONIC_SPEED_SEA_LEVEL: Final[float] = 340.29  # m/s

PRESSURE_POWER: Final[float] = -GRAVITATIONAL_ACCELERATION * AIR_MOLAR_MASS / \
                               (UNIVERSAL_GAS_CONSTANT * TEMPERATURE_LAPSE_RATE_SEA_LEVEL)
BAROMETRIC_SEA_LEVEL_COEFF: Final[float] = TEMPERATURE_LAPSE_RATE_SEA_LEVEL / TEMPERATURE_SEA_LEVEL

BAROMETRIC_EXP_COEFF: Final[float] = -GRAVITATIONAL_ACCELERATION * AIR_MOLAR_MASS / \
                                     (UNIVERSAL_GAS_CONSTANT * TEMPERATURE_SEA_LEVEL)
BAROMETRIC_BASE: Final[float] = math.exp(BAROMETRIC_EXP_COEFF)
PRESSURE_11000_METERS_ALTITUDE_OFFSET: Final[float] = 11000 + math.log(PRESSURE_11000_METERS, BAROMETRIC_BASE)
DENSITY_POWER: Final[float] = PRESSURE_POWER - 1
DENSITY_11000_METERS_ALTITUDE_OFFSET: Final[float] = 11000 + math.log(DENSITY_11000_METERS, BAROMETRIC_BASE)

TEMPERATURE_SD: Final[float] = 26.735
TEMPERATURE_MEAN: Final[float] = 235.745
PRESSURE_SD: Final[float] = 26545
PRESSURE_MEAN: Final[float] = 37590

# Lambert Conformal Conic Projection Principal Latitudes
LCCP_PHI1 = 0  # radians
LCCP_PHI2 = 1.4030661650965  # radians
