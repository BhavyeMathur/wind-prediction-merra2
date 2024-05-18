"""Atmospheric variables related functions"""


import math


_UPPER_ATMOSPHERE = 10972  # meters, start of the Tropopause

PRESSURE_SEA_LEVEL = 101325  # Pascals
PRESSURE_11000_METERS = 22632.1  # Pascals
DENSITY_SEA_LEVEL = 1.225  # kg/m3
DENSITY_11000_METERS = 0.36392  # kg/m3

TEMPERATURE_SEA_LEVEL = 288.15  # Kelvin
TEMPERATURE_11000_METERS = 216.65  # Kelvin
TEMPERATURE_LAPSE_RATE_SEA_LEVEL = -0.0065  # K/m
TEMPERATURE_LAPSE_RATE_11000_METERS = 0.0  # K/m

UNIVERSAL_GAS_CONSTANT = 8.3144598  # J/(mol K)
GRAVITATIONAL_ACCELERATION = 9.80665  # m/s^2
AIR_MOLAR_MASS = 0.0289644  # kg/mol
IDEAL_GAS_LAW_COEFF = AIR_MOLAR_MASS / UNIVERSAL_GAS_CONSTANT

PRESSURE_POWER = -GRAVITATIONAL_ACCELERATION * AIR_MOLAR_MASS / \
                               (UNIVERSAL_GAS_CONSTANT * TEMPERATURE_LAPSE_RATE_SEA_LEVEL)
BAROMETRIC_SEA_LEVEL_COEFF = TEMPERATURE_LAPSE_RATE_SEA_LEVEL / TEMPERATURE_SEA_LEVEL

BAROMETRIC_EXP_COEFF = -GRAVITATIONAL_ACCELERATION * AIR_MOLAR_MASS / \
                                     (UNIVERSAL_GAS_CONSTANT * TEMPERATURE_SEA_LEVEL)
BAROMETRIC_BASE = math.exp(BAROMETRIC_EXP_COEFF)
PRESSURE_11000_METERS_ALTITUDE_OFFSET = 11000 + math.log(PRESSURE_11000_METERS, BAROMETRIC_BASE)
DENSITY_POWER = PRESSURE_POWER - 1
DENSITY_11000_METERS_ALTITUDE_OFFSET = 11000 + math.log(DENSITY_11000_METERS, BAROMETRIC_BASE)


def isa_pressure(altitude: float) -> float:
    """
    Calculates the Atmospheric Pressure, P, at the Altitude given

    @param altitude: the altitude in meters (below 20000 m)
    @return: the atmospheric pressure, in Pascals, at the altitude
    """
    if altitude < 11000:
        return PRESSURE_SEA_LEVEL * (BAROMETRIC_SEA_LEVEL_COEFF * altitude + 1) ** PRESSURE_POWER
    else:
        return PRESSURE_11000_METERS * math.exp(BAROMETRIC_EXP_COEFF * (altitude - 11000))


def isa_temperature(altitude: float) -> float:
    """Calculates the Air Temperature, T, at the Altitude given

    @param altitude: the altitude in meters (below 20000 m)
    @return: the air temperature (K) at the altitude
    """
    if altitude < 11000:
        return TEMPERATURE_SEA_LEVEL + altitude * TEMPERATURE_LAPSE_RATE_SEA_LEVEL
    else:
        return TEMPERATURE_11000_METERS


def isa_temperature_celsius(altitude: float) -> float:
    """Calculates the Air Temperature, T, at the Altitude given

    @param altitude: the altitude in meters (below 20000 m)
    @return: the air temperature (°C) at the altitude
    """
    if altitude < 11000:
        return TEMPERATURE_SEA_LEVEL + altitude * TEMPERATURE_LAPSE_RATE_SEA_LEVEL - 273.15
    else:
        return TEMPERATURE_11000_METERS - 273.15


def isa_density(altitude: float) -> float:
    """Calculates the Atmospheric Density, ρ, at the Altitude given

    @param altitude: the altitude in meters (below 20000 m)
    @return: the atmospheric density, in kilograms/meter cubed, at the altitude
    """
    if altitude < 11000:
        return DENSITY_SEA_LEVEL * (BAROMETRIC_SEA_LEVEL_COEFF * altitude + 1) ** DENSITY_POWER
    else:
        return DENSITY_11000_METERS * math.exp(BAROMETRIC_EXP_COEFF * (altitude - 11000))


def height_from_pressure(pressure: float) -> float:
    """
    Calculates the inverse of the get_isa_pressure() function

    @param pressure: the pressure (in Pascals)
    @return: the altitude (in meters)
    """
    if pressure > PRESSURE_11000_METERS:
        return ((pressure / PRESSURE_SEA_LEVEL) ** (1 / PRESSURE_POWER) - 1) / BAROMETRIC_SEA_LEVEL_COEFF
    else:
        return 11000 + math.log(pressure / PRESSURE_11000_METERS) / BAROMETRIC_EXP_COEFF


def height_from_temperature(temperature: float) -> float:
    """
    Calculates the inverse of the get_isa_temperature() function

    @param temperature: the temperature (in Kelvin)
    @return: the altitude (in meters)
    """
    if temperature == TEMPERATURE_11000_METERS:
        return 11000
    else:
        return (temperature - TEMPERATURE_SEA_LEVEL) / TEMPERATURE_LAPSE_RATE_SEA_LEVEL


def density_from_ideal_gas_law(temperature: float,
                               pressure: float):
    """
    Calculates the density using the ideal gas law

    @param temperature: the temperature of the air (in Kelvin)
    @param pressure: the pressure of the air (in Pascals)
    @return:
    """
    return IDEAL_GAS_LAW_COEFF * pressure / temperature
