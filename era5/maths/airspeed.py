"""Airspeed related functions"""

import math

from barometric import isa_pressure, PRESSURE_SEA_LEVEL, isa_density, DENSITY_SEA_LEVEL

SONIC_SPEED_SEA_LEVEL = 340.29  # m/s


def impact_pressure(speed: float) -> float:
    """Calculates the Impact Pressure, q_c, at the given speed

    @param speed: the speed in meters/second
    @return: the impact pressure in pascals
    """
    return PRESSURE_SEA_LEVEL * (((speed / SONIC_SPEED_SEA_LEVEL) ** 2 / 5 + 1) ** 3.5 - 1)


def mach_number(speed: float,
                altitude: float) -> float:
    """Calculates the Mach number, M, at the given speed and altitude

    @param speed: the speed in meters/second
    @param altitude: the altitude in meters (below 20000 m)
    @return: the mach number, or ratio of the speed to the speed of sound at that altitude
    """
    return math.sqrt(5 * ((impact_pressure(speed=speed) /
                           isa_pressure(altitude=altitude) + 1) ** 0.28571428571 - 1))


def expected_airspeed(calibrated_airspeed: float,
                      altitude: float) -> float:
    """Calculates the expected airspeed (EAS) from the calibrated airspeed (CAS)

    @param calibrated_airspeed: the calibrated airspeed (CAS) in meters/second
    @param altitude: the altitude of the aircraft in meters (below 20000 m)
    @return: the expected airspeed (EAS) in meters/second
    """
    return SONIC_SPEED_SEA_LEVEL * mach_number(speed=calibrated_airspeed, altitude=altitude) * math.sqrt(
        isa_pressure(altitude=altitude) / PRESSURE_SEA_LEVEL)


def true_airspeed_from_calibrated_airspeed(calibrated_airspeed: float,
                                           altitude: float) -> float:
    """Calculates the true airspeed (TAS) from the calibrated airspeed (CAS)

    @param calibrated_airspeed: the calibrated airspeed (CAS) in meters/second
    @param altitude: the altitude of the aircraft in meters (below 20000 m)
    @return: the true airspeed (TAS) in meters/second
    """
    return expected_airspeed(calibrated_airspeed=calibrated_airspeed, altitude=altitude) * math.sqrt(
        DENSITY_SEA_LEVEL / isa_density(altitude=altitude))


def calibrated_airspeed_from_true_airspeed(true_airspeed: float,
                                           altitude: float) -> float:
    """Calculates the true airspeed (TAS) from the calibrated airspeed (CAS)

    @param true_airspeed: the true airspeed (TAS) in meters/second
    @param altitude: the altitude of the aircraft in meters (below 20000 m)
    @return: the calibrated airspeed (CAS) in meters/second
    """
    pressure = isa_pressure(altitude=altitude)

    true_airspeed **= 2
    true_airspeed *= isa_density(altitude=altitude) / (7 * pressure)
    true_airspeed += 1
    true_airspeed **= 3.5
    true_airspeed -= 1
    true_airspeed *= pressure / PRESSURE_SEA_LEVEL
    true_airspeed += 1
    true_airspeed **= 2 / 7
    true_airspeed -= 1
    true_airspeed *= 7 * PRESSURE_SEA_LEVEL / DENSITY_SEA_LEVEL

    return true_airspeed ** 0.5


tas_from_cas = true_airspeed_from_calibrated_airspeed
cas_from_tas = calibrated_airspeed_from_true_airspeed


def mach_number_(speed: float,
                 atmospheric_pressure: float) -> float:
    """
    Calculates the Mach number, M, at the given speed and altitude

    @param speed: the speed in meters/second
    @param atmospheric_pressure: the pressure of the atmosphere around the aircraft (in Pascals)
    @return: the mach number, or ratio of the speed to the speed of sound at that altitude
    """
    return math.sqrt(5 * ((impact_pressure(speed=speed) / atmospheric_pressure + 1) ** 0.28571428571 - 1))


def expected_airspeed_(calibrated_airspeed: float,
                       atmospheric_pressure: float) -> float:
    """
    Calculates the expected airspeed (EAS) from the calibrated airspeed (CAS)

    @param calibrated_airspeed: the calibrated airspeed (CAS) in meters/second
    @param atmospheric_pressure: the pressure of the atmosphere around the aircraft (in Pascals)
    @return: the expected airspeed (EAS) in meters/second
    """
    return SONIC_SPEED_SEA_LEVEL * mach_number_(speed=calibrated_airspeed,
                                                atmospheric_pressure=atmospheric_pressure) * math.sqrt(
        atmospheric_pressure / PRESSURE_SEA_LEVEL)


def true_airspeed_(calibrated_airspeed: float,
                   air_density: float,
                   atmospheric_pressure: float) -> float:
    """Calculates the true airspeed (TAS) from the calibrated airspeed (CAS)

    @param calibrated_airspeed: the calibrated airspeed (CAS) in meters/second
    @param air_density: the density of the air around the aircraft (in kg/m3)
    @param atmospheric_pressure: the pressure of the atmosphere around the aircraft (in Pascals)
    @return: the true airspeed (TAS) in meters/second
    """
    return expected_airspeed_(calibrated_airspeed=calibrated_airspeed,
                              atmospheric_pressure=atmospheric_pressure) * math.sqrt(
        DENSITY_SEA_LEVEL / air_density)


def expected_airspeed_from_mach(mach: float,
                                altitude: float) -> float:
    """Calculates the expected airspeed (EAS) from the MACH number

    @param mach: the MACH number
    @param altitude: the altitude of the aircraft in meters (below 20000 m)
    @return: the expected airspeed (EAS) in meters/second
    """
    return SONIC_SPEED_SEA_LEVEL * mach * math.sqrt(isa_pressure(altitude=altitude) / PRESSURE_SEA_LEVEL)


def true_airspeed_from_mach(mach: float,
                            altitude: float) -> float:
    """Calculates the true airspeed (TAS) from the MACH number

    @param mach: the MACH number
    @param altitude: the altitude of the aircraft in meters (below 20000 m)
    @return: the true airspeed (TAS) in meters/second
    """
    return expected_airspeed_from_mach(mach=mach, altitude=altitude) * math.sqrt(
        DENSITY_SEA_LEVEL / isa_density(altitude=altitude))


def calibrated_airspeed_from_mach(mach: float,
                                  altitude: float) -> float:
    """Calculates the calibrated airspeed (CAS) from the MACH number

    @param mach: the MACH number
    @param altitude: the altitude of the aircraft in meters (below 20000 m)
    @return: the calibrated airspeed (CAS) in meters/second
    """
    return math.sqrt(5 * (((((mach ** 2) / 5 + 1) ** 3.5 - 1) * isa_pressure(altitude=altitude) /
                           PRESSURE_SEA_LEVEL + 1) ** 0.2857142857142857 - 1)) * SONIC_SPEED_SEA_LEVEL
