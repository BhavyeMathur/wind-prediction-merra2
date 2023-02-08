"""
This submodule contains important mathematical function used in the flight calculations, primarily distance
calculation functions.
"""

import math
from typing import Callable

import constants as const

POLAR_RADIUS: float = (1 - const.FLATTENING) * const.EQUATORIAL_RADIUS  # radius at poles in kilometers
FLATTENING_INVERSE: float = 1 - const.FLATTENING
RADIUS_QUOTIENT: float = (const.EQUATORIAL_RADIUS ** 2 - POLAR_RADIUS ** 2) / (POLAR_RADIUS ** 2)

PI_BY_4: float = math.pi / 4
PI_BY_2: float = math.pi / 2

"""Bearing & Distance Calculation"""


def bearing_degrees(lat1: float,
                    lon1: float,
                    lat2: float,
                    lon2: float) -> float:
    """
    Calculates the Bearing between 2 coordinates

    @param lat1: 1st latitude in degrees
    @param lon1: 1st longitude in degrees
    @param lat2: 2nd latitude in degrees
    @param lon2: 2nd longitude in degrees
    @return: the bearing between the 2 coordinates in degrees
    """

    lat1 = math.radians(lat1)  # all coordinates must be converted to radians for the trigonometric functions to work
    lat2 = math.radians(lat2)
    dlon = math.radians(abs(lon2 - lon1))

    return math.degrees(math.atan2(math.cos(lat2) * math.sin(dlon),
                                   math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)))


def get_bearing(lat1: float,
                lon1: float,
                lat2: float,
                lon2: float) -> float:
    """
    Calculates the Bearing between 2 coordinates

    @param lat1: 1st latitude in radians
    @param lon1: 1st longitude in radians
    @param lat2: 2nd latitude in radians
    @param lon2: 2nd longitude in radians
    @return: the bearing between the 2 coordinates in radians
    """
    dlon = abs(lon2 - lon1)

    return math.atan2(math.cos(lat2) * math.sin(dlon),
                      math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))


def coordinates_from_bearing(bearing: float,
                             distance: float,
                             lat: float,
                             lon: float) -> tuple[float, float]:
    """
    Calculates the coordinate some distance away from another using a bearing

    @param bearing: the bearing (in radians) of the path
    @param distance: the length (in meters) of the path
    @param lat: latitude (in radians) to calculate coordinate from
    @param lon: longitude (in radians) to calculate coordinate from
    @return: the 2nd coordinate, in radians
    """

    sigma = distance / (const.MEAN_RADIUS * 1000)
    sin_s = math.sin(sigma)
    cos_s = math.cos(sigma)
    sin_l = math.sin(lat)
    cos_l = math.cos(lat)

    lat2 = math.asin(sin_l * cos_s + cos_l * sin_s * math.cos(bearing))

    return lat2, lon + math.atan2(math.sin(bearing) * sin_s * cos_l, cos_s - sin_l * math.sin(lat2))


def coordinates_from_bearing_degrees(bearing: float,
                                     distance: float,
                                     lat: float,
                                     lon: float) -> tuple[float, float]:
    """
    Calculates the coordinate some distance away from another using a bearing

    @param bearing: the bearing (in radians) of the path
    @param distance: the length (in meters) of the path
    @param lat: latitude (in radians) to calculate coordinate from
    @param lon: longitude (in radians) to calculate coordinate from
    @return: the 2nd coordinate, in radians
    """

    lat, lon = coordinates_from_bearing(bearing=math.radians(bearing),
                                        distance=distance,
                                        lat=math.radians(lat),
                                        lon=math.radians(lon))

    return math.degrees(lat), math.degrees(lon)


def coordinates_from_bearing_(distance: float,
                              lon: float,
                              sin_lat: float,
                              cos_lat_times_cos_bearing: float,
                              cos_lat_times_sin_bearing: float) -> tuple[float, float]:
    """
    Calculates the coordinate some distance away from another using a bearing

    @param distance: the length (in meters) of the path
    @param lon: longitude (in radians) to calculate coordinate from
    @param sin_lat: the sine of the latitude
    @param cos_lat_times_cos_bearing: cosine of the latitude times the cosine of the bearing
    @param cos_lat_times_sin_bearing: cosine of the latitude times the sine of the bearing

    @return: the 2nd coordinate, in radians
    """

    sigma = distance / (const.MEAN_RADIUS * 1000)
    sin_s = math.sin(sigma)
    cos_s = math.cos(sigma)

    lat2 = math.asin(sin_lat * cos_s + cos_lat_times_cos_bearing * sin_s)

    return lat2, lon + math.atan2(cos_lat_times_sin_bearing * sin_s, cos_s - sin_lat * math.sin(lat2))


"""GCD - Great Circle Distance Formulae"""


# Finds the GCD between 2 coordinates given in degrees
def great_circle_distance_degrees(lat1: float,
                                  lon1: float,
                                  lat2: float,
                                  lon2: float) -> float:
    """Finds the length of the shortest path, in kilometers, along the surface of a sphere between a pair of coordinates

    @param lat1: latitude of coordinate 1 (in degrees)
    @param lon1: longitude of coordinate 1 (in degrees)
    @param lat2: latitude of coordinate 2 (in degrees)
    @param lon2: longitude of coordinate 2 (in degrees)

    @return: the distance in kilometers between the pair of coordinates
    """
    return great_circle_distance(lat1=math.radians(lat1),
                                 lon1=math.radians(lon1),
                                 lat2=math.radians(lat2),
                                 lon2=math.radians(lon2))


# Finds the GCD between 2 coordinates given in radians
def great_circle_distance(lat1: float,
                          lon1: float,
                          lat2: float,
                          lon2: float) -> float:
    """Finds the length of the shortest path, in kilometers, along the surface of a sphere between a pair of coordinates

    @param lat1: latitude of coordinate 1 (in radians)
    @param lon1: longitude of coordinate 1 (in radians)
    @param lat2: latitude of coordinate 2 (in radians)
    @param lon2: longitude of coordinate 2 (in radians)

    @return: the distance in kilometers between the pair of coordinates
    """
    return const.MEAN_RADIUS * math.acos(math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) *
                                         math.cos(abs(lon1 - lon2)))


def great_circle_distance_(sin_lat1: float,
                           cos_lat1: float,
                           lon1: float,
                           lat2: float,
                           lon2: float) -> float:
    """Finds the length of the shortest path, in kilometers, along the surface of a sphere between a pair of coordinates

    @param sin_lat1: sine of the latitude of coordinate 1
    @param cos_lat1: cosine of the latitude of coordinate 1
    @param lon1: longitude of coordinate 1 (in radians)
    @param lat2: latitude of coordinate 2 (in radians)
    @param lon2: longitude of coordinate 2 (in radians)

    @return: the distance in kilometers between the pair of coordinates
    """

    return const.MEAN_RADIUS * math.acos(sin_lat1 * math.sin(lat2) + cos_lat1 * math.cos(lat2) *
                                         math.cos(math.radians(abs(lon1 - lon2))))


def haversine_great_circle_distance(lat1: float,
                                    lon1: float,
                                    lat2: float,
                                    lon2: float) -> float:
    """Finds the length of the shortest path, in kilometers, along the surface of a sphere between a pair of coordinates

    Uses the Haversine formula to calculate the GCD

    @param lat1: latitude of coordinate 1 (in degrees)
    @param lon1: longitude of coordinate 1 (in degrees)
    @param lat2: latitude of coordinate 2 (in degrees)
    @param lon2: longitude of coordinate 2 (in degrees)

    @return: the distance in kilometers between the pair of coordinates
    """
    lat1 = math.radians(lat1)  # all coordinates must be converted to radians for the trigonometric functions to work
    lat2 = math.radians(lat2)

    return 12742 * math.asin(math.sqrt(math.sin(abs(lat1 - lat2) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) *
                                       math.sin(math.radians(abs(lon1 - lon2) / 2)) ** 2))


# Finds the GCD between 2 coordinates given in degrees
def vincenty_great_circle_distance(lat1: float,
                                   lon1: float,
                                   lat2: float,
                                   lon2: float) -> float:
    """Finds the length of the shortest path, in kilometers, along the surface of a sphere between a pair of coordinates

    Uses a special case of Vincenty's Inverse to calculate the GCD

    @param lat1: latitude of coordinate 1 (in degrees)
    @param lon1: longitude of coordinate 1 (in degrees)
    @param lat2: latitude of coordinate 2 (in degrees)
    @param lon2: longitude of coordinate 2 (in degrees)

    @return: the distance in kilometers between the pair of coordinates
    """
    lat1 = math.radians(lat1)  # all coordinates must be converted to radians for the trigonometric functions to work
    lat2 = math.radians(lat2)

    sin_lat1 = math.sin(lat1)  # pre-calculating these trigonometric functions because they are used multiple times
    cos_lat1 = math.cos(lat1)
    sin_lat2 = math.sin(lat2)
    cos_lat2 = math.cos(lat2)

    lat_difference = math.radians(lon1 - lon2)  # the difference in the latitudes
    cos_lat_difference = math.cos(lat_difference)

    return const.MEAN_RADIUS * math.atan(math.hypot(cos_lat2 * math.sin(lat_difference),
                                                    cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_lat_difference)
                                         / (sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_lat_difference))


"""Vincenty's Inverse Formulae"""


# Finds the distance between 2 coordinates given in degrees on an oblate sphere
def vincenty_inverse(lat1: float,
                     lon1: float,
                     lat2: float,
                     lon2: float,
                     iterations: int = 200,
                     tol: float = 10 ** -12) -> float:
    """Finds the length of the shortest path, in kilometers, along the surface of an oblate sphere between a pair of
    coordinates

    @param lat1: latitude of coordinate 1 (in degrees)
    @param lon1: longitude of coordinate 1 (in degrees)
    @param lat2: latitude of coordinate 2 (in degrees)
    @param lon2: longitude of coordinate 2 (in degrees)

    @param iterations: the number of times to iterate the function
    @param tol: the tolerance level at which to stop iteration

    @return: the distance in kilometers between the pair of coordinates
    """
    aux_lat1 = math.atan(FLATTENING_INVERSE * math.tan(math.radians(lat1)))
    aux_lat2 = math.atan(FLATTENING_INVERSE * math.tan(math.radians(lat2)))
    sin_aux_lat1 = math.sin(aux_lat1)
    cos_aux_lat1 = math.cos(aux_lat1)

    longitude_difference = math.radians(lon2 - lon1)

    return __vincenty_inverse(longitude_difference, aux_lat2, sin_aux_lat1, cos_aux_lat1, iterations, tol)


# Finds the distance between a coordinate and pre-calculated values of the other coordinate on an oblate sphere
def vincenty_inverse_(sin_aux_lat1: float,
                      cos_aux_lat1: float,
                      lon1: float,
                      lat2: float,
                      lon2: float,
                      iterations: int = 200,
                      tol: float = 10 ** -12) -> float:
    """Finds the length of the shortest path, in kilometers, along the surface of an oblate sphere between a pair of
    coordinates

    For internal use

    @param sin_aux_lat1: the sine of the auxiliary latitude calculated from coordinate 1
    @param cos_aux_lat1: the cosine of the auxiliary latitude calculated from coordinate 1
    @param lon1: longitude of coordinate 1 (in radians)

    @param lat2: latitude of coordinate 2 (in radians)
    @param lon2: longitude of coordinate 2 (in radians)

    @param iterations: the number of times to iterate the function
    @param tol: the tolerance level at which to stop iteration

    @return: the distance in kilometers between the pair of coordinates
    """
    aux_lat2 = math.atan(FLATTENING_INVERSE * math.tan(lat2))
    longitude_difference = lon2 - lon1

    return __vincenty_inverse(longitude_difference, aux_lat2, sin_aux_lat1, cos_aux_lat1, iterations, tol)


def __vincenty_inverse(longitude_difference, aux_lat2, sin_aux_lat1, cos_aux_lat1, iterations, tol):
    lambda_ = longitude_difference  # set initial value of lambda to L

    sin_aux_lat2 = math.sin(aux_lat2)
    cos_aux_lat2 = math.cos(aux_lat2)

    sin_aux_lat1_times_sin_aux_lat2 = sin_aux_lat1 * sin_aux_lat2
    cos_aux_lat1_times_cos_aux_lat2 = cos_aux_lat1 * cos_aux_lat2
    cos_aux_lat1_times_sin_aux_lat2 = cos_aux_lat1 * sin_aux_lat2
    cos_aux_lat2_times_sin_aux_lat1 = cos_aux_lat2 * sin_aux_lat1

    sigma = 0
    cos_sq_alpha = 0
    cos2_sigma_m = 0
    sin_sigma = 0
    cos_sigma = 0

    for i in range(iterations):
        sin_lambda = math.sin(lambda_)
        cos_lambda = math.cos(lambda_)

        sin_sigma = math.hypot(sin_lambda * cos_aux_lat2,
                               cos_aux_lat1_times_sin_aux_lat2 - cos_aux_lat2_times_sin_aux_lat1 * cos_lambda)
        cos_sigma = sin_aux_lat1_times_sin_aux_lat2 + cos_aux_lat1_times_cos_aux_lat2 * cos_lambda

        sigma = math.atan2(sin_sigma, cos_sigma)

        sin_alpha = (cos_aux_lat1_times_cos_aux_lat2 * sin_lambda) / sin_sigma
        cos_sq_alpha = 1 - sin_alpha ** 2

        cos2_sigma_m = cos_sigma - ((2 * sin_aux_lat1_times_sin_aux_lat2) / cos_sq_alpha)

        c = const.FLATTENING / 16 * cos_sq_alpha * (const.FLATTENING + 4) * (4 - 3 * cos_sq_alpha)

        lambda_prev = lambda_
        lambda_ = longitude_difference + (1 - c) * const.FLATTENING * sin_alpha * (
                sigma + c * sin_sigma * (cos2_sigma_m + c * cos_sigma * (-1 + 2 * cos2_sigma_m ** 2)))

        if abs(lambda_prev - lambda_) < tol:  # successful convergence
            break

    u_sq = cos_sq_alpha * RADIUS_QUOTIENT
    b = (u_sq / 1024) * (256 + u_sq * (u_sq * (74 - 47 * u_sq) - 128))

    cos4_sigma_m = 2 * cos2_sigma_m ** 2
    delta_sig = b * sin_sigma * (cos2_sigma_m + 0.25 * b * (cos_sigma * (cos4_sigma_m - 1) - 0.16667 * b * cos2_sigma_m
                                                            * (4 * sin_sigma ** 2 - 3) * (2 * cos4_sigma_m - 3)))

    return POLAR_RADIUS * (1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))) * (sigma - delta_sig)


# Returns a closure function for use in testing
def get_vincenty_inverse(iterations: int = 200,
                         tol: float = 10 ** -12) \
        -> Callable[[float, float, float, float, int, float], float]:
    """Returns a Vincenty's Inverse closure function for use in testing

    @param iterations: the number of iterations for the closure function to have
    @param tol: the tolerance for the closure function to have
    @return: a closure function that calculates the Vincenty's Inverse between a pair of coordinates
    """

    def vincenty_inverse_closure(lat1, lon1, lat2, lon2) -> float:
        """
        A closure function around vincenty_inverse()
        """
        return vincenty_inverse(lat1, lon1, lat2, lon2, iterations=iterations, tol=tol)

    return vincenty_inverse_closure


# Atmospheric variables related functions


def isa_pressure(altitude: float) -> float:
    """
    Calculates the Atmospheric Pressure, P, at the Altitude given

    @param altitude: the altitude in meters (below 20000 m)
    @return: the atmospheric pressure, in Pascals, at the altitude
    """
    if altitude < 11000:
        return const.PRESSURE_SEA_LEVEL * (const.BAROMETRIC_SEA_LEVEL_COEFF * altitude + 1) ** const.PRESSURE_POWER
    else:
        return const.PRESSURE_11000_METERS * math.exp(const.BAROMETRIC_EXP_COEFF * (altitude - 11000))


def isa_temperature(altitude: float) -> float:
    """Calculates the Air Temperature, T, at the Altitude given

    @param altitude: the altitude in meters (below 20000 m)
    @return: the air temperature (K) at the altitude
    """
    if altitude < 11000:
        return const.TEMPERATURE_SEA_LEVEL + altitude * const.TEMPERATURE_LAPSE_RATE_SEA_LEVEL
    else:
        return const.TEMPERATURE_11000_METERS


def isa_temperature_celsius(altitude: float) -> float:
    """Calculates the Air Temperature, T, at the Altitude given

    @param altitude: the altitude in meters (below 20000 m)
    @return: the air temperature (°C) at the altitude
    """
    if altitude < 11000:
        return const.TEMPERATURE_SEA_LEVEL + altitude * const.TEMPERATURE_LAPSE_RATE_SEA_LEVEL - 273.15
    else:
        return const.TEMPERATURE_11000_METERS - 273.15


def isa_density(altitude: float) -> float:
    """Calculates the Atmospheric Density, ρ, at the Altitude given

    @param altitude: the altitude in meters (below 20000 m)
    @return: the atmospheric density, in kilograms/meter cubed, at the altitude
    """
    if altitude < 11000:
        return const.DENSITY_SEA_LEVEL * (const.BAROMETRIC_SEA_LEVEL_COEFF * altitude + 1) ** const.DENSITY_POWER
    else:
        return const.DENSITY_11000_METERS * math.exp(const.BAROMETRIC_EXP_COEFF * (altitude - 11000))


def height_from_pressure(pressure: float) -> float:
    """
    Calculates the inverse of the get_isa_pressure() function

    @param pressure: the pressure (in Pascals)
    @return: the altitude (in meters)
    """
    if pressure > const.PRESSURE_11000_METERS:
        return ((pressure / const.PRESSURE_SEA_LEVEL) ** (1 / const.PRESSURE_POWER) - 1) / const.BAROMETRIC_SEA_LEVEL_COEFF
    else:
        return 11000 + math.log(pressure / const.PRESSURE_11000_METERS) / const.BAROMETRIC_EXP_COEFF


def height_from_temperature(temperature: float) -> float:
    """
    Calculates the inverse of the get_isa_temperature() function

    @param temperature: the temperature (in Kelvin)
    @return: the altitude (in meters)
    """
    if temperature == const.TEMPERATURE_11000_METERS:
        return 11000
    else:
        return (temperature - const.TEMPERATURE_SEA_LEVEL) / const.TEMPERATURE_LAPSE_RATE_SEA_LEVEL


def density_from_ideal_gas_law(temperature: float,
                               pressure: float):
    """
    Calculates the density using the ideal gas law

    @param temperature: the temperature of the air (in Kelvin)
    @param pressure: the pressure of the air (in Pascals)
    @return:
    """
    return const.IDEAL_GAS_LAW_COEFF * pressure / temperature


# Speed related functions


def impact_pressure(speed: float) -> float:
    """Calculates the Impact Pressure, q_c, at the given speed

    @param speed: the speed in meters/second
    @return: the impact pressure in pascals
    """
    return const.PRESSURE_SEA_LEVEL * (((speed / const.SONIC_SPEED_SEA_LEVEL) ** 2 / 5 + 1) ** 3.5 - 1)


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
    return const.SONIC_SPEED_SEA_LEVEL * mach_number(speed=calibrated_airspeed, altitude=altitude) * math.sqrt(
        isa_pressure(altitude=altitude) / const.PRESSURE_SEA_LEVEL)


def true_airspeed_from_calibrated_airspeed(calibrated_airspeed: float,
                                           altitude: float) -> float:
    """Calculates the true airspeed (TAS) from the calibrated airspeed (CAS)

    @param calibrated_airspeed: the calibrated airspeed (CAS) in meters/second
    @param altitude: the altitude of the aircraft in meters (below 20000 m)
    @return: the true airspeed (TAS) in meters/second
    """
    return expected_airspeed(calibrated_airspeed=calibrated_airspeed, altitude=altitude) * math.sqrt(
        const.DENSITY_SEA_LEVEL / isa_density(altitude=altitude))


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
    true_airspeed *= pressure / const.PRESSURE_SEA_LEVEL
    true_airspeed += 1
    true_airspeed **= 2 / 7
    true_airspeed -= 1
    true_airspeed *= 7 * const.PRESSURE_SEA_LEVEL / const.DENSITY_SEA_LEVEL

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
    return const.SONIC_SPEED_SEA_LEVEL * mach_number_(speed=calibrated_airspeed,
                                                      atmospheric_pressure=atmospheric_pressure) * math.sqrt(
        atmospheric_pressure / const.PRESSURE_SEA_LEVEL)


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
        const.DENSITY_SEA_LEVEL / air_density)


def expected_airspeed_from_mach(mach: float,
                                altitude: float) -> float:
    """Calculates the expected airspeed (EAS) from the MACH number

    @param mach: the MACH number
    @param altitude: the altitude of the aircraft in meters (below 20000 m)
    @return: the expected airspeed (EAS) in meters/second
    """
    return const.SONIC_SPEED_SEA_LEVEL * mach * math.sqrt(isa_pressure(altitude=altitude) / const.PRESSURE_SEA_LEVEL)


def true_airspeed_from_mach(mach: float,
                            altitude: float) -> float:
    """Calculates the true airspeed (TAS) from the MACH number

    @param mach: the MACH number
    @param altitude: the altitude of the aircraft in meters (below 20000 m)
    @return: the true airspeed (TAS) in meters/second
    """
    return expected_airspeed_from_mach(mach=mach, altitude=altitude) * math.sqrt(
        const.DENSITY_SEA_LEVEL / isa_density(altitude=altitude))


def calibrated_airspeed_from_mach(mach: float,
                                  altitude: float) -> float:
    """Calculates the calibrated airspeed (CAS) from the MACH number

    @param mach: the MACH number
    @param altitude: the altitude of the aircraft in meters (below 20000 m)
    @return: the calibrated airspeed (CAS) in meters/second
    """
    return math.sqrt(5 * (((((mach ** 2) / 5 + 1) ** 3.5 - 1) * isa_pressure(altitude=altitude) /
                           const.PRESSURE_SEA_LEVEL + 1) ** 0.2857142857142857 - 1)) * const.SONIC_SPEED_SEA_LEVEL


"""Lambert Conformal Conic Projection Formulae"""


def lcc_projection_n(phi1: float,
                     phi2: float) -> float:
    """Returns the n-value for the Lambert Conformal Conic Projection

    @param phi1: principal latitude 1 (radians)
    @param phi2: principal latitude 2 (radians)
    @return: the n-value
    """
    return math.log(math.cos(phi1) / math.cos(phi2)) / math.log(
        math.tan(PI_BY_4 + phi2 / 2) / math.tan(PI_BY_4 + phi1 / 2))


def __lcc_projection_n(phi2: float,
                       cos_phi1: float,
                       tan_pi_by_4_plus_phi1_by_2: float) -> float:
    """Returns the n-value for the Lambert Conformal Conic Projection

    @param phi2: principal latitude 2 (radians)
    @param cos_phi1: cosine of phi1
    @param tan_pi_by_4_plus_phi1_by_2: tangent of (pi/4 + phi1/2)
    @return: the n-value
    """
    return math.log(cos_phi1 / math.cos(phi2)) / math.log(math.tan(PI_BY_4 + phi2 / 2) / tan_pi_by_4_plus_phi1_by_2)


def __lcc_projection_nf(phi1: float,
                        phi2: float) -> tuple[float, float]:
    """Returns the n- and F-values for the Lambert Conformal Conic Projection

    @param phi1: principal latitude 1 (radians)
    @param phi2: principal latitude 2 (radians)
    @return: a tuple with the n- and F-values
    """
    cos_phi1 = math.cos(phi1)
    tan_phi1 = math.tan(PI_BY_4 + phi1 / 2)
    n = __lcc_projection_n(phi2=phi2, cos_phi1=cos_phi1, tan_pi_by_4_plus_phi1_by_2=tan_phi1)
    return n, cos_phi1 * tan_phi1 ** n / n


def lcc_projection_f(phi1: float,
                     phi2: float) -> float:
    """Returns the F-value for the Lambert Conformal Conic Projection

    @param phi1: principal latitude 1 (radians)
    @param phi2: principal latitude 2 (radians)
    @return: the F-value
    """
    _, f = __lcc_projection_nf(phi1=phi1, phi2=phi2)
    return f


def lcc_projection_rho0(phi1: float,
                        phi2: float) -> float:
    """Returns the rho0-value for the Lambert Conformal Conic Projection

    @param phi1: principal latitude 1 (radians)
    @param phi2: principal latitude 2 (radians)
    @return: the rho0-value
    """
    return lcc_projection_f(phi1=phi1, phi2=phi2)


def __lcc_projection_n_f_rho(phi: float,
                             phi1: float,
                             phi2: float) -> tuple[float, float, float]:
    """Returns the rho-value for a latitude in the Lambert Conformal Conic Projection

    @param phi: latitude of coordinate (radians)
    @param phi1: principal latitude 1 (radians)
    @param phi2: principal latitude 2 (radians)
    @return: the n, f, and rho values for a latitude
    """
    n, f = __lcc_projection_nf(phi1=phi1, phi2=phi2)
    return n, f, f * math.tan(PI_BY_4 + phi / 2) ** -n


def lcc_projection_rho(phi: float,
                       phi1: float,
                       phi2: float) -> float:
    """Returns the rho-value for a latitude in the Lambert Conformal Conic Projection

    @param phi: latitude of coordinate (radians)
    @param phi1: principal latitude 1 (radians)
    @param phi2: principal latitude 2 (radians)
    @return: the rho-value for a latitude
    """
    _, _, rho = __lcc_projection_n_f_rho(phi=phi, phi1=phi1, phi2=phi2)
    return rho


def lcc_projection_x(latitude: float,
                     longitude: float,
                     phi1: float,
                     phi2: float) -> float:
    """Returns the x-value for a coordinate in the Lambert Conformal Conic Projection

    @param latitude: latitude of coordinate (radians)
    @param longitude: longitude of coordinate (radians)
    @param phi1: principal latitude 1 (radians)
    @param phi2: principal latitude 2 (radians)
    @return: the x-value for a coordinate
    """
    n, _, rho = __lcc_projection_n_f_rho(phi=latitude, phi1=phi1, phi2=phi2)
    return rho * math.sin(n * longitude)


def lcc_projection_y(latitude: float,
                     longitude: float,
                     phi1: float,
                     phi2: float) -> float:
    """Returns the y-value for a coordinate in the Lambert Conformal Conic Projection

    @param latitude: latitude of coordinate (radians)
    @param longitude: longitude of coordinate (radians)
    @param phi1: principal latitude 1 (radians)
    @param phi2: principal latitude 2 (radians)
    @return: the y-value for a coordinate
    """
    n, f, rho = __lcc_projection_n_f_rho(phi=latitude, phi1=phi1, phi2=phi2)
    return f - rho * math.cos(n * longitude)


def lcc_projection(latitude: float,
                   longitude: float,
                   phi1: float,
                   phi2: float) -> tuple[float, float]:
    """Returns the x,y value for a coordinate in the Lambert Conformal Conic Projection

    @param latitude: latitude of coordinate (radians)
    @param longitude: longitude of coordinate (radians)
    @param phi1: principal latitude 1 (radians)
    @param phi2: principal latitude 2 (radians)
    @return: the x, y value for a coordinate
    """
    n, f, rho = __lcc_projection_n_f_rho(phi=latitude, phi1=phi1, phi2=phi2)
    longitude *= n
    return rho * math.sin(longitude), f - rho * math.cos(longitude)


def lcc_projection_to_latitude(x: float,
                               y: float,
                               phi1: float,
                               phi2: float) -> float:
    """Returns the latitude for an x, y value in the Lambert Conformal Conic Projection

    @param x: x value in projection
    @param y: y value in projection
    @param phi1: principal latitude 1 (radians)
    @param phi2: principal latitude 2 (radians)
    @return: the latitude of the coordinate (radians)
    """
    n, f = __lcc_projection_nf(phi1=phi1, phi2=phi2)
    return PI_BY_2 - 2 * math.atan(((x ** 2 + (f - y) ** 2) ** 0.5 / f) ** (1 / n))


def lcc_projection_to_longitude(x: float,
                                y: float,
                                phi1: float,
                                phi2: float) -> float:
    """Returns the longitude for an x, y value in the Lambert Conformal Conic Projection

    @param x: x value in projection
    @param y: y value in projection
    @param phi1: principal latitude 1 (radians)
    @param phi2: principal latitude 2 (radians)
    @return: the longitude of the coordinate (radians)
    """
    n = lcc_projection_n(phi1=phi1, phi2=phi2)
    if y > 1:
        if x < 0:
            return math.atan(x / (1 - y)) / n - math.pi / n
        else:
            return math.atan(x / (1 - y)) / n + math.pi / n
    else:
        return math.atan(x / (1 - y)) / n


def lcc_projection_to_coord(x: float,
                            y: float,
                            phi1: float,
                            phi2: float) -> tuple[float, float]:
    """Returns the coordinate for an x, y value in the Lambert Conformal Conic Projection

    @param x: x value in projection
    @param y: y value in projection
    @param phi1: principal latitude 1 (radians)
    @param phi2: principal latitude 2 (radians)
    @return: the latitude, longitude of the coordinate (radians)
    """

    n, f = __lcc_projection_nf(phi1=phi1, phi2=phi2)

    lat = PI_BY_2 - 2 * math.atan((((f * x) ** 2 + (f - f * y) ** 2) ** 0.5 / f) ** (1 / n))

    if y > 1:
        if x < 0:
            return lat, math.atan(x / (1 - y)) / n - math.pi / n
        else:
            return lat, math.atan(x / (1 - y)) / n + math.pi / n
    else:
        return lat, math.atan(x / (1 - y)) / n


"""Error Functions"""


def me(data, prediction):
    return (data - prediction).mean()


def mae(data, prediction):
    return abs(data - prediction).mean()


def mape(data, prediction):
    ((data - prediction) / data).mean()


def mse(data, prediction):
    return ((data - prediction) ** 2).mean()


def rmse(data, prediction):
    return math.sqrt(mse(data, prediction))


"""Interpolation Functions"""


def linear_interpolate(values, index: int, t: float):
    return values[index] * (1 - t) + values[index + 1] * t


def cosine_interpolate(values, index: int, t: float):
    return linear_interpolate(values, index, (1 - math.cos(t * math.pi)) / 2)


def cubic_interpolate(values, index: int, t: float):
    p0 = values[max(0, index - 1)]
    p1 = values[index]
    p2 = values[index + 1]
    p3 = values[min(len(values) - 1, index + 2)]

    t2 = t * t
    a0 = p3 - p2 - p0 + p1
    a1 = p0 - p1 - a0
    a2 = p2 - p0
    a3 = p1

    return a0 * t * t2 + a1 * t2 + a2 * t + a3


def catmull_rom_interpolate(values, index: int, t: float):
    p0 = values[max(0, index - 1)]
    p1 = values[index]
    p2 = values[index + 1]
    p3 = values[min(len(values) - 1, index + 2)]
    
    t2 = t * t

    a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
    a1 = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3
    a2 = -0.5 * p0 + 0.5 * p2
    a3 = p1

    return a0 * t * t2 + a1 * t2 + a2 * t + a3


def hermite_interpolate(values, index: int, t: float, tension: float = 0, bias: float = 0):
    p0 = values[max(0, index - 1)]
    p1 = values[index]
    p2 = values[index + 1]
    p3 = values[min(len(values) - 1, index + 2)]
    
    t2 = t * t
    t3 = t2 * t

    p2_minus_p1 = p2 - p1

    m0 = (p1 - p0) * (1 + bias)
    m0 += p2_minus_p1 * (1 - bias)

    m1 = p2_minus_p1 * (1 + bias)
    m1 += (p3 - p2) * (1 - bias)

    a0 = 2 * t3 - 3 * t2 + 1
    a1 = t3 - 2 * t2 + t
    a2 = t3 - t2
    a3 = -2 * t3 + 3 * t2

    return a0 * p1 + (1 - tension) * (a1 * m0 + a2 * m1) / 2 + a3 * p2


def get_hermite_interpolate(tension: float = 0, bias: float = 0):
    def _interpolate(*args):
        return hermite_interpolate(*args, tension=tension, bias=bias)

    return _interpolate


def fit_kochanek_bartels_spline(data, values, index: int, t: float, learning_rate: float = 0.5, iterations: int = 15):
    p0 = values[max(0, index - 1)]
    p1 = values[index]
    p2 = values[index + 1]
    p3 = values[min(len(values) - 1, index + 2)]

    t2 = t * t
    t3 = t2 * t

    best_loss = float("inf")
    best_tension = 0
    best_bias = 0

    a0 = 2 * t3 - 3 * t2 + 1
    a1 = (t3 - 2 * t2 + t) / 2
    a2 = (t3 - t2) / 2
    a3 = -2 * t3 + 3 * t2

    const = a0 * p1 + a3 * p2
    baseline = data - const

    m00 = p1 - p0
    m01 = p2 - p1
    m10 = m01
    m11 = p3 - p2

    bias = 0
    tension = 0

    for _ in range(iterations):
        m0 = m00 * (1 + bias) + m01 * (1 - bias)
        m1 = m10 * (1 + bias) + m11 * (1 - bias)

        if (loss := ((baseline - (a1 * m0 + a2 * m1) * (1 - tension)) ** 2).sum()) < best_loss:
            best_loss = loss
            best_tension = tension
        else:
            learning_rate /= 1.5

        d_outer = 2 * (baseline - (a1 * m0 + a2 * m1) * (1 - tension))
        dt = (d_outer * (a1 * m0 + a2 * m1)).mean()
        tension -= learning_rate * dt

    return best_tension, best_bias
