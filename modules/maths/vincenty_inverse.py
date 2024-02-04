"""Vincenty's Inverse Formulae"""

import math


# Correction factor as used by DEFRA
# see https://www.eci.ox.ac.uk/research/energy/downloads/jardine09-carboninflights.pdf for comparisions
VINCENTY_CORRECTION_FACTOR = 1.1
VINCENTY_ITERATIONS = 200
VINCENTY_TOLERANCE = 10 ** -12  # accuracy to within 0.06 mm

EARTH_FLATTENING = 1 / 298.257223563  # flattening of the ellipsoid (WGS-84)
EARTH_EQUATORIAL_RADIUS = 6378.137  # radius at equator in kilometers (WGS-84)
POLAR_RADIUS: float = (1 - EARTH_FLATTENING) * EARTH_EQUATORIAL_RADIUS  # radius at poles in kilometers

FLATTENING_INVERSE: float = 1 - EARTH_FLATTENING
RADIUS_QUOTIENT: float = (EARTH_EQUATORIAL_RADIUS ** 2 - POLAR_RADIUS ** 2) / (POLAR_RADIUS ** 2)


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

        c = EARTH_FLATTENING / 16 * cos_sq_alpha * (EARTH_FLATTENING + 4) * (4 - 3 * cos_sq_alpha)

        lambda_prev = lambda_
        lambda_ = longitude_difference + (1 - c) * EARTH_FLATTENING * sin_alpha * (
                sigma + c * sin_sigma * (cos2_sigma_m + c * cos_sigma * (-1 + 2 * cos2_sigma_m ** 2)))

        if abs(lambda_prev - lambda_) < tol:  # successful convergence
            break

    u_sq = cos_sq_alpha * RADIUS_QUOTIENT
    b = (u_sq / 1024) * (256 + u_sq * (u_sq * (74 - 47 * u_sq) - 128))

    cos4_sigma_m = 2 * cos2_sigma_m ** 2
    delta_sig = b * sin_sigma * (cos2_sigma_m + 0.25 * b * (cos_sigma * (cos4_sigma_m - 1) - 0.16667 * b * cos2_sigma_m
                                                            * (4 * sin_sigma ** 2 - 3) * (2 * cos4_sigma_m - 3)))

    return POLAR_RADIUS * (1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))) * (sigma - delta_sig)
