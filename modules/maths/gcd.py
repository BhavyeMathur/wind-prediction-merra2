"""GCD - Great Circle Distance Formulae"""


import math

from .constants import EARTH_MEAN_RADIUS


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
    return EARTH_MEAN_RADIUS * math.acos(math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) *
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

    return EARTH_MEAN_RADIUS * math.acos(sin_lat1 * math.sin(lat2) + cos_lat1 * math.cos(lat2) *
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

    return EARTH_MEAN_RADIUS * math.atan(math.hypot(cos_lat2 * math.sin(lat_difference),
                                                    cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_lat_difference)
                                         / (sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_lat_difference))
