"""Bearing & Distance Calculation"""


import math

from .constants import EARTH_MEAN_RADIUS


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

    sigma = distance / (EARTH_MEAN_RADIUS * 1000)
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

    sigma = distance / (EARTH_MEAN_RADIUS * 1000)
    sin_s = math.sin(sigma)
    cos_s = math.cos(sigma)

    lat2 = math.asin(sin_lat * cos_s + cos_lat_times_cos_bearing * sin_s)

    return lat2, lon + math.atan2(cos_lat_times_sin_bearing * sin_s, cos_s - sin_lat * math.sin(lat2))
