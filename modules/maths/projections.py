"""Lambert Conformal Conic Projection Formulae"""


import math

_PI_BY_4 = math.pi / 4
_PI_BY_2 = math.pi / 2


def lcc_projection_n(phi1: float,
                     phi2: float) -> float:
    """Returns the n-value for the Lambert Conformal Conic Projection

    @param phi1: principal latitude 1 (radians)
    @param phi2: principal latitude 2 (radians)
    @return: the n-value
    """
    return math.log(math.cos(phi1) / math.cos(phi2)) / math.log(
        math.tan(_PI_BY_4 + phi2 / 2) / math.tan(_PI_BY_4 + phi1 / 2))


def __lcc_projection_n(phi2: float,
                       cos_phi1: float,
                       tan_pi_by_4_plus_phi1_by_2: float) -> float:
    """Returns the n-value for the Lambert Conformal Conic Projection

    @param phi2: principal latitude 2 (radians)
    @param cos_phi1: cosine of phi1
    @param tan_pi_by_4_plus_phi1_by_2: tangent of (pi/4 + phi1/2)
    @return: the n-value
    """
    return math.log(cos_phi1 / math.cos(phi2)) / math.log(math.tan(_PI_BY_4 + phi2 / 2) / tan_pi_by_4_plus_phi1_by_2)


def __lcc_projection_nf(phi1: float,
                        phi2: float) -> tuple[float, float]:
    """Returns the n- and F-values for the Lambert Conformal Conic Projection

    @param phi1: principal latitude 1 (radians)
    @param phi2: principal latitude 2 (radians)
    @return: a tuple with the n- and F-values
    """
    cos_phi1 = math.cos(phi1)
    tan_phi1 = math.tan(_PI_BY_4 + phi1 / 2)
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
    return n, f, f * math.tan(_PI_BY_4 + phi / 2) ** -n


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
    return _PI_BY_2 - 2 * math.atan(((x ** 2 + (f - y) ** 2) ** 0.5 / f) ** (1 / n))


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

    lat = _PI_BY_2 - 2 * math.atan((((f * x) ** 2 + (f - f * y) ** 2) ** 0.5 / f) ** (1 / n))

    if y > 1:
        if x < 0:
            return lat, math.atan(x / (1 - y)) / n - math.pi / n
        else:
            return lat, math.atan(x / (1 - y)) / n + math.pi / n
    else:
        return lat, math.atan(x / (1 - y)) / n