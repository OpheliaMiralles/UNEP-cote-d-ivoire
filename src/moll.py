from math import sin, cos, pi, sqrt, asin

sqrt2 = sqrt(2)
R = (6.371)*10**6

def solveNR(lat, epsilon=1e-6):
    """Solve the equation $2\theta\sin(2\theta)=\pi\sin(\mathrm{lat})$
    using Newtons method"""
    if abs(lat) == pi / 2:
        return lat  # avoid division by zero
    theta = lat
    while True:
        nexttheta = theta - (
                (2 * theta + sin(2 * theta) - pi * sin(lat)) /
                (2 + 2 * cos(2 * theta))
        )
        if abs(theta - nexttheta) < epsilon:
            break
        theta = nexttheta
    return nexttheta


def checktheta(theta, lat):
    """Testing function to confirm that the NR method worked"""
    return (2 * theta + sin(2 * theta), pi * sin(lat))


def mollweide(lat, lon, lon_0=0, R=R, degrees=False):
    """Convert latitude and longitude to cartesian mollweide coordinates
    arguments
    lat, lon -- Latitude and longitude with South and West as Negative
        both as decimal values
    lon_0 -- the longitude of the central meridian, default = Greenwich
    R -- radius of the globe
    degrees -- if True, interpret the latitude and longitude as degrees

    Return
    x, y a tuple of coorinates in range $x\in\pm 2R\sqrt{2}$,
      $y\in\pm R\sqrt{2}$
    """
    if degrees:
        lat = lat * pi / 180
        lon = lon * pi / 180
        lon_0 = lon_0 * pi / 180  # convert to radians
    theta = solveNR(lat)
    return (R * 2 * sqrt2 * (lon - lon_0) * cos(theta) / pi,
            R * sqrt2 * sin(theta))


def inv_mollweide(x, y, lon_0=0, R=R, degrees=True):
    """Invert the mollweide transform. Arguments are as for that function"""
    theta = asin(y / (R * sqrt2))
    if degrees:
        factor = 180 / pi
    else:
        factor = 1
    return (
        asin((2 * theta + sin(2 * theta)) / pi) * factor,
        (lon_0 + pi * x / (2 * R * sqrt(2) * cos(theta))) * factor
    )
