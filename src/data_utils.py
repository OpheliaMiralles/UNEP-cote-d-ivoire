import cartopy.crs as ccrs
from pyproj import Transformer

class HigherResPlateCarree(ccrs.PlateCarree):
    @property
    def threshold(self):
        return super().threshold / 100


def lonlat_to_xy(lon, lat, crs_xy, crs_latlon):
    transformer = Transformer.from_crs(crs_latlon, crs_xy)
    return transformer.transform(lat, lon)


def get_boundaries(country=None):
    if country == "COTE D'IVOIRE":
        return (-8.58, 4.36, -2.492, 10.75)
    elif country == "BURKINA FASO":
        return (-5.846659, 9.340672, 2.631718, 15.559544)
    elif country == "GHANA":
        return (-3.626339, 4.28068, 1.447039, 11.329253)
    elif country == "LARGE_ZONE":
        return (-8.923824, 3.381824, 2.541752, 15.665354)
    else:
        # larger zone with the polygon of interest
        return (-6.1, 6.0, 0.7, 11.2)
