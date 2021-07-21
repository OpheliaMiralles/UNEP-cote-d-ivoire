import cartopy.crs as ccrs


class HigherResPlateCarree(ccrs.PlateCarree):
    @property
    def threshold(self):
        return super().threshold / 100


def get_boundaries(country=None):
    if country == "COTE D'IVOIRE":
        return (-8.58, 4.36, -2.492, 10.75)
    elif country == "BURKINA FASO":
        return (-5.846659, 9.340672, 2.631718, 15.559544)
    elif country == "GHANA":
        return (-3.626339, 4.28068, 1.447039, 11.329253)
    else:
        return (-8.923824, 3.381824, 2.541752, 15.665354)
