import pathlib
from glob import glob
from io import StringIO

import cartopy
import cartopy.crs as ccrs
import geopandas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from data_utils import HigherResPlateCarree, get_boundaries

matplotlib.rcParams['text.usetex'] = True


def get_monthly_river_discharge_from_ADHI(path_to_files, stations_path):
    stations = pd.read_csv(stations_path)
    CID_stations = stations[stations["Country"] == "Cote d'Ivoire"]
    to_concat = []
    for file in glob(path_to_files):
        file = pathlib.Path(file)
        st_nb = f"ADHI_{file.with_suffix('').name.replace('monthly_ADHI_', '')}"
        if st_nb in CID_stations["ID"].unique():
            station_data = stations[stations["ID"] == st_nb]
            rd = pd.read_csv(StringIO(file.read_text('latin1')),
                             sep=',',
                             names=["year", "month", "mean_runoff", "max_runoff", "min_runoff", "nb_missing_days"])
            rd = rd[rd["nb_missing_days"] < 5]
            rd["station_id"] = st_nb
            rd = rd.merge(station_data.rename(columns={"ID": "station_id"}), on="station_id", how="left")
            rd = rd.set_index(["year", "month", "station_id"])
            to_concat.append(rd)
    data = pd.concat(to_concat)
    data.to_csv(pathlib.Path(f"{pathlib.Path(path_to_files).parent.parent}", "river_discharge.csv"))


def plot_ADHI_stations(stations_path, figname=''):
    stations = pd.read_csv(stations_path)
    stations = stations[stations['Country'] == "Cote d'Ivoire"]

    class HigherResPlateCarree(ccrs.PlateCarree):
        @property
        def threshold(self):
            return super().threshold / 100

    proj = HigherResPlateCarree()
    subplot_kw = {'projection': proj}
    fig, ax = plt.subplots(ncols=1, figsize=(5, 5), subplot_kw=subplot_kw)
    ax.set_extent(
        [stations.Longitude.min() - 0.3, stations.Longitude.max() + 0.3, stations.Latitude.min() - 0.3,
         stations.Latitude.max() + 0.3])
    ax.coastlines()
    ax.stock_img()
    ax.add_feature(cartopy.feature.LAND, color='goldenrod')
    ax.add_feature(cartopy.feature.LAKES.with_scale('10m'), color='royalblue')
    ax.add_feature(cartopy.feature.OCEAN, color='skyblue')
    ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
    ax.add_feature(cartopy.feature.RIVERS.with_scale('10m'), color='grey')
    c_scheme = ax.scatter(x=stations.Longitude, y=stations.Latitude,
                          s=50,
                          c=stations.lc_forest,
                          cmap='Greens',
                          alpha=0.5,
                          transform=HigherResPlateCarree())
    plt.colorbar(c_scheme, location='bottom', pad=0.07,
                 label='Forest cover', ax=ax)
    fig.suptitle("Map of stations")
    savename = figname if figname != '' else 'map_stations'
    fig.savefig(pathlib.Path(pathlib.Path(stations_path).parent.parent.parent, f"plots/river_discharge/{savename}.png"))


def merge_river_discharge_regions(data, boundaries_file):
    gj = geopandas.read_file(boundaries_file)
    to_concat = []
    for station, d in data.groupby("station_id"):
        lon, lat = d["Longitude"].unique()[0], d["Latitude"].unique()[0]
        point = Point(lon, lat)
        which_polygon = gj.loc[[p.contains(point) for p in gj.geometry].index(True), :].to_frame().T
        d1 = pd.DataFrame(np.tile(which_polygon, (len(d), 1)), columns=which_polygon.columns, index=d.index)
        d2 = pd.concat([d, d1], axis=1)
        to_concat.append(d2)
    return pd.concat(to_concat)


def plot_timelapse_region(data, boundaries_file,
                          plot_path=f'/Users/opheliamiralles/Desktop/PhD/EPFL/DeforestationFlooding_EPFL_UNEP/plots/river_discharge'):
    database = merge_river_discharge_regions(data, boundaries_file)
    gj = geopandas.read_file(boundaries_file).assign(color='navy')
    mean_region = database.groupby(["year", "region"]).agg({"min_runoff": "mean",
                                                            "mean_runoff": "mean",
                                                            "max_runoff": "mean"}) \
        .reset_index().rename(columns={"region": "admin2Name"})
    vmin, vmax = mean_region["mean_runoff"].min(), mean_region["mean_runoff"].max()
    for y, df in mean_region.groupby("year"):
        gj_year = gj.merge(df, on="admin2Name", how="left")
        ax = plt.axes(projection=HigherResPlateCarree())
        b0, b2, b1, b3 = get_boundaries("COTE D'IVOIRE")
        ax.set_extent([b0, b1, b2, b3])
        gj_year.plot(column='mean_runoff', cmap="YlGnBu", ax=ax, legend=True, vmin=vmin, vmax=vmax)
        ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
        ax.add_feature(cartopy.feature.LAND.with_scale('10m'), color='burlywood')
        blue = (0, 0.27450980392, 0.78431372549)
        ax.add_feature(cartopy.feature.OCEAN.with_scale('10m'), color=blue)
        ax.set_title(r'Mean river discharge per administrative region ($m^3.s^{-1}$)' + f'\n{y}')
        bounkani = gj[gj["admin2Name"] == 'Bounkani']
        g = bounkani.geometry
        minx, miny, maxx, maxy = tuple([v for v in g.bounds.values[0]])
        middle_lon = (maxx + minx) / 2
        middle_lat = (maxy + miny) / 2
        ax.text(middle_lon - 0.5, middle_lat, 'Bounkani', transform=HigherResPlateCarree(), color='grey')
        gontougo = gj[gj["admin2Name"] == 'Gontougo']
        g = gontougo.geometry
        minx, miny, maxx, maxy = tuple([v for v in g.bounds.values[0]])
        middle_lon = (maxx + minx) / 2
        middle_lat = (maxy + miny) / 2
        ax.text(middle_lon - 0.5, middle_lat, 'Gontougo', transform=HigherResPlateCarree(), color='grey')
        plt.savefig(f'{plot_path}/regional_discharge_{y}.png')
        plt.clf()
    from PIL import Image
    images = []
    for y in sorted(mean_region["year"].unique()):
        images.append(
            Image.open(f'{plot_path}/regional_discharge_{y}.png'))
    images[0].save(f'{plot_path}/river_discharge_timelapse.gif',
                   save_all=True, append_images=images[1:], duration=300, loop=0)


def merge_catchment_areas(data: xr.Dataset, catchment_file: str):
    gj = geopandas.read_file(catchment_file)
    to_concat = []
    for station, d in data.groupby("station_id"):
        lon, lat = d["Longitude"].unique()[0], d["Latitude"].unique()[0]
        point = Point(lon, lat)
        which_polygon = gj.loc[[p.contains(point) for p in gj.geometry].index(True), :].to_frame().T
        d1 = pd.DataFrame(np.tile(which_polygon, (len(d), 1)), columns=which_polygon.columns, index=d.index)
        d2 = pd.concat([d, d1], axis=1)
        to_concat.append(d2)
    return pd.concat(to_concat)
