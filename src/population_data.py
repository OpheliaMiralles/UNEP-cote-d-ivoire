from glob import glob

import cartopy
import cartopy.crs as ccrs
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import from_levels_and_colors
from pyproj import CRS
from shapely.geometry import mapping

from data_utils import get_boundaries, HigherResPlateCarree, lonlat_to_xy


def compute_population_area(data: xr.Dataset, boundaries_file:str):
    gj = geopandas.read_file(boundaries_file)
    xds = data
    xds = xds[['population_density']].transpose('time', 'x', 'y')
    xds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    xds.rio.write_crs(data.population_density.crs, inplace=True)
    clipped = xds.rio.clip(gj.geometry.apply(mapping), "epsg:3857")
    dataframe = clipped.to_dataframe()
    dataframe = dataframe.drop(columns=["spatial_ref"])
    to_concat = []
    for time, df in dataframe.groupby(level="time", as_index=True):
        pop = df.population_density.sum()
        area = float(gj.geometry.area / (1000 * 1000))
        pop_dens_km = pop / area
        to_concat.append(pd.DataFrame([[time] + [pop] + [pop_dens_km]],
                                      columns=["time", "pop", "pop_density_km"]).set_index(["time"]))
    final_df = pd.concat(to_concat).sort_index()\
        .assign(pop_cum_increase_pct=lambda x: 100 * x["pop"].diff().cumsum().fillna(0) / x["pop"].iloc[0]) \
        .assign(pop_increase_pct=lambda x: 100 * x["pop"].diff().fillna(0) / x["pop"].shift(1).fillna(1))
    final_df["nb_years"] = (pd.to_datetime(final_df["time"]).diff() / 365.25).dt.days.fillna(1).astype(float)
    final_df["pop_increase_annual_pct"] = 100 * (
            (1 + final_df["pop_increase_pct"] / 100) ** (1 / final_df["nb_years"]) - 1)
    final_df = final_df.drop(columns="nb_years")
    return final_df

def compute_population_villages(data: xr.Dataset,
                                         boundaries_file:str):
    gj = geopandas.read_file(boundaries_file)
    xds = data
    xds = xds[['population_density']].transpose('time', 'x', 'y')
    xds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    xds.rio.write_crs(data.population_density.crs, inplace=True)
    to_concat = []
    for c in gj.Name:
        g = gj[gj.Name == c]
        clipped = xds.rio.clip(g.geometry.apply(mapping), "epsg:4326")
        dataframe = clipped.to_dataframe()
        dataframe = dataframe.drop(columns=["spatial_ref"])
        for time, df in dataframe.groupby(level="time", as_index=True):
            pop = df.population_density.sum()
            to_concat.append(pd.DataFrame([[time] + [c] + [pop] ],
                                          columns=["time", "village", "pop"]).set_index(
                ["village", "time"]))
    final_df = pd.concat(to_concat).sort_index()
    return final_df


def compute_population_african_countries(data: xr.Dataset,
                                         boundaries_file:str,
                                         countries=None):
    gj = geopandas.read_file(boundaries_file)
    if countries is not None:
        gj = gj[gj["ADM0_NAME"].isin(countries)]
    xds = data
    xds = xds[['population_density']].transpose('time', 'x', 'y')
    xds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    xds.rio.write_crs(data.population_density.crs, inplace=True)
    to_concat = []
    for c in countries:
        g = gj[gj["ADM0_NAME"] == c]
        clipped = xds.rio.clip(g.geometry.apply(mapping), "epsg:4326")
        dataframe = clipped.to_dataframe()
        dataframe = dataframe.drop(columns=["spatial_ref"])
        for time, df in dataframe.groupby(level="time", as_index=True):
            pop = df.population_density.sum()
            area = float(g.to_crs("epsg:3857").geometry.area / (1000 * 1000))
            pop_dens_km = pop / area
            to_concat.append(pd.DataFrame([[time] + [c.replace("Cï¿½te", "Cote")] + [pop] + [pop_dens_km]],
                                          columns=["time", "country", "pop", "pop_density_km"]).set_index(
                ["country", "time"]))
    temp_df = pd.concat(to_concat).sort_index()
    concat = []
    for c, df in temp_df.reset_index().groupby("country"):
        df = df \
            .assign(pop_cum_increase_pct=lambda x: 100 * x["pop"].diff().cumsum().fillna(0) / x["pop"].iloc[0])\
            .assign(pop_increase_pct=lambda x: 100 * x["pop"].diff().fillna(0) / x["pop"].shift(1).fillna(1))
        df["nb_years"] = (pd.to_datetime(df["time"]).diff() / 365.25).dt.days.fillna(1).astype(float)
        df["pop_increase_annual_pct"] = 100 * (
                    (1 + df["pop_increase_pct"] / 100) ** (1 / df["nb_years"]) - 1)
        df = df.drop(columns = "nb_years")
        concat.append(df)
    final_df = pd.concat(concat).set_index(["country", "time"])
    return final_df


def mapping_change(x):
    if x <= 0.0:
        return 0
    elif x <= 0.5:
        return 0.5
    elif x <= 1:
        return 1
    elif x <= 2:
        return 2
    elif x > 2:
        return 10


def get_cmap():
    levels = [0, 0.5, 1, 2, 10]
    colors = ["slateblue", "darkorange", "orangered", "firebrick", "darkred"]
    cmap_dict = {v: c for v, c in zip(levels, colors)}
    cmap, norm = from_levels_and_colors(levels=list(cmap_dict.keys()) + [20],
                                        colors=list(cmap_dict.values()),
                                        extend="neither")
    return cmap, norm


def plot_pop_density_change(y0: str, y1: str, data: xr.Dataset, boundaries_area: str = None):
    b0, b2, b1, b3 = get_boundaries()
    data_area = pop.where(data.population_density > 0)
    df0 = data_area.sel(time=y0).drop("time").population_density.to_dataframe()
    df1 = data_area.sel(time=y1).drop("time").population_density.to_dataframe()
    diff_df = ((df1 - df0) / df0)
    diff_df = diff_df.assign(population_change=lambda x: x["population_density"])
    diff_df["population_change"] = diff_df["population_change"].apply(mapping_change)
    diff = np.array(diff_df["population_change"]).reshape(len(data_area.y), len(data_area.x))

    # Plot
    subplot_kw = {'projection': HigherResPlateCarree()}
    fig, ax = plt.subplots(ncols=1, figsize=(10, 7.5), subplot_kw=subplot_kw)
    cmap, norm = get_cmap()
    ax.set_extent([b0, b1, b2, b3])
    if boundaries_area is not None:
        area = geopandas.read_file(boundaries_area)
        ax.add_geometries(area.geometry, crs=ccrs.epsg(3857), alpha=0.1, facecolor="b", edgecolor='navy')
    ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
    ax.add_feature(cartopy.feature.LAND.with_scale('10m'), color='burlywood')
    blue = (0, 0.27450980392, 0.78431372549)
    ax.add_feature(cartopy.feature.OCEAN.with_scale('10m'), color=blue)
    ax.add_feature(cartopy.feature.LAKES.with_scale('10m'), color=blue)
    c_scheme = ax.pcolormesh(data_area.x[:], data_area.y[:], diff,
                             cmap=cmap, norm=norm,
                             transform=ccrs.Mollweide())
    cb = plt.colorbar(c_scheme, location='bottom', pad=0.05, ax=ax)
    cb.ax.tick_params(labelsize=7)
    t = cb.get_ticks()
    middle_points = (t[:-1] + t[1:]) / 2
    cb.set_ticks(middle_points)
    labels = ["Pop. decrease", "Pop. increase \n up to 50%", "Pop. increase \n up to 100%",
              "Pop. increase \n up to 200%", "Pop. increase \n of more than 200%"]
    cb.set_ticklabels(labels)
    fig.suptitle(f"Population density increase from {y0} to {y1}")
    return fig

def create_netcdf_from_tifs(path_to_rasters: str, path_to_netcdf: str):
    b0, b2, b1, b3 = get_boundaries("LARGE_ZONE")
    crs_latlon = CRS.from_epsg(4326)
    to_concat = []
    for file in glob(path_to_rasters):
        year = file.split("/")[-1].split(".")[0].split('_')[2].replace("GPW4", "")
        data = xr.open_rasterio(file).isel(band=0)
        crs = data.crs
        crs_xy = CRS.from_proj4(crs)
        b0_, b2_ = lonlat_to_xy(b0, b2, crs_xy, crs_latlon)
        b1_, b3_ = lonlat_to_xy(b1, b3, crs_xy, crs_latlon)
        dataset = data.to_dataset(name="population_density")
        data_area = dataset.sel(x=slice(b0_, b1_), y=slice(b3_, b2_)).expand_dims(time=[pd.to_datetime(year)]).drop(
            "band")
        to_concat.append(data_area)
    dataset = xr.concat(to_concat, dim="time").sortby('time')
    dataset.to_netcdf(path_to_netcdf)

def compute_and_plot_annual_growth_per_country(data: pd.DataFrame):
    concat = []
    for c, df in data.groupby("country"):
        df = df \
            .assign(pop_increase_pct=lambda x: 100 * x["pop"].diff().fillna(0) / x["pop"].shift(1).fillna(1))
        df["nb_years"] = (pd.to_datetime(df["time"]).diff() / 365.25).dt.days.fillna(1).astype(float)
        df["pop_increase_pct"] = 100 * ((1 + df["pop_increase_pct"] / 100) ** (1 / df["nb_years"]) - 1)
        concat.append(df)
    c = pd.concat(concat)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(5, 5))
    for (country, df), ax, color in zip(c.set_index("time").groupby("country"), [ax1, ax2, ax3],
                                        ["slateblue", "darkorange", "firebrick"]):
        print(df)
        ax.bar(df.index, df['pop_increase_pct'], label=country, color=color)
        ax.set_ylabel(r"Pop. increase (\%)")
        ax.legend(loc="upper left")
    fig.suptitle("Population increase annualised")
    return fig

if __name__ == '__main__':
    path_to_netcdf = ""
    path_to_rasters = ""
    #create_netcdf_from_tifs(path_to_rasters, path_to_netcdf)

    area_boundaries = ""
    countries_boundaries = ""
    countries_of_interest = [c for c in geopandas.read_file(countries_boundaries)["ADM0_NAME"] if
                             "Ivoire" in c or "Ghana" in c or "Burkina" in c]
    village_shapefile = ""
    pop = xr.open_mfdataset(path_to_netcdf)
    #fig = plot_pop_density_change('1975', '2015', pop, area_boundaries)
