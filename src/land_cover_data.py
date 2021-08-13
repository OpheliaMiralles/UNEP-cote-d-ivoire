import cartopy
import cartopy.crs as ccrs
import geopandas
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import from_levels_and_colors
from numpy import copy
from shapely.geometry import mapping

from data_utils import HigherResPlateCarree, get_boundaries


def get_land_cover_colormap_for_data(data):
    cmap_list = [(0, 0, 0), (255, 255, 100), (170, 240, 240), (220, 240, 100), (200, 200, 100),
                 (0, 100, 0), (0, 160, 0), (0, 60, 0), (40, 80, 0), (120, 130, 0), (140, 160, 0),
                 (190, 150, 0), (150, 100, 0), (255, 180, 50), (255, 220, 210), (255, 235, 175),
                 (0, 120, 90), (0, 150, 120), (0, 220, 130), (195, 20, 0), (255, 245, 215),
                 (0, 70, 200), (255, 255, 255)]
    keys = np.arange(0, 230, 10)
    cmap_dict = {v: c for v, c in zip(keys, cmap_list)}
    # Regional values
    cmap_dict[11] = (255, 255, 100)
    cmap_dict[12] = (255, 255, 0)
    cmap_dict[61] = (0, 160, 0)
    cmap_dict[62] = (170, 200, 0)
    cmap_dict[71] = (0, 60, 0)
    cmap_dict[72] = (0, 80, 0)
    cmap_dict[81] = (40, 80, 0)
    cmap_dict[82] = (40, 100, 0)
    cmap_dict[121] = (150, 100, 0)
    cmap_dict[122] = (150, 100, 0)
    cmap_dict[151] = (255, 200, 100)
    cmap_dict[152] = (255, 210, 120)
    cmap_dict[153] = (255, 235, 175)
    cmap_dict[201] = (220, 220, 220)
    cmap_dict[202] = (255, 245, 215)
    normalize = mcolors.Normalize(vmin=0, vmax=255)
    sorted_dict = {k: normalize(cmap_dict[k]) for k in sorted(list(cmap_dict))}
    sorted_dict_restricted = {k: sorted_dict[k] for k in np.unique(data)}
    cmap, norm = from_levels_and_colors(levels=list(sorted_dict_restricted.keys()) + [220],
                                        colors=list(sorted_dict_restricted.values()),
                                        extend="neither")
    return cmap, norm


def get_forest_nonforest_dict_for_data(data, default):
    data_values = sorted(np.unique(data))
    forest_levels = [50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 160, 170]
    mapping = {k: k if k in forest_levels else default for k in data_values}
    mapping[210] = 210  # oceans
    forest_nonforest_data = copy(data)
    for k, v in mapping.items(): forest_nonforest_data[data == k] = v
    return forest_nonforest_data


def plot_land_cover(dataset, bounds_longitude, bounds_latitude, time):
    df2 = dataset.sel(lon=slice(bounds_longitude[0], bounds_longitude[1]),
                      lat=slice(bounds_latitude[-1], bounds_latitude[0]))
    longitudes = df2.lon[:]
    latitudes = df2.lat[:]
    year = pd.to_datetime(time).strftime('%Y')
    ds = df2.sel(time=time).lccs_class
    data = ds[:]
    default_level_nonforest = 10
    forest_nonforest_data = get_forest_nonforest_dict_for_data(data, default_level_nonforest)
    cmap, norm = get_land_cover_colormap_for_data(data)
    subplot_kw = {'projection': HigherResPlateCarree()}
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(25, 12.5), subplot_kw=subplot_kw)
    for ax in (ax1, ax2):
        ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()])
        ax.add_feature(cartopy.feature.LAKES.with_scale('10m'), color='royalblue')
        ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
        ax.add_feature(cartopy.feature.RIVERS.with_scale('10m'), color='royalblue')
    # Land cover plot
    c_scheme = ax1.pcolormesh(longitudes, latitudes, data, cmap=cmap, norm=norm, transform=HigherResPlateCarree())
    plt.colorbar(c_scheme, location='bottom', pad=0.05,
                 label="LCCS classification of land cover", ax=ax1)
    ax1.set_title(f"Land Cover in Côte d'Ivoire in {year}", fontsize=14)
    # Forest / Non forest plot
    cmap2, norm2 = get_land_cover_colormap_for_data(forest_nonforest_data)
    c_scheme = ax2.pcolormesh(longitudes, latitudes, forest_nonforest_data, cmap=cmap2,
                              norm=norm2, transform=HigherResPlateCarree())
    cb = plt.colorbar(c_scheme, location='bottom', pad=0.05, ax=ax2)
    map_meanings = {k: ' '.join(v.split('_')).capitalize() for k, v in
                    zip(ds.flag_values, ds.flag_meanings.split(' '))}
    map_meanings[default_level_nonforest] = "Non forest"
    ticks = sorted(np.unique(forest_nonforest_data))
    cb.ax.tick_params(labelsize=8, labelrotation=90)
    t = cb.get_ticks()
    middle_points = (t[:-1] + t[1:]) / 2
    cb.set_ticks(middle_points)
    dic_tick_labels = {k: map_meanings[k] for k in ticks}
    dic_tick_labels[50] = 'Tree cover, broadleaved, evergreen'
    dic_tick_labels[61] = 'Tree cover, broadleaved, deciduous, open'
    dic_tick_labels[62] = 'Tree cover, broadleaved, deciduous, closed'
    dic_tick_labels[60] = 'Tree cover, broadleaved, deciduous, closed to open'
    cb.set_ticklabels(list(dic_tick_labels.values()))
    ax2.set_title(f"Forest/Non forest map for Côte d'Ivoire in {year}", fontsize=14)
    plt.tight_layout()
    return fig


def get_new_lccs_classes():
    dic = dict(
        forest=[50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 160, 170],
        agriculture=[10, 11, 12, 20, 30, 40],
        grassland=[110, 130],
        wetland=[180],
        urban=[190],
        shrubland=[121, 122, 120],
        sparse_vegetation=[140, 150, 151, 152, 153],
        bare_land=[200, 201, 202],
        water=[210],
        permanent_snow_ice=[220],
        lichen_mosses=[140],
        no_data=[0],
    )
    inverted = {n: key for key, v in dic.items() for n in v}
    return inverted


def compute_landuse_region(data: xr.Dataset, boundaries_file: str):
    """

    :param data: land cover data from LCCS (.nc file read with xarray)
    :param boundaries_file: the path to region boundaries
    :return: table with land use proportion per region per year between 0 and 1
    """
    gj = geopandas.read_file(boundaries_file)
    dic_landuse_classes = get_new_lccs_classes()
    regional = []
    for region in gj.admin2Name:
        boun = gj[gj["admin2Name"] == region]
        xds = data.drop_vars(["crs", "lat_bounds", "lon_bounds", "time_bounds"])
        xds = xds[['lccs_class']].transpose('time', 'lat', 'lon')
        xds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        xds.rio.write_crs(data.crs, inplace=True)
        clipped = xds.rio.clip(boun.geometry.apply(mapping))
        dataframe = clipped.to_dataframe()
        dataframe = dataframe.drop(columns=["spatial_ref"]).assign(land_use=lambda x: x["lccs_class"])
        dataframe["land_use"] = dataframe["land_use"].apply(lambda x: dic_landuse_classes[x])
        for time, df in dataframe.groupby(level="time", as_index=True):
            regional_data = df.groupby('land_use').agg({"land_use": lambda x: x.count() / len(df)})
            perc_forest = regional_data.loc["forest", "land_use"]
            perc_missing = regional_data.loc["no_data", "land_use"]
            biggest_landuse = regional_data.drop("no_data").idxmax().values[0]
            regional.append(pd.DataFrame([[time, region, perc_forest, perc_missing, biggest_landuse]],
                                         columns=["time", "region", "forest_proportion", "missing_data_prop",
                                                  "main_land_use"]).set_index(["time", "region"]))
    cid = pd.concat(regional).sort_index()
    return cid


def compute_deforestation_per_region(data: xr.Dataset, boundaries_file: str):
    landcover = compute_landuse_region(data, boundaries_file)
    defo = []
    for r, df in landcover.groupby(level="region", as_index=False):
        deforestation = df["forest_proportion"].diff().cumsum().rename("forest_gain").to_frame().assign(region=r)
        defo.append(deforestation.fillna(0.))
    defo = pd.concat(defo)
    return defo


def compute_landuse_villages(data: xr.Dataset, boundaries_file: str):
    gj = geopandas.read_file(boundaries_file)
    dic_landuse_classes = get_new_lccs_classes()
    columns_dataframe = [k for k in np.unique(list(dic_landuse_classes.values()))]
    regional = []
    for village in gj.Name:
        boun = gj[gj["Name"] == village]
        xds = data.drop_vars(["crs", "lat_bounds", "lon_bounds", "time_bounds"])
        xds = xds[['lccs_class']].transpose('time', 'lat', 'lon')
        xds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        xds.rio.write_crs(data.crs, inplace=True)
        clipped = xds.rio.clip(boun.geometry.apply(mapping))
        dataframe = clipped.to_dataframe()
        dataframe = dataframe.drop(columns=["spatial_ref"]).assign(land_use=lambda x: x["lccs_class"])
        dataframe["land_use"] = dataframe["land_use"].apply(lambda x: dic_landuse_classes[x])
        for time, df in dataframe.groupby(level="time", as_index=True):
            regional_data = df.groupby('land_use').agg({"land_use": lambda x: x.count() / len(df)})
            perc_cover = []
            for col in list(columns_dataframe):
                if col in regional_data.index:
                    perc = regional_data.loc[col, "land_use"]
                else:
                    perc = 0.0
                perc_cover.append(perc)
            biggest_landuse = regional_data.idxmax().values[0]
            regional.append(pd.DataFrame([[time, village] + perc_cover + [biggest_landuse]],
                                         columns=["time", "village"] + list(columns_dataframe) + [
                                             "main_land_use"]).set_index(["village", "time"]))
    cid = pd.concat(regional).sort_index()
    return cid


def compute_landuse_area(data: xr.Dataset, boundaries_file: str):
    gj = geopandas.read_file(boundaries_file)
    dic_landuse_classes = get_new_lccs_classes()
    columns_dataframe = [k for k in np.unique(list(dic_landuse_classes.values()))]
    regional = []
    xds = data.drop_vars(["crs", "lat_bounds", "lon_bounds", "time_bounds"])
    xds = xds[['lccs_class']].transpose('time', 'lat', 'lon')
    xds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    xds.rio.write_crs(data.crs, inplace=True)
    clipped = xds.rio.clip(gj.geometry.apply(mapping), "epsg:3857")
    dataframe = clipped.to_dataframe()
    dataframe = dataframe.drop(columns=["spatial_ref"]).assign(land_use=lambda x: x["lccs_class"])
    dataframe["land_use"] = dataframe["land_use"].apply(lambda x: dic_landuse_classes[x])
    for time, df in dataframe.groupby(level="time", as_index=True):
        regional_data = df.groupby('land_use').agg({"land_use": lambda x: x.count() / len(df)})
        perc_cover = []
        for col in list(columns_dataframe):
            if col in regional_data.index:
                perc = regional_data.loc[col, "land_use"]
            else:
                perc = 0.0
            perc_cover.append(perc)
        biggest_landuse = regional_data.idxmax().values[0]
        regional.append(pd.DataFrame([[time] + perc_cover + [biggest_landuse]],
                                     columns=["time"] + list(columns_dataframe) + [
                                         "main_land_use"]).set_index(["time"]))
    cid = pd.concat(regional).sort_index()
    return cid


def compute_landuse_change_area(data: xr.Dataset, boundaries_file: str):
    landuse_area = compute_landuse_area(data, boundaries_file)
    cumdiff_pct = 100 * landuse_area.drop(columns=["main_land_use"]).diff().cumsum().fillna(0.)
    return cumdiff_pct


def barplot_graph_landuse_area(data: pd.DataFrame):
    degList = [i for i in data.columns]
    bar_l = range(data.shape[0])
    cm = plt.get_cmap('tab20b')
    from cycler import cycler
    f, ax = plt.subplots(1, figsize=(12, 7))
    ax.set_prop_cycle(cycler('color', [cm(1. * i / len(degList)) for i in range(len(degList))]))
    bottom = np.zeros_like(bar_l).astype('float')
    for i, deg in enumerate(degList):
        ax.bar(bar_l, data[deg], bottom=bottom, label=deg.replace("_", " ").capitalize())
        bottom += data[deg].values

    ax.set_xticks(bar_l)
    ax.set_xticklabels([pd.to_datetime(i).strftime("%Y") for i in data.index], rotation=90,
                       size='x-small')
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize='x-small')
    f.tight_layout()
    return f


def landuse_area_plot_with_village_names(data: xr.Dataset, boundaries_file: str, villages_pos_file: str):
    """

    :param data: land cover data from LCCS
    :param boundaries_file: path to area boundaries
    :param villages_pos_file: geographical position of villages of interest
    :return: matplotlib figure, do "f.show()" to see the figure or "f.savefig(path)" to save it somewhere
    """
    b0, b2, b1, b3 = get_boundaries()  # chosen after iterations of plotting the zone
    villages = geopandas.read_file(villages_pos_file)
    gj = geopandas.read_file(boundaries_file)
    subplot_kw = {'projection': HigherResPlateCarree()}
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12.5, 5), subplot_kw=subplot_kw)
    for y, ax in zip(['2000', '2020'], [ax1, ax2]):
        clip = data.sel(time=y).sel(lon=slice(b0, b1), lat=slice(b3, b2)).lccs_class[:]
        longitudes = clip.lon[:]
        latitudes = clip.lat[:]
        dic = get_new_lccs_classes()
        colors = pd.DataFrame.from_dict(dic, orient="index").reset_index().rename(
            columns={0: "land_use", "index": "val"}).groupby("land_use").agg("min")
        dicc = colors.to_dict()["val"]
        finaldic = {k: dicc[v] for k, v in dic.items()}
        colored_ds = xr.Dataset.from_dataframe(clip.to_dataframe()["lccs_class"].map(finaldic).to_frame())
        cmap, norm = get_land_cover_colormap_for_data(colored_ds.lccs_class[:])
        colored_array = np.array(colored_ds.lccs_class[:]).reshape(len(latitudes), len(longitudes))
        ax.set_extent([b0, b1, b2, b3])
        ax.add_geometries(gj.geometry, crs=ccrs.epsg(3857), alpha=0.1, facecolor="b", edgecolor='navy')
        # Land cover plot
        c_scheme = ax.pcolormesh(longitudes, latitudes, colored_array, cmap=cmap, norm=norm,
                                 transform=HigherResPlateCarree())
        cb = plt.colorbar(c_scheme, location='bottom', pad=0.05,
                          label="LCCS classification of land cover", ax=ax)
        cb.ax.tick_params(labelsize=8, labelrotation=45)
        t = cb.get_ticks()
        middle_points = (t[:-1] + t[1:]) / 2
        cb.set_ticks(middle_points)
        levels = [{v: k for k, v in dicc.items()}[t].replace("_", " ").capitalize() for t in t]
        cb.set_ticks(middle_points)
        cb.set_ticklabels(levels)
        ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
        ax.add_feature(cartopy.feature.LAND.with_scale('10m'), color='burlywood')
        blue = (0, 0.27450980392, 0.78431372549)
        ax.add_feature(cartopy.feature.OCEAN.with_scale('10m'), color=blue)
        ax.set_title(f'{y}')
        for name in villages.Name:
            centroid = villages[villages.Name == name].geometry.centroid
            middle_lon, middle_lat = float(centroid.x), float(centroid.y)
            ax.text(middle_lon - 0.5, middle_lat, name, transform=HigherResPlateCarree(),
                    color='darkred', fontsize=8)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    # Example of use
    path_to_landuse_datadir = ''
    path_to_boundaries_shapefile = ''
    data = xr.open_mfdataset(f"{path_to_landuse_datadir}/*nc")
    landuse_prop = compute_landuse_area(data, path_to_boundaries_shapefile)
    cumdiff_pct = compute_landuse_change_area(data, path_to_boundaries_shapefile)
    f = barplot_graph_landuse_area(landuse_prop)
    f.show()
