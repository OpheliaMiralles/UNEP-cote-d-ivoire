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


class HigherResPlateCarree(ccrs.PlateCarree):
    @property
    def threshold(self):
        return super().threshold / 100


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


def merge_land_cover_regions_and_compute_summary_stats(data: xr.Dataset, boundaries_file: str):
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
    landcover = merge_land_cover_regions_and_compute_summary_stats(data, boundaries_file)
    defo = []
    for r, df in landcover.groupby(level="region", as_index=False):
        deforestation = df["forest_proportion"].diff().cumsum().rename("forest_gain").to_frame().assign(region=r)
        defo.append(deforestation.fillna(0.))
    defo = pd.concat(defo)
    return defo


if __name__ == '__main__':
    # Example of use
    path_to_landuse_datadir = ''
    path_to_adminregions_boundaries_geojson = ''
    plot_path = ''
    data = xr.open_mfdataset(f"{path_to_landuse_datadir}/*nc")
    defo = compute_deforestation_per_region(data, path_to_adminregions_boundaries_geojson)  # aggregates data per region
    last_defo = (100 * defo.loc["2020-01-01", slice(None)][
        "forest_gain"]).reset_index()  # gets the total deforestation that happened between the y1 of land use data and last year
