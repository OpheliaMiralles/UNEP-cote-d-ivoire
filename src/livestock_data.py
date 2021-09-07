import cartopy
import cartopy.crs as ccrs
import geopandas
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from data_utils import get_boundaries, HigherResPlateCarree


def livestock_plots(glw_data_path_goats: str, glw_data_path_cattle: str,
                    boundaries_file: str, villages_pos_file: str):
    data_goats = xr.open_rasterio(glw_data_path_goats).isel(band=0)
    data_goats_ds = data_goats.to_dataset(name="goats")
    data_goats_ds = data_goats_ds.where(data_goats_ds.goats != data_goats.nodatavals[0])
    data_cattle = xr.open_rasterio(glw_data_path_cattle).isel(band=0)
    data_cattle_ds = data_cattle.to_dataset(name="cattle")
    data_cattle_ds = data_cattle_ds.where(data_cattle_ds.cattle != data_cattle.nodatavals[0])
    b0, b2, b1, b3 = get_boundaries()  # chosen after iterations of plotting the zone
    data_goats_ds = data_goats_ds.sel(x=slice(b0, b1), y=slice(b3, b2))
    data_cattle_ds = data_cattle_ds.sel(x=slice(b0, b1), y=slice(b3, b2))
    villages = geopandas.read_file(villages_pos_file)
    gj = geopandas.read_file(boundaries_file)
    subplot_kw = {'projection': HigherResPlateCarree()}
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12.5, 5), subplot_kw=subplot_kw)
    for ax, ds, animal in zip([ax1, ax2], [data_goats_ds.goats, data_cattle_ds.cattle], ["goats", "cattle"]):
        ax.set_extent([b0, b1, b2, b3])
        ax.add_geometries(gj.geometry, crs=ccrs.epsg(3857), alpha=0.1, facecolor="b", edgecolor='navy')
        ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
        ax.add_feature(cartopy.feature.LAND.with_scale('10m'), color='burlywood')
        blue = (0, 0.27450980392, 0.78431372549)
        ax.add_feature(cartopy.feature.OCEAN.with_scale('10m'), color=blue)
        ax.add_feature(cartopy.feature.LAKES.with_scale('10m'), color=blue)
        c_scheme = ax.pcolormesh(ds.x[:], ds.y[:], ds, cmap="YlOrBr",
                                 transform=HigherResPlateCarree())
        plt.colorbar(c_scheme, location='bottom', pad=0.05, ax=ax)
        ax.set_title(f"Absolute number of {animal} per pixel")
        for name in villages.Name:
            centroid = villages[villages.Name == name].geometry.centroid
            middle_lon, middle_lat = float(centroid.x), float(centroid.y)
            ax.text(middle_lon - 0.5, middle_lat, name, transform=HigherResPlateCarree(),
                    color='darkred', fontsize=8)
    fig.tight_layout()

def compute_yearly_pct_country(data_path: str):
    data = pd.read_csv(data_path)
    to_concat = []
    for c, df in data.groupby('Area'):
        df = (df.set_index(["Year", "Item"])["Value"].unstack("Item")
              .assign(total=lambda x: x["Goats"] + x["Cattle"])
              .assign(goat_pct=lambda x: x["Goats"] / x["total"])
              .assign(cattle_pct=lambda x: x["Cattle"] / x["total"])
              .assign(country=c))
        to_concat.append(df)
    return pd.concat(to_concat)
