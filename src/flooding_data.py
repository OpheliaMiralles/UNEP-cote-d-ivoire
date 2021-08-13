import os
from typing import Sequence

import cartopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wget

from data_utils import HigherResPlateCarree, get_boundaries


# MODIS
def download_mfw(years: Sequence, path_to_directory: str, location: str):
    for year in np.arange(*years, step=1):
        start = (pd.to_datetime(f"{year}-06-01") - pd.to_datetime(f"{year}-01-01")).days
        end = (pd.to_datetime(f"{year}-10-01") - pd.to_datetime(f"{year}01-01")).days
        for d in np.arange(start, end, step=1):
            target_url = f"https://floodmap.modaps.eosdis.nasa.gov/Products/{location}/{year}/MSW_{year}{d:03}_{location}_A14x3D3OT.tif"
            if not os.path.isfile(f"{path_to_directory}/MSW_{year}{d:03}_{location}_A14x3D3OT.tif"):
                try:
                    wget.download(target_url, path_to_directory)
                    print(f"Downloading url {target_url} to file {path_to_directory}")
                except Exception as err:
                    print(f"     ---> Can't access {target_url}: {err}")
            else:
                print(f"Day {d} of year {year} has already been downloaded to {path_to_directory}")


def download_mwp(years: Sequence, path_to_directory: str, location: str):
    for year in np.arange(*years, step=1):
        start = (pd.to_datetime(f"{year}-06-01") - pd.to_datetime(f"{year}-01-01")).days
        end = (pd.to_datetime(f"{year}-10-01") - pd.to_datetime(f"{year}-01-01")).days
        for d in np.arange(start, end, step=1):
            target_url = f"https://floodmap.modaps.eosdis.nasa.gov/Products/{location}/{year}/MWP_{year}{d:03}_{location}_3D3OT.tif"
            if not os.path.isfile(f"{path_to_directory}/MWP_{year}{d:03}_{location}_3D3OT.tif"):
                try:
                    wget.download(target_url, path_to_directory)
                    print(f"Downloading url {target_url} to file {path_to_directory}")
                except Exception as err:
                    print(f"     ---> Can't access {target_url}: {err}")
            else:
                print(f"Day {d} of year {year} has already been downloaded to {path_to_directory}")


# DFO
def plot_dfo_for_cid_and_area(data: pd.DataFrame):
    data["Country"] = data["Country"].apply(
        lambda x: x.replace('\xa0', "").replace("Cote D'Iavoir", "Cote d'Ivoire").replace("Burkino Faso",
                                                                                          "Burkina Faso"))
    cid_floods = data[data["Country"] == "Cote d'Ivoire"]
    ghana_floods = data[data["Country"] == "Ghana"]
    bf_floods = data[data["Country"] == "Burkina Faso"]
    floods = pd.concat([cid_floods, ghana_floods, bf_floods])
    b0, b2, b1, b3 = get_boundaries()
    proj = HigherResPlateCarree()
    subplot_kw = {'projection': proj}
    fig, ax = plt.subplots(ncols=1, figsize=(8, 8), subplot_kw=subplot_kw)
    ax.set_extent(
        [b0, b1, b2, b3])
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND, color='burlywood')
    blue = (0, 0.27450980392, 0.78431372549)
    ax.add_feature(cartopy.feature.LAKES.with_scale('10m'), color=blue)
    ax.add_feature(cartopy.feature.OCEAN, color=blue)
    ax.add_feature(cartopy.feature.RIVERS.with_scale('10m'), color='grey')
    ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
    factor = floods.Area.median() / 50
    c_scheme = ax.scatter(x=floods.long, y=floods.lat,
                          s=floods.Area / factor,
                          c=np.log(1 + floods.Displaced),
                          cmap='YlOrBr',
                          alpha=0.5,
                          transform=HigherResPlateCarree())
    cb = plt.colorbar(c_scheme, location='right', pad=0.07,
                      label='Estimated number of displaced persons', ax=ax)
    levels = [0, 6, 50, 400, 3000, 20000, 150000, 500000]
    cb.set_ticks(np.log(1 + np.array(levels)))
    cb.set_ticklabels(levels)
    cid_centroid = (-5.5, 7.5)
    bf_centroid = (-1.6, 12.2)
    ghana_centroid = (-1.02, 7.94)
    for count, centr in zip(["Cote d'Ivoire", "Burkina Faso", "Ghana"], [cid_centroid, bf_centroid, ghana_centroid]):
        ax.text(centr[0], centr[1], count.upper(), transform=HigherResPlateCarree(),
                fontsize=12, c='black', horizontalalignment='center', verticalalignment='center')
    fig.suptitle("Floods in Côte d'Ivoire and surrounding areas \n between Aug. 1988 and June 2020 \n"
                 r"(bullet sizes correspond to affected area in $km^2$)")
    plt.tight_layout()
    return fig

