import gzip
import math
import os
import pathlib
import shutil
from glob import glob
from typing import Sequence

import cartopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wget
import xarray as xr

from data_utils import HigherResPlateCarree


def download_CHIRPS(years: Sequence, path_to_directory: str):
    for year in np.arange(*years, step=1):
        start = pd.to_datetime(f"{year}-06-01")
        end = pd.to_datetime(f"{year}-10-01")
        for d in pd.date_range(start, end):
            dstr = d.strftime('%Y.%m.%d')
            suffix = f'chirps-v2.0.{dstr}.tif.gz'
            target_url = f'https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05/{year}/{suffix}'
            if not os.path.isfile(f"{path_to_directory}/{suffix}"):
                try:
                    wget.download(target_url, path_to_directory)
                    print(f"Downloading url {target_url} to file {path_to_directory}")
                except Exception as err:
                    print(f"     ---> Can't access {target_url}: {err}")
            else:
                print(f"Day {d} of year {year} has already been downloaded to {path_to_directory}")


def get_dataset_for_CHIRPS_rainfall_data(path_to_dir, lon_range=None, lat_range=None, date_range=None):
    def get_data_for_tif(file):
        try:
            ds = xr.open_rasterio(file).isel(band=0, drop=True)
            ds = ds.where(ds > -9999.0)
            ds = ds.sel(y=slice(lat_range[1], lat_range[0])) if lat_range is not None else ds
            ds = ds.sel(x=slice(lon_range[0], lon_range[1])) if lon_range is not None else ds
            da = xr.DataArray(ds,
                              coords=ds.coords, name='precipitation')
            return da
        except:
            print(file)
            return None

    for file in glob(path_to_dir):
        if file.endswith('gz'):
            tif_name = file[:-3]
            date = pd.to_datetime(file.split('/')[-1].replace('chirps-v2.0.', '').replace(".tif.gz", ''))
            if date in date_range:
                with gzip.open(file, 'rb') as zip_ref:
                    with open(tif_name, 'wb') as f_out:
                        shutil.copyfileobj(zip_ref, f_out)
                os.remove(file)
    datasets = []
    for file in glob(path_to_dir):
        if file.endswith('tif'):
            date = pd.to_datetime(file.split('/')[-1].replace('chirps-v2.0.', '').replace(".tif", ''))
            if date in date_range:
                ds = get_data_for_tif(file)
                if ds is not None:
                    ds = ds.expand_dims({"time": [date]})
                    datasets.append(ds.to_dataset())
    data = xr.concat(datasets, dim="time")
    data.to_netcdf(pathlib.Path(pathlib.Path(file).parent, 'rainfall_cid_1.nc'))


def plot_rainfall_for_dates(dates, data: xr.Dataset, plot_path: str):
    global_max = math.ceil(np.nanmax(np.array(data.sel(time=dates, method='nearest').precipitation[:])))
    longitudes = np.array(data.x[:])
    latitudes = np.array(data.y[:])
    for time in dates:
        date = pd.to_datetime(time).strftime('%Y-%m-%d')
        ds = data.sel(time=time).precipitation
        data_to_plot = ds[:]
        subplot_kw = {'projection': HigherResPlateCarree()}
        fig, ax = plt.subplots(ncols=1, figsize=(5, 5), subplot_kw=subplot_kw)
        blue = (0, 0.27450980392, 0.78431372549)
        ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()])
        ax.add_feature(cartopy.feature.LAKES.with_scale('10m'), color=blue)
        ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
        ax.add_feature(cartopy.feature.RIVERS.with_scale('10m'), color=blue)
        ax.add_feature(cartopy.feature.OCEAN.with_scale('10m'), color=blue)
        c_scheme = ax.pcolormesh(longitudes, latitudes, data_to_plot, cmap='Blues',
                                 transform=HigherResPlateCarree(),
                                 vmin=0, vmax=global_max)
        plt.colorbar(c_scheme, location='bottom', pad=0.05,
                     label="mm/day", ax=ax)
        ax.set_title(f"CHIRPS rainfall data in Côte d'Ivoire on date {date}", fontsize=10)
        ax.scatter(x=-4.024429, y=5.345317, s=20, c="black")
        ax.text(-4.024429, 5.345317 + 0.3, 'Abidjan, CID', transform=HigherResPlateCarree(),
                fontsize=10, fontweight='semibold', c='black', horizontalalignment='center', verticalalignment='center')
        fig.tight_layout()
        fig.savefig(f'{plot_path}/{date}.png')
        plt.clf()
