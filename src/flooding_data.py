import os
from typing import Sequence

import numpy as np
import pandas as pd
import wget


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
