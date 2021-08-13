from pathlib import Path

import pandas as pd

if __name__ == '__main__':
    data_path = Path(__file__).parent.parent / "data/cattle/FAOSTAT_data_8-10-2021.csv"
    data = pd.read_csv(data_path)
    total_animals = data.groupby(["Area", "Year"]).agg({"Value": 'sum'})["Value"].unstack("Area")
    to_concat = []
    for c, df in data.groupby('Area'):
        df = (df.set_index(["Year", "Item"])["Value"].unstack("Item")
              .assign(total=lambda x: x["Goats"] + x["Cattle"])
              .assign(goat_pct=lambda x: x["Goats"] / x["total"])
              .assign(cattle_pct=lambda x: x["Cattle"] / x["total"])
              .assign(country=c))
        to_concat.append(df)
