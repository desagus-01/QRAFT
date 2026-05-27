import polars as pl
from utils.fred import download_fred

i = download_fred()
# %%
x = pl.read_csv("~/Documents/projects/fund/QRAFT/data/cash.csv", try_parse_dates=True)

x.filter(pl.col("date") == pl.col("date").max()).drop("date").to_numpy()
