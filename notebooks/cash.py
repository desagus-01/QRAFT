import polars as pl
from utils.fred import download_fred, save_fred_csv

i = download_fred()
save_fred_csv("~/Documents/projects/fund/qraft/data/cash.csv")

# %%
x = pl.read_csv("~/Documents/projects/fund/QRAFT/data/cash.csv", try_parse_dates=True)
