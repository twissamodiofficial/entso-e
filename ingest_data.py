# ingest_data.py
import os
import time
from dotenv import load_dotenv
import pandas as pd
from entsoe import EntsoePandasClient


load_dotenv()

API_KEY = os.environ.get("ENTSOE_API_KEY")

client = EntsoePandasClient(api_key=API_KEY)
country_code = "NL"
OUT_DIR = "data/raw"
os.makedirs(OUT_DIR, exist_ok=True)

# train: 2019-01-01 through end of 2025 
# val:   2026-01-01 through end of March 2026
# test:  2026-04-01 through end of June 2026
SPLITS = {
    "train": [
        (pd.Timestamp(f"{y}0101", tz="Europe/Amsterdam"), pd.Timestamp(f"{y+1}0101", tz="Europe/Amsterdam"))
        for y in range(2019, 2026)
    ],
    "val": [
        (pd.Timestamp("20260101", tz="Europe/Amsterdam"), pd.Timestamp("20260401", tz="Europe/Amsterdam")),
    ],
    "test": [
        (pd.Timestamp("20260401", tz="Europe/Amsterdam"), pd.Timestamp("20260701", tz="Europe/Amsterdam")),
    ],
}


def pull_range(start, end):
    print(f"  pulling {start.date()} to {end.date()}...")
    load = client.query_load(country_code, start=start, end=end)
    time.sleep(1)
    return load


for split_name, ranges in SPLITS.items():
    print(f"\n{split_name.upper()}")
    chunks = []
    for start, end in ranges:
        try:
            chunks.append(pull_range(start, end))
        except Exception as e:
            print(f"  FAILED {start.date()}-{end.date()}: {e}")

    if not chunks:
        print(f"  no data pulled for {split_name}, skipping file write")
        continue

    combined = pd.concat(chunks)
    out_path = f"{OUT_DIR}/load_{split_name}.csv"
    combined.to_csv(out_path)
    print(f"  {split_name}: {len(combined)} rows -> {out_path}")