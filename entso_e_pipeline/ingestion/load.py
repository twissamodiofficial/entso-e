import os
import time

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

from .. import config

load_dotenv()


def _client():
    return EntsoePandasClient(api_key=os.environ.get("ENTSOE_API_KEY"))


def pull_range(client, start, end):
    print(f"  pulling {start.date()} to {end.date()}...")
    load = client.query_load(config.COUNTRY_CODE, start=start, end=end)
    time.sleep(1)
    return load


def ingest_all(out_dir: str = config.RAW_DIR):
    client = _client()
    os.makedirs(out_dir, exist_ok=True)

    for split_name, ranges in config.SPLITS.items():
        print(f"\n{split_name.upper()}")
        chunks = []
        for start_str, end_str in ranges:
            start = pd.Timestamp(start_str, tz=config.TIMEZONE)
            end = pd.Timestamp(end_str, tz=config.TIMEZONE)
            try:
                chunks.append(pull_range(client, start, end))
            except Exception as e:
                print(f"  FAILED {start.date()}-{end.date()}: {e}")

        if not chunks:
            print(f"  no data pulled for {split_name}, skipping file write")
            continue

        combined = pd.concat(chunks)
        out_path = f"{out_dir}/load_{split_name}.csv"
        combined.to_csv(out_path)
        print(f"  {split_name}: {len(combined)} rows -> {out_path}")
