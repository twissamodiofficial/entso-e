"""Fetch historical source observations into Supabase."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .. import config
from ..ingestion.load import fetch_load
from ..ingestion.weather import pull_historical_weather
from ..storage.supabase import SupabaseRawStore
from ..time_utils import as_local


FrameFetcher = Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame]


@dataclass(frozen=True)
class BackfillSpec:
    """The half-open local windows required by offline training and testing."""

    train_start: pd.Timestamp = pd.Timestamp("2019-01-01", tz=config.TIMEZONE)
    test_end: pd.Timestamp = pd.Timestamp("2026-09-16", tz=config.TIMEZONE)
    load_history_days: int = config.LOOKBACK_DAYS
    chunk_days: int = 14

    @property
    def load_start(self) -> pd.Timestamp:
        return self.train_start - pd.DateOffset(days=self.load_history_days)

    def __post_init__(self):
        for name in ("train_start", "test_end"):
            object.__setattr__(self, name, as_local(getattr(self, name), config.TIMEZONE))
        if self.train_start >= self.test_end:
            raise ValueError("Backfill train_start must precede test_end.")
        if self.load_history_days < config.LOOKBACK_DAYS:
            raise ValueError("Backfill load history must cover the feature lookback.")
        if self.chunk_days <= 0:
            raise ValueError("Backfill chunk_days must be positive.")


def _chunks(start: pd.Timestamp, end: pd.Timestamp, days: int):
    current = start
    while current < end:
        following = min(current + pd.DateOffset(days=days), end)
        yield current, following
        current = following


class BackfillDownloader:
    """Fetch and persist source readings without creating local files."""

    def __init__(
        self,
        spec: BackfillSpec | None = None,
        load_fetcher: FrameFetcher = fetch_load,
        weather_fetcher: FrameFetcher = pull_historical_weather,
        store: SupabaseRawStore | None = None,
    ):
        self.spec = spec or BackfillSpec()
        self.load_fetcher = load_fetcher
        self.weather_fetcher = weather_fetcher
        self.store = store or SupabaseRawStore()

    def _download_load(self) -> int:
        total = 0
        for chunk_start, chunk_end in _chunks(
            self.spec.load_start,
            self.spec.test_end,
            self.spec.chunk_days,
        ):
            print(
                f"fetching load {chunk_start.isoformat()} to {chunk_end.isoformat()}",
                flush=True,
            )
            fetched = self.load_fetcher(chunk_start, chunk_end)
            fetched = fetched.loc[
                (fetched.index >= chunk_start) & (fetched.index < chunk_end)
            ]
            total += self.store.upsert_load(fetched)
        return total

    def _download_weather(self) -> int:
        print(
            f"fetching weather {self.spec.train_start.isoformat()} "
            f"to {self.spec.test_end.isoformat()}",
            flush=True,
        )
        fetched = self.weather_fetcher(self.spec.train_start, self.spec.test_end)
        fetched = fetched.loc[
            (fetched.index >= self.spec.train_start)
            & (fetched.index < self.spec.test_end)
        ]
        return self.store.upsert_weather(fetched)

    def download(self) -> dict[str, int]:
        """Fetch the load warmup and target-period weather into Supabase."""

        return {
            "load_rows": self._download_load(),
            "weather_rows": self._download_weather(),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-days", type=int, default=14)
    args = parser.parse_args(argv)
    result = BackfillDownloader(
        spec=BackfillSpec(chunk_days=args.chunk_days),
    ).download()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
