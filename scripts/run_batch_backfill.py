"""
One-time historical batch pull of train/val/test load + weather data into data/raw/*.csv, per config.SPLITS.
Part of offline training setup.
"""
import sys

from entso_e_pipeline.pipeline import ForecastPipeline

if __name__ == "__main__":
    try:
        pipeline = ForecastPipeline()
        pipeline.ingest()
    except Exception as e:
        print(f"BACKFILL FAILED: {e}")
        sys.exit(1)

    print("Backfill OK: data/raw/*.csv refreshed")
