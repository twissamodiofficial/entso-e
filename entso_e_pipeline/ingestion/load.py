import os

from dotenv import load_dotenv
from entsoe import EntsoePandasClient

from .. import config
from ..time_utils import as_local

load_dotenv()


def _client():
    return EntsoePandasClient(api_key=os.environ.get("ENTSOE_API_KEY"))


def fetch_load(start, end, client=None):
    """Fetch source-resolution readings; leave cleaning to preprocessing."""

    if client is None:
        client = _client()
    start_local, end_local = as_local(start, config.TIMEZONE), as_local(end, config.TIMEZONE)
    if start_local >= end_local:
        raise ValueError("Load start must be before end.")
    raw = client.query_load(
        config.COUNTRY_CODE,
        start=start_local,
        end=end_local,
    )
    # EntsoePandasClient already returns an aware Amsterdam index for NL.
    return raw.loc[(raw.index >= start_local) & (raw.index < end_local)].copy()
