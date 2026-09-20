import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from entso_e_pipeline import config
from entso_e_pipeline.ingestion import weather
from entso_e_pipeline.time_utils import local_day_hours


class WeatherAdapterTests(unittest.TestCase):
    def response_for(self, params):
        index = pd.date_range(
            pd.Timestamp(params["start_date"], tz="UTC"),
            pd.Timestamp(params["end_date"], tz="UTC") + pd.Timedelta(days=1),
            freq="h", inclusive="left",
        )
        columns = params["hourly"].split(",")
        payload = {
            "hourly": {
                "time": index.strftime("%Y-%m-%dT%H:%M").tolist(),
                **{column: [10. if column.startswith("apparent") else 5.] * len(index)
                   for column in columns},
                # Latest-run fields must never be substituted for the selected
                # historical forecast fields, even if the API includes them.
                "apparent_temperature": [999.] * len(index),
                "dew_point_2m": [999.] * len(index),
            },
            "hourly_units": {column: "°C" for column in columns},
        }
        response = Mock()
        response.json.return_value = payload
        return response

    def api_response(self, url, *, params, timeout):
        return self.response_for(params)

    @patch("entso_e_pipeline.ingestion.weather.requests.get")
    def test_forecast_pins_model_lead_units_and_returns_half_open_local_window(self, get):
        get.side_effect = self.api_response
        result = weather.pull_forecast("2026-09-18 10:00", "2026-09-18 12:00")
        url = get.call_args.args[0]
        params = get.call_args.kwargs["params"]
        self.assertEqual(url, "https://previous-runs-api.open-meteo.com/v1/forecast")
        self.assertEqual(params["models"], "jma_gsm")
        self.assertEqual(params["hourly"],
                         "apparent_temperature_previous_day2,dew_point_2m_previous_day2")
        self.assertEqual(params["timezone"], "UTC")
        self.assertEqual(params["temperature_unit"], "celsius")
        self.assertEqual(params["start_date"], "2026-09-18")
        self.assertEqual(params["end_date"], "2026-09-18")
        expected = pd.date_range("2026-09-18 10:00", periods=2, freq="h", tz=config.TIMEZONE)
        self.assertTrue(result.index.equals(expected))
        self.assertEqual(result.columns.tolist(), config.WEATHER_FEATURES)
        self.assertEqual(result["apparent_temperature"].tolist(), [10., 10.])
        self.assertEqual(result["dew_point_2m"].tolist(), [5., 5.])
        self.assertEqual(result.attrs["weather_model"], "jma_gsm")
        self.assertEqual(result.attrs["weather_forecast_previous_days"], 2)

    @patch("entso_e_pipeline.ingestion.weather.requests.get")
    def test_historical_and_live_entry_points_use_identical_requests_and_features(self, get):
        get.side_effect = self.api_response
        historical = weather.pull_historical_weather("2019-01-10", "2019-01-11")
        historical_calls = get.call_args_list.copy()
        get.reset_mock()
        live = weather.pull_forecast("2019-01-10", "2019-01-11")
        self.assertEqual(historical_calls, get.call_args_list)
        pd.testing.assert_frame_equal(historical, live)
        self.assertEqual(historical.attrs, live.attrs)

    @patch("entso_e_pipeline.ingestion.weather.requests.get")
    def test_utc_requests_preserve_all_real_hours_across_dst(self, get):
        get.side_effect = self.api_response
        for day, hours in (("2026-03-29", 23), ("2026-10-25", 25)):
            with self.subTest(day=day):
                horizon = local_day_hours(day, config.TIMEZONE)
                end = horizon[0] + pd.DateOffset(days=1)
                result = weather.pull_forecast(horizon[0], end)
                self.assertTrue(result.index.equals(horizon))
                self.assertEqual(len(result), hours)
                self.assertTrue(result.index.is_unique)
                params = get.call_args.kwargs["params"]
                self.assertEqual(params["start_date"], horizon[0].tz_convert("UTC").date().isoformat())
                self.assertEqual(params["end_date"], horizon[-1].tz_convert("UTC").date().isoformat())
                if hours == 25:
                    repeated = result.index[result.index.hour == 2]
                    self.assertEqual(len(repeated), 2)
                    self.assertNotEqual(repeated[0].utcoffset(), repeated[1].utcoffset())

    @patch("entso_e_pipeline.ingestion.weather.requests.get")
    def test_backfill_chunks_utc_dates_without_gaps_or_overlaps(self, get):
        get.side_effect = self.api_response
        start = pd.Timestamp("2026-03-15", tz=config.TIMEZONE)
        end = pd.Timestamp("2026-04-20", tz=config.TIMEZONE)
        result = weather.pull_historical_weather(start, end)
        expected = pd.date_range(start, end, freq="h", inclusive="left")
        self.assertTrue(result.index.equals(expected))
        self.assertTrue(result.index.is_unique)
        self.assertEqual(get.call_count, 3)
        previous_end = None
        for call in get.call_args_list:
            params = call.kwargs["params"]
            request_start = pd.Timestamp(params["start_date"])
            request_end = pd.Timestamp(params["end_date"])
            self.assertLessEqual((request_end - request_start).days + 1, 14)
            if previous_end is not None:
                self.assertEqual(request_start, previous_end + pd.Timedelta(days=1))
            previous_end = request_end

    @patch("entso_e_pipeline.ingestion.weather.requests.get")
    def test_rejects_leads_that_do_not_leave_a_cutoff_buffer(self, get):
        for days in (0, 1, 8, 2.5, True):
            with self.subTest(days=days), patch.object(config, "WEATHER_FORECAST_PREVIOUS_DAYS", days):
                with self.assertRaisesRegex(ValueError, "2 to 7 days"):
                    weather.pull_forecast("2026-01-10", "2026-01-11")
        get.assert_not_called()

    @patch("entso_e_pipeline.ingestion.weather.requests.get")
    def test_invalid_bounds_are_rejected_before_request(self, get):
        for start, end in (
            ("2026-01-10", "2026-01-10"),
            ("2026-01-11", "2026-01-10"),
            ("2026-01-10 00:15", "2026-01-11"),
            ("2026-01-10", "2026-01-11 00:15"),
            (pd.NaT, "2026-01-11"),
        ):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                weather.pull_forecast(start, end)
        get.assert_not_called()

    @patch("entso_e_pipeline.ingestion.weather.requests.get")
    def test_missing_forecast_column_never_falls_back_to_latest_values(self, get):
        def response(url, *, params, timeout):
            result = self.response_for(params)
            del result.json.return_value["hourly"]["apparent_temperature_previous_day2"]
            return result
        get.side_effect = response
        with self.assertRaisesRegex(ValueError, "missing forecast columns"):
            weather.pull_forecast("2026-01-10", "2026-01-11")
        self.assertEqual(get.call_count, 1)

    @patch("entso_e_pipeline.ingestion.weather.requests.get")
    def test_missing_hour_is_preserved_for_preprocessing(self, get):
        def response(url, *, params, timeout):
            result = self.response_for(params)
            for values in result.json.return_value["hourly"].values():
                del values[25]
            return result
        get.side_effect = response
        result = weather.pull_forecast("2026-01-10", "2026-01-11")
        self.assertEqual(len(result), 23)

    @patch("entso_e_pipeline.ingestion.weather.requests.get")
    def test_unavailable_or_invalid_forecast_values_reach_preprocessing(self, get):
        for value in (None, np.nan, np.inf, -np.inf, "bad"):
            with self.subTest(value=value):
                def response(url, *, params, timeout):
                    result = self.response_for(params)
                    result.json.return_value["hourly"]["dew_point_2m_previous_day2"][25] = value
                    return result
                get.side_effect = response
                result = weather.pull_forecast("2026-01-10", "2026-01-11")
                reading = result.loc[pd.Timestamp("2026-01-10 02:00", tz=config.TIMEZONE), "dew_point_2m"]
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    self.assertTrue(pd.isna(reading))
                else:
                    self.assertEqual(reading, value)

    @patch("entso_e_pipeline.ingestion.weather.requests.get")
    def test_wrong_or_missing_units_are_errors(self, get):
        for unit in ("°F", None):
            with self.subTest(unit=unit):
                def response(url, *, params, timeout):
                    result = self.response_for(params)
                    result.json.return_value["hourly_units"]["dew_point_2m_previous_day2"] = unit
                    return result
                get.side_effect = response
                with self.assertRaisesRegex(ValueError, "Celsius"):
                    weather.pull_forecast("2026-01-10", "2026-01-11")

    @patch("entso_e_pipeline.ingestion.weather.requests.get")
    def test_http_failure_propagates_without_switching_sources(self, get):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("429 rate limit")
        get.return_value = response
        with self.assertRaises(requests.HTTPError):
            weather.pull_forecast("2026-01-10", "2026-01-11")
        self.assertEqual(get.call_count, 1)
        response.json.assert_not_called()

    @patch("entso_e_pipeline.ingestion.weather.requests.get")
    def test_missing_hourly_payload_is_an_error(self, get):
        response = Mock()
        response.json.return_value = {"error": True, "reason": "unavailable"}
        get.return_value = response
        with self.assertRaisesRegex(ValueError, "hourly payload"):
            weather.pull_forecast("2026-01-10", "2026-01-11")


if __name__ == "__main__":
    unittest.main()
