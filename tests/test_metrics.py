import unittest

import numpy as np

from entso_e_pipeline.modeling.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    pinball_loss,
    point_metrics,
    quantile_metrics,
    interval_coverage,
    conformal_interval_adjustment,
)


class MetricsTests(unittest.TestCase):
    def test_point_metrics_report_mae_and_percentage_mape(self):
        actual = np.array([100.0, 200.0])
        predicted = np.array([90.0, 220.0])

        self.assertEqual(mean_absolute_error(actual, predicted), 15.0)
        self.assertEqual(mean_absolute_percentage_error(actual, predicted), 10.0)
        self.assertEqual(
            point_metrics(actual, predicted),
            {"mae": 15.0, "mape_percent": 10.0},
        )

    def test_quantile_metrics_include_pinball_loss(self):
        actual = np.array([100.0, 200.0])
        predicted = np.array([90.0, 220.0])

        result = quantile_metrics(actual, predicted, 0.5)

        self.assertEqual(result["mae"], 15.0)
        self.assertEqual(result["mape_percent"], 10.0)
        self.assertEqual(result["pinball_loss"], 7.5)
        self.assertEqual(pinball_loss(actual, predicted, 0.5), 7.5)

    def test_mape_rejects_zero_actuals(self):
        with self.assertRaisesRegex(ValueError, "zero values"):
            mean_absolute_percentage_error([0.0], [1.0])

    def test_conformal_adjustment_is_fit_on_calibration_only(self):
        actual = np.array([0.0, 10.0, 20.0, 30.0])
        lower = np.array([2.0, 8.0, 18.0, 25.0])
        upper = np.array([8.0, 12.0, 22.0, 26.0])

        adjustment = conformal_interval_adjustment(actual, lower, upper, 0.75)

        self.assertEqual(adjustment, 4.0)
        self.assertEqual(
            interval_coverage(actual, lower - adjustment, upper + adjustment),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
