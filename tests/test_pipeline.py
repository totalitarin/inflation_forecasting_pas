"""Проверки временной причинности, единиц измерения и целостности данных."""
import unittest
import numpy as np
import pandas as pd
from inflation_pas.data import validate, make_features, parse_cpi_html
from inflation_pas.analysis import evaluate, interval


def fixture():
    rng = np.random.default_rng(42)
    return pd.DataFrame({'date': pd.date_range('2010-01-01', periods=96, freq='MS'),
                         'cpi_mom': rng.normal(0.5, 0.2, 96), 'usd_rub_mom': rng.normal(0, 2, 96)})


class PipelineTests(unittest.TestCase):
    def test_invalid_history(self):
        for frame in [fixture().drop(index=20), pd.concat([fixture(), fixture().iloc[[0]]]),
                      fixture().assign(cpi_mom=100.5), fixture().assign(cpi_mom=np.nan)]:
            with self.assertRaises(ValueError):
                validate(frame)

    def test_future_cannot_change_past_features(self):
        original = validate(fixture())
        modified = original.copy()
        cutoff = original.index[60]
        modified.loc[cutoff:, :] += 5
        x, _, _ = make_features(original)
        z, _, _ = make_features(modified)
        pd.testing.assert_frame_equal(x.loc[:cutoff], z.loc[:cutoff])
        self.assertEqual(x.loc[cutoff, 'cpi_lag_12'], original.cpi_mom.iloc[48])

    def test_forecast_month_and_lag(self):
        original = validate(fixture())
        _, _, future = make_features(original)
        self.assertEqual(future.index[0], original.index[-1] + pd.offsets.MonthBegin())
        self.assertEqual(future.cpi_lag_1.iloc[0], original.cpi_mom.iloc[-1])

    def test_html_units(self):
        html = '<table><tr><th>Месяц</th><th>2020</th></tr><tr><td>январь</td><td>100,4</td></tr></table>'
        self.assertAlmostEqual(parse_cpi_html(html).cpi_mom.iloc[0], 0.4)

    def test_outer_test_does_not_change_previous_predictions(self):
        original = validate(fixture())
        x, y, _ = make_features(original)
        first, _ = evaluate(x, y, original.cpi_mom, 24)
        changed = original.copy()
        changed.iloc[-1, 0] += 3
        z, target, _ = make_features(changed)
        second, _ = evaluate(z, target, changed.cpi_mom, 24)
        np.testing.assert_allclose(first.prediction, second.prediction)
        np.testing.assert_allclose(first.lower_95, second.lower_95, equal_nan=True)
        for _, group in first.groupby("model"):
            self.assertTrue(group.lower_95.iloc[:12].isna().all())
            self.assertTrue(group.lower_95.iloc[12:].notna().all())

    def test_intervals_need_past_errors(self):
        self.assertTrue(np.isnan(interval([1, 2], 0, .95)[0]))
        low, high = interval(list(range(-6, 6)), 1, .95)
        self.assertLessEqual(low, 1)
        self.assertGreaterEqual(high, 1)


if __name__ == '__main__':
    unittest.main()
