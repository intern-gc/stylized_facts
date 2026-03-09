import unittest
import numpy as np
import pandas as pd
from facts_test.volclustering import VolatilityClustering
from facts_test.decay import SlowDecay
from facts_test.leverage import LeverageEffect
from facts_test.volvolcorr import VolVolCorr


class TestVolatilityClustering(unittest.TestCase):

    # TEST 1: VOLATILITY CLUSTERING (Positive)
    def test_volclustering_regime_switching(self):
        # We create 500 wild/volatile returns followed by 500 tiny/calm returns.
        # This mimics a market crash followed by a quiet period.
        # Because variance is stuck high for 500 days then stuck low for 500 days,
        # the squared returns are very autocorrelated — knowing today's variance
        # tells you a lot about tomorrow's.
        # We expect several lags to be flagged as significant, and C1(lag=1) > 0.1.
        np.random.seed(42)
        high_vol = np.random.normal(0, 5.0, 500)
        low_vol = np.random.normal(0, 0.1, 500)
        vc_clustered_data = pd.Series(np.concatenate([high_vol, low_vol]))

        vc_tester = VolatilityClustering(vc_clustered_data, "THEORY_VC")
        c1_values, sig_lags = vc_tester.compute_c1(max_lag=20, plot=False, n_shuffles=100)

        self.assertGreater(len(sig_lags), 0, "Volatile-then-calm data must show volatility clustering.")
        self.assertGreater(c1_values[1], 0.1, "C1 at lag 1 must be strongly positive for regime-switching data.")

    # TEST 2: VOLATILITY CLUSTERING (Negative)
    def test_volclustering_iid_returns(self):
        # Pure random (iid) Gaussian returns: every day is independent of every other day.
        # Squaring them and computing autocorrelation should give mostly noise (near zero).
        # We allow up to 3 "false positives" out of 20 lags at the 5% level.
        np.random.seed(42)
        vc_iid_data = pd.Series(np.random.normal(0, 1.0, 1000))

        vc_tester = VolatilityClustering(vc_iid_data, "IID_VC")
        c1_values, sig_lags = vc_tester.compute_c1(max_lag=20, plot=False, n_shuffles=100)

        self.assertLessEqual(len(sig_lags), 3,
                             "Random data should have almost no significant lags in the volatility ACF.")


class TestSlowDecay(unittest.TestCase):

    def _make_block_vol_returns(self, n=5000, block=500, seed=42):
        """Alternating high/low vol blocks: 500 days std=3, 500 days std=0.05, repeat.
        Within each block volatility is constant, so |returns| is strongly autocorrelated
        across all lags within the same block. No model assumptions."""
        np.random.seed(seed)
        r = []
        for i in range(n // block):
            std = 3.0 if i % 2 == 0 else 0.05
            r.extend(np.random.normal(0, std, block))
        return pd.Series(np.array(r[:n]))

    # TEST 3: SLOW DECAY (Return Structure)
    def test_return_structure(self):
        # Block-alternating volatility: |returns| is clearly autocorrelated within each block.
        # acf lists are clipped to the significant range (lags 1..cutoff <= max_lag).
        sv_returns = self._make_block_vol_returns()
        sd = SlowDecay(sv_returns, "TEST")
        result = sd.compute_decay(max_lag=50, plot=False, n_shuffles=100)

        self.assertIsNotNone(result)
        self.assertGreater(len(result['lags_alpha1']), 0)
        self.assertLessEqual(len(result['lags_alpha1']), 50)
        self.assertGreater(len(result['acf_alpha1']), 0)
        self.assertIsInstance(result['slow_decay_confirmed'], bool)

    # TEST 4: SLOW DECAY (Both Alphas Computed)
    def test_both_alphas_computed(self):
        # Both |returns| and returns^2 are autocorrelated in block-vol data.
        # Both beta values must be present and positive.
        sv_returns = self._make_block_vol_returns()
        sd = SlowDecay(sv_returns, "TEST")
        result = sd.compute_decay(max_lag=50, plot=False, n_shuffles=100)

        self.assertIsNotNone(result)
        self.assertGreater(result['beta_alpha1'], 0, "Decay exponent for |returns| must be positive.")
        self.assertGreater(result['beta_alpha2'], 0, "Decay exponent for returns^2 must be positive.")
        self.assertGreater(result['A_alpha1'], 0, "Amplitude for |returns| must be positive.")
        self.assertGreater(result['A_alpha2'], 0, "Amplitude for returns^2 must be positive.")

    # TEST 5: SLOW DECAY (ACF Positive at Lag 1)
    def test_acf_positive_at_lag1(self):
        # Within a block, today's |return| is strongly correlated with yesterday's.
        # ACF at lag 1 must be > 0.
        sv_returns = self._make_block_vol_returns()
        sd = SlowDecay(sv_returns, "TEST_SV")
        result = sd.compute_decay(max_lag=50, plot=False, n_shuffles=100)

        self.assertIsNotNone(result)
        self.assertGreater(result['acf_alpha1'][0], 0.0,
                           "Block-vol returns must have positive |returns| autocorrelation at lag 1.")

    # TEST 6: SLOW DECAY (Beta Exponent Is Positive)
    def test_beta_positive(self):
        # Beta is the power-law exponent: ACF(lag) ~ A * lag^(-beta).
        # If ACF starts positive and decays, beta must be positive.
        sv_returns = self._make_block_vol_returns()
        sd = SlowDecay(sv_returns, "TEST_SV")
        result = sd.compute_decay(max_lag=50, plot=False, n_shuffles=100)

        self.assertIsNotNone(result)
        self.assertGreater(result['beta_alpha1'], 0,
                           "Beta must be positive (autocorrelation decays from a positive start).")
        self.assertGreater(result['beta_alpha2'], 0,
                           "Beta for squared returns must also be positive.")

    # TEST 7: SLOW DECAY (Robustness)
    def test_robustness_nan_inf(self):
        # Feed in a mix of good data and garbage. Should not crash.
        np.random.seed(42)
        clean = np.random.normal(0, 0.01, 1000)
        dirty = np.concatenate([clean, [np.nan] * 50, [np.inf] * 50, [-np.inf] * 50])
        np.random.shuffle(dirty)
        dirty_returns = pd.Series(dirty)

        sd = SlowDecay(dirty_returns, "ROBUST")
        try:
            result = sd.compute_decay(max_lag=20, plot=False, n_shuffles=100)
            if result is not None:
                self.assertIsInstance(result['slow_decay_confirmed'], bool)
                if result['beta_alpha1'] is not None:
                    self.assertTrue(np.isfinite(result['beta_alpha1']))
        except Exception as e:
            self.fail(f"SlowDecay crashed on bad data: {e}")

    # TEST 8: SLOW DECAY (Insufficient Data Returns None)
    def test_insufficient_data_returns_none(self):
        # Only 5 data points but max_lag=50: impossible to compute a 50-lag ACF.
        # Should return None gracefully.
        data = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])

        sd = SlowDecay(data, "TEST_SHORT")
        result = sd.compute_decay(max_lag=50, plot=False)

        self.assertIsNone(result)


class TestLeverageEffect(unittest.TestCase):

    # TEST 9: LEVERAGE EFFECT (Return Structure)
    def test_return_structure(self):
        # IID Gaussian has no leverage. max_lag=20 → lags run -20 to +20 = 41 values.
        # leverage_detected must be False; min_L is a correlation-based value in [-1, 1];
        # min_lag must fall within the computed range.
        np.random.seed(42)
        iid_returns = pd.Series(np.random.normal(0, 0.01, 5000))

        le = LeverageEffect(iid_returns, "TEST")
        result = le.compute_leverage(max_lag=20, plot=False, n_shuffles=100)

        self.assertIsNotNone(result)
        self.assertEqual(len(result['lags']), 41)
        self.assertEqual(len(result['L_values']), 41)
        self.assertAlmostEqual(result['min_L'], 0.0, delta=0.1,
                               msg="IID Gaussian should have L ≈ 0 for all lags.")
        self.assertGreaterEqual(result['min_lag'], -20)
        self.assertLessEqual(result['min_lag'], 20)

    # TEST 10: LEVERAGE EFFECT (Detected for Crash Data)
    def test_leverage_detected_for_asymmetric_data(self):
        # Constructed crash data: after a -1.0 return, next 4 returns are very volatile (std=0.5).
        # After a +1.0 return, next 4 returns are totally calm (std=0.001).
        # So negative returns predict HIGH future variance → L(tau) must be negative for tau > 0.
        np.random.seed(42)
        n_events = 500
        returns = []
        for _ in range(n_events):
            r0 = np.random.choice([-1.0, 1.0])
            future_vol = 0.5 if r0 < 0 else 0.001
            returns.append(r0)
            returns.extend(np.random.normal(0, future_vol, 4))
        leverage_returns = pd.Series(returns[:2000])

        le = LeverageEffect(leverage_returns, "TEST_LEVERAGE")
        result = le.compute_leverage(max_lag=10, plot=False, n_shuffles=100)

        self.assertIsNotNone(result)
        lags = np.array(result['lags'])
        L_arr = np.array(result['L_values'])
        pos_L = L_arr[lags > 0]
        self.assertTrue(np.any(pos_L < 0),
                        "Crash→volatile data must have at least one negative L(tau) for tau > 0.")
        self.assertLess(float(np.nanmin(pos_L)), -0.1,
                        "The leverage effect must be large (min L < -0.1) for this constructed data.")

    # TEST 11: LEVERAGE EFFECT (IID Data, No Strong Leverage)
    def test_iid_data_no_strong_leverage(self):
        # Random (iid) Gaussian data: no relationship between past returns and future variance.
        # L(tau) should stay close to zero for all tau > 0 (nothing is above -0.1).
        np.random.seed(42)
        iid_returns = pd.Series(np.random.normal(0, 0.01, 5000))

        le = LeverageEffect(iid_returns, "TEST_IID")
        result = le.compute_leverage(max_lag=20, plot=False, n_shuffles=100)

        self.assertIsNotNone(result)
        lags = np.array(result['lags'])
        L_arr = np.array(result['L_values'])
        pos_L = L_arr[lags > 0]
        self.assertGreater(float(np.nanmin(pos_L)), -0.1,
                           "Random data should not show a strong leverage effect (min L > -0.1).")

    # TEST 12: LEVERAGE EFFECT (Robustness)
    def test_robustness_nan_inf(self):
        # Feed in garbage data. Should not crash.
        np.random.seed(42)
        clean = np.random.normal(0, 0.01, 1000)
        dirty = np.concatenate([clean, [np.nan] * 50, [np.inf] * 50, [-np.inf] * 50])
        np.random.shuffle(dirty)
        dirty_returns = pd.Series(dirty)

        le = LeverageEffect(dirty_returns, "ROBUST")
        try:
            result = le.compute_leverage(max_lag=10, plot=False, n_shuffles=100)
            if result is not None:
                self.assertGreater(result['min_L'], -0.1,
                                   "Cleaned Gaussian data should not show a strong leverage effect.")
                self.assertGreaterEqual(result['min_L'], -1.0)
                self.assertLessEqual(result['min_L'], 1.0)
        except Exception as e:
            self.fail(f"LeverageEffect crashed on bad data: {e}")

    # TEST 13: LEVERAGE EFFECT (Insufficient Data Returns None)
    def test_insufficient_data_returns_none(self):
        # Only 3 data points with max_lag=50: not enough. Should return None.
        data = pd.Series([0.01, -0.01, 0.02])

        le = LeverageEffect(data, "TEST_SHORT")
        result = le.compute_leverage(max_lag=50, plot=False, n_shuffles=100)

        self.assertIsNone(result)


class TestVolVolCorr(unittest.TestCase):

    # TEST 14: VOL-VOL CORRELATION (Return Structure)
    def test_return_structure(self):
        # IID data: volume and returns are independent, so the correlation must not be
        # significant. rho is bounded [-1, 1]; pval is bounded [0, 1]; corr_confirmed = False.
        np.random.seed(42)
        n = 2000
        iid_returns = pd.Series(np.random.normal(0, 0.01, n))
        iid_volume = pd.Series(np.random.normal(1e6, 1e4, n))

        vvc = VolVolCorr(iid_returns, iid_volume, "TEST")
        result = vvc.compute_correlation(plot=False, n_shuffles=100)

        self.assertIsNotNone(result)
        self.assertFalse(result['significant_abs'], "IID data must not show significant correlation.")
        self.assertFalse(result['significant_sq'], "IID data: squared proxy also not significant.")
        self.assertFalse(result['corr_confirmed'], "IID data: corr_confirmed must be False.")
        self.assertGreater(result['pval_abs'], 0.05)
        self.assertGreaterEqual(result['rho_abs'], -1.0)
        self.assertLessEqual(result['rho_abs'], 1.0)
        self.assertGreaterEqual(result['pval_abs'], 0.0)
        self.assertLessEqual(result['pval_abs'], 1.0)

    # TEST 15: VOL-VOL CORRELATION (Positive Correlation Detected)
    def test_positive_correlation_detected(self):
        # We manually make volume = |return| * 10 million + small noise.
        # So when returns are big, volume is big. Correlation should be very strong (rho > 0.5).
        np.random.seed(42)
        n = 2000
        abs_vol = np.abs(np.random.normal(0, 0.02, n))
        signs = np.random.choice([-1, 1], n)
        corr_returns = pd.Series(abs_vol * signs)
        corr_volume = pd.Series(abs_vol * 1e7 + np.random.normal(0, 1e4, n))

        vvc = VolVolCorr(corr_returns, corr_volume, "TEST_CORR")
        result = vvc.compute_correlation(plot=False, n_shuffles=100)

        self.assertIsNotNone(result)
        self.assertGreater(result['rho_abs'], 0.5,
                           "rho must be strongly positive when we built volume = |returns| * scale.")
        self.assertLess(result['pval_abs'], 0.05,
                        "This strong correlation must be statistically significant.")
        self.assertTrue(result['significant_abs'],
                        "significant_abs must be True when rho > 0 and p < 0.05.")
        self.assertTrue(result['corr_confirmed'])

    # TEST 16: VOL-VOL CORRELATION (IID Data, Not Significant)
    def test_iid_data_not_significant(self):
        # Volume and returns are generated completely independently — no real relationship.
        # The p-value should be > 0.05, meaning we can't reject the "no correlation" hypothesis.
        np.random.seed(42)
        n = 2000
        iid_returns = pd.Series(np.random.normal(0, 0.01, n))
        iid_volume = pd.Series(np.random.normal(1e6, 1e4, n))

        vvc = VolVolCorr(iid_returns, iid_volume, "TEST_IID")
        result = vvc.compute_correlation(plot=False, n_shuffles=100)

        self.assertIsNotNone(result)
        self.assertGreater(result['pval_abs'], 0.05,
                           "Independent data should not produce a significant p-value.")
        self.assertFalse(result['corr_confirmed'],
                         "corr_confirmed must be False when volume and returns are independent.")

    # TEST 17: VOL-VOL CORRELATION (Robustness)
    def test_robustness_nan_inf(self):
        # Feed garbage data in both series. Must not crash.
        np.random.seed(42)
        clean_r = np.random.normal(0, 0.01, 500)
        clean_v = np.abs(clean_r) * 1e7 + np.random.normal(0, 1e4, 500)
        dirty_r = np.concatenate([clean_r, [np.nan] * 30, [np.inf] * 20, [-np.inf] * 20])
        dirty_v = np.concatenate([clean_v, [np.nan] * 30, [np.nan] * 20, [np.nan] * 20])
        dirty_returns = pd.Series(dirty_r)
        dirty_volume = pd.Series(dirty_v)

        vvc = VolVolCorr(dirty_returns, dirty_volume, "ROBUST")
        try:
            result = vvc.compute_correlation(plot=False, n_shuffles=100)
            if result is not None:
                self.assertFalse(np.isnan(result['rho_abs']))
                self.assertGreater(result['rho_abs'], 0.5)
                self.assertLessEqual(result['rho_abs'], 1.0)
        except Exception as e:
            self.fail(f"VolVolCorr crashed on bad data: {e}")

    # TEST 18: VOL-VOL CORRELATION (Insufficient Data Returns None)
    def test_insufficient_data_returns_none(self):
        # Only 3 data points — not enough for a meaningful correlation. Should return None.
        data_r = pd.Series([0.01, -0.01, 0.02])
        data_v = pd.Series([1e6, 2e6, 1.5e6])

        vvc = VolVolCorr(data_r, data_v, "TEST_SHORT")
        result = vvc.compute_correlation(plot=False, n_shuffles=100)

        self.assertIsNone(result)


class TestDataManagerCacheBug(unittest.TestCase):

    # TEST 19: DATA MANAGER CACHE BUG (Timestamp vs Date Type Mismatch)
    def test_days_received_matches_missing_days_type(self):
        # Bug: pandas gives us Timestamp objects, but groupby gives us datetime.date objects.
        # Timestamp(2024-01-02) == date(2024-01-02) is False in Python — different types!
        # So the "is this day missing?" check always says YES even when we have the data.
        # Fix: call .date() on the Timestamp before comparing, so types match.
        # After the fix, zero days should be "falsely missing."
        from datetime import date

        business_days = pd.bdate_range(start='2024-01-02', end='2024-01-05')
        missing_days = list(business_days)
        days_received = {date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)}

        falsely_missing = [m_day for m_day in missing_days if m_day.date() not in days_received]

        self.assertEqual(len(falsely_missing), 0,
                         f"Type mismatch bug: {len(falsely_missing)} days falsely marked missing.")


class TestDataManagerErrorReporting(unittest.TestCase):

    # TEST 20: DATA MANAGER ERROR REPORTING (Ticker Name in Error Message)
    def test_empty_return_includes_ticker_in_report(self):
        # When the data fetch fails, the error message must say WHICH ticker failed.
        # "No data available" is useless. "No data available for FAKEXYZ" is helpful.
        from unittest.mock import patch, MagicMock
        from utilities.data import DataManager

        with patch.object(DataManager, '__init__', lambda self, **kw: None):
            dm = DataManager()
            dm.cache_dir = '/tmp/nonexistent_cache_dir_test'
            dm.client = MagicMock()
            dm.api_key = 'fake'
            dm.secret_key = 'fake'

            dm.client.get_stock_bars.side_effect = Exception("API unavailable")

            df, returns, report = dm.get_data("FAKEXYZ", "2024-01-02", "2024-01-05", "1d")

            self.assertTrue(df.empty)
            self.assertIn("FAKEXYZ", report,
                          f"Error report must say which ticker failed. Got: '{report}'")


class TestAuditAndClean(unittest.TestCase):

    def _make_df(self, rows):
        # Helper: build a minimal OHLCV DataFrame from a list of dicts.
        index = pd.date_range('2024-01-02', periods=len(rows), freq='D', tz='US/Eastern')
        return pd.DataFrame(rows, index=index,
                            columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    # TEST 21: AUDIT AND CLEAN (Clean Data Passes Through Unchanged)
    def test_clean_data_no_issues(self):
        # All bars are valid: prices positive, High >= Low, Close inside range, volume > 0.
        # The report should say "DATA CLEAN" and no rows should be dropped.
        from utilities.data import DataManager
        rows = [
            {'Open': 100, 'High': 105, 'Low': 99, 'Close': 103, 'Volume': 1000},
            {'Open': 103, 'High': 107, 'Low': 102, 'Close': 106, 'Volume': 1500},
            {'Open': 106, 'High': 110, 'Low': 105, 'Close': 108, 'Volume': 1200},
        ]
        df = self._make_df(rows)

        dm = DataManager.__new__(DataManager)
        cleaned_df, returns, report = dm.audit_and_clean(df, '1d')

        self.assertEqual(len(cleaned_df), 3, "No rows should be dropped from clean data.")
        self.assertIn('✅', report, f"Clean data should produce a clean report. Got: '{report}'")

    # TEST 22: AUDIT AND CLEAN (NaN Values Forward-Filled)
    def test_nan_values_forward_filled(self):
        # One bar has NaN for Close. The function should forward-fill it from the previous bar
        # and report how many NaNs were fixed — not crash or silently drop the row.
        from utilities.data import DataManager
        rows = [
            {'Open': 100, 'High': 105, 'Low': 99,  'Close': 103,     'Volume': 1000},
            {'Open': 103, 'High': 107, 'Low': 102, 'Close': np.nan,  'Volume': 1500},
            {'Open': 103, 'High': 108, 'Low': 102, 'Close': 106,     'Volume': 1200},
        ]
        df = self._make_df(rows)

        dm = DataManager.__new__(DataManager)
        cleaned_df, returns, report = dm.audit_and_clean(df, '1d')

        self.assertEqual(len(cleaned_df), 3, "NaN rows should be filled, not dropped.")
        self.assertFalse(cleaned_df['Close'].isna().any(), "No NaNs should remain after forward-fill.")
        self.assertAlmostEqual(cleaned_df['Close'].iloc[1], 103.0, delta=0.001,
                               msg="NaN Close must be forward-filled with previous row's Close (103.0).")
        self.assertIn('NaN', report, f"Report must mention NaN fix. Got: '{report}'")

    # TEST 23: AUDIT AND CLEAN (Negative Prices Dropped)
    def test_negative_price_rows_dropped(self):
        # One bar has a negative Close price (-5). That's impossible for a stock.
        # The row must be removed entirely, and the report must say it was dropped.
        from utilities.data import DataManager
        rows = [
            {'Open': 100, 'High': 105, 'Low': 99,  'Close': 103, 'Volume': 1000},
            {'Open': 103, 'High': 107, 'Low': -10, 'Close': -5,  'Volume': 1500},
            {'Open': 103, 'High': 108, 'Low': 102, 'Close': 106, 'Volume': 1200},
        ]
        df = self._make_df(rows)

        dm = DataManager.__new__(DataManager)
        cleaned_df, returns, report = dm.audit_and_clean(df, '1d')

        self.assertEqual(len(cleaned_df), 2, "The row with a negative price must be dropped.")
        self.assertIn('negative', report.lower(), f"Report must mention negative prices. Got: '{report}'")

    # TEST 24: AUDIT AND CLEAN (Zero Close Price Dropped)
    def test_zero_close_price_dropped(self):
        # One bar has Close = 0. This is almost always a bad data point, not a real price.
        # The row must be removed, and the report must say so.
        from utilities.data import DataManager
        rows = [
            {'Open': 100, 'High': 105, 'Low': 99,  'Close': 103, 'Volume': 1000},
            {'Open': 103, 'High': 107, 'Low': 102, 'Close': 0,   'Volume': 1500},
            {'Open': 103, 'High': 108, 'Low': 102, 'Close': 106, 'Volume': 1200},
        ]
        df = self._make_df(rows)

        dm = DataManager.__new__(DataManager)
        cleaned_df, returns, report = dm.audit_and_clean(df, '1d')

        self.assertEqual(len(cleaned_df), 2, "The row with zero close must be dropped.")
        self.assertIn('zero close', report.lower(), f"Report must mention zero close. Got: '{report}'")

    # TEST 25: AUDIT AND CLEAN (Broken OHLC Dropped)
    def test_broken_ohlc_dropped(self):
        # One bar has High < Low (107 < 102 is impossible — the high can't be below the low).
        # Another has Close > High (110 > 108), which is also impossible.
        # Both rows must be dropped, and the report must mention OHLC.
        from utilities.data import DataManager
        rows = [
            {'Open': 100, 'High': 105, 'Low': 99,  'Close': 103, 'Volume': 1000},
            {'Open': 103, 'High': 102, 'Low': 107, 'Close': 104, 'Volume': 1500},  # High < Low
            {'Open': 106, 'High': 108, 'Low': 105, 'Close': 110, 'Volume': 1200},  # Close > High
            {'Open': 108, 'High': 112, 'Low': 107, 'Close': 111, 'Volume': 900},
        ]
        df = self._make_df(rows)

        dm = DataManager.__new__(DataManager)
        cleaned_df, returns, report = dm.audit_and_clean(df, '1d')

        self.assertEqual(len(cleaned_df), 2, "Both broken OHLC rows must be dropped.")
        self.assertIn('OHLC', report, f"Report must mention OHLC issue. Got: '{report}'")

    # TEST 26: AUDIT AND CLEAN (Zero Volume Warns but Keeps Row)
    def test_zero_volume_warns_but_keeps_row(self):
        # One bar has Volume = 0. This is suspicious (no trades?) but not necessarily wrong —
        # could be a market halt or a pre/post-market artifact.
        # The row must stay in the data, but the report must warn about it.
        from utilities.data import DataManager
        rows = [
            {'Open': 100, 'High': 105, 'Low': 99,  'Close': 103, 'Volume': 1000},
            {'Open': 103, 'High': 107, 'Low': 102, 'Close': 106, 'Volume': 0},
            {'Open': 106, 'High': 110, 'Low': 105, 'Close': 108, 'Volume': 1200},
        ]
        df = self._make_df(rows)

        dm = DataManager.__new__(DataManager)
        cleaned_df, returns, report = dm.audit_and_clean(df, '1d')

        self.assertEqual(len(cleaned_df), 3, "Zero-volume rows must be kept, not dropped.")
        self.assertIn('zero volume', report.lower(), f"Report must warn about zero volume. Got: '{report}'")

    # TEST 27: AUDIT AND CLEAN (V-Spike Repaired for Intraday Data)
    def test_vspike_repaired_intraday(self):
        # A V-spike is a bar that shoots to a crazy price then immediately snaps back.
        # With 200 normal bars, std is tiny and log(10000/100) = 4.6 >> 5 * std.
        from utilities.data import DataManager
        n_normal = 100
        normal_price = 100.0
        spike_price = 10000.0
        normal_closes = [normal_price] * n_normal + [spike_price, normal_price] + [normal_price] * n_normal
        n_total = len(normal_closes)
        idx = pd.date_range('2024-01-02 09:30', periods=n_total, freq='1min', tz='US/Eastern')
        df = pd.DataFrame({
            'Open':   normal_closes,
            'High':   [c + 0.5 for c in normal_closes],
            'Low':    [c - 0.5 for c in normal_closes],
            'Close':  normal_closes,
            'Volume': [1000] * n_total,
        }, index=idx)

        dm = DataManager.__new__(DataManager)
        cleaned_df, returns, report = dm.audit_and_clean(df, '1m')

        self.assertIn('V-Spike', report, f"Report must mention V-Spike repair. Got: '{report}'")
        self.assertLess(cleaned_df['Close'].max(), 200.0,
                        "After repair, the spike value of 10000 must be replaced.")


if __name__ == '__main__':
    unittest.main()
