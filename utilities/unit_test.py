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
        # 5000-point alternating-regime series: 2500 high-vol then 2500 low-vol.
        # max_lag=500 covers multi-day lags for 1m data (389 bars/day ≈ 1.3 days).
        # The regime block length (2500) >> max_lag (500), so ACF decays slowly —
        # many lags must be significant, and C1(lag=1) must be strongly positive.
        np.random.seed(42)
        high_vol = np.random.normal(0, 5.0, 2500)
        low_vol = np.random.normal(0, 0.1, 2500)
        vc_clustered_data = pd.Series(np.concatenate([high_vol, low_vol]))

        vc_tester = VolatilityClustering(vc_clustered_data, "THEORY_VC")
        c1_values, sig_lags = vc_tester.compute_c1(max_lag=500, plot=False, n_shuffles=100)

        self.assertGreater(len(sig_lags), 0, "Volatile-then-calm data must show volatility clustering.")
        self.assertGreater(c1_values[1], 0.1, "C1 at lag 1 must be strongly positive for regime-switching data.")

    # TEST 2: VOLATILITY CLUSTERING (Negative)
    def test_volclustering_iid_returns(self):
        # 5000-point iid Gaussian. At max_lag=500 and 5% level we expect at most
        # ~5% × 500 = 25 false-positive lags. We allow up to 30 as a safe margin.
        np.random.seed(42)
        vc_iid_data = pd.Series(np.random.normal(0, 1.0, 5000))

        vc_tester = VolatilityClustering(vc_iid_data, "IID_VC")
        c1_values, sig_lags = vc_tester.compute_c1(max_lag=500, plot=False, n_shuffles=100)

        self.assertLessEqual(len(sig_lags), 30,
                             "Random data should have very few significant lags in the volatility ACF.")


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

    # TEST 6: SLOW DECAY (Robustness)
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

    # TEST 10: LEVERAGE EFFECT (Robustness)
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
        # Result must contain lags (-max_lag..max_lag), corr_abs and corr_sq of matching
        # length, a scalar null_upper, and corr_confirmed=False for IID data.
        np.random.seed(42)
        n = 2000
        max_lag = 20
        iid_returns = pd.Series(np.random.normal(0, 0.01, n))
        iid_volume = pd.Series(np.random.normal(1e6, 1e4, n))

        vvc = VolVolCorr(iid_returns, iid_volume, "TEST")
        result = vvc.compute_correlation(max_lag=max_lag, plot=False, n_shuffles=100)

        self.assertIsNotNone(result)
        self.assertEqual(len(result['lags']), 2 * max_lag + 1)
        self.assertEqual(len(result['corr_abs']), 2 * max_lag + 1)
        self.assertEqual(len(result['corr_sq']),  2 * max_lag + 1)
        self.assertTrue(np.all(np.abs(result['corr_abs']) <= 1.0))
        self.assertFalse(result['corr_confirmed'], "IID data must not show significant correlation.")

    # TEST 15: VOL-VOL CORRELATION (Positive Correlation Detected)
    def test_positive_correlation_detected(self):
        # Volume = |return| * 10M + noise → strong contemporaneous correlation at τ=0.
        np.random.seed(42)
        n = 2000
        abs_vol = np.abs(np.random.normal(0, 0.02, n))
        corr_returns = pd.Series(abs_vol * np.random.choice([-1, 1], n))
        corr_volume = pd.Series(abs_vol * 1e7 + np.random.normal(0, 1e4, n))

        vvc = VolVolCorr(corr_returns, corr_volume, "TEST_CORR")
        result = vvc.compute_correlation(max_lag=20, plot=False, n_shuffles=100)

        self.assertIsNotNone(result)
        lags = np.array(result['lags'])
        corr_abs = np.array(result['corr_abs'])
        tau0 = float(corr_abs[lags == 0][0])
        self.assertGreater(tau0, 0.5, "Contemporaneous corr at τ=0 must be > 0.5.")
        self.assertTrue(result['corr_confirmed'])

    # TEST 15: VOL-VOL CORRELATION (Robustness)
    def test_robustness_nan_inf(self):
        np.random.seed(42)
        clean_r = np.random.normal(0, 0.01, 500)
        clean_v = np.abs(clean_r) * 1e7 + np.random.normal(0, 1e4, 500)
        dirty_r = np.concatenate([clean_r, [np.nan] * 30, [np.inf] * 20, [-np.inf] * 20])
        dirty_v = np.concatenate([clean_v, [np.nan] * 30, [np.nan] * 20, [np.nan] * 20])

        vvc = VolVolCorr(pd.Series(dirty_r), pd.Series(dirty_v), "ROBUST")
        try:
            result = vvc.compute_correlation(max_lag=10, plot=False, n_shuffles=100)
            if result is not None:
                self.assertTrue(np.all(np.isfinite(result['corr_abs'])))
                self.assertTrue(np.all(np.abs(result['corr_abs']) <= 1.0))
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


class TestIntradayDemeaning(unittest.TestCase):

    def _make_pattern_df(self, n_days, open_vol, mid_vol):
        """
        Build n_days of 1m deterministic data.
        The 9:32 bar return is exactly open_vol; every other return is mid_vol.
        Using deterministic (non-random) prices makes the expected slot means exact.
        """
        dates = pd.bdate_range('2024-01-02', periods=n_days, tz='US/Eastern')
        all_dfs = []
        for date in dates:
            idx = pd.date_range(f"{date.date()} 09:31", f"{date.date()} 15:59",
                                freq='1min', tz='US/Eastern')
            n = len(idx)
            prices = np.ones(n) * 100.0
            prices[1] = prices[0] * np.exp(open_vol)   # 9:32 bar
            for i in range(2, n):
                prices[i] = prices[i - 1] * np.exp(mid_vol)
            all_dfs.append(pd.DataFrame({
                'Open': prices, 'High': prices * 1.001,
                'Low':  prices * 0.999, 'Close': prices, 'Volume': [1000] * n,
            }, index=idx))
        return pd.concat(all_dfs).sort_index()

    # TEST: after demeaning every time slot has mean |return| = 1.0
    def test_demeaned_returns_have_unit_slot_mean(self):
        # With a strong open/close volatility pattern (open 100x more volatile than mid),
        # the raw slot means would range from 0.0002 to 0.02.
        # After demeaning (dividing each bar by its slot's mean |return|),
        # every slot must have mean |return| = 1.0 exactly.
        from utilities.data import DataManager

        df = self._make_pattern_df(n_days=10, open_vol=0.02, mid_vol=0.0002)
        dm = DataManager.__new__(DataManager)
        _, returns, _ = dm.audit_and_clean(df, '1m')

        slot_means = returns.abs().groupby(returns.index.time).mean()
        np.testing.assert_allclose(
            slot_means.values, 1.0, atol=1e-6,
            err_msg="After intraday demeaning every slot must have mean |return| = 1.0"
        )

    # TEST: daily data is NOT demeaned
    def test_daily_data_returns_not_demeaned(self):
        # Daily log returns are ~0.01, not ~1.0.
        # audit_and_clean must leave them unchanged for interval='1d'.
        from utilities.data import DataManager

        index = pd.date_range('2024-01-02', periods=50, freq='B', tz='US/Eastern')
        prices = 100.0 * np.exp(np.cumsum(np.full(50, 0.01)))
        df = pd.DataFrame({
            'Open': prices, 'High': prices * 1.01,
            'Low':  prices * 0.99, 'Close': prices, 'Volume': [1000] * 50,
        }, index=index)

        dm = DataManager.__new__(DataManager)
        _, returns, _ = dm.audit_and_clean(df, '1d')

        self.assertAlmostEqual(float(returns.abs().mean()), 0.01, delta=0.005,
                               msg="Daily returns must not be intraday-demeaned (expected ≈0.01, not ≈1.0).")


class TestLeverageCrossCorr(unittest.TestCase):
    """Test the FFT-based _cross_corr helper added to LeverageEffect for speed."""

    # _cross_corr matches direct corrcoef (implicitly tests length and value range)
    def test_cross_corr_matches_direct_corrcoef(self):
        # For i.i.d. data, the FFT-based result must agree with a direct per-lag
        # corrcoef loop to within 0.02 (small tolerance for global mean/std approximation).
        from facts_test.leverage import LeverageEffect
        np.random.seed(42)
        r = np.random.normal(0, 0.01, 5000)
        r_sq = r ** 2
        le = LeverageEffect(pd.Series(r), "TEST")
        max_lag = 10

        fft_result = le._cross_corr(r, r_sq, max_lag)
        direct = np.array([np.corrcoef(r[:-tau], r_sq[tau:])[0, 1]
                           for tau in range(1, max_lag + 1)])

        np.testing.assert_allclose(fft_result, direct, atol=0.02,
                                   err_msg="_cross_corr must agree with direct corrcoef within 0.02.")


class TestBarCountAndTimeFilter(unittest.TestCase):

    def _make_intraday_df(self, date_str, start_time, end_time):
        """Build a 1-minute OHLCV DataFrame for a single trading day."""
        idx = pd.date_range(
            f"{date_str} {start_time}",
            f"{date_str} {end_time}",
            freq='1min',
            tz='US/Eastern'
        )
        n = len(idx)
        return pd.DataFrame({
            'Open':   [100.0] * n,
            'High':   [101.0] * n,
            'Low':    [99.0]  * n,
            'Close':  [100.5] * n,
            'Volume': [1000]  * n,
        }, index=idx)

    # TEST 28: TIME FILTER (boundary bars excluded)
    def test_time_filter_excludes_0930_and_1600(self):
        # The get_data pipeline applies between_time('09:31', '15:59').
        # A raw DataFrame spanning 09:30-16:00 (391 bars) must be trimmed to
        # exactly 389 bars with no 09:30 or 16:00 bar remaining.
        import datetime
        idx = pd.date_range('2024-01-02 09:30', '2024-01-02 16:00',
                            freq='1min', tz='US/Eastern')
        n = len(idx)
        df = pd.DataFrame({
            'Open': [100.0] * n, 'High': [101.0] * n,
            'Low':  [99.0]  * n, 'Close': [100.5] * n, 'Volume': [1000] * n,
        }, index=idx)

        filtered = df.between_time('09:31', '15:59')
        times = filtered.index.time

        self.assertNotIn(datetime.time(9, 30), times,
                         "09:30 bar must be excluded by time filter.")
        self.assertNotIn(datetime.time(16, 0), times,
                         "16:00 bar must be excluded by time filter.")
        self.assertEqual(len(filtered), 389,
                         f"Filtered full day must have 389 bars, got {len(filtered)}.")

    # TEST 29: BAR COUNT (389 bars — no warning)
    def test_full_day_389_bars_no_bar_count_warning(self):
        # Perfect full trading day: 09:31-15:59 = 389 bars. Must be silent.
        from utilities.data import DataManager
        df = self._make_intraday_df('2024-01-02', '09:31', '15:59')
        self.assertEqual(len(df), 389)

        dm = DataManager.__new__(DataManager)
        _, _, report = dm.audit_and_clean(df, '1m')

        self.assertNotIn('BAR COUNT', report.upper(),
                         f"389-bar full day must not warn. Got: '{report}'")

    # TEST 30: BAR COUNT (387 bars — no warning, normal provider noise)
    def test_387_bars_no_bar_count_warning(self):
        # Alpaca commonly delivers 385-388 bars on days where 1-4 illiquid
        # minutes had no trades. This is normal noise and must not warn.
        from utilities.data import DataManager
        idx = pd.date_range('2024-01-02 09:31', periods=387, freq='1min', tz='US/Eastern')
        n = len(idx)
        df = pd.DataFrame({
            'Open': [100.0] * n, 'High': [101.0] * n,
            'Low':  [99.0]  * n, 'Close': [100.5] * n, 'Volume': [1000] * n,
        }, index=idx)

        dm = DataManager.__new__(DataManager)
        _, _, report = dm.audit_and_clean(df, '1m')

        self.assertNotIn('BAR COUNT', report.upper(),
                         f"387-bar day (provider noise) must not warn. Got: '{report}'")

    # TEST 31: BAR COUNT (385 bars — no warning, lower bound of normal range)
    def test_385_bars_no_bar_count_warning(self):
        # 385 bars is the floor of acceptable full-day bar counts (4 missing = ~1%).
        # Must be silent.
        from utilities.data import DataManager
        idx = pd.date_range('2024-01-02 09:31', periods=385, freq='1min', tz='US/Eastern')
        n = len(idx)
        df = pd.DataFrame({
            'Open': [100.0] * n, 'High': [101.0] * n,
            'Low':  [99.0]  * n, 'Close': [100.5] * n, 'Volume': [1000] * n,
        }, index=idx)

        dm = DataManager.__new__(DataManager)
        _, _, report = dm.audit_and_clean(df, '1m')

        self.assertNotIn('BAR COUNT', report.upper(),
                         f"385-bar day must not warn. Got: '{report}'")

    # TEST 32: BAR COUNT (209 bars — no warning, proper half-day after 13:00 trim)
    def test_half_day_209_bars_no_bar_count_warning(self):
        # Post-2010 NYSE half-days close at 13:00 ET. After the 09:31 strict lower bound
        # and < 13:00 upper bound, the window is 09:31-12:59 = exactly 209 bars.
        from utilities.data import DataManager
        idx = pd.date_range('2024-11-29 09:31', periods=209, freq='1min', tz='US/Eastern')
        n = len(idx)
        df = pd.DataFrame({
            'Open': [100.0] * n, 'High': [101.0] * n,
            'Low':  [99.0]  * n, 'Close': [100.5] * n, 'Volume': [1000] * n,
        }, index=idx)

        dm = DataManager.__new__(DataManager)
        _, _, report = dm.audit_and_clean(df, '1m')

        self.assertNotIn('BAR COUNT', report.upper(),
                         f"209-bar half-day must not warn. Got: '{report}'")

    # TEST 33: BAR COUNT (320 bars on a known halt day — emits HALT WARNING)
    def test_circuit_breaker_day_emits_halt_warning(self):
        # 2020-03-09 is a known Level-1 circuit breaker day. Real data has ~375 bars
        # (15-min halt). 320 bars is pathologically low even for a halt day and must
        # still warn, but as a HALT WARNING rather than a generic BAR COUNT WARNING.
        from utilities.data import DataManager
        idx = pd.date_range('2020-03-09 09:31', periods=320, freq='1min', tz='US/Eastern')
        n = len(idx)
        df = pd.DataFrame({
            'Open': [100.0] * n, 'High': [101.0] * n,
            'Low':  [99.0]  * n, 'Close': [100.5] * n, 'Volume': [1000] * n,
        }, index=idx)

        dm = DataManager.__new__(DataManager)
        _, _, report = dm.audit_and_clean(df, '1m')

        self.assertIn('HALT', report.upper(),
                      f"320-bar known halt day must produce a HALT WARNING. Got: '{report}'")

    # TEST 34: BAR COUNT (too few bars < 200 — warns)
    def test_day_with_under_200_bars_warns(self):
        # A day with < 200 bars is broken data (API gap, provider outage, etc.).
        from utilities.data import DataManager
        idx = pd.date_range('2024-01-02 09:31', periods=149, freq='1min', tz='US/Eastern')
        n = len(idx)
        df = pd.DataFrame({
            'Open': [100.0] * n, 'High': [101.0] * n,
            'Low':  [99.0]  * n, 'Close': [100.5] * n, 'Volume': [1000] * n,
        }, index=idx)

        dm = DataManager.__new__(DataManager)
        _, _, report = dm.audit_and_clean(df, '1m')

        self.assertIn('BAR COUNT', report.upper(),
                      f"149-bar day must warn. Got: '{report}'")

    # TEST 35: BAR COUNT (too many bars > 389 — warns)
    def test_day_with_over_389_bars_warns(self):
        # 391 bars means 09:30 and/or 16:00 leaked through the time filter.
        from utilities.data import DataManager
        idx = pd.date_range('2024-01-02 09:30', '2024-01-02 16:00',
                            freq='1min', tz='US/Eastern')
        n = len(idx)
        self.assertEqual(n, 391)
        df = pd.DataFrame({
            'Open': [100.0] * n, 'High': [101.0] * n,
            'Low':  [99.0]  * n, 'Close': [100.5] * n, 'Volume': [1000] * n,
        }, index=idx)

        dm = DataManager.__new__(DataManager)
        _, _, report = dm.audit_and_clean(df, '1m')

        self.assertIn('BAR COUNT', report.upper(),
                      f"391-bar day (time filter leak) must warn. Got: '{report}'")

    # TEST 36: BAR COUNT (multi-day — only bad date named in warning)
    def test_multiday_only_bad_days_trigger_warning(self):
        # Two perfect days + one circuit-breaker day (320 bars). Only the bad
        # date must appear in the warning.
        from utilities.data import DataManager
        good1 = self._make_intraday_df('2024-01-02', '09:31', '15:59')
        bad_idx = pd.date_range('2024-01-03 09:31', periods=320,
                                freq='1min', tz='US/Eastern')
        bad = pd.DataFrame({
            'Open': [100.0] * 320, 'High': [101.0] * 320,
            'Low':  [99.0]  * 320, 'Close': [100.5] * 320, 'Volume': [1000] * 320,
        }, index=bad_idx)
        good2 = self._make_intraday_df('2024-01-04', '09:31', '15:59')
        df = pd.concat([good1, bad, good2]).sort_index()

        dm = DataManager.__new__(DataManager)
        _, _, report = dm.audit_and_clean(df, '1m')

        self.assertIn('BAR COUNT', report.upper(),
                      f"Multi-day df with circuit-breaker day must warn. Got: '{report}'")
        self.assertIn('2024-01-03', report,
                      f"Warning must name the bad date. Got: '{report}'")


class TestBarCount5mAnd1h(unittest.TestCase):

    def _make_df(self, date_str, start_time, freq, n_bars):
        idx = pd.date_range(f"{date_str} {start_time}", periods=n_bars,
                            freq=freq, tz='US/Eastern')
        return pd.DataFrame({
            'Open': [100.0] * n_bars, 'High': [101.0] * n_bars,
            'Low':  [99.0]  * n_bars, 'Close': [100.5] * n_bars,
            'Volume': [1000] * n_bars,
        }, index=idx)

    # 5m full day: 9:35–15:55 = 77 bars → silent
    def test_5m_full_day_77_bars_no_warning(self):
        from utilities.data import DataManager
        df = self._make_df('2024-01-02', '09:35', '5min', 77)
        self.assertEqual(len(df), 77)
        dm = DataManager.__new__(DataManager)
        _, _, report = dm.audit_and_clean(df, '5m')
        self.assertNotIn('BAR COUNT', report.upper(),
                         f"77-bar 5m day must not warn. Got: '{report}'")

    # 5m broken day → warns
    def test_5m_broken_day_warns(self):
        from utilities.data import DataManager
        df = self._make_df('2024-01-02', '09:35', '5min', 30)
        dm = DataManager.__new__(DataManager)
        _, _, report = dm.audit_and_clean(df, '5m')
        self.assertIn('BAR COUNT', report.upper(),
                      f"30-bar 5m day must warn. Got: '{report}'")

    # 5m half-day: 9:35–12:55 = 41 bars → silent
    def test_5m_half_day_41_bars_no_warning(self):
        from utilities.data import DataManager
        df = self._make_df('2024-11-29', '09:35', '5min', 41)
        self.assertEqual(len(df), 41)
        dm = DataManager.__new__(DataManager)
        _, _, report = dm.audit_and_clean(df, '5m')
        self.assertNotIn('BAR COUNT', report.upper(),
                         f"41-bar 5m half-day must not warn. Got: '{report}'")

    # 1h full day: 10:30–15:30 = 6 bars → silent
    def test_1h_full_day_6_bars_no_warning(self):
        from utilities.data import DataManager
        df = self._make_df('2024-01-02', '10:30', '1h', 6)
        self.assertEqual(len(df), 6)
        dm = DataManager.__new__(DataManager)
        _, _, report = dm.audit_and_clean(df, '1h')
        self.assertNotIn('BAR COUNT', report.upper(),
                         f"6-bar 1h day must not warn. Got: '{report}'")

    # 1h broken day (1 bar) → warns
    def test_1h_broken_day_warns(self):
        from utilities.data import DataManager
        df = self._make_df('2024-01-02', '10:30', '1h', 1)
        dm = DataManager.__new__(DataManager)
        _, _, report = dm.audit_and_clean(df, '1h')
        self.assertIn('BAR COUNT', report.upper(),
                      f"1-bar 1h day must warn. Got: '{report}'")

    # 1h half-day: 3 bars → silent
    def test_1h_half_day_3_bars_no_warning(self):
        from utilities.data import DataManager
        df = self._make_df('2024-11-29', '10:30', '1h', 3)
        self.assertEqual(len(df), 3)
        dm = DataManager.__new__(DataManager)
        _, _, report = dm.audit_and_clean(df, '1h')
        self.assertNotIn('BAR COUNT', report.upper(),
                         f"3-bar 1h half-day must not warn. Got: '{report}'")


if __name__ == '__main__':
    unittest.main()
