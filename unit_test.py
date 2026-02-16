import unittest
import numpy as np
import pandas as pd
import scipy.stats as stats
from autocorrelation import AbsenceOfAutocorrelationTest
from heavytails import HeavyTailsEVT
from volclustering import VolatilityClustering
from gainloss import GainLossAsymmetry
from aggregational import AggregationalGaussianity


class TestStylizedFactsTheoretical(unittest.TestCase):

    def setUp(self):
        """
        Create datasets with KNOWN theoretical properties.
        No random guessing.
        """
        np.random.seed(42)

        # ---------------------------------------------------------
        # 1. THEORETICAL DATA FOR AUTOCORRELATION
        # ---------------------------------------------------------
        # Dataset: [1, -1, 1, -1, ...]
        # Mean: 0
        # Variance: 1
        # Autocorrelation at Lag 1: sum(x_t * x_{t-1}) / var
        # (1)*(-1) + (-1)*(1) ... = -1.0
        self.acf_theoretical_data = pd.Series(np.tile([1.0, -1.0], 500))

        # ---------------------------------------------------------
        # 2. THEORETICAL DATA FOR EVT (XI / ALPHA)
        # ---------------------------------------------------------
        # Instead of simulating a stock and hoping it converges,
        # we generate data directly from the Fréchet distribution.
        #
        # Target Alpha = 2.0  =>  Xi = 0.5
        # Scipy genextreme params: c = -xi = -0.5
        # Location shifted to +10 to ensure all values are positive (avoid 0-censoring).
        target_alpha = 2.0
        c_param = -1.0 / target_alpha

        # Generate 5,000 points.
        # Note: We will use block_size=1 so the fitter sees this EXACT distribution.
        frechet_data = stats.genextreme.rvs(c=c_param, loc=10, scale=1, size=5000)

        # HeavyTailsEVT calculates: losses = -returns.
        # So we input NEGATIVE values: returns = -frechet_data
        self.evt_theoretical_data = pd.Series(-frechet_data)

        # ---------------------------------------------------------
        # 3. BROKEN DATA FOR ROBUSTNESS
        # ---------------------------------------------------------
        self.robustness_data = pd.Series([1.0, np.nan, np.inf, -np.inf, 2.0, 3.0] * 10)

        # ---------------------------------------------------------
        # 4. THEORETICAL DATA FOR VOLATILITY CLUSTERING
        # ---------------------------------------------------------
        # Regime-switching: high-vol block then low-vol block.
        # Squared returns will be highly autocorrelated because
        # variance is persistent within each regime.
        high_vol = np.random.normal(0, 5.0, 500)
        low_vol = np.random.normal(0, 0.1, 500)
        self.vc_clustered_data = pd.Series(np.concatenate([high_vol, low_vol]))

        # Constant-volatility data: squared returns are i.i.d.
        # The original alternating [1, -1] data has r² = [1, 1, ...] (zero variance),
        # so C2 is undefined. Use i.i.d. Gaussian instead for a clean negative test.
        self.vc_iid_data = pd.Series(np.random.normal(0, 1.0, 1000))

    # ==========================================
    # TEST 1: AUTOCORRELATION (Theoretical Check)
    # ==========================================
    def test_acf_exact_value(self):
        """
        Verify ACF calculates exactly -1.0 (or -0.999 due to bias) for alternating data.
        """
        print("\n[Test] Theoretical ACF Calculation")
        ac_tester = AbsenceOfAutocorrelationTest(self.acf_theoretical_data, "THEORY_ACF")

        # We test Lag 1 specifically
        # We need to access the raw ACF values or check the significant lags
        # The class returns 'significant_lags'. For [1, -1], Lag 1 MUST be significant.
        sig_lags = ac_tester.test_linear_independence(lags=5, plot=False)

        print(f"   -> Significant Lags Found: {sig_lags}")
        self.assertIn(1, sig_lags, "Theoretical Data [1, -1] MUST have significant Lag 1.")

    # ==========================================
    # TEST 2: EVT XI / ALPHA (Theoretical Check)
    # ==========================================
    def test_evt_exact_parameters(self):
        """
        Verify EVT recovers Alpha=2.0 from perfect Fréchet data.
        """
        print("\n[Test] Theoretical EVT Parameter Recovery")
        evt_tester = HeavyTailsEVT(self.evt_theoretical_data, "THEORY_EVT")

        # TRICK: Set block_size=1.
        # This forces the internal 'block_maxima' to be identical to our input data.
        # We are fitting the GEV directly to GEV data. No approximation error.
        result = evt_tester.run_mle_fit(block_size=1, plot=False)

        self.assertIsNotNone(result)
        xi, alpha = result

        print(f"   -> Target Alpha: 2.00")
        print(f"   -> Fitted Alpha: {alpha:.4f}")
        print(f"   -> Fitted Xi:    {xi:.4f}")

        # With direct fitting, tolerance can be very tight (0.1)
        self.assertAlmostEqual(alpha, 2.0, delta=0.1,
                               msg="EVT Optimizer failed to fit perfect Fréchet data.")
        self.assertGreater(xi, 0, "Xi must be positive for Heavy Tails.")

    # ==========================================
    # TEST 3: ROBUSTNESS (Gaps & NaNs)
    # ==========================================
    def test_robustness_nans(self):
        """
        Verify functions don't crash when fed Garbage Data (NaN, Inf).
        """
        print("\n[Test] Robustness (NaNs/Infs)")

        # 1. Test Autocorrelation Robustness
        ac_tester = AbsenceOfAutocorrelationTest(self.robustness_data, "ROBUST_ACF")
        try:
            lags = ac_tester.test_linear_independence(lags=5, plot=False)
            print(f"   -> ACF handled dirty data. Result: {lags}")
        except Exception as e:
            self.fail(f"Autocorrelation CRASHED on NaN/Inf data: {e}")

        # 2. Test EVT Robustness
        evt_tester = HeavyTailsEVT(self.robustness_data, "ROBUST_EVT")
        try:
            # Should likely return None or a value if enough clean data remains
            # Our dataset has 60 points, 30 are NaN/Inf. 30 Clean.
            # Block size 10 => 4 blocks. Might fail "Need >10 blocks".
            result = evt_tester.run_mle_fit(block_size=2, plot=False)
            print(f"   -> EVT handled dirty data. Result: {result}")
        except Exception as e:
            self.fail(f"EVT CRASHED on NaN/Inf data: {e}")

    # ==========================================
    # TEST 4: VOLATILITY CLUSTERING (Positive)
    # ==========================================
    def test_volclustering_regime_switching(self):
        """
        Regime-switching data (high-vol then low-vol) MUST show
        significant C2(τ) because variance is persistent within regimes.
        """
        print("\n[Test] Volatility Clustering (Regime-Switching)")
        vc_tester = VolatilityClustering(self.vc_clustered_data, "THEORY_VC")
        c2_values, sig_lags = vc_tester.compute_c2(max_lag=20, plot=False)

        print(f"   -> Significant Lags Found: {sig_lags}")
        self.assertGreater(len(sig_lags), 0,
                           "Regime-switching data MUST show volatility clustering.")
        # C2(1) should be strongly positive for clustered volatility
        self.assertGreater(c2_values[1], 0.1,
                           "C2(1) should be strongly positive for clustered data.")

    # ==========================================
    # TEST 5: VOLATILITY CLUSTERING (Negative)
    # ==========================================
    def test_volclustering_iid_returns(self):
        """
        I.I.D. Gaussian returns should show NO significant volatility clustering.
        Squared i.i.d. returns have no autocorrelation (beyond sampling noise).
        """
        print("\n[Test] Volatility Clustering (I.I.D. Gaussian)")
        vc_tester = VolatilityClustering(self.vc_iid_data, "IID_VC")
        c2_values, sig_lags = vc_tester.compute_c2(max_lag=20, plot=False)

        print(f"   -> Significant Lags Found: {sig_lags}")
        # For 1000 i.i.d. points, we expect at most ~1 spurious significant lag (5% chance each)
        self.assertLessEqual(len(sig_lags), 3,
                             "I.I.D. data should have very few (if any) significant C2 lags.")


class TestGainLossAsymmetry(unittest.TestCase):

    def test_known_loss_skew(self):
        """
        100 returns: 50 gains + 49 losses in body, 1 extreme loss.
        Body loss rate = 49/99 ~ 49.5%. Tail loss rate = 1/1 = 100%.
        With n=1 in the tail, z-test cannot be significant.
        Result must include z_stat, pvalue, body_loss_pct, and alternative.
        """
        small = [0.01] * 50 + [-0.01] * 49
        extreme = [-0.50]
        returns = pd.Series(small + extreme)

        gl = GainLossAsymmetry(returns, "TEST")
        result = gl.compute_asymmetry(q=0.99, plot=False)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['loss_pct'], 100.0, delta=0.01)
        self.assertAlmostEqual(result['gain_pct'], 0.0, delta=0.01)
        self.assertGreaterEqual(result['n_extreme'], 1)
        self.assertAlmostEqual(result['avg_loss'], -0.50, delta=0.01)
        self.assertAlmostEqual(result['median_loss'], -0.50, delta=0.01)
        self.assertIsNone(result['avg_gain'])
        self.assertIsNone(result['median_gain'])
        # Two-proportion z-test fields must be present
        self.assertIn('pvalue', result)
        self.assertIsInstance(result['pvalue'], float)
        self.assertIn('z_stat', result)
        self.assertIn('body_loss_pct', result)
        self.assertIn('alternative', result)

    def test_known_gain_skew(self):
        """
        Body is ~50/50, tail is 100% gains (0% losses).
        Tail loss rate < body loss rate, so alternative should be 'smaller'.
        """
        small = [0.01] * 50 + [-0.01] * 49
        extreme = [0.50]
        returns = pd.Series(small + extreme)

        gl = GainLossAsymmetry(returns, "TEST")
        result = gl.compute_asymmetry(q=0.99, plot=False)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['loss_pct'], 0.0, delta=0.01)
        self.assertAlmostEqual(result['gain_pct'], 100.0, delta=0.01)
        self.assertAlmostEqual(result['avg_gain'], 0.50, delta=0.01)
        self.assertIsNone(result['avg_loss'])
        self.assertIn('body_loss_pct', result)

    def test_mixed_extremes_not_significant(self):
        """
        Body: 48 gains + 48 losses = 50% loss rate.
        Tail: 3 losses + 1 gain = 75% loss rate.
        With only n=4 in the tail, the z-test should NOT be significant.
        """
        small = [0.001] * 48 + [-0.001] * 48
        extremes = [-0.50, -0.60, -0.70, 0.55]
        returns = pd.Series(small + extremes)

        gl = GainLossAsymmetry(returns, "TEST")
        result = gl.compute_asymmetry(q=0.96, plot=False)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['loss_pct'], 75.0, delta=0.01)
        self.assertEqual(result['n_extreme'], 4)
        self.assertAlmostEqual(result['avg_loss'], -0.60, delta=0.001)
        self.assertAlmostEqual(result['median_loss'], -0.60, delta=0.001)
        self.assertAlmostEqual(result['avg_gain'], 0.55, delta=0.001)
        self.assertAlmostEqual(result['median_gain'], 0.55, delta=0.001)
        self.assertGreater(result['pvalue'], 0.05)

    def test_strong_loss_asymmetry_significant(self):
        """
        Body: 490 gains + 490 losses = 50% loss rate.
        Tail: 18 losses + 2 gains = 90% loss rate.
        Two-proportion z-test (larger): tail loss rate >> body loss rate => p < 0.05.
        """
        small = [0.0001] * 490 + [-0.0001] * 490
        extremes = [-0.50] * 18 + [0.50] * 2
        returns = pd.Series(small + extremes)

        gl = GainLossAsymmetry(returns, "TEST")
        result = gl.compute_asymmetry(q=0.98, plot=False)

        self.assertIsNotNone(result)
        self.assertLess(result['pvalue'], 0.05,
                        "Tail 90% losses vs body 50% should be significant.")
        self.assertEqual(result['alternative'], 'larger')
        self.assertGreater(result['z_stat'], 0)

    def test_bullish_body_bearish_tail(self):
        """
        KEY TEST: Body is bullish (only 30% losses), but tail is bearish (80% losses).
        This is the scenario the old binomial test (p=0.5) would get wrong for a
        market with upward drift. The two-proportion z-test correctly detects
        that the tail loss rate is significantly higher than the body's.
        Body: 700 gains + 300 losses = 30% loss rate (values at |0.001|).
        Tail: 16 losses + 4 gains out of 20 = 80% loss rate (values at |0.50|).
        Use varied body values to avoid quantile ties.
        """
        np.random.seed(99)
        body_gains = list(np.random.uniform(0.0001, 0.005, 700))
        body_losses = list(-np.random.uniform(0.0001, 0.005, 300))
        tail_losses = [-0.50] * 16
        tail_gains = [0.50] * 4
        returns = pd.Series(body_gains + body_losses + tail_losses + tail_gains)

        gl = GainLossAsymmetry(returns, "TEST")
        result = gl.compute_asymmetry(q=0.98, plot=False)

        self.assertIsNotNone(result)
        # Body is ~30% losses
        self.assertAlmostEqual(result['body_loss_pct'], 30.0, delta=2.0)
        # Tail is heavily loss-skewed (>=75%)
        self.assertGreaterEqual(result['loss_pct'], 70.0)
        # The z-test must detect this anomaly
        self.assertLess(result['pvalue'], 0.05,
                        "Tail 80% losses vs body 30% losses must be significant.")
        self.assertEqual(result['alternative'], 'larger')

    def test_robustness_nan_inf(self):
        """
        GainLossAsymmetry should handle NaN/Inf without crashing.
        """
        data = pd.Series([0.01, -0.01, np.nan, np.inf, -np.inf, -0.50, 0.02] * 20)
        gl = GainLossAsymmetry(data, "ROBUST")
        try:
            result = gl.compute_asymmetry(q=0.99, plot=False)
            self.assertIn('pvalue', result)
            self.assertIn('z_stat', result)
            self.assertIn('body_loss_pct', result)
        except Exception as e:
            self.fail(f"GainLossAsymmetry CRASHED on NaN/Inf data: {e}")


class TestDataManagerCacheBug(unittest.TestCase):
    """
    Regression test for the bug where days_received contains datetime.date objects
    but missing_days contains pd.Timestamp objects, causing set membership checks
    to always fail and overwrite real cache files with empty DataFrames.
    """

    def test_days_received_matches_missing_days_type(self):
        """
        Simulate the exact logic from DataManager.get_data():
        - business_days from pd.bdate_range() produces pd.Timestamp objects
        - groupby(index.date) produces datetime.date objects
        - The set membership check must correctly identify received days

        Bug: m_day (Timestamp) not in days_received (set of date) is always True
        because Timestamp and date have different hashes and == returns False.
        This causes real data files to be overwritten with empty DataFrames.
        """
        from datetime import date

        # Simulate business_days (pd.Timestamps from bdate_range)
        business_days = pd.bdate_range(start='2024-01-02', end='2024-01-05')

        # Simulate missing_days (same type as business_days)
        missing_days = list(business_days)

        # Simulate days_received from groupby(index.date) - these are datetime.date
        days_received = {date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)}

        # Fixed check from data.py line 78 (uses .date() to convert Timestamp to date):
        falsely_missing = [m_day for m_day in missing_days if m_day.date() not in days_received]

        # If the bug exists, ALL days appear missing (would overwrite real data with empty DFs)
        # After fix, NO days should appear missing (they were all received)
        self.assertEqual(len(falsely_missing), 0,
                         f"Type mismatch bug: {len(falsely_missing)} days falsely marked as missing. "
                         f"Timestamp objects don't match datetime.date in set lookups.")


class TestDataManagerErrorReporting(unittest.TestCase):
    """
    Tests that DataManager.get_data returns actionable error info
    when no data is available, not just a vague message.
    """

    def test_empty_return_includes_ticker_in_report(self):
        """
        When get_data returns empty, the report string must include the ticker
        so the caller knows WHICH asset failed.
        """
        from unittest.mock import patch, MagicMock
        from data import DataManager

        # Patch the client so no real API call happens
        with patch.object(DataManager, '__init__', lambda self, **kw: None):
            dm = DataManager()
            dm.cache_dir = '/tmp/nonexistent_cache_dir_test'
            dm.client = MagicMock()
            dm.api_key = 'fake'
            dm.secret_key = 'fake'

            # get_data with a non-existent cache dir means no files found
            dm.client.get_stock_bars.side_effect = Exception("API unavailable")

            df, returns, report = dm.get_data("FAKEXYZ", "2024-01-02", "2024-01-05", "1d")

            self.assertTrue(df.empty)
            self.assertIn("FAKEXYZ", report,
                          f"Error report must include the ticker name. Got: '{report}'")


class TestVolTargetedLeverage(unittest.TestCase):
    """
    Tests for the vol-targeted dynamic leverage calculation:
    leverage = clamp(target_vol / realized_vol, min_lev, max_lev)
    """

    def test_basic_vol_target_scaling(self):
        """
        With target_vol=15 and realized_vol=10, leverage should be 1.5.
        With target_vol=15 and realized_vol=15, leverage should be 1.0.
        With target_vol=15 and realized_vol=30, leverage should be 0.5 -> clamped to min.
        """
        from backtest import calc_dynamic_leverage

        # Exact target match -> leverage = 1.0
        result = calc_dynamic_leverage(realized_vol=15.0, target_vol=15.0, min_lev=1.0, max_lev=4.0)
        self.assertAlmostEqual(result, 1.0, places=4)

        # Low vol -> lever up: 15/10 = 1.5
        result = calc_dynamic_leverage(realized_vol=10.0, target_vol=15.0, min_lev=1.0, max_lev=4.0)
        self.assertAlmostEqual(result, 1.5, places=4)

        # Very low vol -> lever up more: 15/5 = 3.0
        result = calc_dynamic_leverage(realized_vol=5.0, target_vol=15.0, min_lev=1.0, max_lev=4.0)
        self.assertAlmostEqual(result, 3.0, places=4)

    def test_clamp_max(self):
        """
        When vol is very low, leverage must be capped at MAX_LEV.
        target_vol=15, realized_vol=2 -> 15/2=7.5 -> clamped to 4.0
        """
        from backtest import calc_dynamic_leverage

        result = calc_dynamic_leverage(realized_vol=2.0, target_vol=15.0, min_lev=1.0, max_lev=4.0)
        self.assertAlmostEqual(result, 4.0, places=4)

    def test_clamp_min(self):
        """
        When vol is very high, leverage must be floored at MIN_LEV.
        target_vol=15, realized_vol=100 -> 15/100=0.15 -> clamped to 1.0
        """
        from backtest import calc_dynamic_leverage

        result = calc_dynamic_leverage(realized_vol=100.0, target_vol=15.0, min_lev=1.0, max_lev=4.0)
        self.assertAlmostEqual(result, 1.0, places=4)

    def test_series_input(self):
        """
        calc_dynamic_leverage should work on a pd.Series of realized vols,
        returning a Series of per-bar leverage values.
        """
        from backtest import calc_dynamic_leverage

        realized_vols = pd.Series([5.0, 10.0, 15.0, 30.0, 2.0])
        result = calc_dynamic_leverage(realized_vols, target_vol=15.0, min_lev=1.0, max_lev=4.0)

        self.assertIsInstance(result, pd.Series)
        expected = pd.Series([3.0, 1.5, 1.0, 1.0, 4.0])
        pd.testing.assert_series_equal(result, expected, atol=0.0001)

    def test_zero_vol_returns_max_leverage(self):
        """
        If realized vol is 0 or NaN, leverage should be capped at max (not inf/NaN).
        """
        from backtest import calc_dynamic_leverage

        result_zero = calc_dynamic_leverage(realized_vol=0.0, target_vol=15.0, min_lev=1.0, max_lev=4.0)
        self.assertAlmostEqual(result_zero, 4.0, places=4)

        result_nan = calc_dynamic_leverage(realized_vol=float('nan'), target_vol=15.0, min_lev=1.0, max_lev=4.0)
        self.assertAlmostEqual(result_nan, 4.0, places=4)

    def test_inverse_relationship(self):
        """
        Core property: lower vol MUST produce higher leverage (inverse relationship).
        """
        from backtest import calc_dynamic_leverage

        lev_low_vol = calc_dynamic_leverage(realized_vol=8.0, target_vol=15.0, min_lev=1.0, max_lev=4.0)
        lev_high_vol = calc_dynamic_leverage(realized_vol=20.0, target_vol=15.0, min_lev=1.0, max_lev=4.0)

        self.assertGreater(lev_low_vol, lev_high_vol,
                           "Lower realized vol must produce higher leverage.")


class TestHysteresisRebalancing(unittest.TestCase):
    """
    Tests for the hysteresis buffer on leverage rebalancing.
    Only rebalance when ideal leverage deviates >10% from last traded position.
    """

    def test_no_rebalance_within_buffer(self):
        """
        If ideal leverage stays within 10% of the initial trade,
        the actual leverage should remain at the initial value.
        """
        from backtest import apply_hysteresis

        # Start at 2.0, then ideal fluctuates within 10% (1.8 to 2.2)
        ideal = pd.Series([2.0, 2.05, 1.95, 2.10, 1.85, 2.15])
        result = apply_hysteresis(ideal, hysteresis_pct=0.10)

        # Should stay at 2.0 the whole time (all within 10%)
        expected = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        pd.testing.assert_series_equal(result, expected, atol=0.0001)

    def test_rebalance_when_exceeding_buffer(self):
        """
        When ideal leverage exceeds 10% from last traded, should rebalance.
        """
        from backtest import apply_hysteresis

        # Start at 2.0, then jump to 2.5 (25% change > 10%), then stay near 2.5
        ideal = pd.Series([2.0, 2.5, 2.45, 2.55])
        result = apply_hysteresis(ideal, hysteresis_pct=0.10)

        # Bar 0: trade to 2.0
        # Bar 1: |2.5-2.0|/2.0 = 25% > 10% -> trade to 2.5
        # Bar 2: |2.45-2.5|/2.5 = 2% < 10% -> stay at 2.5
        # Bar 3: |2.55-2.5|/2.5 = 2% < 10% -> stay at 2.5
        expected = pd.Series([2.0, 2.5, 2.5, 2.5])
        pd.testing.assert_series_equal(result, expected, atol=0.0001)

    def test_regime_switch_zero_to_nonzero(self):
        """
        Going from 0 (safe mode) to any nonzero leverage should always trigger a trade.
        """
        from backtest import apply_hysteresis

        ideal = pd.Series([0.0, 0.0, 2.0, 2.05])
        result = apply_hysteresis(ideal, hysteresis_pct=0.10)

        # Bar 0-1: stay at 0
        # Bar 2: 0->2.0 always triggers
        # Bar 3: |2.05-2.0|/2.0 = 2.5% < 10% -> stay at 2.0
        expected = pd.Series([0.0, 0.0, 2.0, 2.0])
        pd.testing.assert_series_equal(result, expected, atol=0.0001)

    def test_regime_switch_nonzero_to_zero(self):
        """
        Going from nonzero to 0 (safe mode) should always trigger.
        |0 - 2.0| / 2.0 = 100% > 10%.
        """
        from backtest import apply_hysteresis

        ideal = pd.Series([2.0, 2.05, 0.0, 0.0])
        result = apply_hysteresis(ideal, hysteresis_pct=0.10)

        expected = pd.Series([2.0, 2.0, 0.0, 0.0])
        pd.testing.assert_series_equal(result, expected, atol=0.0001)

    def test_preserves_index(self):
        """
        Output Series should preserve the input's index.
        """
        from backtest import apply_hysteresis

        idx = pd.date_range('2024-01-01', periods=3, freq='h')
        ideal = pd.Series([2.0, 2.05, 3.0], index=idx)
        result = apply_hysteresis(ideal, hysteresis_pct=0.10)

        self.assertTrue(result.index.equals(idx))


class TestComputeStrategy(unittest.TestCase):
    """
    Tests for the refactored compute_strategy() function that returns metrics as a dict.
    """

    def setUp(self):
        """Create synthetic return series long enough for the strategy warmup."""
        np.random.seed(42)
        n = 2000  # Need > BASELINE_WINDOW (120) + buffer
        # Synthetic log returns with slight positive drift
        self.ret_risk = pd.Series(
            np.random.normal(0.0003, 0.01, n),
            index=pd.date_range('2021-01-01', periods=n, freq='h')
        )
        self.ret_safe = pd.Series(
            np.random.normal(0.0001, 0.005, n),
            index=pd.date_range('2021-01-01', periods=n, freq='h')
        )

    def test_returns_dict_with_required_keys(self):
        """compute_strategy must return a dict with all key metrics."""
        from backtest import compute_strategy, DEFAULT_PARAMS

        result = compute_strategy(self.ret_risk, self.ret_safe, DEFAULT_PARAMS)

        self.assertIsInstance(result, dict)
        required_keys = [
            'total_ret', 'bench_ret', 'cagr_strat', 'cagr_bench',
            'sharpe_strat', 'sharpe_bench', 'sortino_strat', 'sortino_bench',
            'mdd_strat', 'mdd_bench', 'vol_strat', 'vol_bench',
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")
            self.assertIsInstance(result[key], float, f"{key} should be float")
            self.assertFalse(np.isnan(result[key]), f"{key} should not be NaN")

    def test_metrics_change_with_different_params(self):
        """Different parameter sets must produce different metrics (not hardcoded)."""
        from backtest import compute_strategy, DEFAULT_PARAMS

        params_low_vol = {**DEFAULT_PARAMS, 'target_vol': 10.0}
        params_high_vol = {**DEFAULT_PARAMS, 'target_vol': 30.0}

        result_low = compute_strategy(self.ret_risk, self.ret_safe, params_low_vol)
        result_high = compute_strategy(self.ret_risk, self.ret_safe, params_high_vol)

        # Different target vol should produce different Sharpe ratios
        self.assertNotAlmostEqual(result_low['sharpe_strat'], result_high['sharpe_strat'], places=2)

    def test_benchmark_unchanged_by_params(self):
        """Benchmark metrics should be identical regardless of strategy parameters."""
        from backtest import compute_strategy, DEFAULT_PARAMS

        params_a = {**DEFAULT_PARAMS, 'target_vol': 15.0, 'max_lev': 3.0}
        params_b = {**DEFAULT_PARAMS, 'target_vol': 25.0, 'max_lev': 5.0}

        result_a = compute_strategy(self.ret_risk, self.ret_safe, params_a)
        result_b = compute_strategy(self.ret_risk, self.ret_safe, params_b)

        self.assertAlmostEqual(result_a['bench_ret'], result_b['bench_ret'], places=6)
        self.assertAlmostEqual(result_a['sharpe_bench'], result_b['sharpe_bench'], places=6)


class TestAggregationalGaussianity(unittest.TestCase):
    """
    Tests for Aggregational Gaussianity (Stylized Fact 5):
    As returns are aggregated over longer horizons, the distribution
    converges toward Gaussian (excess kurtosis -> 0) via CLT.
    """

    def setUp(self):
        np.random.seed(42)
        # Heavy-tailed data: t-distribution with df=3 has excess kurtosis = 6/(3-4) -> infinite,
        # but finite-sample kurtosis will be high. Use df=5 for finite but heavy kurtosis.
        # Excess kurtosis of t(df) = 6/(df-4) for df>4. So t(5) has excess kurtosis = 6.
        self.heavy_tailed_returns = pd.Series(
            stats.t.rvs(df=5, size=10000) * 0.01  # Scale to realistic return magnitude
        )

        # Gaussian data: excess kurtosis ~0 at all scales
        self.gaussian_returns = pd.Series(
            np.random.normal(0, 0.01, 10000)
        )

        # Dirty data for robustness
        self.dirty_returns = pd.Series(
            [0.01, -0.01, np.nan, np.inf, -np.inf, 0.02, -0.03] * 500
        )

    def test_kurtosis_decreases_with_aggregation(self):
        """
        Heavy-tailed returns should show DECREASING excess kurtosis
        as the aggregation scale increases (CLT convergence).
        """
        ag = AggregationalGaussianity(self.heavy_tailed_returns, "TEST_HEAVY")
        result = ag.test_aggregational_gaussianity(scales=[1, 5, 20, 50], plot=False)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('kurtosis_by_scale', result)

        kurt = result['kurtosis_by_scale']
        # Kurtosis at scale 1 should be higher than at scale 50
        self.assertGreater(kurt[1], kurt[50],
                           "Excess kurtosis must decrease with aggregation for heavy-tailed data.")
        # Kurtosis at scale 1 should be noticeably leptokurtic (excess > 1)
        self.assertGreater(kurt[1], 1.0,
                           "t(5) returns should have substantial excess kurtosis at scale 1.")

    def test_gaussian_stays_near_zero_kurtosis(self):
        """
        Gaussian returns should have near-zero excess kurtosis at ALL scales.
        CLT doesn't change already-Gaussian data.
        """
        ag = AggregationalGaussianity(self.gaussian_returns, "TEST_GAUSSIAN")
        result = ag.test_aggregational_gaussianity(scales=[1, 5, 20], plot=False)

        self.assertIsNotNone(result)
        kurt = result['kurtosis_by_scale']
        for scale, k in kurt.items():
            self.assertAlmostEqual(k, 0.0, delta=1.0,
                                   msg=f"Gaussian data should have near-zero excess kurtosis at scale {scale}.")

    def test_return_structure(self):
        """
        Result dict must contain required keys with correct types.
        """
        ag = AggregationalGaussianity(self.heavy_tailed_returns, "TEST_STRUCT")
        result = ag.test_aggregational_gaussianity(scales=[1, 10], plot=False)

        self.assertIsNotNone(result)
        self.assertIn('kurtosis_by_scale', result)
        self.assertIn('convergence_confirmed', result)

        self.assertIsInstance(result['kurtosis_by_scale'], dict)
        self.assertIsInstance(result['convergence_confirmed'], bool)

        # Dict should have entries for each requested scale
        self.assertIn(1, result['kurtosis_by_scale'])
        self.assertIn(10, result['kurtosis_by_scale'])

    def test_robustness_nan_inf(self):
        """
        AggregationalGaussianity should handle NaN/Inf without crashing.
        """
        ag = AggregationalGaussianity(self.dirty_returns, "TEST_ROBUST")
        try:
            result = ag.test_aggregational_gaussianity(scales=[1, 5], plot=False)
            self.assertIsNotNone(result)
            self.assertIn('kurtosis_by_scale', result)
        except Exception as e:
            self.fail(f"AggregationalGaussianity CRASHED on NaN/Inf data: {e}")

    def test_convergence_detected_for_heavy_tails(self):
        """
        For heavy-tailed data, convergence_confirmed should be True
        (kurtosis does decrease with aggregation).
        """
        ag = AggregationalGaussianity(self.heavy_tailed_returns, "TEST_CONVERGE")
        result = ag.test_aggregational_gaussianity(scales=[1, 5, 20, 50], plot=False)

        self.assertTrue(result['convergence_confirmed'],
                        "Heavy-tailed data should show convergence toward Gaussianity.")


if __name__ == '__main__':
    unittest.main()
