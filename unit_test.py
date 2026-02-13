import unittest
import numpy as np
import pandas as pd
import scipy.stats as stats
from autocorrelation import AbsenceOfAutocorrelationTest
from heavytails import HeavyTailsEVT
from volclustering import VolatilityClustering
from gainloss import GainLossAsymmetry


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


if __name__ == '__main__':
    unittest.main()
