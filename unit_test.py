import unittest
import numpy as np
import pandas as pd
import scipy.stats as stats
from autocorrelation import AbsenceOfAutocorrelationTest
from heavytails import HeavyTailsEVT
from volclustering import VolatilityClustering


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


if __name__ == '__main__':
    unittest.main()
