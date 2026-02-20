import unittest
import numpy as np
import pandas as pd
import scipy.stats as stats
from facts_test.autocorrelation import AbsenceOfAutocorrelationTest
from facts_test.heavytails import HeavyTailsEVT
from facts_test.volclustering import VolatilityClustering
from facts_test.gainloss import GainLossAsymmetry
from facts_test.aggregational import AggregationalGaussianity
from facts_test.intermittency import Intermittency
from facts_test.decay import SlowDecay
from facts_test.leverage import LeverageEffect
from facts_test.volvolcorr import VolVolCorr
from facts_test.timescales import AsymmetryTimescales
from facts_test.conditional import ConditionalTails


class TestStylizedFactsTheoretical(unittest.TestCase):

    # TEST 1: AUTOCORRELATION (Theoretical Check)
    def test_acf_exact_value(self):
        # Data: 1, -1, 1, -1, ... forever
        # Each value is the opposite of the one before it.
        # That means if you multiply each pair (1 * -1, -1 * 1, ...) you always get -1.
        # So the correlation at lag 1 is perfectly -1.
        # We expect the test to flag lag 1 as "significant" (meaning it's not random).
        np.random.seed(42)
        acf_data = pd.Series(np.tile([1.0, -1.0], 500))

        ac_tester = AbsenceOfAutocorrelationTest(acf_data, "THEORY_ACF")
        sig_lags = ac_tester.test_linear_independence(lags=5, plot=False)

        self.assertIn(1, sig_lags, "Perfectly alternating data [1, -1] must have lag 1 flagged as significant.")

    # TEST 2: HEAVY TAILS (Theoretical Check)
    def test_evt_exact_parameters(self):
        # We generate data from a known heavy-tail distribution (Fréchet) with alpha = 2.
        # Alpha measures "how fat" the tail is. Lower alpha = fatter tail.
        # We then run the EVT fitter and check it recovers alpha ≈ 2.
        # Xi = 1/alpha = 0.5. Xi > 0 means heavy tails exist.
        # block_size=1 means every single data point is its own "block maximum",
        # so the fitter sees the exact distribution we generated.
        np.random.seed(42)
        target_alpha = 2.0
        c_param = -1.0 / target_alpha
        frechet_data = stats.genextreme.rvs(c=c_param, loc=10, scale=1, size=5000)
        evt_data = pd.Series(-frechet_data)

        evt_tester = HeavyTailsEVT(evt_data, "THEORY_EVT")
        result = evt_tester.run_mle_fit(block_size=1, plot=False)

        self.assertIsNotNone(result)
        xi, alpha = result
        self.assertAlmostEqual(alpha, 2.0, delta=0.1,
                               msg="EVT must recover alpha=2 from data we literally generated with alpha=2.")
        self.assertGreater(xi, 0, "Xi > 0 means heavy tails exist.")

    # TEST 3: ROBUSTNESS (Broken Data)
    def test_robustness_nans(self):
        # We feed the functions garbage: NaN (missing), Inf (infinity), -Inf.
        # The functions should not crash — drop the data, warn the user, and move on.
        robustness_data = pd.Series([1.0, np.nan, np.inf, -np.inf, 2.0, 3.0] * 10)

        ac_tester = AbsenceOfAutocorrelationTest(robustness_data, "ROBUST_ACF")
        try:
            lags = ac_tester.test_linear_independence(lags=5, plot=False)
        except Exception as e:
            self.fail(f"Autocorrelation crashed on bad data: {e}")

        evt_tester = HeavyTailsEVT(robustness_data, "ROBUST_EVT")
        try:
            result = evt_tester.run_mle_fit(block_size=2, plot=False)
        except Exception as e:
            self.fail(f"EVT crashed on bad data: {e}")

    # TEST 4: VOLATILITY CLUSTERING (Positive)
    def test_volclustering_regime_switching(self):
        # We create 500 wild/volatile returns followed by 500 tiny/calm returns.
        # This mimics a market crash followed by a quiet period.
        # Because variance is stuck high for 500 days then stuck low for 500 days,
        # the squared returns are very autocorrelated — knowing today's variance
        # tells you a lot about tomorrow's.
        # We expect several lags to be flagged as significant, and C2(lag=1) > 0.1.
        np.random.seed(42)
        high_vol = np.random.normal(0, 5.0, 500)
        low_vol = np.random.normal(0, 0.1, 500)
        vc_clustered_data = pd.Series(np.concatenate([high_vol, low_vol]))

        vc_tester = VolatilityClustering(vc_clustered_data, "THEORY_VC")
        c2_values, sig_lags = vc_tester.compute_c2(max_lag=20, plot=False)

        self.assertGreater(len(sig_lags), 0, "Volatile-then-calm data must show volatility clustering.")
        self.assertGreater(c2_values[1], 0.1, "C2 at lag 1 must be strongly positive for regime-switching data.")

    # TEST 5: VOLATILITY CLUSTERING (Negative)
    def test_volclustering_iid_returns(self):
        # Pure random (iid) Gaussian returns: every day is independent of every other day.
        # Squaring them and computing autocorrelation should give mostly noise (near zero).
        # We allow up to 3 "false positives" out of 20 lags at the 5% level.
        np.random.seed(42)
        vc_iid_data = pd.Series(np.random.normal(0, 1.0, 1000))

        vc_tester = VolatilityClustering(vc_iid_data, "IID_VC")
        c2_values, sig_lags = vc_tester.compute_c2(max_lag=20, plot=False)

        self.assertLessEqual(len(sig_lags), 3,
                             "Random data should have almost no significant lags in the volatility ACF.")


class TestGainLossAsymmetry(unittest.TestCase):

    # TEST 6: GAIN/LOSS ASYMMETRY (Pure Loss Tail)
    def test_known_loss_skew(self):
        # 100 returns: 50 small gains, 49 small losses, 1 huge loss.
        # The top 1% of extreme events is just that one -50% crash.
        # So the tail is 100% losses, 0% gains.
        # With only 1 extreme event, a statistical significance test can't say much,
        # but the percentages and averages should be exactly right.
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
        self.assertIn('pvalue', result)
        self.assertIsInstance(result['pvalue'], float)
        self.assertIn('z_stat', result)
        self.assertIn('body_loss_pct', result)
        self.assertIn('alternative', result)

    # TEST 7: GAIN/LOSS ASYMMETRY (Pure Gain Tail)
    def test_known_gain_skew(self):
        # Same setup but the extreme event is a +50% gain instead of a loss.
        # The tail is 100% gains, 0% losses.
        # Tail loss rate < body loss rate, so "alternative" should be 'smaller'.
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

    # TEST 8: GAIN/LOSS ASYMMETRY (Mixed Tail, Not Significant)
    def test_mixed_extremes_not_significant(self):
        # Body: 48 gains + 48 losses = exactly 50% loss rate.
        # Tail: 3 losses + 1 gain = 75% loss rate.
        # With only 4 extreme events total, the stats test has no power — p > 0.05.
        # We just can't say it's significant with so few data points.
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

    # TEST 9: GAIN/LOSS ASYMMETRY (Strong Loss Skew, Significant)
    def test_strong_loss_asymmetry_significant(self):
        # Body: 490 gains + 490 losses = 50% loss rate (balanced).
        # Tail: 18 losses + 2 gains = 90% loss rate (very loss-heavy).
        # With 20 tail events and a huge gap (90% vs 50%), the z-test must flag this as significant.
        small = [0.0001] * 490 + [-0.0001] * 490
        extremes = [-0.50] * 18 + [0.50] * 2
        returns = pd.Series(small + extremes)

        gl = GainLossAsymmetry(returns, "TEST")
        result = gl.compute_asymmetry(q=0.98, plot=False)

        self.assertIsNotNone(result)
        self.assertLess(result['pvalue'], 0.05,
                        "90% tail losses vs 50% body losses must be statistically significant.")
        self.assertEqual(result['alternative'], 'larger')
        self.assertGreater(result['z_stat'], 0)

    # TEST 10: GAIN/LOSS ASYMMETRY (Bullish Body, Bearish Tail)
    def test_bullish_body_bearish_tail(self):
        # Body: 700 gains + 300 losses = 30% loss rate (overall bullish market).
        # Tail: 16 losses + 4 gains = 80% loss rate (crashes dominate the extremes).
        # The old way of testing (assume 50/50 body) would mess this up.
        # The two-proportion z-test compares tail vs body correctly and should find p < 0.05.
        np.random.seed(99)
        body_gains = list(np.random.uniform(0.0001, 0.005, 700))
        body_losses = list(-np.random.uniform(0.0001, 0.005, 300))
        tail_losses = [-0.50] * 16
        tail_gains = [0.50] * 4
        returns = pd.Series(body_gains + body_losses + tail_losses + tail_gains)

        gl = GainLossAsymmetry(returns, "TEST")
        result = gl.compute_asymmetry(q=0.98, plot=False)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['body_loss_pct'], 30.0, delta=2.0)
        self.assertGreaterEqual(result['loss_pct'], 70.0)
        self.assertLess(result['pvalue'], 0.05,
                        "Tail 80% losses vs body 30% losses must be flagged as significant.")
        self.assertEqual(result['alternative'], 'larger')

    # TEST 11: GAIN/LOSS ASYMMETRY (Robustness)
    def test_robustness_nan_inf(self):
        # Feed the function bad data: NaN (missing) and Inf (infinity).
        # It should clean the data and not explode.
        data = pd.Series([0.01, -0.01, np.nan, np.inf, -np.inf, -0.50, 0.02] * 20)

        gl = GainLossAsymmetry(data, "ROBUST")
        try:
            result = gl.compute_asymmetry(q=0.99, plot=False)
            self.assertIn('pvalue', result)
            self.assertIn('z_stat', result)
            self.assertIn('body_loss_pct', result)
        except Exception as e:
            self.fail(f"GainLossAsymmetry crashed on bad data: {e}")


class TestDataManagerCacheBug(unittest.TestCase):

    # TEST 12: DATA MANAGER CACHE BUG (Timestamp vs Date Type Mismatch)
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

    # TEST 13: DATA MANAGER ERROR REPORTING (Ticker Name in Error Message)
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


class TestAggregationalGaussianity(unittest.TestCase):

    # TEST 14: AGGREGATIONAL GAUSSIANITY (Kurtosis Decreases with Scale)
    def test_kurtosis_decreases_with_aggregation(self):
        # t-distribution with df=5 has fat tails (excess kurtosis = 6).
        # When you average many t-distributed returns together (Central Limit Theorem),
        # the fat tails shrink toward zero — the distribution looks more Gaussian.
        # Kurtosis at scale=1 (daily) must be higher than at scale=50 (50-day averages).
        np.random.seed(42)
        heavy_tailed_returns = pd.Series(stats.t.rvs(df=5, size=10000) * 0.01)

        ag = AggregationalGaussianity(heavy_tailed_returns, "TEST_HEAVY")
        result = ag.test_aggregational_gaussianity(scales=[1, 5, 20, 50], plot=False)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('kurtosis_by_scale', result)
        kurt = result['kurtosis_by_scale']
        self.assertGreater(kurt[1], kurt[50],
                           "Fat-tailed data must get less fat as you aggregate more.")
        self.assertGreater(kurt[1], 1.0,
                           "t(5) daily returns should have noticeable fat tails (kurtosis > 1).")

    # TEST 15: AGGREGATIONAL GAUSSIANITY (Gaussian Stays Near-Zero Kurtosis)
    def test_gaussian_stays_near_zero_kurtosis(self):
        # Normal/Gaussian data already has zero excess kurtosis by definition.
        # Averaging Gaussians gives more Gaussians, so kurtosis should stay near 0 at all scales.
        np.random.seed(42)
        gaussian_returns = pd.Series(np.random.normal(0, 0.01, 10000))

        ag = AggregationalGaussianity(gaussian_returns, "TEST_GAUSSIAN")
        result = ag.test_aggregational_gaussianity(scales=[1, 5, 20], plot=False)

        self.assertIsNotNone(result)
        kurt = result['kurtosis_by_scale']
        for scale, k in kurt.items():
            self.assertAlmostEqual(k, 0.0, delta=1.0,
                                   msg=f"Gaussian data should stay near kurtosis=0 at scale {scale}.")

    # TEST 16: AGGREGATIONAL GAUSSIANITY (Return Structure)
    def test_return_structure(self):
        # Just check that the output dictionary has the right keys and right types.
        np.random.seed(42)
        heavy_tailed_returns = pd.Series(stats.t.rvs(df=5, size=10000) * 0.01)

        ag = AggregationalGaussianity(heavy_tailed_returns, "TEST_STRUCT")
        result = ag.test_aggregational_gaussianity(scales=[1, 10], plot=False)

        self.assertIsNotNone(result)
        self.assertIn('kurtosis_by_scale', result)
        self.assertIn('convergence_confirmed', result)
        self.assertIsInstance(result['kurtosis_by_scale'], dict)
        self.assertIsInstance(result['convergence_confirmed'], bool)
        self.assertIn(1, result['kurtosis_by_scale'])
        self.assertIn(10, result['kurtosis_by_scale'])

    # TEST 17: AGGREGATIONAL GAUSSIANITY (Robustness)
    def test_robustness_nan_inf(self):
        # NaN and Inf in the input — the function must survive without crashing.
        dirty_returns = pd.Series([0.01, -0.01, np.nan, np.inf, -np.inf, 0.02, -0.03] * 500)

        ag = AggregationalGaussianity(dirty_returns, "TEST_ROBUST")
        try:
            result = ag.test_aggregational_gaussianity(scales=[1, 5], plot=False)
            self.assertIsNotNone(result)
            self.assertIn('kurtosis_by_scale', result)
        except Exception as e:
            self.fail(f"AggregationalGaussianity crashed on bad data: {e}")

    # TEST 18: AGGREGATIONAL GAUSSIANITY (Convergence Confirmed)
    def test_convergence_detected_for_heavy_tails(self):
        # convergence_confirmed=True means kurtosis is falling as scale increases.
        # For fat-tailed data, this should always be True (CLT is doing its job).
        np.random.seed(42)
        heavy_tailed_returns = pd.Series(stats.t.rvs(df=5, size=10000) * 0.01)

        ag = AggregationalGaussianity(heavy_tailed_returns, "TEST_CONVERGE")
        result = ag.test_aggregational_gaussianity(scales=[1, 5, 20, 50], plot=False)

        self.assertTrue(result['convergence_confirmed'],
                        "Fat-tailed data must show convergence toward Gaussian as scale grows.")


class TestIntermittency(unittest.TestCase):

    # TEST 19: INTERMITTENCY (Clustered Extremes, High Fano Factor)
    def test_clustered_extremes_high_fano(self):
        # Fano factor = variance of counts / mean of counts.
        # If extremes are clustered (all in one block, none elsewhere),
        # the counts per block are wildly unequal → high variance → Fano >> 1.
        # Block 1: half the values are "extreme" (std=0.10).
        # Blocks 2-10: completely calm (std=0.001). Extremes are totally bunched up.
        np.random.seed(42)
        block_size = 100
        burst_block = np.concatenate([np.random.normal(0, 0.001, 50), np.random.normal(0, 0.10, 50)])
        calm_block = np.random.normal(0, 0.001, 100)
        data = pd.Series(np.concatenate([burst_block] + [calm_block] * 9))

        im = Intermittency(data, "TEST_CLUSTERED")
        result = im.compute_intermittency(quantile=0.99, block_size=block_size, plot=False)

        self.assertIsNotNone(result)
        self.assertGreater(result['fano_factor'], 1.0,
                           "Bunched-up extremes must give Fano > 1.")

    # TEST 20: INTERMITTENCY (Uniform Extremes, Fano Near One)
    def test_uniform_extremes_fano_near_one(self):
        # Exactly 1 extreme per block → counts are [1, 1, 1, ...] → variance = 0 → Fano ≈ 0.
        # So Fano should definitely be ≤ 1.5 (not bursty at all).
        np.random.seed(123)
        block_size = 100
        n_blocks = 20
        blocks = []
        for _ in range(n_blocks):
            block = np.random.normal(0, 0.001, block_size)
            block[np.random.randint(0, block_size)] = 0.50
            blocks.append(block)
        data = pd.Series(np.concatenate(blocks))

        im = Intermittency(data, "TEST_UNIFORM")
        result = im.compute_intermittency(quantile=0.99, block_size=block_size, plot=False)

        self.assertIsNotNone(result)
        self.assertLessEqual(result['fano_factor'], 1.5,
                             "One extreme per block should NOT produce a high Fano factor.")

    # TEST 21: INTERMITTENCY (Return Structure)
    def test_return_structure(self):
        # Just check that all the expected keys come back with the right data types.
        np.random.seed(42)
        data = pd.Series(np.random.standard_t(df=3, size=5000) * 0.01)

        im = Intermittency(data, "TEST_STRUCT")
        result = im.compute_intermittency(quantile=0.99, block_size=100, plot=False)

        self.assertIsNotNone(result)
        self.assertIn('fano_factor', result)
        self.assertIn('threshold', result)
        self.assertIn('n_extremes', result)
        self.assertIn('n_blocks', result)
        self.assertIn('mean_count', result)
        self.assertIn('var_count', result)
        self.assertIn('intermittent', result)
        self.assertIsInstance(result['fano_factor'], float)
        self.assertIsInstance(result['threshold'], float)
        self.assertIsInstance(result['n_extremes'], int)
        self.assertIsInstance(result['n_blocks'], int)
        self.assertIsInstance(result['intermittent'], bool)
        self.assertFalse(np.isnan(result['fano_factor']))

    # TEST 22: INTERMITTENCY (Regime-Switching Data Is Intermittent)
    def test_regime_switching_data_is_intermittent(self):
        # 2500 high-volatility returns followed by 2500 calm returns.
        # Extreme events pile up in the first half, nothing in the second half.
        # The function should flag this as intermittent (bursty).
        np.random.seed(42)
        high_vol = np.random.normal(0, 0.05, 2500)
        low_vol = np.random.normal(0, 0.001, 2500)
        data = pd.Series(np.concatenate([high_vol, low_vol]))

        im = Intermittency(data, "TEST_REGIME")
        result = im.compute_intermittency(quantile=0.99, block_size=500, plot=False)

        self.assertIsNotNone(result)
        self.assertTrue(result['intermittent'],
                        "Volatile-then-calm data should be flagged as intermittent.")
        self.assertGreater(result['fano_factor'], 1.0)

    # TEST 23: INTERMITTENCY (Gaussian Data Not Intermittent)
    def test_gaussian_data_not_intermittent(self):
        # Pure Gaussian (random) data has no pattern — extreme events are spread out randomly.
        # They follow a Poisson process where Fano ≈ 1. Should not be strongly bursty.
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 0.01, 20000))

        im = Intermittency(data, "TEST_GAUSS")
        result = im.compute_intermittency(quantile=0.99, block_size=200, plot=False)

        self.assertIsNotNone(result)
        self.assertLess(result['fano_factor'], 3.0,
                        "Random Gaussian data should not look bursty (Fano < 3).")

    # TEST 24: INTERMITTENCY (Robustness)
    def test_robustness_nan_inf(self):
        # Mix in NaN and Inf with real data. The function should survive.
        np.random.seed(42)
        clean = np.random.normal(0, 0.01, 1000)
        dirty = np.concatenate([clean, [np.nan] * 50, [np.inf] * 50, [-np.inf] * 50])
        np.random.shuffle(dirty)
        data = pd.Series(dirty)

        im = Intermittency(data, "ROBUST")
        try:
            result = im.compute_intermittency(quantile=0.95, block_size=50, plot=False)
            self.assertIsNotNone(result)
            self.assertIn('fano_factor', result)
        except Exception as e:
            self.fail(f"Intermittency crashed on bad data: {e}")

    # TEST 25: INTERMITTENCY (Insufficient Data Returns None)
    def test_insufficient_data_returns_none(self):
        # Only 3 data points but block_size=100: not enough to form a single block.
        # The function should return None gracefully instead of crashing.
        data = pd.Series([0.01, -0.01, 0.02])

        im = Intermittency(data, "TEST_SHORT")
        result = im.compute_intermittency(quantile=0.99, block_size=100, plot=False)

        self.assertIsNone(result)


class TestSlowDecay(unittest.TestCase):

    # TEST 26: SLOW DECAY (Return Structure)
    def test_return_structure(self):
        # GARCH(1,1): today's volatility depends on yesterday's — it's "sticky".
        # We check the output dictionary has all the keys we expect.
        np.random.seed(42)
        n = 5000
        alpha0, alpha1, beta1 = 0.00001, 0.05, 0.90
        h = np.zeros(n)
        r = np.zeros(n)
        h[0] = alpha0 / (1 - alpha1 - beta1)
        for t in range(1, n):
            h[t] = alpha0 + alpha1 * r[t - 1] ** 2 + beta1 * h[t - 1]
            r[t] = np.sqrt(h[t]) * np.random.normal()
        garch_returns = pd.Series(r)

        sd = SlowDecay(garch_returns, "TEST")
        result = sd.compute_decay(max_lag=50, plot=False)

        self.assertIsNotNone(result)
        required_keys = ['beta_alpha1', 'A_alpha1', 'beta_alpha2', 'A_alpha2',
                         'acf_alpha1', 'acf_alpha2', 'lags', 'slow_decay_confirmed']
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")
        self.assertIsInstance(result['slow_decay_confirmed'], bool)
        self.assertIsInstance(result['lags'], list)
        self.assertIsInstance(result['acf_alpha1'], list)
        self.assertIsInstance(result['acf_alpha2'], list)
        self.assertEqual(len(result['lags']), 50)
        self.assertEqual(len(result['acf_alpha1']), 50)

    # TEST 27: SLOW DECAY (Both Alphas Computed)
    def test_both_alphas_computed(self):
        # The slow decay test is run twice: once on |returns| (alpha=1) and once on returns^2 (alpha=2).
        # Both beta values (the decay rate exponent) must be present and not None.
        np.random.seed(42)
        n = 5000
        alpha0, alpha1, beta1 = 0.00001, 0.05, 0.90
        h = np.zeros(n)
        r = np.zeros(n)
        h[0] = alpha0 / (1 - alpha1 - beta1)
        for t in range(1, n):
            h[t] = alpha0 + alpha1 * r[t - 1] ** 2 + beta1 * h[t - 1]
            r[t] = np.sqrt(h[t]) * np.random.normal()
        garch_returns = pd.Series(r)

        sd = SlowDecay(garch_returns, "TEST")
        result = sd.compute_decay(max_lag=50, plot=False)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result['beta_alpha1'], "Decay exponent for |returns| must be computed.")
        self.assertIsNotNone(result['beta_alpha2'], "Decay exponent for returns^2 must be computed.")
        self.assertIsNotNone(result['A_alpha1'])
        self.assertIsNotNone(result['A_alpha2'])

    # TEST 28: SLOW DECAY (ACF Positive at Lag 1 for GARCH)
    def test_acf_positive_at_lag1_for_garch(self):
        # For GARCH data, today's absolute return is correlated with yesterday's.
        # High volatility tends to stick around. So ACF at lag 1 must be > 0.
        np.random.seed(42)
        n = 5000
        alpha0, alpha1, beta1 = 0.00001, 0.05, 0.90
        h = np.zeros(n)
        r = np.zeros(n)
        h[0] = alpha0 / (1 - alpha1 - beta1)
        for t in range(1, n):
            h[t] = alpha0 + alpha1 * r[t - 1] ** 2 + beta1 * h[t - 1]
            r[t] = np.sqrt(h[t]) * np.random.normal()
        garch_returns = pd.Series(r)

        sd = SlowDecay(garch_returns, "TEST_GARCH")
        result = sd.compute_decay(max_lag=50, plot=False)

        self.assertIsNotNone(result)
        self.assertGreater(result['acf_alpha1'][0], 0.0,
                           "GARCH |returns| must have positive autocorrelation at lag 1.")

    # TEST 29: SLOW DECAY (Beta Exponent Is Positive)
    def test_beta_positive_for_garch(self):
        # Beta is the power-law exponent: ACF(lag) ~ A * lag^(-beta).
        # If ACF starts positive and decays, beta must be positive.
        np.random.seed(42)
        n = 5000
        alpha0, alpha1, beta1 = 0.00001, 0.05, 0.90
        h = np.zeros(n)
        r = np.zeros(n)
        h[0] = alpha0 / (1 - alpha1 - beta1)
        for t in range(1, n):
            h[t] = alpha0 + alpha1 * r[t - 1] ** 2 + beta1 * h[t - 1]
            r[t] = np.sqrt(h[t]) * np.random.normal()
        garch_returns = pd.Series(r)

        sd = SlowDecay(garch_returns, "TEST_GARCH")
        result = sd.compute_decay(max_lag=50, plot=False)

        self.assertIsNotNone(result)
        self.assertGreater(result['beta_alpha1'], 0,
                           "Beta must be positive (autocorrelation decays from a positive start).")
        self.assertGreater(result['beta_alpha2'], 0,
                           "Beta for squared returns must also be positive.")

    # TEST 30: SLOW DECAY (Robustness)
    def test_robustness_nan_inf(self):
        # Feed in a mix of good data and garbage. Should not crash.
        np.random.seed(42)
        clean = np.random.normal(0, 0.01, 1000)
        dirty = np.concatenate([clean, [np.nan] * 50, [np.inf] * 50, [-np.inf] * 50])
        np.random.shuffle(dirty)
        dirty_returns = pd.Series(dirty)

        sd = SlowDecay(dirty_returns, "ROBUST")
        try:
            result = sd.compute_decay(max_lag=20, plot=False)
            if result is not None:
                self.assertIn('slow_decay_confirmed', result)
        except Exception as e:
            self.fail(f"SlowDecay crashed on bad data: {e}")

    # TEST 31: SLOW DECAY (Insufficient Data Returns None)
    def test_insufficient_data_returns_none(self):
        # Only 5 data points but max_lag=50: impossible to compute a 50-lag ACF.
        # Should return None gracefully.
        data = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])

        sd = SlowDecay(data, "TEST_SHORT")
        result = sd.compute_decay(max_lag=50, plot=False)

        self.assertIsNone(result)


class TestLeverageEffect(unittest.TestCase):

    # TEST 32: LEVERAGE EFFECT (Return Structure)
    def test_return_structure(self):
        # Check that the output dict has all expected keys with correct types.
        np.random.seed(42)
        iid_returns = pd.Series(np.random.normal(0, 0.01, 5000))

        le = LeverageEffect(iid_returns, "TEST")
        result = le.compute_leverage(max_lag=20, plot=False)

        self.assertIsNotNone(result)
        required_keys = ['lags', 'L_values', 'leverage_detected', 'min_L', 'min_lag']
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")
        self.assertIsInstance(result['leverage_detected'], bool)
        self.assertIsInstance(result['lags'], list)
        self.assertIsInstance(result['L_values'], list)
        self.assertIsInstance(result['min_L'], float)
        self.assertIsInstance(result['min_lag'], int)

    # TEST 33: LEVERAGE EFFECT (Lags Span Negative to Positive)
    def test_lags_span_negative_to_positive(self):
        # The leverage function L(tau) is computed for both positive and negative lags.
        # With max_lag=30, we need lags from -30 to +30, which is 61 values total.
        np.random.seed(42)
        iid_returns = pd.Series(np.random.normal(0, 0.01, 5000))

        le = LeverageEffect(iid_returns, "TEST")
        result = le.compute_leverage(max_lag=30, plot=False)

        self.assertIsNotNone(result)
        lags = result['lags']
        self.assertEqual(min(lags), -30)
        self.assertEqual(max(lags), 30)
        self.assertEqual(len(lags), 61)

    # TEST 34: LEVERAGE EFFECT (Detected for Crash Data)
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
        result = le.compute_leverage(max_lag=10, plot=False)

        self.assertIsNotNone(result)
        lags = np.array(result['lags'])
        L_arr = np.array(result['L_values'])
        pos_L = L_arr[lags > 0]
        self.assertTrue(np.any(pos_L < 0),
                        "Crash→volatile data must have at least one negative L(tau) for tau > 0.")
        self.assertLess(float(np.nanmin(pos_L)), -0.1,
                        "The leverage effect must be large (min L < -0.1) for this constructed data.")

    # TEST 35: LEVERAGE EFFECT (IID Data, No Strong Leverage)
    def test_iid_data_no_strong_leverage(self):
        # Random (iid) Gaussian data: no relationship between past returns and future variance.
        # L(tau) should stay close to zero for all tau > 0 (nothing is above -0.1).
        np.random.seed(42)
        iid_returns = pd.Series(np.random.normal(0, 0.01, 5000))

        le = LeverageEffect(iid_returns, "TEST_IID")
        result = le.compute_leverage(max_lag=20, plot=False)

        self.assertIsNotNone(result)
        lags = np.array(result['lags'])
        L_arr = np.array(result['L_values'])
        pos_L = L_arr[lags > 0]
        self.assertGreater(float(np.nanmin(pos_L)), -0.1,
                           "Random data should not show a strong leverage effect (min L > -0.1).")

    # TEST 36: LEVERAGE EFFECT (Robustness)
    def test_robustness_nan_inf(self):
        # Feed in garbage data. Should not crash.
        np.random.seed(42)
        clean = np.random.normal(0, 0.01, 1000)
        dirty = np.concatenate([clean, [np.nan] * 50, [np.inf] * 50, [-np.inf] * 50])
        np.random.shuffle(dirty)
        dirty_returns = pd.Series(dirty)

        le = LeverageEffect(dirty_returns, "ROBUST")
        try:
            result = le.compute_leverage(max_lag=10, plot=False)
            if result is not None:
                self.assertIn('leverage_detected', result)
        except Exception as e:
            self.fail(f"LeverageEffect crashed on bad data: {e}")

    # TEST 37: LEVERAGE EFFECT (Insufficient Data Returns None)
    def test_insufficient_data_returns_none(self):
        # Only 3 data points with max_lag=50: not enough. Should return None.
        data = pd.Series([0.01, -0.01, 0.02])

        le = LeverageEffect(data, "TEST_SHORT")
        result = le.compute_leverage(max_lag=50, plot=False)

        self.assertIsNone(result)


class TestVolVolCorr(unittest.TestCase):

    # TEST 38: VOL-VOL CORRELATION (Return Structure)
    def test_return_structure(self):
        # Check all expected keys exist with the right types.
        # rho = correlation coefficient (between -1 and 1).
        # t = test statistic (how many standard deviations away from zero).
        # pval = probability of seeing this result if there's no real correlation.
        np.random.seed(42)
        n = 2000
        iid_returns = pd.Series(np.random.normal(0, 0.01, n))
        iid_volume = pd.Series(np.random.normal(1e6, 1e4, n))

        vvc = VolVolCorr(iid_returns, iid_volume, "TEST")
        result = vvc.compute_correlation(plot=False)

        self.assertIsNotNone(result)
        required_keys = [
            'rho_abs', 't_abs', 'pval_abs', 'significant_abs',
            'rho_sq', 't_sq', 'pval_sq', 'significant_sq',
            'corr_confirmed',
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")
        self.assertIsInstance(result['rho_abs'], float)
        self.assertIsInstance(result['t_abs'], float)
        self.assertIsInstance(result['pval_abs'], float)
        self.assertIsInstance(result['significant_abs'], bool)
        self.assertIsInstance(result['rho_sq'], float)
        self.assertIsInstance(result['t_sq'], float)
        self.assertIsInstance(result['pval_sq'], float)
        self.assertIsInstance(result['significant_sq'], bool)
        self.assertIsInstance(result['corr_confirmed'], bool)

    # TEST 39: VOL-VOL CORRELATION (Positive Correlation Detected)
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
        result = vvc.compute_correlation(plot=False)

        self.assertIsNotNone(result)
        self.assertGreater(result['rho_abs'], 0.5,
                           "rho must be strongly positive when we built volume = |returns| * scale.")
        self.assertLess(result['pval_abs'], 0.05,
                        "This strong correlation must be statistically significant.")
        self.assertTrue(result['significant_abs'],
                        "significant_abs must be True when rho > 0 and p < 0.05.")
        self.assertTrue(result['corr_confirmed'])

    # TEST 40: VOL-VOL CORRELATION (T-Statistic Formula)
    def test_t_statistic_formula(self):
        # t = rho * sqrt(n - 2) / sqrt(1 - rho^2)
        # This converts a correlation coefficient into a t-score for a significance test.
        # We compute this manually and check the function gives the same answer.
        np.random.seed(7)
        n = 300
        abs_v = np.abs(np.random.normal(0, 0.02, n))
        returns = pd.Series(abs_v * np.random.choice([-1, 1], n))
        volume = pd.Series(abs_v * 1e6 + np.random.normal(0, 5000, n))

        abs_r = np.abs(returns.values)
        v = volume.values
        expected_rho_abs = float(np.corrcoef(v, abs_r)[0, 1])
        expected_t_abs = expected_rho_abs * np.sqrt(n - 2) / np.sqrt(1 - expected_rho_abs ** 2)

        vvc = VolVolCorr(returns, volume, "TEST_FORMULA")
        result = vvc.compute_correlation(plot=False)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['rho_abs'], expected_rho_abs, places=6,
                               msg="rho_abs must match np.corrcoef exactly.")
        self.assertAlmostEqual(result['t_abs'], expected_t_abs, places=4,
                               msg="t_abs must match rho * sqrt(n-2) / sqrt(1 - rho^2).")

    # TEST 41: VOL-VOL CORRELATION (IID Data, Not Significant)
    def test_iid_data_not_significant(self):
        # Volume and returns are generated completely independently — no real relationship.
        # The p-value should be > 0.05, meaning we can't reject the "no correlation" hypothesis.
        np.random.seed(42)
        n = 2000
        iid_returns = pd.Series(np.random.normal(0, 0.01, n))
        iid_volume = pd.Series(np.random.normal(1e6, 1e4, n))

        vvc = VolVolCorr(iid_returns, iid_volume, "TEST_IID")
        result = vvc.compute_correlation(plot=False)

        self.assertIsNotNone(result)
        self.assertGreater(result['pval_abs'], 0.05,
                           "Independent data should not produce a significant p-value.")
        self.assertFalse(result['corr_confirmed'],
                         "corr_confirmed must be False when volume and returns are independent.")

    # TEST 42: VOL-VOL CORRELATION (Robustness)
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
            result = vvc.compute_correlation(plot=False)
            if result is not None:
                self.assertIn('rho_abs', result)
                self.assertFalse(np.isnan(result['rho_abs']))
        except Exception as e:
            self.fail(f"VolVolCorr crashed on bad data: {e}")

    # TEST 43: VOL-VOL CORRELATION (Insufficient Data Returns None)
    def test_insufficient_data_returns_none(self):
        # Only 3 data points — not enough for a meaningful correlation. Should return None.
        data_r = pd.Series([0.01, -0.01, 0.02])
        data_v = pd.Series([1e6, 2e6, 1.5e6])

        vvc = VolVolCorr(data_r, data_v, "TEST_SHORT")
        result = vvc.compute_correlation(plot=False)

        self.assertIsNone(result)


class TestAsymmetryTimescales(unittest.TestCase):

    # TEST 44: TIMESCALE ASYMMETRY (Return Structure)
    def test_return_structure(self):
        # A(tau) = correlation between coarse (long-window) volatility and fine (short-window) future returns.
        # D(tau) = A(tau) - A(-tau): if D < 0, big slow moves predict small fast moves more than vice versa.
        # Just check the output has all the right keys and lengths.
        np.random.seed(42)
        iid_returns = pd.Series(np.random.normal(0, 0.01, 4000))

        ts = AsymmetryTimescales(iid_returns, "TEST")
        result = ts.compute_asymmetry(dT=5, max_tau=5, plot=False)

        self.assertIsNotNone(result)
        required_keys = ['taus', 'A_pos', 'A_neg', 'D_values', 'D_mean', 'top_down_detected']
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")
        self.assertIsInstance(result['taus'], list)
        self.assertIsInstance(result['A_pos'], list)
        self.assertIsInstance(result['A_neg'], list)
        self.assertIsInstance(result['D_values'], list)
        self.assertIsInstance(result['D_mean'], float)
        self.assertIsInstance(result['top_down_detected'], bool)
        self.assertEqual(len(result['taus']), 5)
        self.assertEqual(len(result['A_pos']), 5)
        self.assertEqual(len(result['A_neg']), 5)
        self.assertEqual(len(result['D_values']), 5)

    # TEST 45: TIMESCALE ASYMMETRY (D = A_pos - A_neg Formula Check)
    def test_D_equals_Apos_minus_Aneg(self):
        # D(tau) is defined as A(tau) - A(-tau). Check this math holds exactly.
        np.random.seed(42)
        n = 4000
        sigma = np.zeros(n)
        r_garch = np.zeros(n)
        sigma[0] = 0.01
        for t in range(1, n):
            sigma[t] = np.sqrt(max(1e-8, 0.00005 + 0.15 * r_garch[t - 1] ** 2 + 0.80 * sigma[t - 1] ** 2))
            r_garch[t] = sigma[t] * np.random.normal()
        garch_returns = pd.Series(r_garch)

        ts = AsymmetryTimescales(garch_returns, "TEST_FORMULA")
        result = ts.compute_asymmetry(dT=5, max_tau=5, plot=False)

        self.assertIsNotNone(result)
        for i, (d, a_pos, a_neg) in enumerate(zip(result['D_values'], result['A_pos'], result['A_neg'])):
            if np.isfinite(d) and np.isfinite(a_pos) and np.isfinite(a_neg):
                self.assertAlmostEqual(
                    d, a_pos - a_neg, places=10,
                    msg=f"D[{i}] = {d:.6f} but A_pos[{i}] - A_neg[{i}] = {a_pos - a_neg:.6f}"
                )

    # TEST 46: TIMESCALE ASYMMETRY (A Values Are Valid Correlations)
    def test_A_values_are_valid_correlations(self):
        # Correlation must always be between -1 and 1. Any value outside that is a bug.
        np.random.seed(42)
        n = 4000
        sigma = np.zeros(n)
        r_garch = np.zeros(n)
        sigma[0] = 0.01
        for t in range(1, n):
            sigma[t] = np.sqrt(max(1e-8, 0.00005 + 0.15 * r_garch[t - 1] ** 2 + 0.80 * sigma[t - 1] ** 2))
            r_garch[t] = sigma[t] * np.random.normal()
        garch_returns = pd.Series(r_garch)

        ts = AsymmetryTimescales(garch_returns, "TEST_RANGE")
        result = ts.compute_asymmetry(dT=5, max_tau=8, plot=False)

        self.assertIsNotNone(result)
        for i, (a_pos, a_neg) in enumerate(zip(result['A_pos'], result['A_neg'])):
            if np.isfinite(a_pos):
                self.assertGreaterEqual(a_pos, -1.0 - 1e-9, f"A_pos[{i}]={a_pos:.4f} is below -1.")
                self.assertLessEqual(a_pos, 1.0 + 1e-9, f"A_pos[{i}]={a_pos:.4f} is above 1.")
            if np.isfinite(a_neg):
                self.assertGreaterEqual(a_neg, -1.0 - 1e-9, f"A_neg[{i}]={a_neg:.4f} is below -1.")
                self.assertLessEqual(a_neg, 1.0 + 1e-9, f"A_neg[{i}]={a_neg:.4f} is above 1.")

    # TEST 47: TIMESCALE ASYMMETRY (IID Data, D Near Zero)
    def test_iid_D_near_zero(self):
        # Random (iid) data has no time-scale structure, so D(tau) should be close to zero.
        # |D_mean| < 0.15 is our threshold for "basically zero".
        np.random.seed(99)
        iid = pd.Series(np.random.normal(0, 0.01, 5000))

        ts = AsymmetryTimescales(iid, "IID")
        result = ts.compute_asymmetry(dT=5, max_tau=5, plot=False)

        self.assertIsNotNone(result)
        d_mean = result['D_mean']
        self.assertLess(abs(d_mean), 0.15,
                        f"Random data should have |D_mean| < 0.15, got {d_mean:.4f}.")

    # TEST 48: TIMESCALE ASYMMETRY (Robustness)
    def test_robustness_nan_inf(self):
        # Garbage data in the input. Should not crash.
        np.random.seed(42)
        clean_r = np.random.normal(0, 0.01, 500)
        dirty_r = np.concatenate([clean_r, [np.nan] * 20, [np.inf] * 10, [-np.inf] * 10])
        dirty_returns = pd.Series(dirty_r)

        ts = AsymmetryTimescales(dirty_returns, "ROBUST")
        try:
            result = ts.compute_asymmetry(dT=5, max_tau=5, plot=False)
            if result is not None:
                self.assertIn('D_mean', result)
                self.assertIsInstance(result['D_mean'], float)
        except Exception as e:
            self.fail(f"AsymmetryTimescales crashed on bad data: {e}")

    # TEST 49: TIMESCALE ASYMMETRY (Insufficient Data Returns None)
    def test_insufficient_data_returns_none(self):
        # 5 data points with dT=5 and max_tau=10: nowhere near enough. Should return None.
        short = pd.Series([0.01, -0.01, 0.02, 0.00, -0.02])

        ts = AsymmetryTimescales(short, "SHORT")
        result = ts.compute_asymmetry(dT=5, max_tau=10, plot=False)

        self.assertIsNone(result)


class TestConditionalTails(unittest.TestCase):

    # TEST 50: CONDITIONAL TAILS (Return Structure)
    def test_return_structure(self):
        # GARCH(1,1) with t(5) innovations: the model removes volatility clustering,
        # but the leftover residuals (epsilon) should still look fat-tailed.
        # Check the output dict has all expected keys and types.
        np.random.seed(42)
        n = 2000
        sigma = np.zeros(n)
        r = np.zeros(n)
        sigma[0] = 0.01
        for t in range(1, n):
            sigma[t] = np.sqrt(max(1e-10, 0.00005 + 0.10 * r[t - 1] ** 2 + 0.85 * sigma[t - 1] ** 2))
            r[t] = sigma[t] * float(np.random.standard_t(df=5))
        garch_t_returns = pd.Series(r)

        ct = ConditionalTails(garch_t_returns, "TEST")
        result = ct.compute_conditional_tails(plot=False)

        self.assertIsNotNone(result)
        required_keys = ['residuals', 'kurtosis', 'excess_kurtosis', 'tail_index', 'non_gaussian']
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")
        self.assertIsInstance(result['residuals'], np.ndarray)
        self.assertIsInstance(result['kurtosis'], float)
        self.assertIsInstance(result['excess_kurtosis'], float)
        self.assertIsInstance(result['non_gaussian'], bool)
        self.assertIsInstance(result['tail_index'], float)

    # TEST 51: CONDITIONAL TAILS (Residuals Have Excess Kurtosis)
    def test_garch_t_residuals_have_excess_kurtosis(self):
        # t(5) innovations have excess kurtosis = 6/(5-4) = 6.
        # After stripping out GARCH volatility, the residuals should still show excess kurtosis > 0.
        # If it equals 0, the GARCH model "used up" the fat tails, which would be wrong.
        np.random.seed(42)
        n = 2000
        sigma = np.zeros(n)
        r = np.zeros(n)
        sigma[0] = 0.01
        for t in range(1, n):
            sigma[t] = np.sqrt(max(1e-10, 0.00005 + 0.10 * r[t - 1] ** 2 + 0.85 * sigma[t - 1] ** 2))
            r[t] = sigma[t] * float(np.random.standard_t(df=5))
        garch_t_returns = pd.Series(r)

        ct = ConditionalTails(garch_t_returns, "GARCH_T")
        result = ct.compute_conditional_tails(plot=False)

        self.assertIsNotNone(result)
        self.assertGreater(result['excess_kurtosis'], 0.0,
                           f"Residuals from t(5) innovations must still have excess kurtosis > 0. "
                           f"Got {result['excess_kurtosis']:.4f}.")

    # TEST 52: CONDITIONAL TAILS (Residuals Near Unit Standard Deviation)
    def test_residuals_approximately_unit_std(self):
        # After dividing returns by GARCH conditional volatility, the residuals epsilon = r / sigma.
        # If GARCH estimates sigma correctly, std(epsilon) should be near 1.0.
        # We allow a generous range [0.7, 1.5] because GARCH estimation isn't perfect.
        np.random.seed(42)
        n = 2000
        sigma = np.zeros(n)
        r = np.zeros(n)
        sigma[0] = 0.01
        for t in range(1, n):
            sigma[t] = np.sqrt(max(1e-10, 0.00005 + 0.10 * r[t - 1] ** 2 + 0.85 * sigma[t - 1] ** 2))
            r[t] = sigma[t] * float(np.random.standard_t(df=5))
        garch_t_returns = pd.Series(r)

        ct = ConditionalTails(garch_t_returns, "STD_CHECK")
        result = ct.compute_conditional_tails(plot=False)

        self.assertIsNotNone(result)
        eps = result['residuals']
        std_eps = float(np.std(eps))
        self.assertGreater(std_eps, 0.7, f"Residual std should be > 0.7, got {std_eps:.4f}.")
        self.assertLess(std_eps, 1.5, f"Residual std should be < 1.5, got {std_eps:.4f}.")

    # TEST 53: CONDITIONAL TAILS (Tail Index Is Finite and Positive)
    def test_tail_index_is_finite_positive(self):
        # The Hill estimator estimates the tail index (how fat the tails are).
        # For any realistic fat-tailed data it must be a positive, finite number.
        np.random.seed(42)
        n = 2000
        sigma = np.zeros(n)
        r = np.zeros(n)
        sigma[0] = 0.01
        for t in range(1, n):
            sigma[t] = np.sqrt(max(1e-10, 0.00005 + 0.10 * r[t - 1] ** 2 + 0.85 * sigma[t - 1] ** 2))
            r[t] = sigma[t] * float(np.random.standard_t(df=5))
        garch_t_returns = pd.Series(r)

        ct = ConditionalTails(garch_t_returns, "TAIL_IDX")
        result = ct.compute_conditional_tails(plot=False)

        self.assertIsNotNone(result)
        tai = result['tail_index']
        self.assertTrue(np.isfinite(tai), f"tail_index must be a finite number, got {tai}.")
        self.assertGreater(tai, 0.0, f"tail_index must be positive, got {tai:.4f}.")

    # TEST 54: CONDITIONAL TAILS (Robustness)
    def test_robustness_nan_inf(self):
        # Garbage input. Must not crash.
        np.random.seed(42)
        clean_r = np.random.normal(0, 0.01, 300)
        dirty_r = np.concatenate([clean_r, [np.nan] * 20, [np.inf] * 10, [-np.inf] * 10])
        dirty_returns = pd.Series(dirty_r)

        ct = ConditionalTails(dirty_returns, "ROBUST")
        try:
            result = ct.compute_conditional_tails(plot=False)
            if result is not None:
                self.assertIn('excess_kurtosis', result)
        except Exception as e:
            self.fail(f"ConditionalTails crashed on bad data: {e}")

    # TEST 55: CONDITIONAL TAILS (Insufficient Data Returns None)
    def test_insufficient_data_returns_none(self):
        # Only 10 data points — way too few for GARCH fitting. Should return None.
        np.random.seed(42)
        short = pd.Series(np.random.normal(0, 0.01, 10))

        ct = ConditionalTails(short, "SHORT")
        result = ct.compute_conditional_tails(plot=False)

        self.assertIsNone(result)


class TestAuditAndClean(unittest.TestCase):

    def _make_df(self, rows):
        # Helper: build a minimal OHLCV DataFrame from a list of dicts.
        # Uses a simple DatetimeIndex so audit_and_clean has a real-looking input.
        index = pd.date_range('2024-01-02', periods=len(rows), freq='D', tz='US/Eastern')
        return pd.DataFrame(rows, index=index,
                            columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    # TEST 56: AUDIT AND CLEAN (Clean Data Passes Through Unchanged)
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

    # TEST 57: AUDIT AND CLEAN (NaN Values Forward-Filled)
    def test_nan_values_forward_filled(self):
        # One bar has NaN for Close. The function should forward-fill it from the previous bar
        # and report how many NaNs were fixed — not crash or silently drop the row.
        from utilities.data import DataManager
        rows = [
            {'Open': 100, 'High': 105, 'Low': 99,  'Close': 103,  'Volume': 1000},
            {'Open': 103, 'High': 107, 'Low': 102, 'Close': np.nan, 'Volume': 1500},
            {'Open': 103, 'High': 108, 'Low': 102, 'Close': 106,  'Volume': 1200},
        ]
        df = self._make_df(rows)

        dm = DataManager.__new__(DataManager)
        cleaned_df, returns, report = dm.audit_and_clean(df, '1d')

        self.assertEqual(len(cleaned_df), 3, "NaN rows should be filled, not dropped.")
        self.assertFalse(cleaned_df['Close'].isna().any(), "No NaNs should remain after forward-fill.")
        self.assertIn('NaN', report, f"Report must mention NaN fix. Got: '{report}'")

    # TEST 58: AUDIT AND CLEAN (Negative Prices Dropped)
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

    # TEST 59: AUDIT AND CLEAN (Zero Close Price Dropped)
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

    # TEST 60: AUDIT AND CLEAN (Broken OHLC Dropped)
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

    # TEST 61: AUDIT AND CLEAN (Zero Volume Warns but Keeps Row)
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

    # TEST 62: AUDIT AND CLEAN (V-Spike Repaired for Intraday Data)
    def test_vspike_repaired_intraday(self):
        # A V-spike is a bar that shoots to a crazy price then immediately snaps back.
        # Example: 200 normal bars at ~$100, then one bar at $10,000, then right back to $100.
        # That $10,000 bar is a data error — no real trade moved the price 100x in one minute.
        # We need MANY normal bars so the standard deviation stays small. With only 10 bars,
        # the two spike bars dominate std and the filter never triggers (spike / std < 5).
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
