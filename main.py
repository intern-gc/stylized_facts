
import pandas as pd
from data import DataManager
from autocorrelation import AbsenceOfAutocorrelationTest
from heavytails import HeavyTailsEVT
from volclustering import VolatilityClustering
from gainloss import GainLossAsymmetry
from aggregational import AggregationalGaussianity
from intermittency import Intermittency
from decay import SlowDecay
from leverage import LeverageEffect
from volvolcorr import VolVolCorr
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def run_analysis():
    # --- CONFIGURATION ---
    ticker = "SPY"
    interval = "1h"  # Try "1m", "5m", "1h", or "1d"

    # Adjust dates based on timeframe to ensure enough data
    start_date = "2015-01-01"  # Daily needs ~5 years for good EVT
    end_date = "2025-01-01"

    # --- STEP 1: GET DATA ---
    dm = DataManager()
    df, returns, report = dm.get_data(ticker, start_date, end_date, interval)

    if df.empty:
        print(f"\n❌ ABORTING: {report}")
        return

    print(f"\n📊 [DATA READY]: {len(df)} bars | {report}")

    # --- STEP 2: RUN TESTS ---
    print("\n" + "=" * 40)
    print(f"🔎 STARTING STYLIZED FACTS ANALYSIS: {ticker} ({interval})")
    print("=" * 40)

    # FACT 1: Autocorrelation
    ac_tester = AbsenceOfAutocorrelationTest(returns, ticker)
    # Capture the returned lags!
    sig_lags = ac_tester.test_linear_independence(lags=40)

    # FACT 2: Heavy Tails (EVT)
    # Dynamic block size based on interval
    bs_map = {'1m': 390, '5m': 78, '1h': 7, '1d': 21}  # 1d = 21 days (approx 1 month)
    block_size = bs_map.get(interval, 21)

    evt_tester = HeavyTailsEVT(returns, ticker)
    # Capture the returned xi/alpha!
    evt_result = evt_tester.run_mle_fit(block_size=block_size)

    # FACT 3: Volatility Clustering
    vc_tester = VolatilityClustering(returns, ticker)
    c2_values, vc_sig_lags = vc_tester.compute_c2(max_lag=40)

    # FACT 4: Gain/Loss Asymmetry (Ratliff-Crain et al., 2024)
    gl_tester = GainLossAsymmetry(returns, ticker)
    gl_result = gl_tester.compute_asymmetry(q=0.95)

    # FACT 5: Aggregational Gaussianity
    ag_tester = AggregationalGaussianity(returns, ticker)
    ag_scales = {'1m': [1, 5, 30, 78, 390], '5m': [1, 5, 16, 78], '1h': [1, 4, 7, 14, 21, 26], '1d': [1, 5, 10, 21]}
    ag_result = ag_tester.test_aggregational_gaussianity(scales=ag_scales.get(interval, [1, 4, 7, 13, 26]))

    # FACT 6: Intermittency
    im_tester = Intermittency(returns, ticker)
    im_block_sizes = {'1m': 390, '5m': 78, '1h': 168, '1d': 21}
    im_result = im_tester.compute_intermittency(quantile=0.99, block_size=im_block_sizes.get(interval, 168))

    # FACT 7: Slow Decay of Autocorrelation
    sd_tester = SlowDecay(returns, ticker)
    sd_max_lags = {'1m': 200, '5m': 150, '1h': 100, '1d': 60}
    sd_result = sd_tester.compute_decay(max_lag=sd_max_lags.get(interval, 100))

    # FACT 8: Leverage Effect
    le_tester = LeverageEffect(returns, ticker)
    le_max_lags = {'1m': 30, '5m': 30, '1h': 50, '1d': 30}
    le_result = le_tester.compute_leverage(max_lag=le_max_lags.get(interval, 50))

    # FACT 9: Volume-Volatility Correlation
    volume = df['Volume']
    vvc_tester = VolVolCorr(returns, volume, ticker)
    vvc_result = vvc_tester.compute_correlation()

    # --- STEP 3: FINAL REPORT CARD ---
    print("\n" + "=" * 40)
    print(f"📝 FINAL REPORT CARD: {ticker}")
    print("=" * 40)

    # Report Fact 1
    if len(sig_lags) == 0:
        print("✅ FACT 1 (No Autocorrelation): PASSED")
    else:
        print(f"❌ FACT 1 (No Autocorrelation): FAILED (Violated at {len(sig_lags)} lags)")

    # Report Fact 2
    if evt_result:
        xi, alpha = evt_result
        if xi > 0:
            print(f"✅ FACT 2 (Heavy Tails): CONFIRMED (Alpha = {alpha:.2f})")
            print(
                f"   -> Interpretation: Market has finite variance" if alpha > 2 else "   -> Interpretation: INFINITE VARIANCE (Extreme Risk)")
        else:
            print(f"❌ FACT 2 (Heavy Tails): FAILED (Thin/Bounded Tails detected)")
    else:
        print("⚠️ FACT 2 (Heavy Tails): INCONCLUSIVE (Insufficient Data)")

    # Report Fact 3
    if len(vc_sig_lags) > 0:
        print(f"✅ FACT 3 (Volatility Clustering): CONFIRMED (Significant at {len(vc_sig_lags)} lags)")
    else:
        print("❌ FACT 3 (Volatility Clustering): NOT DETECTED")

    # Report Fact 4
    # H0: Extreme moves behave identically to normal market moves
    # (i.e., tail loss rate = body loss rate for the same time period)
    if gl_result:
        loss_pct = gl_result['loss_pct']
        body_pct = gl_result['body_loss_pct']
        n_extreme = gl_result['n_extreme']
        pval = gl_result['pvalue']
        z = gl_result['z_stat']
        alt = gl_result['alternative']
        if pval < 0.05:
            print(f"✅ FACT 4 (Gain/Loss Asymmetry): CONFIRMED (p={pval:.4f}, z={z:.2f}, {alt})")
        else:
            print(f"❌ FACT 4 (Gain/Loss Asymmetry): NOT SIGNIFICANT (p={pval:.4f}, z={z:.2f})")
        print(f"   -> H0: Extreme moves behave identically to normal market moves")
        print(f"   -> Tail loss rate: {loss_pct:.1f}% ({n_extreme} extreme returns)")
        print(f"   -> Body loss rate: {body_pct:.1f}% (null hypothesis baseline)")
        if gl_result['avg_loss'] is not None:
            print(f"   -> Avg extreme loss:    {gl_result['avg_loss']:.6f}  |  Median: {gl_result['median_loss']:.6f}")
        if gl_result['avg_gain'] is not None:
            print(f"   -> Avg extreme gain:   +{gl_result['avg_gain']:.6f}  |  Median: +{gl_result['median_gain']:.6f}")
    else:
        print("⚠️ FACT 4 (Gain/Loss Asymmetry): INCONCLUSIVE (Insufficient Data)")

    # Report Fact 5
    if ag_result:
        if ag_result['convergence_confirmed']:
            kurt = ag_result['kurtosis_by_scale']
            sorted_scales = sorted(kurt.keys())
            print(f"✅ FACT 5 (Aggregational Gaussianity): CONFIRMED (κ: {kurt[sorted_scales[0]]:.2f} -> {kurt[sorted_scales[-1]]:.2f})")
        else:
            print("❌ FACT 5 (Aggregational Gaussianity): NOT DETECTED")
    else:
        print("⚠️ FACT 5 (Aggregational Gaussianity): INCONCLUSIVE (Insufficient Data)")
    # Report Fact 6
    if im_result:
        fano = im_result['fano_factor']
        if im_result['intermittent']:
            print(f"✅ FACT 6 (Intermittency): CONFIRMED (Fano={fano:.2f}, bursty extremes)")
        else:
            print(f"❌ FACT 6 (Intermittency): NOT DETECTED (Fano={fano:.2f}, Poisson-like)")
        print(f"   -> {im_result['n_extremes']} extreme events across {im_result['n_blocks']} blocks")
    else:
        print("⚠️ FACT 6 (Intermittency): INCONCLUSIVE (Insufficient Data)")

    # Report Fact 7
    if sd_result:
        b1 = sd_result['beta_alpha1']
        b2 = sd_result['beta_alpha2']
        b1_str = f"{b1:.3f}" if b1 is not None else "N/A"
        b2_str = f"{b2:.3f}" if b2 is not None else "N/A"
        if sd_result['slow_decay_confirmed']:
            print(f"✅ FACT 7 (Slow Decay): CONFIRMED (β(α=1)={b1_str}, β(α=2)={b2_str})")
        else:
            print(f"❌ FACT 7 (Slow Decay): NOT CONFIRMED (β(α=1)={b1_str}, β(α=2)={b2_str}, outside [0.2,0.4])")
    else:
        print("⚠️ FACT 7 (Slow Decay): INCONCLUSIVE (Insufficient Data)")

    # Report Fact 8
    if le_result:
        min_L = le_result['min_L']
        min_lag = le_result['min_lag']
        if le_result['leverage_detected']:
            print(f"✅ FACT 8 (Leverage Effect): CONFIRMED (min L={min_L:.4f} at τ={min_lag})")
        else:
            print(f"❌ FACT 8 (Leverage Effect): NOT DETECTED (min L={min_L:.4f})")
    else:
        print("⚠️ FACT 8 (Leverage Effect): INCONCLUSIVE (Insufficient Data)")

    # Report Fact 9
    if vvc_result:
        rho_a = vvc_result['rho_abs']
        rho_s = vvc_result['rho_sq']
        p_a = vvc_result['pval_abs']
        p_s = vvc_result['pval_sq']
        if vvc_result['corr_confirmed']:
            print(f"✅ FACT 9 (Vol-Vol Correlation): CONFIRMED "
                  f"(ρ(|r|)={rho_a:.4f} p={p_a:.4f}, ρ(r²)={rho_s:.4f} p={p_s:.4f})")
        else:
            print(f"❌ FACT 9 (Vol-Vol Correlation): NOT DETECTED "
                  f"(ρ(|r|)={rho_a:.4f} p={p_a:.4f}, ρ(r²)={rho_s:.4f} p={p_s:.4f})")
    else:
        print("⚠️ FACT 9 (Vol-Vol Correlation): INCONCLUSIVE (Insufficient Data)")

    print("=" * 40 + "\n")


if __name__ == "__main__":
    run_analysis()
