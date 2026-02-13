
import pandas as pd
from data import DataManager
from autocorrelation import AbsenceOfAutocorrelationTest
from heavytails import HeavyTailsEVT
from volclustering import VolatilityClustering
from gainloss import GainLossAsymmetry
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def run_analysis():
    # --- CONFIGURATION ---
    ticker = "SPY"
    interval = "1h"  # Try "1m", "5m", "1h", or "1d"

    # Adjust dates based on timeframe to ensure enough data
    start_date = "2020-01-01"  # Daily needs ~5 years for good EVT
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
    print("=" * 40 + "\n")


if __name__ == "__main__":
    run_analysis()
