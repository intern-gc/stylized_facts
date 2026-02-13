import numpy as np
import pandas as pd
from data import DataManager
from autocorrelation import AbsenceOfAutocorrelationTest
from heavytails import HeavyTailsEVT
from volclustering import VolatilityClustering
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def run_analysis():
    # --- CONFIGURATION ---
    ticker = "SPY"
    interval = "1d"  # Try "1m", "5m", "1h", or "1d"

    # Adjust dates based on timeframe to ensure enough data
    if interval == "1d":
        start_date = "2020-01-01"  # Daily needs ~5 years for good EVT
    else:
        start_date = "2024-01-01"  # Intraday needs less time
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
    print("=" * 40 + "\n")


if __name__ == "__main__":
    run_analysis()
