
import numpy as np
from utilities.data import DataManager
from facts_test.volclustering import VolatilityClustering
from facts_test.decay import SlowDecay
from facts_test.leverage import LeverageEffect
from facts_test.volvolcorr import VolVolCorr
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def run_analysis():
    # --- CONFIGURATION ---
    ticker = "NFLX"
    interval = "1d"  # Try "1m", "5m", "1h", or "1d"

    start_date = "2015-01-01"
    end_date = "2025-01-01"

    # --- STEP 1: GET DATA ---
    dm = DataManager()
    df, returns, report = dm.get_data(ticker, start_date, end_date, interval)

    if df.empty:
        print(f"\n❌ ABORTING: {report}")
        return

    print(f"\n📊 [DATA READY]: {len(df)} bars | {report}")

    # --- STEP 2: RUN FACTS ---
    print("\n" + "=" * 40)
    print(f"🔎 STARTING STYLIZED FACTS ANALYSIS: {ticker} ({interval})")
    print("=" * 40)

    # FACT 1: Volatility Clustering
    vc_tester = VolatilityClustering(returns, ticker)
    c2_values, vc_sig_lags = vc_tester.compute_c2(max_lag=40)

    # FACT 2: Slow Decay of Autocorrelation
    sd_tester = SlowDecay(returns, ticker)
    sd_max_lags = {'1m': 2000, '5m': 1000, '1h': 500, '1d': 500}
    sd_result = sd_tester.compute_decay(max_lag=sd_max_lags.get(interval, 500))

    # FACT 3: Leverage Effect
    le_tester = LeverageEffect(returns, ticker)
    le_max_lags = {'1m': 30, '5m': 30, '1h': 50, '1d': 30}
    le_result = le_tester.compute_leverage(max_lag=le_max_lags.get(interval, 50))

    # FACT 4: Volume-Volatility Correlation
    volume = df['Volume']
    vvc_tester = VolVolCorr(returns, volume, ticker)
    vvc_result = vvc_tester.compute_correlation()

    # --- STEP 3: FINAL REPORT CARD ---
    print("\n" + "=" * 40)
    print(f"📝 FINAL REPORT CARD: {ticker}")
    print("=" * 40)

    # Report Fact 1
    if len(vc_sig_lags) > 0:
        print(f"✅ FACT 1 (Volatility Clustering): CONFIRMED (Significant at {len(vc_sig_lags)} lags)")
    else:
        print("❌ FACT 1 (Volatility Clustering): NOT DETECTED")

    # Report Fact 2
    if sd_result:
        b1 = sd_result['beta_alpha1']
        b2 = sd_result['beta_alpha2']
        b1_str = f"{b1:.3f}" if b1 is not None else "N/A"
        b2_str = f"{b2:.3f}" if b2 is not None else "N/A"
        if sd_result['slow_decay_confirmed']:
            print(f"✅ FACT 2 (Slow Decay): CONFIRMED (β(α=1)={b1_str}, β(α=2)={b2_str})")
        else:
            print(f"❌ FACT 2 (Slow Decay): FAILED (β(α=1)={b1_str}, β(α=2)={b2_str}, outside [0.2,0.4])")
    else:
        print("⚠️ FACT 2 (Slow Decay): INCONCLUSIVE (Insufficient Data)")

    # Report Fact 3
    if le_result:
        min_L = le_result['min_L']
        min_lag = le_result['min_lag']
        if le_result['leverage_detected']:
            print(f"✅ FACT 3 (Leverage Effect): CONFIRMED (min L={min_L:.4f} at τ={min_lag})")
        else:
            print(f"❌ FACT 3 (Leverage Effect): NOT DETECTED (min L={min_L:.4f})")
    else:
        print("⚠️ FACT 3 (Leverage Effect): INCONCLUSIVE (Insufficient Data)")

    # Report Fact 4
    if vvc_result:
        rho_a = vvc_result['rho_abs']
        rho_s = vvc_result['rho_sq']
        p_a = vvc_result['pval_abs']
        p_s = vvc_result['pval_sq']
        if vvc_result['corr_confirmed']:
            print(f"✅ FACT 4 (Vol-Vol Correlation): CONFIRMED "
                  f"(ρ(|r|)={rho_a:.4f} p={p_a:.4f}, ρ(r²)={rho_s:.4f} p={p_s:.4f})")
        else:
            print(f"❌ FACT 4 (Vol-Vol Correlation): NOT DETECTED "
                  f"(ρ(|r|)={rho_a:.4f} p={p_a:.4f}, ρ(r²)={rho_s:.4f} p={p_s:.4f})")
    else:
        print("⚠️ FACT 4 (Vol-Vol Correlation): INCONCLUSIVE (Insufficient Data)")

    print("=" * 40 + "\n")


if __name__ == "__main__":
    run_analysis()
