import pandas as pd
import numpy as np
from itertools import product
from data import DataManager
from backtest import compute_strategy, DEFAULT_PARAMS

# --- CONFIGURATION ---
RISK_TICKER = "SPY"
SAFE_TICKER = "GLD"
INTERVAL = "1h"

# Full date range - data is fetched once, then sliced into windows
FULL_START = "2020-01-01"
FULL_END = "2025-01-01"

# Rolling 1-year windows to test across different market regimes
TIME_WINDOWS = [
    ("2020-01-01", "2021-01-01"),  # COVID crash + recovery
    ("2021-01-01", "2022-01-01"),  # Bull run
    ("2022-01-01", "2023-01-01"),  # Bear market
    ("2023-01-01", "2024-01-01"),  # Recovery
    ("2024-01-01", "2025-01-01"),  # Recent
]

# Parameter sweep ranges (one-at-a-time around defaults)
PARAM_SWEEPS = {
    'target_vol':       [15.0, 18.0, 20.0, 22.0, 25.0],
    'vol_window':       [18, 21, 24, 27, 30],
    'baseline_window':  [90, 105, 120, 135, 150],
    'smooth_window':    [3, 4, 5, 6, 7],
    'hysteresis':       [0.05, 0.08, 0.10, 0.12, 0.15],
    'max_lev':          [3.0, 3.5, 4.0, 4.5, 5.0],
}

# Key metrics to track
METRICS = ['sharpe_strat', 'cagr_strat', 'mdd_strat', 'sortino_strat']


def fetch_all_data():
    """Fetch data once for the full date range."""
    dm = DataManager()

    print(f"Fetching {RISK_TICKER} ({FULL_START} to {FULL_END})...")
    df_risk, ret_risk, report_risk = dm.get_data(RISK_TICKER, FULL_START, FULL_END, INTERVAL)

    print(f"Fetching {SAFE_TICKER} ({FULL_START} to {FULL_END})...")
    df_safe, ret_safe, report_safe = dm.get_data(SAFE_TICKER, FULL_START, FULL_END, INTERVAL)

    if df_risk.empty:
        print(f"FAILED: {RISK_TICKER} - {report_risk}")
        return None, None
    if df_safe.empty:
        print(f"FAILED: {SAFE_TICKER} - {report_safe}")
        return None, None

    print(f"Loaded {len(ret_risk)} {RISK_TICKER} bars, {len(ret_safe)} {SAFE_TICKER} bars\n")
    return ret_risk, ret_safe


def slice_returns(ret_series, start, end):
    """Slice a return series to a date range."""
    mask = (ret_series.index >= start) & (ret_series.index < end)
    return ret_series[mask]


def run_sensitivity_analysis(ret_risk_full, ret_safe_full):
    """One-at-a-time parameter sweep across all time windows."""
    all_results = []

    total_runs = sum(len(vals) for vals in PARAM_SWEEPS.values()) * len(TIME_WINDOWS)
    run_count = 0

    for param_name, sweep_values in PARAM_SWEEPS.items():
        for value in sweep_values:
            for window_start, window_end in TIME_WINDOWS:
                run_count += 1

                # Slice data to this window
                ret_risk = slice_returns(ret_risk_full, window_start, window_end)
                ret_safe = slice_returns(ret_safe_full, window_start, window_end)

                if len(ret_risk) < 200 or len(ret_safe) < 200:
                    continue

                # Build params: defaults with one param changed
                params = {**DEFAULT_PARAMS, param_name: value}
                result = compute_strategy(ret_risk, ret_safe, params)

                if result is None:
                    continue

                row = {
                    'param': param_name,
                    'value': value,
                    'is_default': (value == DEFAULT_PARAMS[param_name]),
                    'window': f"{window_start[:4]}-{window_end[:4]}",
                    **{m: result[m] for m in METRICS},
                }
                all_results.append(row)

                if run_count % 25 == 0:
                    print(f"  Progress: {run_count}/{total_runs} runs...")

    return pd.DataFrame(all_results)


def compute_robustness(df):
    """Compute robustness scores per parameter."""
    print("=" * 80)
    print("PARAMETER ROBUSTNESS REPORT")
    print(f"Strategy: {RISK_TICKER} <-> {SAFE_TICKER} ({INTERVAL})")
    print(f"Windows: {len(TIME_WINDOWS)} x 1-year periods")
    print("=" * 80)

    for param_name in PARAM_SWEEPS:
        param_df = df[df['param'] == param_name]
        if param_df.empty:
            continue

        print(f"\n{'─' * 80}")
        print(f"PARAMETER: {param_name} (default = {DEFAULT_PARAMS[param_name]})")
        print(f"{'─' * 80}")

        # Per-value averages across all windows
        summary = param_df.groupby('value')[METRICS].agg(['mean', 'std'])

        # Print table header
        print(f"  {'Value':>10}  {'Sharpe':>12}  {'CAGR%':>12}  {'MDD%':>12}  {'Sortino':>12}")
        print(f"  {'':>10}  {'(mean±std)':>12}  {'(mean±std)':>12}  {'(mean±std)':>12}  {'(mean±std)':>12}")
        print(f"  {'─' * 62}")

        for val in PARAM_SWEEPS[param_name]:
            val_rows = param_df[param_df['value'] == val]
            if val_rows.empty:
                continue
            marker = " *" if val == DEFAULT_PARAMS[param_name] else "  "
            s_m = val_rows['sharpe_strat'].mean()
            s_s = val_rows['sharpe_strat'].std()
            c_m = val_rows['cagr_strat'].mean()
            c_s = val_rows['cagr_strat'].std()
            m_m = val_rows['mdd_strat'].mean()
            m_s = val_rows['mdd_strat'].std()
            so_m = val_rows['sortino_strat'].mean()
            so_s = val_rows['sortino_strat'].std()
            print(f"{marker}{val:>9}  {s_m:>6.2f}±{s_s:<5.2f}  {c_m:>6.1f}±{c_s:<5.1f}  {m_m:>6.1f}±{m_s:<5.1f}  {so_m:>6.2f}±{so_s:<5.2f}")

        # Robustness score: CV of Sharpe across all (value, window) combos
        sharpe_vals = param_df['sharpe_strat']
        if sharpe_vals.std() > 0 and sharpe_vals.mean() != 0:
            cv = abs(sharpe_vals.std() / sharpe_vals.mean())
        else:
            cv = 0.0

        # Also check: does performance degrade at non-default values?
        default_sharpe = param_df[param_df['is_default']]['sharpe_strat'].mean()
        non_default_sharpes = param_df[~param_df['is_default']].groupby('value')['sharpe_strat'].mean()
        beats_default = (non_default_sharpes >= default_sharpe * 0.8).sum()
        total_non_default = len(non_default_sharpes)

        if cv < 0.3:
            verdict = "ROBUST"
        elif cv < 0.6:
            verdict = "MODERATE"
        else:
            verdict = "SENSITIVE"

        print(f"\n  Coefficient of Variation (Sharpe): {cv:.3f} -> {verdict}")
        print(f"  Non-default values within 80% of default Sharpe: {beats_default}/{total_non_default}")

    # Overall summary
    print(f"\n{'=' * 80}")
    print("OVERALL ROBUSTNESS SUMMARY")
    print(f"{'=' * 80}")
    print(f"  {'Parameter':<20} {'CV(Sharpe)':>12} {'Verdict':>12}")
    print(f"  {'─' * 48}")

    verdicts = []
    for param_name in PARAM_SWEEPS:
        param_df = df[df['param'] == param_name]
        sharpe_vals = param_df['sharpe_strat']
        if sharpe_vals.std() > 0 and sharpe_vals.mean() != 0:
            cv = abs(sharpe_vals.std() / sharpe_vals.mean())
        else:
            cv = 0.0
        verdict = "ROBUST" if cv < 0.3 else ("MODERATE" if cv < 0.6 else "SENSITIVE")
        verdicts.append(verdict)
        print(f"  {param_name:<20} {cv:>12.3f} {verdict:>12}")

    robust_count = verdicts.count("ROBUST")
    total = len(verdicts)
    print(f"\n  Overall: {robust_count}/{total} parameters are robust.")
    if robust_count == total:
        print("  Strategy appears parameter-robust across tested ranges.")
    elif robust_count >= total * 0.7:
        print("  Strategy is mostly robust. Check SENSITIVE parameters for overfitting risk.")
    else:
        print("  WARNING: Strategy may be overfit. Performance depends heavily on parameter choices.")
    print("=" * 80)


def main():
    ret_risk, ret_safe = fetch_all_data()
    if ret_risk is None:
        return

    print("Running parameter sensitivity analysis...")
    print(f"  Parameters: {len(PARAM_SWEEPS)}")
    print(f"  Sweep values per param: ~{sum(len(v) for v in PARAM_SWEEPS.values()) // len(PARAM_SWEEPS)}")
    print(f"  Time windows: {len(TIME_WINDOWS)}")
    total = sum(len(v) for v in PARAM_SWEEPS.values()) * len(TIME_WINDOWS)
    print(f"  Total runs: {total}\n")

    results_df = run_sensitivity_analysis(ret_risk, ret_safe)

    if results_df.empty:
        print("ERROR: No results. Check data availability.")
        return

    compute_robustness(results_df)


if __name__ == "__main__":
    main()
