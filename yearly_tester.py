# yearly_tester.py
import pandas as pd
import numpy as np
from data import DataManager
from backtest import compute_strategy, DEFAULT_PARAMS
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# CONFIGURATION
RISK_TICKER = "TQQQ"  # Change to test others
SAFE_TICKER = "BIL"
INTERVAL = "1h"
FULL_START = "2015-01-01"
FULL_END = "2026-01-01"

YEARS = [
    ("2015-01-01", "2016-01-01"),  # China crash
    ("2016-01-01", "2017-01-01"),  # Post-crisis recovery
    ("2017-01-01", "2018-01-01"),  # Low-vol bull
    ("2018-01-01", "2019-01-01"),  # Volmageddon
    ("2019-01-01", "2020-01-01"),  # Pre-COVID
    ("2020-01-01", "2021-01-01"),  # COVID crash/recovery
    ("2021-01-01", "2022-01-01"),  # Bull run
    ("2022-01-01", "2023-01-01"),  # Inflation bear
    ("2023-01-01", "2024-01-01"),  # Recovery
    ("2024-01-01", "2025-01-01"),  # Pre-tariff
    ("2025-01-01", "2026-01-01"),  # 2026 YTD (your concern)
]


def slice_returns(ret_series, start, end):
    """Slice return series to exact year."""
    mask = (ret_series.index >= start) & (ret_series.index < end)
    return ret_series[mask]


def run_yearly_analysis():
    print(f"🔬 YEAR-BY-YEAR ANALYSIS: {RISK_TICKER} ↔ {SAFE_TICKER}")
    print("=" * 80)

    # Fetch full data once
    dm = DataManager()
    print(f"Fetching {RISK_TICKER}...")
    _, ret_risk, _ = dm.get_data(RISK_TICKER, FULL_START, FULL_END, INTERVAL)
    print(f"Fetching {SAFE_TICKER}...")
    _, ret_safe, _ = dm.get_data(SAFE_TICKER, FULL_START, FULL_END, INTERVAL)

    results = []

    for year_start, year_end in YEARS:
        print(f"\n📅 Testing {year_start[:4]}...")

        # Slice to exact year
        year_risk = slice_returns(ret_risk, year_start, year_end)
        year_safe = slice_returns(ret_safe, year_start, year_end)

        if len(year_risk) < 500 or len(year_safe) < 500:  # Skip short periods
            print(f"  ⏭️  Skipping {year_start[:4]} (insufficient data)")
            continue

        # Run backtest
        metrics = compute_strategy(year_risk, year_safe, DEFAULT_PARAMS)
        if metrics is None:
            print(f"  ❌ Failed {year_start[:4]}")
            continue

        results.append({
            'year': year_start[:4],
            'total_ret': metrics['total_ret'],
            'bench_ret': metrics['bench_ret'],
            'cagr_strat': metrics['cagr_strat'],
            'cagr_bench': metrics['cagr_bench'],
            'sharpe_strat': metrics['sharpe_strat'],
            'sharpe_bench': metrics['sharpe_bench'],
            'mdd_strat': metrics['mdd_strat'],
            'mdd_bench': metrics['mdd_bench'],
            'win': metrics['total_ret'] > metrics['bench_ret']
        })

        print(f"  ✅ {year_start[:4]}: {metrics['total_ret']:.1f}% vs {metrics['bench_ret']:.1f}% "
              f"(Sharpe {metrics['sharpe_strat']:.2f} vs {metrics['sharpe_bench']:.2f})")

    if not results:
        print("❌ No valid yearly results")
        return

    # Summary table
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("📊 YEARLY PERFORMANCE SUMMARY")
    print("=" * 80)

    print(f"{'Year':<6} {'Strat %':>8} {'Bench %':>8} {'Strat CAGR':>9} {'Bench CAGR':>9} {'Sharpe':>8}")
    print("-" * 70)

    for _, row in df_results.iterrows():
        marker = "⭐" if row['win'] else " "
        print(f"{row['year']:<6} {row['total_ret']:>7.1f}% {row['bench_ret']:>8.1f}% "
              f"{row['cagr_strat']:>8.1f}% {row['cagr_bench']:>9.1f}% {row['sharpe_strat']:.2f}")

    # Key stats
    print("\n" + "-" * 80)
    wins = df_results['win'].sum()
    total_years = len(df_results)
    print(f"📈 WINS: {wins}/{total_years} years beat benchmark ({wins / total_years:.1%})")

    pre_2026 = df_results[df_results['year'] != '2025']
    if len(pre_2026) > 0:
        pre_wins = pre_2026['win'].sum()
        print(f"⏪ Pre-2026: {pre_wins}/{len(pre_2026)} wins ({pre_wins / len(pre_2026):.1%})")

    avg_sharpe_edge = df_results['sharpe_strat'].mean() - df_results['sharpe_bench'].mean()
    print(f"⚡ Average Sharpe edge: +{avg_sharpe_edge:.2f}")

    # 2026-specific callout
    y2026 = df_results[df_results['year'] == '2025']
    if not y2026.empty:
        print(f"\n🚨 2026 YTD: {y2026['total_ret'].iloc[0]:.1f}% vs {y2026['bench_ret'].iloc[0]:.1f}%")
        print(f"   2026 is {'CRUCIAL' if not y2026['win'].iloc[0] else 'NOT the driver'}")

    print("\n" + "=" * 80)
    if wins / total_years >= 0.6:
        print("🎉 STRATEGY IS REGIME-ROBUST (beats benchmark ≥60% of years)")
    else:
        print("⚠️  Strategy may depend on specific market regimes")


if __name__ == "__main__":
    run_yearly_analysis()
