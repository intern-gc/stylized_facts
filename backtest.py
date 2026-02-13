import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import DataManager
import os
from datetime import datetime, timedelta, time

# --- CONFIGURATION ---
RISK_TICKER = "QQQ"
SAFE_TICKER = "BIL"
INTERVAL = "1h"
START_DATE, END_DATE = "2020-01-01", "2025-01-01"

# Strategy Parameters
LEVERAGE = 3  # Adjusted to the "Goldilocks" 1.5x
VOL_WINDOW = 24
BASELINE_WINDOW = 120
SMOOTH_WINDOW = 5
COST_BPS = 2.0


def calc_mdd(prices):
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return drawdown.min()


def run_rotation_strategy():
    dm = DataManager()

    print(f"🚀 FETCHING RISK ASSET: {RISK_TICKER}...")
    df_risk, ret_risk, _ = dm.get_data(RISK_TICKER, START_DATE, END_DATE, INTERVAL)

    print(f"🚀 FETCHING SAFE ASSET: {SAFE_TICKER}...")
    df_safe, ret_safe, _ = dm.get_data(SAFE_TICKER, START_DATE, END_DATE, INTERVAL)

    if df_risk.empty or df_safe.empty:
        print("❌ Error: Missing data.")
        return

    # 1. ALIGN DATA
    # First, check if they are empty before joining
    if df_risk.empty:
        print(f"❌ Error: {RISK_TICKER} is empty!")
        return
    if df_safe.empty:
        print(f"❌ Error: {SAFE_TICKER} is empty!")
        return

    # Use pd.concat for safer alignment
    data = pd.concat([
        pd.Series(ret_risk, name='Ret_Risk'),
        pd.Series(ret_safe, name='Ret_Safe')
    ], axis=1).dropna()

    print(f"  📊 DATA CHECK: {RISK_TICKER}({len(ret_risk)} bars) | {SAFE_TICKER}({len(ret_safe)} bars)")
    print(f"  📊 OVERLAP: {len(data)} bars after alignment.")

    if data.empty:
        print("❌ Error: No timestamp overlap between assets!")
        return

    # 2. CALCULATE SIGNAL
    data['Raw_Vol'] = data['Ret_Risk'].rolling(VOL_WINDOW).std() * np.sqrt(1638) * 100
    data['Syn_VIX'] = data['Raw_Vol'].rolling(SMOOTH_WINDOW).mean()
    data['VIX_MA'] = data['Syn_VIX'].rolling(BASELINE_WINDOW).mean()

    # 3. DEFINE REGIME
    data['Regime'] = np.where(data['Syn_VIX'] > data['VIX_MA'], 0.0, 1.0)
    data['Signal'] = data['Regime'].shift(1).fillna(1.0)

    # 4. CALCULATE RETURNS
    ann_factor = 1638
    borrow_cost_per_bar = (0.05 / ann_factor) * (LEVERAGE - 1)

    risk_leg = data['Signal'] * ((data['Ret_Risk'] * LEVERAGE) - borrow_cost_per_bar)
    safe_leg = (1 - data['Signal']) * data['Ret_Safe']

    trades = data['Signal'].diff().abs().fillna(0)
    turnover_size = np.where(trades > 0, LEVERAGE, 0)
    txn_costs = turnover_size * (COST_BPS / 10000)

    data['Strat_Ret'] = risk_leg + safe_leg - txn_costs

    # 5. EVALUATION
    start_idx = max(VOL_WINDOW, BASELINE_WINDOW, SMOOTH_WINDOW) + 1
    eval_data = data.iloc[start_idx:].copy()

    eval_data['Eq_Strat'] = np.exp(eval_data['Strat_Ret'].cumsum())
    eval_data['Eq_Bench'] = np.exp(eval_data['Ret_Risk'].cumsum())

    # --- ADVANCED STATS ---
    years = len(eval_data) / ann_factor

    # Returns & CAGR
    total_ret = (eval_data['Eq_Strat'].iloc[-1] - 1) * 100
    bench_ret = (eval_data['Eq_Bench'].iloc[-1] - 1) * 100
    cagr_strat = (eval_data['Eq_Strat'].iloc[-1] ** (1 / years) - 1) * 100
    cagr_bench = (eval_data['Eq_Bench'].iloc[-1] ** (1 / years) - 1) * 100

    # Risk Metrics
    mdd_strat = calc_mdd(eval_data['Eq_Strat']) * 100
    mdd_bench = calc_mdd(eval_data['Eq_Bench']) * 100
    vol_strat = eval_data['Strat_Ret'].std() * np.sqrt(ann_factor) * 100
    vol_bench = eval_data['Ret_Risk'].std() * np.sqrt(ann_factor) * 100

    sharpe = (eval_data['Strat_Ret'].mean() / eval_data['Strat_Ret'].std()) * np.sqrt(ann_factor)
    calmar = cagr_strat / abs(mdd_strat) if mdd_strat != 0 else 0

    # Regime Stats
    bull_mask = eval_data['Signal'] == 1
    safe_mask = eval_data['Signal'] == 0
    time_bull = bull_mask.mean() * 100
    time_safe = safe_mask.mean() * 100

    # Win Rates (Hourly)
    win_rate_total = (eval_data['Strat_Ret'] > 0).mean() * 100
    win_rate_bull = (eval_data[bull_mask]['Strat_Ret'] > 0).mean() * 100
    win_rate_safe = (eval_data[safe_mask]['Strat_Ret'] > 0).mean() * 100

    print("\n" + "=" * 60)
    print(f"🚀 LEVERAGED ROTATION: {RISK_TICKER} ({LEVERAGE}x) <-> {SAFE_TICKER}")
    print(f"   Analysis Period: {years:.2f} Years")
    print("=" * 60)

    print(f"{'Metric':<25} {'Strategy':>15} {'Benchmark':>15}")
    print("-" * 58)
    print(f"{'Total Return':<25} {total_ret:>14.2f}% {bench_ret:>14.2f}%")
    print(f"{'CAGR':<25} {cagr_strat:>14.2f}% {cagr_bench:>14.2f}%")
    print(f"{'Max Drawdown':<25} {mdd_strat:>14.2f}% {mdd_bench:>14.2f}%")
    print(f"{'Ann. Volatility':<25} {vol_strat:>14.2f}% {vol_bench:>14.2f}%")
    print("-" * 58)
    print(f"{'Sharpe Ratio':<25} {sharpe:>15.2f}")
    print(f"{'Calmar Ratio':<25} {calmar:>15.2f}")
    print(f"{'Total Trades':<25} {trades.iloc[start_idx:].sum():>15.0f}")
    print("-" * 58)
    print(f"{'Time in Bull Mode':<25} {time_bull:>14.2f}%")
    print(f"{'Time in Safe Mode':<25} {time_safe:>14.2f}%")
    print(f"{'Bull Win Rate (1h)':<25} {win_rate_bull:>14.2f}%")
    print(f"{'Safe Win Rate (1h)':<25} {win_rate_safe:>14.2f}%")
    print("=" * 60)

    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Top Panel: Equity
    ax1.plot(eval_data.index, eval_data['Eq_Strat'], label='Strategy', color='#FFD700', linewidth=2)
    ax1.plot(eval_data.index, eval_data['Eq_Bench'], label='Benchmark', color='black', alpha=0.3)
    ax1.set_title(f"{RISK_TICKER} Rotation Strategy Performance")
    ax1.set_ylabel("Growth of $1")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # Bottom Panel: Regime Heatmap
    ax2.fill_between(eval_data.index, 0, 1, where=bull_mask, facecolor='green', alpha=0.6, label='Bull Mode')
    ax2.fill_between(eval_data.index, 0, 1, where=safe_mask, facecolor='red', alpha=0.6, label='Safe Mode')
    ax2.set_yticks([])
    ax2.set_title("Current Regime")
    ax2.legend(loc='upper right')

    # Overlay VIX Baseline
    ax3 = ax2.twinx()
    ax3.plot(eval_data.index, eval_data['Syn_VIX'], color='white', alpha=0.3, linewidth=0.5)
    ax3.set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_rotation_strategy()