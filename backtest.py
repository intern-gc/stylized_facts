import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import DataManager
import os
from datetime import datetime, timedelta, time

# --- CONFIGURATION ---
RISK_TICKER = "SPY"
SAFE_TICKER = "GLD"
INTERVAL = "1h"
START_DATE, END_DATE = "2021-01-01", "2022-01-01"

# Strategy Parameters
TARGET_VOL = 20.0  # Annualized vol target (%)
MIN_LEV = 1.0
MAX_LEV = 4.0
VOL_WINDOW = 24
BASELINE_WINDOW = 120
SMOOTH_WINDOW = 5
COST_BPS = 2.0
HYSTERESIS = 0.10  # Only rebalance when ideal leverage deviates >10% from last trade

DEFAULT_PARAMS = {
    'target_vol': TARGET_VOL,
    'min_lev': MIN_LEV,
    'max_lev': MAX_LEV,
    'vol_window': VOL_WINDOW,
    'baseline_window': BASELINE_WINDOW,
    'smooth_window': SMOOTH_WINDOW,
    'cost_bps': COST_BPS,
    'hysteresis': HYSTERESIS,
}


def calc_dynamic_leverage(realized_vol, target_vol, min_lev, max_lev):
    """Vol-targeted leverage: target_vol / realized_vol, clamped to [min_lev, max_lev]."""
    is_series = isinstance(realized_vol, pd.Series)
    vol = pd.Series(realized_vol) if not is_series else realized_vol
    raw = vol.replace(0, np.nan).rdiv(target_vol)
    result = raw.fillna(max_lev).clip(lower=min_lev, upper=max_lev)
    if is_series:
        return result
    return float(result.iloc[0]) if len(result) == 1 else result


def apply_hysteresis(ideal_leverage, hysteresis_pct):
    """Only rebalance when ideal leverage deviates >hysteresis_pct from last traded position."""
    actual = np.empty(len(ideal_leverage))
    last_traded = 0.0
    for i in range(len(ideal_leverage)):
        ideal = ideal_leverage.iloc[i]
        if last_traded == 0.0:
            if ideal != 0.0:
                last_traded = ideal
            actual[i] = last_traded
        else:
            pct_change = abs(ideal - last_traded) / last_traded
            if pct_change > hysteresis_pct:
                last_traded = ideal
            actual[i] = last_traded
    return pd.Series(actual, index=ideal_leverage.index)


def calc_mdd(prices):
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return drawdown.min()


def compute_strategy(ret_risk, ret_safe, params):
    """Run the rotation strategy on aligned return series and return metrics as a dict."""
    p = params
    ann_factor = 1638

    data = pd.concat([
        pd.Series(ret_risk, name='Ret_Risk'),
        pd.Series(ret_safe, name='Ret_Safe')
    ], axis=1).dropna()

    if data.empty:
        return None

    # Signal
    data['Raw_Vol'] = data['Ret_Risk'].rolling(int(p['vol_window'])).std() * np.sqrt(ann_factor) * 100
    data['Syn_VIX'] = data['Raw_Vol'].rolling(int(p['smooth_window'])).mean()
    data['VIX_MA'] = data['Syn_VIX'].rolling(int(p['baseline_window'])).mean()

    data['Regime'] = np.where(data['Syn_VIX'] > data['VIX_MA'], 0.0, 1.0)
    data['Signal'] = data['Regime'].shift(1).fillna(1.0)

    # Leverage
    data['Dyn_Leverage'] = calc_dynamic_leverage(data['Syn_VIX'], p['target_vol'], p['min_lev'], p['max_lev'])
    data['Ideal_Leverage'] = data['Signal'] * data['Dyn_Leverage']
    data['Eff_Leverage'] = apply_hysteresis(data['Ideal_Leverage'], p['hysteresis'])

    # Returns
    borrow_cost_per_bar = (0.05 / ann_factor) * (data['Eff_Leverage'] - 1).clip(lower=0)
    in_risk = (data['Eff_Leverage'] > 0).astype(float)
    risk_leg = data['Eff_Leverage'] * data['Ret_Risk'] - borrow_cost_per_bar * in_risk
    safe_leg = (1 - in_risk) * data['Ret_Safe']
    lev_delta = data['Eff_Leverage'].diff().abs().fillna(0)
    txn_costs = lev_delta * (p['cost_bps'] / 10000)
    data['Strat_Ret'] = risk_leg + safe_leg - txn_costs

    # Evaluate
    start_idx = max(int(p['vol_window']), int(p['baseline_window']), int(p['smooth_window'])) + 1
    eval_data = data.iloc[start_idx:].copy()

    if len(eval_data) < 10:
        return None

    eval_data['Eq_Strat'] = np.exp(eval_data['Strat_Ret'].cumsum())
    eval_data['Eq_Bench'] = np.exp(eval_data['Ret_Risk'].cumsum())

    years = len(eval_data) / ann_factor

    total_ret = (eval_data['Eq_Strat'].iloc[-1] - 1) * 100
    bench_ret = (eval_data['Eq_Bench'].iloc[-1] - 1) * 100
    cagr_strat = (eval_data['Eq_Strat'].iloc[-1] ** (1 / years) - 1) * 100
    cagr_bench = (eval_data['Eq_Bench'].iloc[-1] ** (1 / years) - 1) * 100

    mdd_strat = calc_mdd(eval_data['Eq_Strat']) * 100
    mdd_bench = calc_mdd(eval_data['Eq_Bench']) * 100
    vol_strat = eval_data['Strat_Ret'].std() * np.sqrt(ann_factor) * 100
    vol_bench = eval_data['Ret_Risk'].std() * np.sqrt(ann_factor) * 100

    sharpe_strat = (eval_data['Strat_Ret'].mean() / eval_data['Strat_Ret'].std()) * np.sqrt(ann_factor)
    sharpe_bench = (eval_data['Ret_Risk'].mean() / eval_data['Ret_Risk'].std()) * np.sqrt(ann_factor)

    strat_downside = eval_data['Strat_Ret'][eval_data['Strat_Ret'] < 0].std() * np.sqrt(ann_factor)
    bench_downside = eval_data['Ret_Risk'][eval_data['Ret_Risk'] < 0].std() * np.sqrt(ann_factor)
    sortino_strat = (eval_data['Strat_Ret'].mean() * ann_factor) / strat_downside if strat_downside > 0 else 0.0
    sortino_bench = (eval_data['Ret_Risk'].mean() * ann_factor) / bench_downside if bench_downside > 0 else 0.0

    return {
        'total_ret': float(total_ret),
        'bench_ret': float(bench_ret),
        'cagr_strat': float(cagr_strat),
        'cagr_bench': float(cagr_bench),
        'sharpe_strat': float(sharpe_strat),
        'sharpe_bench': float(sharpe_bench),
        'sortino_strat': float(sortino_strat),
        'sortino_bench': float(sortino_bench),
        'mdd_strat': float(mdd_strat),
        'mdd_bench': float(mdd_bench),
        'vol_strat': float(vol_strat),
        'vol_bench': float(vol_bench),
    }


def run_rotation_strategy():
    dm = DataManager()

    print(f"🚀 FETCHING RISK ASSET: {RISK_TICKER}...")
    df_risk, ret_risk, report_risk = dm.get_data(RISK_TICKER, START_DATE, END_DATE, INTERVAL)

    print(f"🚀 FETCHING SAFE ASSET: {SAFE_TICKER}...")
    df_safe, ret_safe, report_safe = dm.get_data(SAFE_TICKER, START_DATE, END_DATE, INTERVAL)

    if df_risk.empty or df_safe.empty:
        if df_risk.empty:
            print(f"❌ {RISK_TICKER} failed: {report_risk}")
        if df_safe.empty:
            print(f"❌ {SAFE_TICKER} failed: {report_safe}")
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

    # 4. DYNAMIC LEVERAGE (vol-targeted) with hysteresis
    data['Dyn_Leverage'] = calc_dynamic_leverage(data['Syn_VIX'], TARGET_VOL, MIN_LEV, MAX_LEV)
    # Ideal effective leverage: full leverage in bull, 0 in safe
    data['Ideal_Leverage'] = data['Signal'] * data['Dyn_Leverage']
    # Apply hysteresis: only rebalance when ideal deviates >HYSTERESIS from last trade
    data['Eff_Leverage'] = apply_hysteresis(data['Ideal_Leverage'], HYSTERESIS)

    # 5. CALCULATE RETURNS
    ann_factor = 1638
    borrow_cost_per_bar = (0.05 / ann_factor) * (data['Eff_Leverage'] - 1).clip(lower=0)

    # When Eff_Leverage > 0, we're in risk asset; when 0, we're in safe asset
    in_risk = (data['Eff_Leverage'] > 0).astype(float)
    risk_leg = data['Eff_Leverage'] * data['Ret_Risk'] - borrow_cost_per_bar * in_risk
    safe_leg = (1 - in_risk) * data['Ret_Safe']

    lev_delta = data['Eff_Leverage'].diff().abs().fillna(0)
    txn_costs = lev_delta * (COST_BPS / 10000)

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

    # Sharpe (both)
    sharpe_strat = (eval_data['Strat_Ret'].mean() / eval_data['Strat_Ret'].std()) * np.sqrt(ann_factor)
    sharpe_bench = (eval_data['Ret_Risk'].mean() / eval_data['Ret_Risk'].std()) * np.sqrt(ann_factor)

    # Sortino (downside deviation only)
    strat_downside = eval_data['Strat_Ret'][eval_data['Strat_Ret'] < 0].std() * np.sqrt(ann_factor)
    bench_downside = eval_data['Ret_Risk'][eval_data['Ret_Risk'] < 0].std() * np.sqrt(ann_factor)
    sortino_strat = (eval_data['Strat_Ret'].mean() * ann_factor) / strat_downside if strat_downside > 0 else 0
    sortino_bench = (eval_data['Ret_Risk'].mean() * ann_factor) / bench_downside if bench_downside > 0 else 0

    # Calmar (both)
    calmar_strat = cagr_strat / abs(mdd_strat) if mdd_strat != 0 else 0
    calmar_bench = cagr_bench / abs(mdd_bench) if mdd_bench != 0 else 0

    # Skew & Kurtosis of returns
    skew_strat = eval_data['Strat_Ret'].skew()
    skew_bench = eval_data['Ret_Risk'].skew()
    kurt_strat = eval_data['Strat_Ret'].kurtosis()
    kurt_bench = eval_data['Ret_Risk'].kurtosis()

    # Regime Stats
    bull_mask = eval_data['Signal'] == 1
    safe_mask = eval_data['Signal'] == 0
    time_bull = bull_mask.mean() * 100
    time_safe = safe_mask.mean() * 100

    # Win Rates (Hourly)
    win_rate_total = (eval_data['Strat_Ret'] > 0).mean() * 100
    win_rate_bull = (eval_data[bull_mask]['Strat_Ret'] > 0).mean() * 100
    win_rate_safe = (eval_data[safe_mask]['Strat_Ret'] > 0).mean() * 100

    # Leverage stats (bull mode only)
    bull_lev = eval_data.loc[bull_mask, 'Dyn_Leverage']
    avg_lev = bull_lev.mean() if len(bull_lev) > 0 else 0
    min_lev_used = bull_lev.min() if len(bull_lev) > 0 else 0
    max_lev_used = bull_lev.max() if len(bull_lev) > 0 else 0

    # Rebalance count (leverage changes > 0.1x)
    rebalances = (eval_data['Eff_Leverage'].diff().abs() > 0.1).sum()

    print("\n" + "=" * 60)
    print(f"🚀 VOL-TARGETED ROTATION: {RISK_TICKER} (target {TARGET_VOL}% vol) <-> {SAFE_TICKER}")
    print(f"   Analysis Period: {years:.2f} Years")
    print("=" * 60)

    print(f"{'Metric':<25} {'Strategy':>15} {'Benchmark':>15}")
    print("-" * 58)
    print(f"{'Total Return':<25} {total_ret:>14.2f}% {bench_ret:>14.2f}%")
    print(f"{'CAGR':<25} {cagr_strat:>14.2f}% {cagr_bench:>14.2f}%")
    print(f"{'Max Drawdown':<25} {mdd_strat:>14.2f}% {mdd_bench:>14.2f}%")
    print(f"{'Ann. Volatility':<25} {vol_strat:>14.2f}% {vol_bench:>14.2f}%")
    print("-" * 58)
    print(f"{'Sharpe Ratio':<25} {sharpe_strat:>15.2f} {sharpe_bench:>15.2f}")
    print(f"{'Sortino Ratio':<25} {sortino_strat:>15.2f} {sortino_bench:>15.2f}")
    print(f"{'Calmar Ratio':<25} {calmar_strat:>15.2f} {calmar_bench:>15.2f}")
    print("-" * 58)
    print(f"{'Skewness':<25} {skew_strat:>15.4f} {skew_bench:>15.4f}")
    print(f"{'Excess Kurtosis':<25} {kurt_strat:>15.4f} {kurt_bench:>15.4f}")
    print("-" * 58)
    print(f"{'Rebalances':<25} {rebalances:>15.0f}")
    print(f"{'Avg Leverage (bull)':<25} {avg_lev:>14.2f}x")
    print(f"{'Min Leverage (bull)':<25} {min_lev_used:>14.2f}x")
    print(f"{'Max Leverage (bull)':<25} {max_lev_used:>14.2f}x")
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