import pandas as pd
import numpy as np
from data import DataManager
from backtest import compute_strategy, DEFAULT_PARAMS

# --- CONFIGURATION ---
INTERVAL = "1h"
START_DATE, END_DATE = "2015-01-01", "2026-01-01"

RISK_TICKERS = ["SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "ARKK", "TQQQ", "UPRO", "VT"]
SAFE_TICKERS = ["GLD", "TLT", "TMF", "TBT", "UUP", "USDU", "SGOV", "BIL", "TIP", "JNK"]

OUTPUT_FILE = "maxtester_results.xlsx"


def main():
    dm = DataManager()

    # Pre-fetch all data
    all_tickers = list(set(RISK_TICKERS + SAFE_TICKERS))
    data_cache = {}

    print(f"Fetching data for {len(all_tickers)} tickers...")
    for ticker in all_tickers:
        print(f"  Fetching {ticker}...")
        df, ret, report = dm.get_data(ticker, START_DATE, END_DATE, INTERVAL)
        if df.empty:
            print(f"  FAILED: {ticker} - {report}")
        else:
            data_cache[ticker] = ret
            print(f"  OK: {ticker} ({len(ret)} bars)")

    # Run all pairs
    rows = []
    total = len(RISK_TICKERS) * len(SAFE_TICKERS)
    count = 0

    for risk in RISK_TICKERS:
        for safe in SAFE_TICKERS:
            count += 1
            pair = f"{risk}/{safe}"

            if risk not in data_cache or safe not in data_cache:
                print(f"  [{count}/{total}] {pair} - SKIPPED (missing data)")
                continue

            print(f"  [{count}/{total}] {pair}...", end=" ")
            result = compute_strategy(data_cache[risk], data_cache[safe], DEFAULT_PARAMS)

            if result is None:
                print("FAILED")
                continue

            metrics, _ = result

            # Compute deltas
            row = {
                'Risk Ticker': risk,
                'Safe Ticker': safe,
                'Strat Total Return (%)': metrics['total_ret'],
                'Bench Total Return (%)': metrics['bench_ret'],
                'Delta Total Return (%)': metrics['total_ret'] - metrics['bench_ret'],
                'Strat CAGR (%)': metrics['cagr_strat'],
                'Bench CAGR (%)': metrics['cagr_bench'],
                'Delta CAGR (%)': metrics['cagr_strat'] - metrics['cagr_bench'],
                'Strat Sharpe': metrics['sharpe_strat'],
                'Bench Sharpe': metrics['sharpe_bench'],
                'Delta Sharpe': metrics['sharpe_strat'] - metrics['sharpe_bench'],
                'Strat Sortino': metrics['sortino_strat'],
                'Bench Sortino': metrics['sortino_bench'],
                'Delta Sortino': metrics['sortino_strat'] - metrics['sortino_bench'],
                'Strat Max DD (%)': metrics['mdd_strat'],
                'Bench Max DD (%)': metrics['mdd_bench'],
                'Delta Max DD (%)': metrics['mdd_strat'] - metrics['mdd_bench'],
                'Strat Vol (%)': metrics['vol_strat'],
                'Bench Vol (%)': metrics['vol_bench'],
                'Delta Vol (%)': metrics['vol_strat'] - metrics['vol_bench'],
            }
            rows.append(row)
            print(f"OK (Strat: {metrics['total_ret']:+.1f}% | Bench: {metrics['bench_ret']:+.1f}% | Delta: {metrics['total_ret'] - metrics['bench_ret']:+.1f}%)")

    if not rows:
        print("No results to write.")
        return

    df_results = pd.DataFrame(rows)

    # Write to Excel with formatting
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='All Pairs', index=False)

        # Auto-size columns
        ws = writer.sheets['All Pairs']
        for col_idx, col in enumerate(df_results.columns, 1):
            max_len = max(len(str(col)), df_results[col].apply(lambda x: len(f"{x:.2f}" if isinstance(x, float) else str(x))).max())
            ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = max_len + 2

    print(f"\nResults written to {OUTPUT_FILE}")
    print(f"  {len(rows)} pairs tested, {total - len(rows)} skipped/failed")


if __name__ == "__main__":
    main()
