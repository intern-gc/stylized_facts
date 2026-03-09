"""
multitester.py
Run all 4 stylized fact tests across multiple tickers and timeframes,
then export results to Excel.

Edit TICKERS / INTERVALS / START_DATE / END_DATE at the top, then run:
    python multitester.py
"""
import io
import sys
import time
import logging
import traceback

import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter

from utilities.data import DataManager
from facts_test.volclustering import VolatilityClustering
from facts_test.decay import SlowDecay
from facts_test.leverage import LeverageEffect
from facts_test.volvolcorr import VolVolCorr

# ── USER CONFIGURATION ─────────────────────────────────────────────────────────
TICKERS = [
    "NVDA", "META", "AMD",         # Tech, Social & Semiconductors
    "HD", "SBUX", "MCD",           # Consumer Discretionary & Dining
    "UNH", "LLY", "PFE",           # Healthcare & Pharmaceuticals
    "XOM", "CVX",                  # Energy & Oil Heavyweights
    "V", "MA",                     # Global Payment Processors
    "CAT", "BA",                   # Industrials & Aerospace
    "DIS", "NFLX",                 # Media & Entertainment
    "KO", "PG", "VNQ"              # Consumer Staples & Real Estate ETF
]
INTERVALS   = ["1m", "1h", "1d"]
START_DATE  = "2015-01-01"
END_DATE    = "2025-01-01"
OUTPUT_FILE = "multitester_results.xlsx"
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# ── INTERVAL-SPECIFIC PARAMETERS ──────────────────────────────────────────────
_SD_MAX_LAGS = {'1m': 2000, '5m': 1000, '1h': 500,  '1d': 500}
_LE_MAX_LAGS = {'1m': 30,   '5m': 30,   '1h': 50,   '1d': 30}
# ──────────────────────────────────────────────────────────────────────────────


class _Quiet:
    """Context manager: suppress stdout and stderr (prints inside testers)."""
    def __enter__(self):
        self._out, self._err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = io.StringIO()

    def __exit__(self, *_):
        sys.stdout, sys.stderr = self._out, self._err


def _safe(fn, fallback='INCONCLUSIVE'):
    """Run fn(); on any exception return fallback string."""
    try:
        return fn()
    except Exception:
        return fallback


# ── INDIVIDUAL FACT FORMATTERS ─────────────────────────────────────────────────

def fact1(returns, ticker):
    with _Quiet():
        t = VolatilityClustering(returns, ticker)
        _, sig = t.compute_c1(max_lag=40, plot=False)
    return f'CONFIRMED ({len(sig)} lags)' if len(sig) > 0 else 'NOT DETECTED'


def fact2(returns, ticker, interval):
    ml = _SD_MAX_LAGS.get(interval, 500)
    with _Quiet():
        t = SlowDecay(returns, ticker)
        r = t.compute_decay(max_lag=ml, plot=False)
    if r is None:
        return 'INCONCLUSIVE'
    if r['slow_decay_confirmed']:
        b1, b2 = r['beta_alpha1'], r['beta_alpha2']
        b = b1 if (b1 is not None and 0.2 <= b1 <= 0.4) else b2
        return f"CONFIRMED (Beta={b:.3f})"
    b1, b2 = r['beta_alpha1'], r['beta_alpha2']
    b = b1 if b1 is not None else b2
    b_str = f'{b:.3f}' if b is not None else 'N/A'
    return f"NOT CONFIRMED (Beta={b_str})"


def fact3(returns, ticker, interval):
    ml = _LE_MAX_LAGS.get(interval, 50)
    with _Quiet():
        t = LeverageEffect(returns, ticker)
        r = t.compute_leverage(max_lag=ml, plot=False)
    if r is None:
        return 'INCONCLUSIVE'
    if r['leverage_detected']:
        return f"CONFIRMED (L={r['min_L']:.2f})"
    min_l = r['min_L']
    l_str = f"{min_l:.2f}" if np.isfinite(min_l) else 'N/A'
    return f"NOT DETECTED (min L={l_str})"


def fact4(returns, volume, ticker):
    with _Quiet():
        t = VolVolCorr(returns, volume, ticker)
        r = t.compute_correlation(plot=False)
    if r is None:
        return 'INCONCLUSIVE'
    if r['corr_confirmed']:
        rho = r['rho_abs'] if np.isfinite(r['rho_abs']) else r['rho_sq']
        return f"CONFIRMED (rho={rho:.2f})"
    rho = r['rho_abs']
    rho_str = f"{rho:.2f}" if np.isfinite(rho) else 'N/A'
    return f"NOT DETECTED (rho={rho_str})"


# ── MAIN RUNNER ───────────────────────────────────────────────────────────────

def run_single(ticker, interval, dm):
    """Fetch data and run all 4 facts. Returns dict f1..f4."""
    df, returns, report = dm.get_data(ticker, START_DATE, END_DATE, interval)
    if df.empty:
        print(f"    NO DATA — {report}")
        return {f'f{i}': 'NO DATA' for i in range(1, 5)}

    print(f"    {len(returns)} bars | {report}")
    results = {}
    volume = df['Volume']

    facts = [
        ('f1', lambda: fact1(returns, ticker)),
        ('f2', lambda: fact2(returns, ticker, interval)),
        ('f3', lambda: fact3(returns, ticker, interval)),
        ('f4', lambda: fact4(returns, volume, ticker)),
    ]

    for key, fn in facts:
        t0 = time.time()
        val = _safe(fn)
        elapsed = time.time() - t0
        tag = '✅' if val.startswith(('CONFIRMED', 'PASSED')) else ('⚠️' if 'INCONCLUSIVE' in val or 'NO DATA' in val else '❌')
        print(f"    {key.upper():<4} {tag}  {val}  ({elapsed:.1f}s)")
        results[key] = val

    return results


# ── EXCEL WRITER ──────────────────────────────────────────────────────────────

HEADERS = [
    'Ticker', 'Time',
    'F1: Vol. Clustering', 'F2: Slow Decay',
    'F3: Leverage Effect', 'F4: Vol-Vol Corr.',
]

_GREEN  = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
_RED    = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')
_YELLOW = PatternFill(start_color='FFEB9C', end_color='FFEB9C', fill_type='solid')
_ALT    = PatternFill(start_color='F2F2F2', end_color='F2F2F2', fill_type='solid')


def _fill_for(value: str):
    v = (value or '').upper()
    if v.startswith(('CONFIRMED', 'PASSED')):
        return _GREEN
    if v.startswith(('FAILED', 'NOT CONFIRMED', 'NOT DETECTED', 'NOT SIGNIFICANT')):
        return _RED
    if v.startswith(('INCONCLUSIVE', 'NO DATA', 'ERROR')):
        return _YELLOW
    return None


def _write_data_rows(ws, rows, start_row):
    """Write data rows into ws starting at start_row, with colour formatting."""
    existing_tickers = []
    for r in range(3, start_row):
        val = ws.cell(r, 1).value
        if val and val not in existing_tickers:
            existing_tickers.append(val)

    new_tickers = list(dict.fromkeys(t for t, _, _ in rows))
    all_tickers = existing_tickers + [t for t in new_tickers if t not in existing_tickers]

    for row_idx, (ticker, interval, facts) in enumerate(rows, start=start_row):
        alt = all_tickers.index(ticker) % 2 == 1

        def cell(col, val=''):
            c = ws.cell(row_idx, col, val)
            c.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            return c

        cell(1, ticker)
        cell(2, interval)

        if alt:
            ws.cell(row_idx, 1).fill = _ALT
            ws.cell(row_idx, 2).fill = _ALT

        for col_offset, key in enumerate(['f1', 'f2', 'f3', 'f4'], start=3):
            val = facts.get(key, '')
            c = cell(col_offset, val)
            fill = _fill_for(val)
            c.fill = fill if fill else (_ALT if alt else PatternFill())


def write_excel(rows, output_file):
    import os

    if os.path.exists(output_file):
        wb = load_workbook(output_file)
        ws = wb['Stylized Facts'] if 'Stylized Facts' in wb.sheetnames else wb.active
        start_row = ws.max_row + 1
        print(f'  Appending {len(rows)} rows after row {start_row - 1} in existing file.')
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = 'Stylized Facts'

        note_cell = ws.cell(1, 1, f'Dates: {START_DATE} if possible till {END_DATE}')
        note_cell.font = Font(italic=True, color='808080')

        hdr_fill = PatternFill(start_color='1F4E79', end_color='1F4E79', fill_type='solid')
        hdr_font = Font(bold=True, color='FFFFFF')
        center   = Alignment(horizontal='center', vertical='center', wrap_text=True)
        for col, h in enumerate(HEADERS, start=1):
            c = ws.cell(2, col, h)
            c.font      = hdr_font
            c.fill      = hdr_fill
            c.alignment = center

        ws.column_dimensions['A'].width = 10
        ws.column_dimensions['B'].width = 6
        for col in range(3, len(HEADERS) + 1):
            ws.column_dimensions[get_column_letter(col)].width = 30
        ws.row_dimensions[2].height = 30
        ws.freeze_panes = 'A3'

        start_row = 3
        print(f'  Creating new file: {output_file}')

    _write_data_rows(ws, rows, start_row)
    wb.save(output_file)
    print(f'\n✅ Results saved to: {output_file}')


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    dm = DataManager()
    rows = []
    total = len(TICKERS) * len(INTERVALS)
    done  = 0

    t_start = time.time()

    for ticker in TICKERS:
        for interval in INTERVALS:
            done += 1
            print(f'\n[{done}/{total}] {ticker} @ {interval}')
            try:
                facts = run_single(ticker, interval, dm)
            except Exception:
                print(f'  ❌ Unhandled exception:\n{traceback.format_exc()}')
                facts = {f'f{i}': 'ERROR' for i in range(1, 5)}
            rows.append((ticker, interval, facts))

    elapsed = time.time() - t_start
    print(f'\nTotal runtime: {elapsed/60:.1f} min')
    write_excel(rows, OUTPUT_FILE)


if __name__ == '__main__':
    main()
