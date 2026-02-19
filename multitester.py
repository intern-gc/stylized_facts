"""
multitester.py
Run all 11 stylized fact tests across multiple tickers and timeframes,
then export results to Excel in the same format as Sheet1 of
"Stylized Facts + Backtest.xlsx".

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

from data import DataManager
from autocorrelation import AbsenceOfAutocorrelationTest
from heavytails import HeavyTailsEVT
from gainloss import GainLossAsymmetry
from aggregational import AggregationalGaussianity
from intermittency import Intermittency
from volclustering import VolatilityClustering
from conditional import ConditionalTails
from decay import SlowDecay
from leverage import LeverageEffect
from volvolcorr import VolVolCorr
from timescales import AsymmetryTimescales

# ── USER CONFIGURATION ─────────────────────────────────────────────────────────
TICKERS = [
    "MSFT", "GOOGL", "AMZN",       # The "Digital Utilities" (Tech Blue Chips)
    "JPM", "BAC", "GS",            # The Banking Giants (Financial Stability)
    "WMT", "JNJ",                  # Defensive Staples (Retail & Healthcare)
    "DIA", "XLF"                   # Major ETFs (Dow Jones & Financial Sector)
]
INTERVALS   = ["1m", "1h", "1d"]
START_DATE  = "2015-01-01"
END_DATE    = "2025-01-01"
OUTPUT_FILE = "multitester_results.xlsx"
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# ── INTERVAL-SPECIFIC PARAMETERS (mirrors main.py) ────────────────────────────
_BS_MAP      = {'1m': 390,  '5m': 78,   '1h': 7,   '1d': 21}
_AG_SCALES   = {'1m': [1, 5, 30, 78, 390], '5m': [1, 5, 16, 78],
                '1h': [1, 4, 7, 14, 21, 26], '1d': [1, 5, 10, 21]}
_IM_BLOCKS   = {'1m': 390,  '5m': 78,   '1h': 168, '1d': 21}
_SD_MAX_LAGS = {'1m': 2000, '5m': 1000, '1h': 500,  '1d': 500}
_LE_MAX_LAGS = {'1m': 30,   '5m': 30,   '1h': 50,   '1d': 30}
_TS_DT_MAP   = {'1m': 390,  '5m': 78,   '1h': 7,   '1d': 5}
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
        t = AbsenceOfAutocorrelationTest(returns, ticker)
        sig = t.test_linear_independence(lags=40, plot=False)
    return 'PASSED' if len(sig) == 0 else f'FAILED ({len(sig)} lags)'


def fact2(returns, ticker, interval):
    bs = _BS_MAP.get(interval, 21)
    with _Quiet():
        t = HeavyTailsEVT(returns, ticker)
        r = t.run_mle_fit(block_size=bs, plot=False)
    if r is None:
        return 'INCONCLUSIVE'
    xi, alpha = r
    if xi > 0:
        return f'CONFIRMED (Alpha={alpha:.2f})'
    return 'FAILED (Thin/Bounded Tail)'


def fact3(returns, ticker):
    with _Quiet():
        t = GainLossAsymmetry(returns, ticker)
        r = t.compute_asymmetry(q=0.95, plot=False)
    if r is None:
        return 'INCONCLUSIVE'
    if r['pvalue'] < 0.05:
        return f"CONFIRMED ({r['loss_pct']:.1f}% Tail vs {r['body_loss_pct']:.1f}% Body)"
    return f"NOT SIGNIFICANT (p={r['pvalue']:.4f}, z={r['z_stat']:.2f})"


def fact4(returns, ticker, interval):
    scales = _AG_SCALES.get(interval, [1, 4, 7, 13, 26])
    with _Quiet():
        t = AggregationalGaussianity(returns, ticker)
        r = t.test_aggregational_gaussianity(scales=scales, plot=False)
    if r is None:
        return 'INCONCLUSIVE'
    if r['convergence_confirmed']:
        k = r['kurtosis_by_scale']
        ss = sorted(k.keys())
        return f"CONFIRMED (k={k[ss[0]]:.2f} @ {ss[0]} bar to {k[ss[-1]]:.2f} @ {ss[-1]} bars)"
    return 'NOT DETECTED'


def fact5(returns, ticker, interval):
    bs = _IM_BLOCKS.get(interval, 168)
    with _Quiet():
        t = Intermittency(returns, ticker)
        r = t.compute_intermittency(quantile=0.99, block_size=bs, plot=False)
    if r is None:
        return 'INCONCLUSIVE'
    fano = r['fano_factor']
    return f"CONFIRMED (Fano={fano:.2f})" if r['intermittent'] else f"NOT DETECTED (Fano={fano:.2f})"


def fact6(returns, ticker):
    with _Quiet():
        t = VolatilityClustering(returns, ticker)
        _, sig = t.compute_c2(max_lag=40, plot=False)
    return f'CONFIRMED ({len(sig)} lags)' if len(sig) > 0 else 'NOT DETECTED'


def fact7(returns, ticker):
    with _Quiet():
        t = ConditionalTails(returns, ticker)
        r = t.compute_conditional_tails(plot=False)
    if r is None:
        return 'INCONCLUSIVE'
    ek = r['excess_kurtosis']
    return f"CONFIRMED (Ex. k={ek:.2f})" if r['non_gaussian'] else f"NOT DETECTED (Ex. k={ek:.2f})"


def fact8(returns, ticker, interval):
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


def fact9(returns, ticker, interval):
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


def fact10(returns, volume, ticker):
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


def fact11(returns, ticker, interval):
    dT = _TS_DT_MAP.get(interval, 7)
    with _Quiet():
        t = AsymmetryTimescales(returns, ticker)
        r = t.compute_asymmetry(dT=dT, max_tau=10, plot=False)
    if r is None:
        return 'INCONCLUSIVE'
    if r['top_down_detected']:
        return 'CONFIRMED (D < 0)'
    return f"NOT CONFIRMED (D_mean={r['D_mean']:.4f})"


# ── MAIN RUNNER ───────────────────────────────────────────────────────────────

def run_single(ticker, interval, dm):
    """Fetch data and run all 11 facts. Returns dict f1..f11."""
    df, returns, report = dm.get_data(ticker, START_DATE, END_DATE, interval)
    if df.empty:
        print(f"    NO DATA — {report}")
        return {f'f{i}': 'NO DATA' for i in range(1, 12)}

    print(f"    {len(returns)} bars | {report}")
    results = {}
    volume = df['Volume']

    facts = [
        ('f1',  lambda: fact1(returns, ticker)),
        ('f2',  lambda: fact2(returns, ticker, interval)),
        ('f3',  lambda: fact3(returns, ticker)),
        ('f4',  lambda: fact4(returns, ticker, interval)),
        ('f5',  lambda: fact5(returns, ticker, interval)),
        ('f6',  lambda: fact6(returns, ticker)),
        ('f7',  lambda: fact7(returns, ticker)),
        ('f8',  lambda: fact8(returns, ticker, interval)),
        ('f9',  lambda: fact9(returns, ticker, interval)),
        ('f10', lambda: fact10(returns, volume, ticker)),
        ('f11', lambda: fact11(returns, ticker, interval)),
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
    'F1: No Autocorr.', 'F2: Heavy Tails', 'F3: Gain/Loss Asym.',
    'F4: Aggr. Gaussianity', 'F5: Intermittency', 'F6: Vol. Clustering',
    'F7: Cond. Heavy Tails', 'F8: Slow Decay', 'F9: Leverage Effect',
    'F10: Vol-Vol Corr.', 'F11: Time-Scale Asym.',
]

_GREEN  = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
_RED    = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')
_YELLOW = PatternFill(start_color='FFEB9C', end_color='FFEB9C', fill_type='solid')
_ALT    = PatternFill(start_color='F2F2F2', end_color='F2F2F2', fill_type='solid')  # alternate ticker rows


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
    # Build alternating-shading order from existing rows + new rows
    # Read tickers already in the sheet (rows 3..start_row-1)
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

        for col_offset, key in enumerate(
            ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11'],
            start=3
        ):
            val = facts.get(key, '')
            c = cell(col_offset, val)
            fill = _fill_for(val)
            c.fill = fill if fill else (_ALT if alt else PatternFill())


def write_excel(rows, output_file):
    import os

    if os.path.exists(output_file):
        wb = load_workbook(output_file)
        ws = wb['Stylized Facts'] if 'Stylized Facts' in wb.sheetnames else wb.active
        # Find first truly empty row after the headers
        start_row = ws.max_row + 1
        print(f'  Appending {len(rows)} rows after row {start_row - 1} in existing file.')
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = 'Stylized Facts'

        # Row 1: note
        note_cell = ws.cell(1, 1, f'Dates: {START_DATE} if possible till {END_DATE}')
        note_cell.font = Font(italic=True, color='808080')

        # Row 2: headers
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
                facts = {f'f{i}': 'ERROR' for i in range(1, 12)}
            rows.append((ticker, interval, facts))

    elapsed = time.time() - t_start
    print(f'\nTotal runtime: {elapsed/60:.1f} min')
    write_excel(rows, OUTPUT_FILE)


if __name__ == '__main__':
    main()
