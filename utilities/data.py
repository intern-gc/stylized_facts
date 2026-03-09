import os
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime, timedelta, time
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()


class DataManager:
    def __init__(self, cache_dir="financial_cache"):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_alpaca_tf(self, interval):
        mapping = {'1m': TimeFrame.Minute, '5m': TimeFrame(5, TimeFrame.Minute), '1h': TimeFrame.Hour,
                   '1d': TimeFrame.Day}
        return mapping.get(interval, TimeFrame.Minute)

    def get_data(self, ticker, start_date, end_date, interval, force_resync=False):
        yesterday = (datetime.now() - timedelta(days=1)).date()
        req_start = datetime.strptime(start_date, '%Y-%m-%d').date()
        req_end = min(datetime.strptime(end_date, '%Y-%m-%d').date(), yesterday)
        business_days = pd.bdate_range(start=req_start, end=req_end)

        needed_files = {day: f"{day.strftime('%y%m%d')}_{interval}_{ticker}.pkl" for day in business_days}

        # If force_resync is True, we ignore existing files
        if force_resync:
            print(f"  🔄 FORCE RESYNC: Ignoring cache for {ticker}...")
            missing_days = list(business_days)
        else:
            missing_days = [day for day, fname in needed_files.items() if
                            not os.path.exists(os.path.join(self.cache_dir, fname))]

        if missing_days:
            fetch_start = min(missing_days)
            fetch_end = max(missing_days)
            print(f"  🌐 API BATCH FETCH: {ticker} ({fetch_start} to {fetch_end})...")

            request_params = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=self._get_alpaca_tf(interval),
                start=datetime.combine(fetch_start, time(0, 0)),
                end=datetime.combine(fetch_end, time(23, 59)),
                feed="sip",        # Securities Information Processor: the consolidated tape aggregating
                                   # quotes from all US exchanges. More complete than "iex" (IEX-only)
                                   # or "otc". Requires a funded Alpaca account.
                adjustment='split' # adjust historical prices for stock splits so that the price
                                   # series is continuous. Without this, a 2-for-1 split would show
                                   # a -50% return on split day, which is not a real return.
            )

            try:
                bars = self.client.get_stock_bars(request_params)
                days_received = set()

                if bars.data and not bars.df.empty:
                    df_all = bars.df
                    if isinstance(df_all.index, pd.MultiIndex):
                        df_all = df_all.xs(ticker, level=0)

                    df_all = df_all.rename(
                        columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})

                    for day_ts, group in df_all.groupby(df_all.index.date):
                        day_fname = f"{day_ts.strftime('%y%m%d')}_{interval}_{ticker}.pkl"
                        group.to_pickle(os.path.join(self.cache_dir, day_fname))
                        days_received.add(day_ts)

                # Only mark as empty if we actually got a successful response from the API
                # and that specific day was simply not in the result.
                for m_day in missing_days:
                    if m_day.date() not in days_received:
                        empty_fname = f"{m_day.strftime('%y%m%d')}_{interval}_{ticker}.pkl"
                        pd.DataFrame().to_pickle(os.path.join(self.cache_dir, empty_fname))

                print(f"  ✅ BATCH SYNC COMPLETE.")
            except Exception as e:
                print(f"  ❌ API ERROR: {e}. Not marking days as empty.")

        # Parallel load
        def load_pkl(day):
            fpath = os.path.join(self.cache_dir, needed_files[day])
            if os.path.exists(fpath):
                df = pd.read_pickle(fpath)
                return None if df.empty else df
            return None

        print(f"  💾 LOADING {ticker} CACHE...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(load_pkl, business_days))

        stitched_dfs = [df for df in results if df is not None]
        if not stitched_dfs:
            n_cached = sum(1 for d in business_days if os.path.exists(os.path.join(self.cache_dir, needed_files[d])))
            return pd.DataFrame(), pd.Series(), (
                f"❌ NO DATA for {ticker} ({interval}): "
                f"{len(business_days)} business days requested, {n_cached} cache files found but all empty. "
                f"Try force_resync=True or delete the cache."
            )

        full_df = pd.concat(stitched_dfs).sort_index()
        if full_df.index.tz is None: full_df.index = full_df.index.tz_localize('UTC')
        full_df = full_df.tz_convert('US/Eastern')
        if interval != '1d': full_df = full_df.between_time('09:30', '16:00')
        full_df = full_df[~full_df.index.duplicated(keep='first')]

        return self.audit_and_clean(full_df, interval)

    def audit_and_clean(self, df, interval):
        if df.empty:
            return df, pd.Series(), "❌ ERROR: No data"

        c_df = df.copy()
        issues = []
        warnings = []

        # --- 1. NaN VALUES ---
        # Check for missing OHLCV values and forward-fill them.
        nan_count = c_df[['Open', 'High', 'Low', 'Close', 'Volume']].isna().sum().sum()
        if nan_count > 0:
            c_df = c_df.ffill()
            issues.append(f"REPAIRED: {nan_count} NaN values (forward-filled)")

        # --- 2. NEGATIVE PRICES ---
        # A stock price below zero is physically impossible. Drop those rows entirely.
        neg_mask = (c_df[['Open', 'High', 'Low', 'Close']] < 0).any(axis=1)
        n_neg = neg_mask.sum()
        if n_neg > 0:
            c_df = c_df[~neg_mask]
            issues.append(f"DROPPED: {n_neg} rows with negative prices")

        # --- 3. ZERO CLOSE PRICE ---
        # A close price of exactly 0 is almost always a data error. Drop those rows.
        zero_close_mask = c_df['Close'] == 0
        n_zero_close = zero_close_mask.sum()
        if n_zero_close > 0:
            c_df = c_df[~zero_close_mask]
            issues.append(f"DROPPED: {n_zero_close} rows with zero close price")

        # --- 4. OHLC CONSISTENCY ---
        # High must be >= Low, and Close must sit inside [Low, High].
        # Violating these means the bar is corrupted.
        ohlc_bad = (
            (c_df['High'] < c_df['Low']) |
            (c_df['Close'] > c_df['High']) |
            (c_df['Close'] < c_df['Low'])
        )
        n_ohlc_bad = ohlc_bad.sum()
        if n_ohlc_bad > 0:
            c_df = c_df[~ohlc_bad]
            issues.append(f"DROPPED: {n_ohlc_bad} rows with broken OHLC (High < Low or Close out of range)")

        # --- 5. ZERO VOLUME ---
        # Zero volume on an active trading bar is suspicious — could be a stale/phantom bar.
        # We warn but keep the data since it may be valid (halted trading, etc.).
        zero_vol_mask = c_df['Volume'] == 0
        n_zero_vol = zero_vol_mask.sum()
        if n_zero_vol > 0:
            warnings.append(f"WARNING: {n_zero_vol} bars with zero volume (kept — verify manually)")

        # --- 6. V-SPIKE DETECTION (intraday only) ---
        # A V-spike is a bar that shoots way up (or down) then immediately reverses.
        # Pattern: |r_t| >> 5σ AND r_t + r_{t+1} ≈ 0 (meaning it reversed next bar).
        # Only relevant for high-frequency data — daily bars don't get V-spikes.
        # Uses global shift so bar_idx maps correctly to c_df.iloc positions.
        # Overnight gaps won't be flagged as spikes because they don't immediately reverse.
        if interval in ['1m', '5m'] and len(c_df) > 10:
            _global_ret = np.log(c_df['Close'] / c_df['Close'].shift(1)).dropna()
            std = _global_ret.std()
            if std > 0:
                r_arr = _global_ret.values
                spike_idx = np.where(
                    (np.abs(r_arr) > 5.0 * std) &
                    (np.abs(np.roll(r_arr, -1) + r_arr) < std)
                )[0]
                if len(spike_idx) > 0:
                    # _global_ret[k] = log(close[k+1] / close[k]), so the spike BAR
                    # is at position k+1 in c_df, not k. Null that bar then ffill.
                    bar_idx = np.clip(spike_idx + 1, 0, len(c_df) - 1)
                    c_df.iloc[bar_idx] = np.nan
                    c_df = c_df.ffill()
                    issues.append(f"REPAIRED: {len(spike_idx)} V-Spikes")

        # --- 7. COMPUTE LOG RETURNS ---
        # Daily: close-to-close overnight return is the standard convention.
        # Intraday: compute within each trading day only. A global shift(1) across
        # the stitched DataFrame would include the overnight gap (last bar of day N
        # → first bar of day N+1), injecting large spurious outliers into the series.
        if interval == '1d':
            returns = np.log(c_df['Close'] / c_df['Close'].shift(1)).dropna()
        else:
            day_rets = []
            for _, group in c_df.groupby(c_df.index.date):
                if len(group) > 1:
                    day_rets.append(np.log(group['Close'] / group['Close'].shift(1)).dropna())
            returns = pd.concat(day_rets).sort_index() if day_rets else pd.Series(dtype=float)

        # --- BUILD REPORT ---
        # Always show counts for every check so you can see at a glance what was found,
        # even when nothing was wrong (e.g. "NaNs: 0 | Neg prices: 0 | ...").
        summary = (
            f"NaNs: {nan_count} | "
            f"Neg prices: {n_neg} | "
            f"Zero close: {n_zero_close} | "
            f"Bad OHLC: {n_ohlc_bad} | "
            f"Zero vol: {n_zero_vol}"
        )
        if issues or warnings:
            report = " | ".join(issues + warnings) + f" [{summary}]"
        else:
            report = f"✅ DATA CLEAN [{summary}]"
        return c_df, returns, report