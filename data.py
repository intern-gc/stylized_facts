import os
import pandas as pd
import numpy as np
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
        """Maps any string interval to Alpaca TimeFrame objects."""
        mapping = {
            '1m': TimeFrame.Minute,
            '5m': TimeFrame(5, TimeFrame.Minute),
            '1h': TimeFrame.Hour,
            '1d': TimeFrame.Day
        }
        return mapping.get(interval, TimeFrame.Minute)

    def get_data(self, ticker, start_date, end_date, interval):
        yesterday = (datetime.now() - timedelta(days=1)).date()
        req_start = datetime.strptime(start_date, '%Y-%m-%d').date()
        req_end = min(datetime.strptime(end_date, '%Y-%m-%d').date(), yesterday)

        business_days = pd.bdate_range(start=req_start, end=req_end)
        stitched_dfs = []

        print(f"\n🚀 [STARTING SYNC]: {ticker} | {interval} | {len(business_days)} potential days")

        for day in business_days:
            # Filename is unique to interval to prevent cache collisions
            file_name = f"{day.strftime('%y%m%d')}_{interval}_{ticker}.pkl"
            file_path = os.path.join(self.cache_dir, file_name)

            if os.path.exists(file_path):
                print(f"  💾 CACHE HIT: {day.strftime('%Y-%m-%d')}")
                day_df = pd.read_pickle(file_path)
            else:
                print(f"  🌐 API FETCH: {day.strftime('%Y-%m-%d')}...", end="\r")
                request_params = StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=self._get_alpaca_tf(interval),
                    start=datetime.combine(day, time(0, 0)),
                    end=datetime.combine(day, time(23, 59)),
                    feed="sip"
                )
                bars = self.client.get_stock_bars(request_params)

                if not bars.data:
                    print(f"  ⚠️  SKIP (HOLIDAY/NO DATA): {day.strftime('%Y-%m-%d')}     ")
                    continue

                day_df = bars.df
                if isinstance(day_df.index, pd.MultiIndex):
                    day_df = day_df.xs(ticker, level=0)

                day_df = day_df.rename(
                    columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                day_df.to_pickle(file_path)
                print(f"  ✅ API SUCCESS: {day.strftime('%Y-%m-%d')} ({len(day_df)} bars)")

            # Frequency-specific cleaning
            if not day_df.empty:
                if interval in ['1m', '5m']:
                    # Reindex for intraday micro-gaps only
                    freq_map = {'1m': '1min', '5m': '5min'}
                    full_range = pd.date_range(start=day_df.index.min(), end=day_df.index.max(),
                                               freq=freq_map[interval])
                    day_df = day_df.reindex(full_range).ffill()
                    day_df['Volume'] = day_df['Volume'].fillna(0)
                stitched_dfs.append(day_df)

        if not stitched_dfs:
            # FIXED: Must return 3 values to match unpacking in main.py
            return pd.DataFrame(), pd.Series(), "❌ NO DATA FOUND"

        full_df = pd.concat(stitched_dfs).sort_index()

        # Timezone Logic
        if full_df.index.tz is None:
            full_df.index = full_df.index.tz_localize('UTC')
        full_df = full_df.tz_convert('US/Eastern')

        # Filter Market Hours ONLY for intraday timeframes
        if interval != '1d':
            print(f"  🕒 Filtering for 09:30-16:00 EST...")
            full_df = full_df.between_time('09:30', '16:00')

        return self.audit_and_clean(full_df, interval)

    def audit_and_clean(self, df, interval):
        if df.empty: return df, pd.Series(), "❌ ERROR: No data"

        c_df = df.copy()
        # Centralized Log Return Calculation
        returns = np.log(c_df['Close'] / c_df['Close'].shift(1)).dropna()

        issues = []
        if interval in ['1m', '5m']:
            std = returns.std()
            # Repair V-Spikes using pre-calculated returns
            spike_idx = np.where((np.abs(returns) > 5.0 * std) &
                                 (np.abs(np.roll(returns, -1) + returns) < std))[0]
            if len(spike_idx) > 0:
                c_df.iloc[spike_idx] = np.nan
                c_df = c_df.ffill()
                # Recalculate returns only if a repair happened
                returns = np.log(c_df['Close'] / c_df['Close'].shift(1)).dropna()
                issues.append(f"REPAIRED: {len(spike_idx)} V-Spikes")

        report = "✅ DATA CLEAN" if not issues else " | ".join(issues)
        return c_df, returns, report