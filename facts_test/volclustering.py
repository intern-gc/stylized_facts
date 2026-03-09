import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf


class VolatilityClustering:
    """
    Calculates C1(τ): the autocorrelation of absolute returns.

    C1(τ) = corr(|r(t,Δt)|, |r(t+τ,Δt)|)

    Positive C1(τ) at multiple lags indicates volatility clustering —
    large moves tend to be followed by large moves.

    Absolute returns are used instead of squared returns because:
      - Only requires finite 2nd moment (vs 4th for r², 8th for its ACF variance)
      - Robust to outliers — a single large move doesn't dominate
      - Ding, Granger & Engle (1993) showed |r|^α maximises autocorrelation at α≈1
    """

    def __init__(self, returns: np.ndarray, ticker: str):
        if returns is None:
            self.returns = np.array([])
        else:
            arr = np.asarray(returns).flatten().astype(float)
            self.returns = arr[np.isfinite(arr)]
        self.ticker = ticker
        self.abs_returns = np.abs(self.returns)

    def _compute_null_ci(self, max_lag, n_shuffles=1000):
        """
        Build a single flat null CI threshold by shuffling returns n_shuffles times.

        For each shuffle we destroy temporal order (i.i.d. null), take absolute
        value, and compute the ACF.  We return the 95th percentile of all
        |ACF_null| values (across all shuffles and all lags) — a single
        horizontal threshold that is easy to interpret on a bar chart.
        """
        null_acfs = np.empty((n_shuffles, max_lag))
        r = self.returns.copy()
        for i in range(n_shuffles):
            shuffled_abs = np.abs(np.random.permutation(r))
            null_acfs[i] = acf(shuffled_abs, nlags=max_lag, fft=True)[1:]
        return float(np.percentile(np.abs(null_acfs), 95))

    def compute_c1(self, max_lag=40, plot=True, n_shuffles=1000):
        """
        Compute C1(τ) for τ = 1, ..., max_lag.

        Significance is assessed against a permutation null: returns are
        shuffled n_shuffles times and the 95th percentile of |ACF_null| across
        all lags is used as a flat threshold (most extreme white-noise bound).

        Returns (c1_values array, list of significant lag indices).
        """
        if len(self.abs_returns) < max_lag:
            print("❌ Error: Insufficient data for volatility clustering.")
            return np.array([]), []

        try:
            c1_values = acf(
                self.abs_returns,
                nlags=max_lag,   # compute ACF at lags 0, 1, ..., max_lag (returns max_lag+1 values)
                fft=True,        # use FFT-based algorithm (Wiener-Khinchin theorem): O(n log n)
                                 # instead of computing each lag individually at O(n²) cost.
                                 # Equivalent result; much faster for large n or high max_lag.
            )
        except Exception as e:
            print(f"❌ Error calculating C1: {e}")
            return np.array([]), []

        null_ci = self._compute_null_ci(max_lag, n_shuffles)
        significant_lags = np.where(np.abs(c1_values[1:]) > null_ci)[0] + 1

        if plot:
            try:
                lags_axis = np.arange(1, max_lag + 1)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(lags_axis, c1_values[1:], color='steelblue', alpha=0.7)
                ax.axhline(y=null_ci, color='r', linestyle='--',
                           label=f'95% null CI (n={n_shuffles} shuffles)')
                ax.axhline(y=-null_ci, color='r', linestyle='--')
                ax.set_title(f"Volatility Clustering C1(τ) = ACF(|r|): {self.ticker}")
                ax.set_xlabel("Lag (τ)")
                ax.set_ylabel("C1(τ)")
                ax.legend()
                plt.show()
            except Exception:
                pass

        n = len(self.abs_returns)
        print(f"--- Volatility Clustering Results: {self.ticker} ---")
        print(f"Sample size (T): {n} | Null CI: 95th pct of {n_shuffles} shuffles")

        if len(significant_lags) > 0:
            print(f"✅ FACT CONFIRMED: Significant C1 at {len(significant_lags)} lags.")
            print(f"   First 5 significant lags: {significant_lags[:5].tolist()}")
        else:
            print("❌ FACT VIOLATED: No significant volatility clustering detected.")

        return c1_values, significant_lags.tolist()
