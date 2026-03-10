import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from concurrent.futures import ThreadPoolExecutor


class VolatilityClustering:
    """
    Calculates C1(τ) = corr(|r(t,Δt)|, |r(t+τ,Δt)|).

    Absolute returns used instead of squared: requires only finite 2nd moment
    (vs 4th for r²), robust to outliers, and Ding, Granger & Engle (1993)
    showed |r|^α maximises autocorrelation at α≈1.
    """

    def __init__(self, returns: np.ndarray, ticker: str):
        if returns is None:
            self.returns = np.array([])
        else:
            arr = np.asarray(returns).flatten().astype(float)
            self.returns = arr[np.isfinite(arr)]
        self.ticker = ticker
        self.abs_returns = np.abs(self.returns)

    @staticmethod
    def _eff_shuffles(n, n_shuffles):
        """Scale down shuffle count for large n: null CI converges quickly."""
        if n >= 200_000: return min(n_shuffles, 100)
        if n >= 50_000:  return min(n_shuffles, 200)
        if n >= 10_000:  return min(n_shuffles, 500)
        return n_shuffles

    def _compute_null_ci(self, max_lag, n_shuffles=1000):
        """Permutation null CI. Shuffles run in parallel threads (numpy releases GIL for FFT)."""
        eff = self._eff_shuffles(len(self.returns), n_shuffles)
        r = self.returns.copy()
        def _one(seed):
            return acf(np.abs(np.random.default_rng(seed).permutation(r)),
                       nlags=max_lag, fft=True)[1:]
        with ThreadPoolExecutor() as ex:
            null_acfs = np.array(list(ex.map(_one, range(eff))))
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
            c1_values = acf(self.abs_returns, nlags=max_lag, fft=True)
        except Exception as e:
            print(f"❌ Error calculating C1: {e}")
            return np.array([]), []

        eff = self._eff_shuffles(len(self.returns), n_shuffles)
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
        print(f"Sample size (T): {n} | Null CI: 95th pct of {eff} shuffles")

        if len(significant_lags) > 0:
            print(f"✅ FACT CONFIRMED: Significant C1 at {len(significant_lags)} lags.")
            print(f"   First 5 significant lags: {significant_lags[:5].tolist()}")
        else:
            print("❌ FACT VIOLATED: No significant volatility clustering detected.")

        return c1_values, significant_lags.tolist()
