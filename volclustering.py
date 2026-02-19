import logging

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

logger = logging.getLogger(__name__)


class VolatilityClustering:
    """
    Calculates C2(τ): the autocorrelation of squared returns.

    C2(τ) = corr(r(t,Δt)², r(t+τ,Δt)²)

    Positive C2(τ) at multiple lags indicates volatility clustering —
    large returns tend to be followed by large returns.
    """

    def __init__(self, returns: np.ndarray, ticker: str):
        logger.warning(
            "Volatility clustering (C2) uses squared returns. When the tail index "
            "α ≤ 4, the fourth moment E[r⁴] is unstable or infinite, making C2 "
            "estimates unreliable."
        )
        if returns is None:
            self.returns = np.array([])
        else:
            arr = np.asarray(returns).flatten().astype(float)
            self.returns = arr[np.isfinite(arr)]
        self.ticker = ticker
        self.squared_returns = self.returns ** 2

    def compute_c2(self, max_lag=40, plot=True):
        """
        Compute C2(τ) for τ = 1, ..., max_lag.
        Returns (c2_values array, list of significant lag indices).
        """
        if len(self.squared_returns) < max_lag:
            print("❌ Error: Insufficient data for volatility clustering.")
            return np.array([]), []

        try:
            c2_values = acf(self.squared_returns, nlags=max_lag, fft=True)
        except Exception as e:
            print(f"❌ Error calculating C2: {e}")
            return np.array([]), []

        n = len(self.squared_returns)
        threshold = 1.96 / np.sqrt(n)
        significant_lags = np.where(np.abs(c2_values[1:]) > threshold)[0] + 1

        if plot:
            try:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(range(1, max_lag + 1), c2_values[1:], color='steelblue', alpha=0.7)
                ax.axhline(y=threshold, color='r', linestyle='--', label=f'95% CI (+{threshold:.4f})')
                ax.axhline(y=-threshold, color='r', linestyle='--', label=f'95% CI (-{threshold:.4f})')
                ax.set_title(f"Volatility Clustering C2(τ): {self.ticker}")
                ax.set_xlabel("Lag (τ)")
                ax.set_ylabel("C2(τ)")
                ax.legend()
                plt.show()
            except Exception:
                pass

        print(f"--- Volatility Clustering Results: {self.ticker} ---")
        print(f"Sample size (T): {n}")
        print(f"95% Confidence Band: +/- {threshold:.4f}")

        if len(significant_lags) > 0:
            print(f"✅ FACT CONFIRMED: Significant C2 at {len(significant_lags)} lags.")
            print(f"   First 5 significant lags: {significant_lags[:5].tolist()}")
        else:
            print("❌ FACT VIOLATED: No significant volatility clustering detected.")

        return c2_values, significant_lags.tolist()
