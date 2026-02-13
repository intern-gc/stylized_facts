import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf


class AbsenceOfAutocorrelationTest:
    def __init__(self, returns: np.ndarray, ticker: str):
        self.ticker = ticker
        self.returns = returns

    def test_linear_independence(self, lags=40, plot=True):
        if self.returns is None or len(self.returns) < lags:
            print("❌ Error: Insufficient or missing returns data.")
            return []

            # Convert to numpy, flatten, and filter bad data (Defensive Programming)
        returns_array = np.asarray(self.returns).flatten()
        returns_array = returns_array[np.isfinite(returns_array)]

        if len(returns_array) < max(10, lags):
            print("❌ Error: Data too short after cleaning.")
            return []

        try:
            acf_values = acf(returns_array, nlags=lags, fft=True)
        except Exception as e:
            print(f"❌ Error calculating ACF: {e}")
            return []

        if plot:
            try:
                fig, ax = plt.subplots(figsize=(10, 5))
                plot_acf(returns_array, lags=lags, ax=ax, alpha=0.05)
                ax.set_title(f"Absence of Autocorrelation Test: {self.ticker}")
                ax.set_xlabel("Lag")
                ax.set_ylabel("Autocorrelation")
                plt.show()
            except Exception:
                pass

        return self._check_significance(acf_values, len(returns_array))

    def _check_significance(self, acf_values, n_samples):
        threshold = 1.96 / np.sqrt(n_samples)
        significant_lags = np.where(np.abs(acf_values[1:]) > threshold)[0] + 1

        print(f"--- Results for {self.ticker} ---")
        print(f"Sample size (T): {n_samples}")
        print(f"95% Confidence Band: +/- {threshold:.4f}")

        if len(significant_lags) == 0:
            print("✅ FACT CONFIRMED: No significant autocorrelations found.")
        else:
            print(f"❌ FACT VIOLATED: Significant correlation at lags: {significant_lags}")

        return significant_lags.tolist()
