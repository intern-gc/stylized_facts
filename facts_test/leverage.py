import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor
_WORKERS = max(1, (os.cpu_count() or 4) // 2)


class LeverageEffect:
    """
    Tests for Leverage Effect (Stylized Fact 8):
    Measures the cross-correlation between returns and future absolute returns:

        L(tau) = Corr(r_t, |r_{t+tau}|)   for tau in [-max_lag, max_lag]

    The leverage effect is present when L(tau) starts negative for small positive tau
    and slowly decays toward zero. Captures the asymmetry: negative price moves
    predict higher future volatility more than positive moves do.

    Indicator: L(tau) < 0 for positive tau.
    """

    def __init__(self, returns, ticker: str):
        if returns is None:
            self.returns = np.array([])
        else:
            arr = np.asarray(returns).flatten().astype(float)
            self.returns = arr[np.isfinite(arr)]
        self.ticker = ticker

    @staticmethod
    def _eff_shuffles(n, n_shuffles):
        """Scale down shuffle count for large n: null CI converges quickly."""
        if n >= 500_000: return min(n_shuffles, 20)
        if n >= 200_000: return min(n_shuffles, 50)
        if n >= 50_000:  return min(n_shuffles, 200)
        if n >= 10_000:  return min(n_shuffles, 500)
        return n_shuffles

    def _cross_corr(self, x, y, max_lag):
        """
        Compute Corr(x_t, y_{t+tau}) for tau=1..max_lag using a single FFT pass.

        Replaces a per-lag corrcoef loop (O(max_lag * n)) with O(n log n).
        Uses global means/stds — approximation error is negligible for n >> max_lag.
        """
        n = len(x)
        x_c = x - x.mean()
        y_c = y - y.mean()
        std_x = x.std(ddof=0)
        std_y = y.std(ddof=0)
        if std_x == 0 or std_y == 0:
            return np.zeros(max_lag)
        fft_len = 1 << (2 * n - 1).bit_length()
        Fx = np.fft.rfft(x_c, n=fft_len)
        Fy = np.fft.rfft(y_c, n=fft_len)
        xcorr = np.fft.irfft(np.conj(Fx) * Fy, n=fft_len)
        return xcorr[1:max_lag + 1] / (n * std_x * std_y)

    def _compute_null_L(self, max_lag, n_shuffles=1000):
        """
        Build a single flat null lower bound for L(τ) by shuffling returns.

        Uses FFT-based cross-correlation to compute all lags in one pass (O(n log n)
        per shuffle) instead of a per-lag corrcoef loop (O(max_lag * n)).
        Auto-scales n_shuffles for large datasets where the null CI converges quickly.
        """
        r = self.returns.copy()
        eff = self._eff_shuffles(len(r), n_shuffles)
        def _one(seed):
            s = np.random.default_rng(seed).permutation(r)
            return self._cross_corr(s, np.abs(s), max_lag)
        with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
            null_L = np.array(list(ex.map(_one, range(eff))))
        return float(np.percentile(null_L, 5))

    def compute_leverage(self, max_lag=50, plot=True, n_shuffles=1000):
        """
        Compute L(tau) = Corr(r_t, |r_{t+tau}|) for tau = -max_lag .. +max_lag.

        Returns dict with:
          lags             : list of integers from -max_lag to +max_lag
          L_values         : list of cross-correlations (float or nan)
          leverage_detected: bool, True if L(tau) < 0 for any positive tau
          min_L            : float, the minimum L value across all finite lags
          min_lag          : int, the lag at which minimum occurs
        Returns None if data is insufficient.
        """
        n = len(self.returns)
        if n < max_lag * 2 + 10:
            print(f"  Insufficient data for leverage effect test.")
            return None

        r = self.returns
        r_abs = np.abs(r)
        lags = np.arange(-max_lag, max_lag + 1)
        L_values = []

        for tau in lags:
            if tau > 0:
                r_t = r[:-tau]
                r_sq_t = r_abs[tau:]
            elif tau < 0:
                abs_tau = -tau
                r_t = r[abs_tau:]
                r_sq_t = r_abs[:n - abs_tau]
            else:
                r_t = r
                r_sq_t = r_abs

            if len(r_t) < 10:
                L_values.append(float('nan'))
                continue

            corr_matrix = np.corrcoef(r_t, r_sq_t)
            corr = corr_matrix[0, 1]
            L_values.append(float(corr) if np.isfinite(corr) else float('nan'))

        L_arr = np.array(L_values)

        eff = self._eff_shuffles(len(self.returns), n_shuffles)
        null_lower = self._compute_null_L(max_lag, n_shuffles)

        pos_mask = lags > 0
        pos_L = L_arr[pos_mask]
        finite_pos = pos_L[np.isfinite(pos_L)]
        leverage_detected = bool(len(finite_pos) > 0 and np.any(finite_pos < null_lower))

        # Minimum over positive lags only (leverage = predictive, not contemporaneous)
        pos_lags_arr = lags[pos_mask]
        finite_pos_mask = np.isfinite(pos_L)
        if finite_pos_mask.any():
            min_idx = int(np.nanargmin(pos_L))
            min_L = float(pos_L[min_idx])
            min_lag = int(pos_lags_arr[min_idx])
        else:
            min_L = float('nan')
            min_lag = 0

        if plot:
            try:
                fig, ax = plt.subplots(figsize=(12, 5))
                finite = np.isfinite(L_arr)
                ax.plot(lags[finite], L_arr[finite], color='steelblue', linewidth=1.0, label='L(τ)')
                ax.axhline(y=null_lower, color='r', linestyle='--', linewidth=0.8,
                           label=f'5% null lower bound (n={n_shuffles})')
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
                ax.set_title(f"Leverage Effect L(τ) = Corr(r_t, |r_{{t+τ}}|): {self.ticker}")
                ax.set_xlabel("Lag τ")
                ax.set_ylabel("L(τ)")
                ax.legend()
                plt.tight_layout()
                plt.show()
            except Exception:
                pass

        print(f"--- Leverage Effect Results: {self.ticker} ---")
        print(f"Sample size: {len(self.returns)} | Null CI: 5th pct of {eff} shuffles")
        print(f"Lag range: [{-max_lag}, {max_lag}]")
        if np.isfinite(min_L):
            print(f"Min L(τ): {min_L:.4f} at τ={min_lag}")
        if leverage_detected:
            print(f"✅ FACT CONFIRMED: Leverage effect detected (L(τ) < 0 for positive τ).")
        else:
            print(f"❌ FACT NOT DETECTED: No negative L values for positive τ.")

        return {
            'lags': lags.tolist(),
            'L_values': L_arr.tolist(),
            'null_lower': null_lower,
            'leverage_detected': leverage_detected,
            'min_L': min_L,
            'min_lag': min_lag,
        }
